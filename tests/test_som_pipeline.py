from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from research.data import (
    ActivationData,
    DataError,
    load_activations,
    load_prompt_jsonl,
    prompt_dataset_sha256,
    validate_activation_match,
    validate_prompt_rows,
)
from research.som import (
    ArtifactError,
    SOMArtifact,
    load_artifact,
    make_artifact,
    orthonormal_basis,
    project,
    save_artifact,
    train_som,
    validate_artifact,
)
from research.som_pipeline import _projection_hook, evaluate, main, run_synthetic_smoke


def _small_rows() -> list[dict[str, str]]:
    return [
        {
            "question_id": qid,
            "split": split,
            "condition": condition,
            "prompt": f"Question for {qid} under {condition}? Answer:",
            "target": " correct",
            "foil": " incorrect",
        }
        for qid, split in (("train-a", "train"), ("val-a", "validation"), ("test-a", "test"))
        for condition in ("neutral", "false_pressure", "true_claim", "valid_correction")
    ]


def test_projection_is_orthogonal_and_rank_safe() -> None:
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    basis = orthonormal_basis(vectors.T)
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6)

    values = np.asarray([[2.0, 3.0, 4.0], [-1.0, 2.0, 5.0]], dtype=np.float32)
    retained = project(values, basis, mode="keep")
    removed = project(values, basis, mode="remove")
    np.testing.assert_allclose(retained + removed, values, atol=1e-6)
    np.testing.assert_allclose(project(removed, basis, mode="keep"), 0, atol=1e-6)
    np.testing.assert_allclose(project(retained, basis, mode="remove"), 0, atol=1e-6)

    empty = orthonormal_basis(np.zeros((3, 5), dtype=np.float32))
    assert empty.shape == (3, 0)
    np.testing.assert_array_equal(project(values, empty, mode="remove"), values)
    np.testing.assert_array_equal(project(values, empty, mode="keep"), np.zeros_like(values))


def test_projection_hook_preserves_half_dtype_without_float64_tensors(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    block = torch.nn.Identity()
    basis = np.asarray([[1.0], [0.0]], dtype=np.float32)
    cast_dtypes = []
    original_as_tensor = torch.as_tensor

    def record_as_tensor(*args, **kwargs):
        cast_dtypes.append(kwargs.get("dtype"))
        return original_as_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", record_as_tensor)
    hook = _projection_hook(block, basis, "remove")
    try:
        values = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], dtype=torch.float16)
        result = block(values)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"
        assert cast_dtypes == [torch.float32]
        assert result.dtype != torch.float64
        torch.testing.assert_close(result, torch.tensor([[[0.0, 3.0], [0.0, 5.0]]], dtype=torch.float16))
    finally:
        hook.remove()


def test_half_projection_is_finite_and_nearly_idempotent() -> None:
    torch = pytest.importorskip("torch")
    block = torch.nn.Identity()
    basis = np.asarray([[2**-0.5], [2**-0.5], [0.0]], dtype=np.float32)
    hook = _projection_hook(block, basis, "remove")
    try:
        source = torch.tensor([[[1.25, -2.0, 0.5], [4.0, 3.5, -1.0]]], dtype=torch.float16)
        once = block(source)
        twice = block(once)
        assert bool(torch.isfinite(once).all())
        assert once.dtype == source.dtype
        torch.testing.assert_close(twice, once, atol=5e-3, rtol=5e-3)
    finally:
        hook.remove()


def test_som_training_is_deterministic() -> None:
    rng = np.random.default_rng(51)
    samples = rng.normal(size=(17, 9)).astype(np.float32)
    first = train_som(samples, rows=3, cols=4, epochs=25, seed=7)
    second = train_som(samples, rows=3, cols=4, epochs=25, seed=7)
    other_seed = train_som(samples, rows=3, cols=4, epochs=25, seed=8)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_seed)


def test_prompt_schema_rejects_question_split_overlap() -> None:
    rows = _small_rows()
    rows.append({**rows[0], "split": "test", "condition": "true_claim"})
    with pytest.raises(DataError, match="appears in both train and test"):
        validate_prompt_rows(rows)


def test_prompt_schema_requires_all_conditions_and_consistent_targets() -> None:
    rows = _small_rows()
    rows = [row for row in rows if not (row["question_id"] == "train-a" and row["condition"] == "valid_correction")]
    with pytest.raises(DataError, match="missing conditions valid_correction"):
        validate_prompt_rows(rows)

    rows = _small_rows()
    rows[1] = {**rows[1], "target": " other"}
    with pytest.raises(DataError, match="inconsistent target or foil"):
        validate_prompt_rows(rows)


def test_prompt_hash_ignores_row_order_but_covers_prompt_and_targets() -> None:
    rows = _small_rows()
    assert prompt_dataset_sha256(rows) == prompt_dataset_sha256(list(reversed(rows)))
    changed = [dict(row) for row in rows]
    changed[0]["prompt"] += " changed"
    assert prompt_dataset_sha256(rows) != prompt_dataset_sha256(changed)


def test_artifact_roundtrip_and_orthonormal_validation(tmp_path) -> None:
    prototypes = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    artifact = make_artifact(
        prototypes,
        model_id="test/model",
        revision="abc123",
        layer=4,
        hook="decoder.layers.4.output",
        grid=(1, 2),
        seed=4,
        question_count=2,
        training_question_ids=["train-a", "train-b"],
        activation_sha256="a" * 64,
    )
    path = tmp_path / "som.npz"
    save_artifact(path, artifact)
    loaded = load_artifact(path)
    np.testing.assert_array_equal(loaded.prototypes, artifact.prototypes)
    np.testing.assert_array_equal(loaded.basis, artifact.basis)
    assert loaded.metadata["training_question_id_hashes"] == sorted(
        hashlib.sha256(qid.encode()).hexdigest() for qid in ("train-a", "train-b")
    )

    broken = SOMArtifact(artifact.prototypes, np.ones_like(artifact.basis), artifact.metadata)
    with pytest.raises(ArtifactError, match="not orthonormal"):
        validate_artifact(broken)


def test_artifact_rejects_training_question_hash_mismatch() -> None:
    artifact = make_artifact(
        np.eye(3, dtype=np.float32),
        model_id="test/model",
        revision="abc123",
        layer=0,
        hook="decoder.layers.0.output",
        grid=(1, 3),
        seed=0,
        question_count=1,
        training_question_ids=["train-a"],
        activation_sha256="b" * 64,
    )
    invalid_metadata = dict(artifact.metadata)
    invalid_metadata["training_question_id_hashes"] = []
    with pytest.raises(ArtifactError, match="training question hashes"):
        validate_artifact(SOMArtifact(artifact.prototypes, artifact.basis, invalid_metadata))


def test_activation_file_rejects_duplicate_question_condition(tmp_path) -> None:
    rows = _small_rows()
    vectors = np.ones((len(rows), 4), dtype=np.float32)
    data = ActivationData(
        vectors=vectors,
        question_ids=np.asarray([r["question_id"] for r in rows], dtype=str),
        splits=np.asarray([r["split"] for r in rows], dtype=str),
        conditions=np.asarray([r["condition"] for r in rows], dtype=str),
        last_token_ids=np.zeros(len(rows), dtype=np.int64),
        prompt_token_counts=np.ones(len(rows), dtype=np.int64),
        metadata={"model_id": "test/model"},
    )
    path = tmp_path / "activations.npz"
    from research.data import save_activations

    save_activations(path, data)
    loaded = load_activations(path)
    validate_activation_match(rows, loaded)
    duplicate = replace(loaded, question_ids=np.concatenate([loaded.question_ids[:-1], loaded.question_ids[:1]]))
    # A duplicated key is caught by the loader's persistent-file validation.
    from research.data import save_activations

    duplicate_path = tmp_path / "duplicate.npz"
    save_activations(duplicate_path, duplicate)
    with pytest.raises(DataError, match="duplicate question_id/condition"):
        load_activations(duplicate_path)


def test_synthetic_smoke_runs_without_model_weights(tmp_path) -> None:
    result = run_synthetic_smoke(tmp_path / "smoke")
    assert result["model_loaded"] is False
    assert result["prompt_records"] == 60
    assert (tmp_path / "smoke" / "synthetic_som.npz").exists()
    report = json.loads((tmp_path / "smoke" / "synthetic_evaluation.json").read_text())
    assert report["scientific_claim"].startswith("none;")
    assert load_artifact(tmp_path / "smoke" / "synthetic_som.npz").metadata["training_question_count"] == 8


def test_cli_dry_run_preflights_without_loading_model(tmp_path, capsys) -> None:
    example = "research/examples/paired_prompts.jsonl"
    assert main(["collect", "--prompts", example, "--out", str(tmp_path / "acts.npz"), "--layer", "10", "--dry-run"]) == 0
    assert 'status: "dry_run"' in capsys.readouterr().out

    run_synthetic_smoke(tmp_path / "smoke")
    root = tmp_path / "smoke"
    assert main([
        "fit", "--activations", str(root / "synthetic_activations.npz"),
        "--out", str(root / "planned.npz"), "--dry-run",
    ]) == 0
    assert "artifact_written: false" in capsys.readouterr().out
    assert main([
        "evaluate", "--prompts", str(root / "synthetic_prompts.jsonl"),
        "--activations", str(root / "synthetic_activations.npz"),
        "--artifact", str(root / "synthetic_som.npz"),
        "--out", str(root / "planned-evaluation.json"), "--split", "test", "--dry-run",
    ]) == 0
    dry_run_output = capsys.readouterr().out
    assert 'status: "dry_run"' in dry_run_output
    assert "model_loaded: false" in dry_run_output
    assert not (root / "planned-evaluation.json").exists()


def test_evaluate_rejects_prompt_identity_and_relabelled_train_leak_before_model_load(tmp_path) -> None:
    run_synthetic_smoke(tmp_path / "base")
    prompts_path = tmp_path / "base" / "synthetic_prompts.jsonl"
    activation_path = tmp_path / "base" / "synthetic_activations.npz"
    artifact_path = tmp_path / "base" / "synthetic_som.npz"
    rows = load_prompt_jsonl(prompts_path)
    activations = load_activations(activation_path)
    artifact = load_artifact(artifact_path)

    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["prompt"] += " altered"
    with pytest.raises(ValueError, match="dataset content does not match"):
        evaluate(
            changed_rows, activations, artifact, split="test", device="cpu",
            score_mode="sequence_logprob", model_override=None, revision_override=None,
        )

    renamed_rows = [dict(row) for row in rows]
    id_map = {f"train-{index:02d}": f"renamed-train-{index:02d}" for index in range(8)}
    for row in renamed_rows:
        if row["question_id"] in id_map:
            row["question_id"] = id_map[row["question_id"]]
        elif row["question_id"] == "test-00":
            row["question_id"] = "train-00"
    renamed_vectors = activations.question_ids.copy()
    for old, new in id_map.items():
        renamed_vectors[renamed_vectors == old] = new
    renamed_vectors[renamed_vectors == "test-00"] = "train-00"
    renamed_activations = replace(
        activations,
        question_ids=renamed_vectors,
        metadata={**activations.metadata, "dataset_sha256": prompt_dataset_sha256(renamed_rows)},
    )
    with pytest.raises(DataError, match="overlaps SOM training questions"):
        evaluate(
            renamed_rows, renamed_activations, artifact, split="test", device="cpu",
            score_mode="sequence_logprob", model_override=None, revision_override=None,
        )
