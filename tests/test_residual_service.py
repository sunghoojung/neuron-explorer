"""CPU-only residual group service tests with a deterministic fake runtime."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import explorer.residual_service as residual_service_module
from explorer.residual_service import (
    ResidualService,
    ResidualServiceError,
    validate_dataset,
)


def _dataset() -> dict:
    samples = []
    for split in ("train", "validation", "test"):
        family_id = f"family-{split}"
        pair_id = f"pair-{split}"
        for group in ("neutral", "pressure"):
            sample_id = f"{split}-{group}"
            samples.append({
                "id": sample_id,
                "pair_id": pair_id,
                "family_id": family_id,
                "split": split,
                "group": group,
                "messages": [{"role": "user", "content": f"Question for {split} {group}."}],
                "target": " The answer is blue.",
                "foil": " The answer is green.",
            })
    return {"name": "synthetic-fixture", "samples": samples}


class _FakeAdapter:
    model_id = "org/fake-llama"
    revision = "fixture-revision"
    cache_namespace = "fixture-residual-test"
    family = "llama"
    device = "cpu"
    config_metadata: ClassVar[dict[str, int]] = {"num_hidden_layers": 5, "hidden_size": 4}
    layers: ClassVar[list[object]] = [object() for _ in range(5)]

    @staticmethod
    def apply_chat_template(messages, *, add_generation_prompt=True):
        text = " ".join(message["content"] for message in messages)
        codes = [10 + (ord(char) % 70) for char in text[:20]]
        return [1, *codes, 2] if add_generation_prompt else [1, *codes]


class _FakeRuntime:
    def __init__(self):
        self.model_id = _FakeAdapter.model_id
        self.revision = _FakeAdapter.revision
        self.cache_namespace = _FakeAdapter.cache_namespace
        self.adapter = _FakeAdapter()
        self.model = object()
        self.dev = "cpu"
        self.calls = 0

    def ensure_loaded(self):
        self.calls += 1


def _fake_capture(rt, samples, layers, points, *, window=8, progress=None, cancel_event=None):
    n, hidden = len(samples), 4
    vectors = np.zeros((n, len(layers), len(points), window, hidden), dtype=np.float32)
    token_ids = np.full((n, window), -1, dtype=np.int64)
    positions = np.full((n, window), -1, dtype=np.int64)
    valid = np.zeros((n, window), dtype=bool)
    tokens = [[None] * window for _ in samples]
    lengths = []
    for index, sample in enumerate(samples):
        if progress:
            progress("tokenizing", index + 1, n)
        ids = rt.adapter.apply_chat_template(sample["messages"], add_generation_prompt=True)
        count = min(window, len(ids))
        start = window - count
        token_ids[index, start:] = ids[-count:]
        positions[index, start:] = np.arange(len(ids) - count, len(ids), dtype=np.int64)
        valid[index, start:] = True
        tokens[index][start:] = [f"token-{token_id}" for token_id in ids[-count:]]
        lengths.append(len(ids))
        group_offset = 1.0 if sample["group"] == "pressure" else 0.0
        pair_offset = 0.0 if sample["split"] == "train" else 0.25 if sample["split"] == "validation" else 0.5
        vector = np.asarray([pair_offset, group_offset, 1.0 + pair_offset, 0.5 * group_offset], dtype=np.float32)
        vectors[index, :, :, :, :] = vector
        if progress:
            progress("capturing", index + 1, n)
    return {
        "vectors": vectors,
        "token_ids": token_ids,
        "positions": positions,
        "valid": valid,
        "tokens": tokens,
        "lengths": lengths,
        "metadata": {
            "schema_version": 1,
            "model_id": rt.adapter.model_id,
            "revision": rt.adapter.revision,
            "family": rt.adapter.family,
            "device": rt.adapter.device,
            "hidden_size": hidden,
            "layers": list(layers),
            "points": list(points),
            "window": window,
            "capture_dtype": "float32",
            "model_activation_dtypes": ["torch.float32"],
            "template_sha256": "a" * 64,
            "tokenization": "fake fixture",
            "token_position": "right-aligned final prompt tokens",
            "causal_context": "prompt only",
            "record_count": n,
            "estimated_capture_bytes": vectors.nbytes,
        },
    }


@pytest.fixture
def service_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(residual_service_module, "capture_residuals", _fake_capture)
    score_calls = []

    def fake_score(rt, sample, layer, point, basis=None, alpha=0.0, score_mode="sequence_logprob", continuation=None):
        score_calls.append({"alpha": alpha, "score_mode": score_mode, "basis_rank": 0 if basis is None else basis.shape[1]})
        assert score_mode == "mean_token_logprob"
        word_count = len(continuation.strip().split())
        score = (-0.3 if continuation.strip().endswith("blue.") else -0.6) + alpha * 0.01
        return {"score": score, "token_count": word_count, "score_mode": score_mode}

    monkeypatch.setattr(residual_service_module, "score_residual_continuation", fake_score)

    def create(root: Path | None = None):
        return ResidualService(_FakeRuntime(), root or tmp_path)

    return create, score_calls


def _wait(service: ResidualService, job_id: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def _capture(service: ResidualService) -> tuple[str, dict]:
    job = service.start_capture({"dataset": _dataset(), "layers": [1], "points": ["resid_pre"], "window": 4})
    completed = _wait(service, job["job_id"])
    assert completed["status"] == "completed", completed
    assert completed["integrity_status"] == "verified"
    return completed["result"]["capture_id"], completed


def test_dataset_validation_rejects_family_split_leakage_and_duplicate_pair_group():
    dataset = _dataset()
    valid = validate_dataset(dataset)
    assert len(valid["samples"]) == 6

    leaked = _dataset()
    leaked["samples"][2]["family_id"] = leaked["samples"][0]["family_id"]
    with pytest.raises(ResidualServiceError, match="family") as error:
        validate_dataset(leaked)
    assert error.value.code == "split_leakage"

    duplicated = _dataset()
    duplicated["samples"][1]["group"] = duplicated["samples"][0]["group"]
    with pytest.raises(ResidualServiceError) as duplicate_error:
        validate_dataset(duplicated)
    assert duplicate_error.value.code == "duplicate_pair_group"


def test_capture_summary_and_verified_history(service_factory, tmp_path):
    create, _ = service_factory
    service = create(tmp_path)
    try:
        capture_id, completed = _capture(service)
        summary = service.get_capture_summary(capture_id, layer=1, point="resid_pre", group_a="neutral", group_b="pressure")
        assert summary["selected"] == {"layer": 1, "point": "resid_pre"}
        assert len(summary["groups"]) == 2
        assert summary["pairs"][0]["n_pairs"] == 3
        assert all(row["tokens"][-1]["relative_position"] == 0 for row in summary["samples"])
        assert all(row["has_candidates"] for row in summary["samples"])
        assert service.list_captures()["captures"][0]["capture_id"] == capture_id
        assert completed["result"] == {"capture_id": capture_id}
    finally:
        service.close()


def test_train_only_paired_delta_fit_and_candidate_probe(service_factory, tmp_path):
    create, score_calls = service_factory
    service = create(tmp_path)
    try:
        capture_id, _ = _capture(service)
        fit_job = service.start_fit({
            "capture_id": capture_id,
            "layer": 1,
            "point": "resid_pre",
            "mode": "paired_delta",
            "group_a": "neutral",
            "group_b": "pressure",
            "rows": 2,
            "cols": 2,
            "epochs": 2,
            "seed": 7,
        })
        fit_done = _wait(service, fit_job["job_id"])
        assert fit_done["status"] == "completed", fit_done
        fit_id = fit_done["result"]["fit_id"]
        fit = service.get_fit(fit_id)
        assert fit["fit"]["mode"] == "paired_delta"
        assert fit["training_samples"] == 2
        assert fit["training_units"] == 1
        assert fit["training_pairs"] == 1
        assert fit["training_families"] == 1
        assert fit["basis_rank"] >= 1
        assert fit["projection_note"].startswith("Projection uses the global span")
        assert sum(cell["split_counts"].get("test", 0) for cell in fit["cells"]) == 1
        cell = service.get_fit_cell(fit_id, 0)
        assert "samples" in cell

        probe_job = service.start_probe({"fit_id": fit_id, "split": "test", "alpha": 0.75, "max_samples": 4})
        probe_done = _wait(service, probe_job["job_id"])
        assert probe_done["status"] == "completed", probe_done
        probe = service.get_probe(probe_done["job_id"])
        assert probe["score_mode"] == "mean_token_logprob"
        assert probe["selection"]["selected_samples"] == 2
        assert probe["zero_equivalence"]["all_samples_within_1e-6"]
        assert probe["interpretation"].startswith("Candidate log-prob preference only")
        assert probe["samples"][0]["token_counts"]["target_token_count"] > 0
        assert any(call["alpha"] == -0.75 for call in score_calls)
        assert service.list_fits()["fits"][0]["fit_id"] == fit_id
    finally:
        service.close()


def test_state_fit_cannot_start_candidate_probe(service_factory, tmp_path):
    create, _ = service_factory
    service = create(tmp_path)
    try:
        capture_id, _ = _capture(service)
        fit_job = service.start_fit({"capture_id": capture_id, "layer": 1, "point": "resid_pre", "mode": "states", "rows": 1, "cols": 1, "epochs": 1})
        fit_done = _wait(service, fit_job["job_id"])
        assert fit_done["status"] == "completed", fit_done
        with pytest.raises(ResidualServiceError) as error:
            service.start_probe({"fit_id": fit_done["result"]["fit_id"]})
        assert error.value.code == "probe_requires_pairs"
    finally:
        service.close()


def test_legacy_paired_fit_counts_are_derived_without_claiming_receipt_verification(service_factory, tmp_path):
    create, _ = service_factory
    service = create(tmp_path)
    try:
        capture_id, _ = _capture(service)
        fit_job = service.start_fit({
            "capture_id": capture_id, "layer": 1, "point": "resid_pre", "mode": "paired_delta",
            "group_a": "neutral", "group_b": "pressure", "rows": 1, "cols": 1, "epochs": 1,
        })
        fit_done = _wait(service, fit_job["job_id"])
        assert fit_done["status"] == "completed", fit_done
        fit_id = fit_done["result"]["fit_id"]
        root = tmp_path / "research_outputs" / "residual_groups" / _FakeAdapter.cache_namespace / fit_id
        fit_path = root / "fit.json"
        old_metadata = json.loads(fit_path.read_text(encoding="utf-8"))
        for key in ("training_units", "training_pairs", "training_data_kind", "quantization_unit"):
            old_metadata.pop(key, None)
        old_metadata["training_samples"] = 1  # earlier paired maps counted fitted deltas here
        fit_path.write_text(json.dumps(old_metadata), encoding="utf-8")
        job_path = root / "job.json"
        job_record = json.loads(job_path.read_text(encoding="utf-8"))
        job_record.pop("artifact_sha256", None)
        job_path.write_text(json.dumps(job_record), encoding="utf-8")

        fit = service.get_fit(fit_id)
        assert fit["training_units"] == 1
        assert fit["training_pairs"] == 1
        assert fit["training_samples"] == 2
        assert fit["integrity_status"] == "structure_only"
    finally:
        service.close()


def test_tampered_capture_fails_hash_check_and_is_not_listed_as_verified(service_factory, tmp_path):
    create, _ = service_factory
    service = create(tmp_path)
    try:
        capture_id, _ = _capture(service)
        capture_path = tmp_path / "research_outputs" / "residual_groups" / _FakeAdapter.cache_namespace / capture_id / "capture.npz"
        capture_path.write_bytes(capture_path.read_bytes() + b"changed")
        with pytest.raises(ResidualServiceError) as error:
            service.get_capture_summary(capture_id)
        assert error.value.code == "integrity_failed"
        job = next(item for item in service.list_jobs()["jobs"] if item["job_id"] == capture_id)
        assert job["integrity_status"] == "invalid"
    finally:
        service.close()


def test_cancelled_capture_does_not_publish_a_result_or_verified_artifact(tmp_path, monkeypatch):
    entered_capture = threading.Event()
    release_capture = threading.Event()

    def blocking_capture(*args, **kwargs):
        entered_capture.set()
        assert release_capture.wait(5)
        return _fake_capture(*args, **kwargs)

    monkeypatch.setattr(residual_service_module, "capture_residuals", blocking_capture)
    service = ResidualService(_FakeRuntime(), tmp_path)
    try:
        job = service.start_capture({"dataset": _dataset(), "layers": [1], "points": ["resid_pre"], "window": 4})
        assert entered_capture.wait(2)
        service.cancel_job(job["job_id"])
        release_capture.set()
        cancelled = _wait(service, job["job_id"])
        assert cancelled["status"] == "cancelled"
        assert cancelled.get("result") is None
        assert cancelled.get("integrity_status") == "incomplete"
        assert service.list_captures()["captures"] == []
    finally:
        release_capture.set()
        service.close()


def test_second_service_does_not_recover_a_live_process_job(tmp_path, monkeypatch):
    entered_capture = threading.Event()
    release_capture = threading.Event()

    def blocking_capture(*args, **kwargs):
        entered_capture.set()
        assert release_capture.wait(5)
        return _fake_capture(*args, **kwargs)

    monkeypatch.setattr(residual_service_module, "capture_residuals", blocking_capture)
    first = ResidualService(_FakeRuntime(), tmp_path)
    second = None
    try:
        job = first.start_capture({"dataset": _dataset(), "layers": [1], "points": ["resid_pre"], "window": 4})
        assert entered_capture.wait(2)
        second = ResidualService(_FakeRuntime(), tmp_path)
        still_running = second.get_job(job["job_id"])
        assert still_running["status"] == "running"
        assert still_running.get("error_code") != "interrupted"
        release_capture.set()
        completed = _wait(first, job["job_id"])
        assert completed["status"] == "completed", completed
    finally:
        release_capture.set()
        first.close()
        if second is not None:
            second.close()


def test_restart_marks_persisted_running_job_interrupted(tmp_path):
    runtime = _FakeRuntime()
    service = ResidualService(runtime, tmp_path)
    service.close()
    job_id = "1234567890abcdef1234567890abcdef"
    directory = tmp_path / "research_outputs" / "residual_groups" / _FakeAdapter.cache_namespace / job_id
    directory.mkdir(parents=True)
    (directory / "job.json").write_text(json.dumps({
        "schema_version": 1,
        "job_id": job_id,
        "kind": "capture",
        "status": "running",
        "stage": "capturing",
        "progress": {"completed": 2, "total": 6},
        "created_at": "2026-10-26T00:00:00Z",
        "updated_at": "2026-10-26T00:00:00Z",
    }), encoding="utf-8")
    recovered = ResidualService(runtime, tmp_path)
    try:
        job = recovered.get_job(job_id)
        assert job["status"] == "interrupted"
        assert job["error_code"] == "interrupted"
        assert recovered.list_jobs()["jobs"][0]["status"] == "interrupted"
    finally:
        recovered.close()
