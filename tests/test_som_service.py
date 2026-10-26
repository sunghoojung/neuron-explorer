"""Local SOM service tests using generated fixtures and no model weights."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from explorer.som_service import (
    MAX_PILOT_QUESTION_IDS,
    SOMService,
    SOMServiceError,
)
from research.data import (
    ActivationData,
    load_activations,
    prompt_dataset_sha256,
    save_activations,
)
from research.som import make_artifact, save_artifact
from research.som_pipeline import DEFAULT_HOOK, pressure_deltas, train_som

CONDITIONS = ("neutral", "false_pressure", "true_claim", "valid_correction")


def _write_prompts(path: Path, question_count: int = 3) -> list[dict[str, str]]:
    rows = []
    splits = ("train", "validation", "test")
    for index in range(question_count):
        question_id = f"fixture-q{index:03d}"
        split = splits[index % len(splits)]
        for condition in CONDITIONS:
            rows.append({
                "question_id": question_id,
                "split": split,
                "condition": condition,
                "prompt": f"Fixture prompt {index} under {condition}.",
                "target": " yes",
                "foil": " no",
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def _service(tmp_path: Path, prompts: Path) -> SOMService:
    return SOMService(
        run_root=tmp_path / "runs",
        source_roots=[prompts.parent],
        default_model="org/fixture-model",
        default_revision="fixture-revision",
    )


def _configured_source(service: SOMService) -> str:
    sources = service.list_sources()["sources"]
    return next(source["source_id"] for source in sources if source["dataset_kind"] == "configured_local_input")


def _fake_pipeline_command(service: SOMService, rows: list[dict[str, str]]):
    def run(job_id, stage, argv, job_dir, record, cancel_event, started):
        output = Path(argv[argv.index("--out") + 1])
        if stage == "collect":
            base = np.arange(len(rows), dtype=np.float32)
            offsets = {name: float(index) for index, name in enumerate(CONDITIONS)}
            vectors = np.stack([
                np.asarray([base[index] + offsets[row["condition"]], 0.2 * base[index] + 1.5 * offsets[row["condition"]], 1.0])
                for index, row in enumerate(rows)
            ])
            data = ActivationData(
                vectors=vectors,
                question_ids=np.asarray([row["question_id"] for row in rows]),
                splits=np.asarray([row["split"] for row in rows]),
                conditions=np.asarray([row["condition"] for row in rows]),
                last_token_ids=np.arange(len(rows), dtype=np.int64),
                prompt_token_counts=np.full(len(rows), 4, dtype=np.int64),
                metadata={
                    "schema_version": 1,
                    "model_id": record["model_id"],
                    "revision": record["revision_requested"],
                    "layer": record["layer"],
                    "hook": record["hook"],
                    "dataset_sha256": prompt_dataset_sha256(rows),
                    "model_activation_dtype": "float32",
                },
            )
            save_activations(output, data)
        elif stage == "fit":
            activation_path = Path(argv[argv.index("--activations") + 1])
            data = load_activations(activation_path)
            train_ids, deltas = pressure_deltas(data)
            fit = record["fit"]
            prototypes = train_som(
                deltas,
                rows=fit["rows"],
                cols=fit["cols"],
                epochs=fit["epochs"],
                seed=fit["seed"],
            )
            artifact = make_artifact(
                prototypes,
                model_id=record["model_id"],
                revision=record["revision"],
                layer=record["layer"],
                hook=record["hook"],
                grid=(fit["rows"], fit["cols"]),
                seed=fit["seed"],
                question_count=len(train_ids),
                training_question_ids=train_ids,
                activation_sha256=hashlib.sha256(activation_path.read_bytes()).hexdigest(),
            )
            save_artifact(output, artifact)
        else:
            evaluation_questions = sorted({row["question_id"] for row in rows if row["split"] == record["evaluation"]["split"]})
            methods = ["baseline", "som_remove", "global_mean_remove", "pca_remove", "random_remove"]
            selected = evaluation_questions[: record["evaluation"]["max_questions"]]
            records = [
                {
                    "question_id": question_id,
                    "split": record["evaluation"]["split"],
                    "condition": condition,
                    "method": method,
                    "target_logprob": -0.5,
                    "foil_logprob": -1.5,
                    "margin": 1.0,
                    "target_tokens": 1,
                    "foil_tokens": 1,
                }
                for question_id in selected
                for condition in CONDITIONS
                for method in methods
            ]
            summary = {
                "conditions": {
                    method: {condition: {"mean_target_minus_foil": 1.0} for condition in CONDITIONS}
                    for method in methods
                },
                "false_pressure_effect": {
                    method: {"n": len(selected), "mean_false_pressure_minus_neutral_margin": -0.25}
                    for method in methods
                },
                "truthful_correction_guardrails": {
                    method: {
                        condition: {
                            "mean_margin_change_vs_baseline": 0.0,
                            "preference_rate_change_vs_baseline": 0.0,
                            "regression_detected": False,
                        }
                        for condition in ("true_claim", "valid_correction")
                    }
                    for method in methods
                    if method != "baseline"
                },
            }
            output.write_text(json.dumps({
                "schema_version": 1,
                "model_id": record["model_id"],
                "revision": record["revision"],
                "layer": record["layer"],
                "hook": record["hook"],
                "split": record["evaluation"]["split"],
                "question_count": len(selected),
                "methods": methods,
                "basis_ranks": {"som": 2, "global_mean": 1, "pca": 2, "random": 2},
                "score_mode": "sequence_logprob",
                "score_summary": summary,
                "scored_records": records,
            }), encoding="utf-8")

    return run


def test_bundled_source_is_labeled_and_artifact_roots_are_not_exposed(tmp_path):
    service = SOMService(run_root=tmp_path / "runs")
    bundled = [source for source in service.list_sources()["sources"] if source["dataset_kind"] == "bundled_example"]
    assert len(bundled) == 1
    assert bundled[0]["label"] == "Bundled example prompts"
    assert bundled[0]["records"] == 24
    assert "path" not in bundled[0]
    status = service.status()
    assert "run_root" not in status
    assert "source_roots" not in status
    service.close()


def test_external_job_summary_is_historical_and_manifest_paths_are_not_returned(tmp_path, monkeypatch):
    manifest = tmp_path / "private-local-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "external_jobs": [{
            "job_id": "123456_0",
            "label": "Teacher-forced SOM shard 0",
            "recorded_status": "PENDING",
            "recorded_as_of": "2026-10-24T22:17:00Z",
            "source_ref": "PR #20",
            "source_url": "https://github.com/example/project/pull/20",
            "artifact_status": "not_available",
            "experiment_schema": "example-teacher-map-v1",
            "model_label": "example-model",
            "layers": [1, 2],
            "hook": "example.layers.{layer}.attention.output",
            "pipeline_compatible": False,
            "snapshot_root": "/must/not/be-returned",
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("EXPLORER_SOM_MANIFEST", str(manifest))
    service = SOMService(run_root=tmp_path / "runs")

    job = service.get_job("123456_0")
    assert job["status"] == "unverified"
    assert job["recorded_status"] == "PENDING"
    assert job["recorded_as_of"] == "2026-10-24T22:17:00Z"
    assert job["artifact_status"] == "not_available"
    assert job["pipeline_compatible"] is False
    assert "snapshot_root" not in job
    assert "/must/not/be-returned" not in json.dumps(service.status())
    service.close()


def test_default_local_manifest_persists_source_roots_but_explicit_empty_roots_win(tmp_path):
    repo_root = tmp_path / "repo"
    local_sources = tmp_path / "authorized-local-inputs"
    source_file = local_sources / "paired.jsonl"
    _write_prompts(source_file)
    manifest_dir = repo_root / "explorer"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "som_sources.local.json").write_text(json.dumps({
        "schema_version": 1,
        "source_roots": [str(local_sources)],
        "external_jobs": [],
    }), encoding="utf-8")

    configured = SOMService(repo_root=repo_root, run_root=tmp_path / "configured-runs")
    assert any(source["dataset_kind"] == "configured_local_input" for source in configured.list_sources()["sources"])
    configured.close()

    explicit_empty = SOMService(
        repo_root=repo_root,
        run_root=tmp_path / "empty-runs",
        source_roots=[],
    )
    assert not any(source["dataset_kind"] == "configured_local_input" for source in explicit_empty.list_sources()["sources"])
    explicit_empty.close()


def test_invalid_local_manifest_is_reported_without_exposing_its_path(tmp_path):
    repo_root = tmp_path / "repo"
    manifest_dir = repo_root / "explorer"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "som_sources.local.json").write_text("not-json", encoding="utf-8")
    service = SOMService(repo_root=repo_root, run_root=tmp_path / "runs")

    status = service.status()
    assert status["status"] == "degraded"
    assert "configuration_warning" in status
    assert str(repo_root) not in json.dumps(status)
    service.close()


def test_existing_artifact_requires_active_model_revision_and_hook_match(tmp_path):
    source_root = tmp_path / "external-artifacts"
    artifact_path = source_root / "som.npz"
    artifact_path.parent.mkdir()
    artifact = make_artifact(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        model_id="org/other-model",
        revision="revision-a",
        layer=3,
        hook=DEFAULT_HOOK.format(layer=3),
        grid=(1, 2),
        seed=4,
        question_count=1,
        training_question_ids=["fixture-q001"],
        activation_sha256="a" * 64,
    )
    save_artifact(artifact_path, artifact)
    service = SOMService(
        run_root=tmp_path / "runs",
        source_roots=[source_root],
        default_model="org/current-model",
        default_revision="revision-b",
    )

    artifacts = service.list_artifacts()["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["status"] == "valid"
    assert artifacts[0]["model_revision_hook_match"] is False
    assert "path" not in artifacts[0]
    inspected = service.get_artifact(artifacts[0]["artifact_id"])
    assert inspected["model_id"] == "org/other-model"
    assert inspected["revision"] == "revision-a"
    assert "prototypes" not in inspected
    service.close()


def test_bounded_pilot_runs_collect_fit_evaluate_with_output_validation(tmp_path, monkeypatch):
    prompts_path = tmp_path / "input" / "paired.jsonl"
    rows = _write_prompts(prompts_path)
    service = _service(tmp_path, prompts_path)
    monkeypatch.setattr(service, "_run_command", _fake_pipeline_command(service, rows))

    job = service.start_pilot(
        source_id=_configured_source(service),
        model_id="org/fixture-model",
        revision="fixture-revision",
        layer=1,
        rows=2,
        cols=2,
        epochs=3,
        split="test",
        max_questions=2,
    )
    service._futures[job["job_id"]].result(timeout=5)
    completed = service.get_job(job["job_id"])
    assert completed["status"] == "completed", completed
    assert completed["integrity_status"] == "verified"
    assert completed["result_validity"] == "valid"
    assert completed["stage"] == "complete"
    assert completed["dataset_kind"] == "configured_local_input"
    assert completed["scientific_interpretation"] == "descriptive_diagnostics_only"
    assert set(completed["artifact_kinds"]) == {"activations", "som", "evaluation"}
    assert completed["evaluation_summary"]["question_count"] == 1
    assert completed["evaluation_summary"]["score_mode"] == "sequence_logprob"
    assert set(completed["evaluation_summary"]["score_summary"]) == {
        "conditions", "false_pressure_effect", "truthful_correction_guardrails",
    }
    assert "scored_records" not in json.dumps(completed["evaluation_summary"])
    assert "prompt_dataset_sha256" not in completed
    job_record_path = service.run_root / job["job_id"] / "job.json"
    stored = json.loads(job_record_path.read_text(encoding="utf-8"))
    assert set(stored["artifact_sha256"]) == {"activations.npz", "som.npz", "evaluation.json"}

    stored.pop("artifact_sha256")
    stored.pop("evaluation_summary")
    job_record_path.write_text(json.dumps(stored), encoding="utf-8")
    legacy_record_bytes = job_record_path.read_bytes()
    legacy = service.get_job(job["job_id"])
    assert legacy["status"] == "completed"
    assert legacy["integrity_status"] == "unverified"
    assert legacy["evaluation_summary"]["question_count"] == 1
    assert job_record_path.read_bytes() == legacy_record_bytes
    service.close()


def test_changed_completed_output_is_flagged_invalid_and_summary_is_withheld(tmp_path, monkeypatch):
    prompts_path = tmp_path / "input" / "paired.jsonl"
    rows = _write_prompts(prompts_path)
    service = _service(tmp_path, prompts_path)
    monkeypatch.setattr(service, "_run_command", _fake_pipeline_command(service, rows))
    job = service.start_pilot(
        source_id=_configured_source(service),
        model_id="org/fixture-model",
        revision="fixture-revision",
        layer=1,
        rows=2,
        cols=2,
        epochs=3,
        split="test",
        max_questions=2,
    )
    service._futures[job["job_id"]].result(timeout=5)

    evaluation_path = service.run_root / job["job_id"] / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["score_summary"]["conditions"]["baseline"]["neutral"]["mean_target_minus_foil"] = -99.0
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    result = service.get_job(job["job_id"])
    assert result["status"] == "invalid"
    assert result["integrity_status"] == "invalid"
    assert result["error_code"] == "integrity_failed"
    assert "evaluation_summary" not in result
    service.close()


def test_structurally_incomplete_evaluation_cannot_complete_job(tmp_path, monkeypatch):
    prompts_path = tmp_path / "input" / "paired.jsonl"
    rows = _write_prompts(prompts_path)
    service = _service(tmp_path, prompts_path)
    valid_runner = _fake_pipeline_command(service, rows)

    def incomplete_runner(job_id, stage, argv, job_dir, record, cancel_event, started):
        valid_runner(job_id, stage, argv, job_dir, record, cancel_event, started)
        if stage == "evaluate":
            output = Path(argv[argv.index("--out") + 1])
            result = json.loads(output.read_text(encoding="utf-8"))
            result["scored_records"].pop()
            output.write_text(json.dumps(result), encoding="utf-8")

    monkeypatch.setattr(service, "_run_command", incomplete_runner)
    job = service.start_pilot(
        source_id=_configured_source(service),
        model_id="org/fixture-model",
        revision="fixture-revision",
        layer=1,
        rows=2,
        cols=2,
        epochs=3,
        split="test",
        max_questions=2,
    )
    service._futures[job["job_id"]].result(timeout=5)
    failed = service.get_job(job["job_id"])
    assert failed["status"] == "failed"
    assert failed["error_code"] == "invalid_artifact"
    service.close()


def test_pilot_rejects_more_than_fixed_question_bound_and_hook_mismatch(tmp_path):
    prompts_path = tmp_path / "input" / "large.jsonl"
    _write_prompts(prompts_path, question_count=MAX_PILOT_QUESTION_IDS + 1)
    service = _service(tmp_path, prompts_path)
    source_id = _configured_source(service)

    with pytest.raises(SOMServiceError, match="complete questions"):
        service.start_pilot(source_id=source_id, layer=2, split="test")
    small_path = tmp_path / "small" / "paired.jsonl"
    _write_prompts(small_path)
    small_service = _service(tmp_path / "small-service", small_path)
    with pytest.raises(SOMServiceError, match="hook"):
        small_service.start_pilot(source_id=_configured_source(small_service), layer=2, hook="blocks.2.hook_resid_post")
    service.close()
    small_service.close()


def test_pending_queue_is_bounded(tmp_path, monkeypatch):
    prompts_path = tmp_path / "input" / "paired.jsonl"
    _write_prompts(prompts_path)
    service = _service(tmp_path, prompts_path)
    entered = threading.Event()
    release = threading.Event()

    def blocking_command(job_id, stage, argv, job_dir, record, cancel_event, started):
        entered.set()
        release.wait(timeout=5)
        service._check_cancel(cancel_event)
        raise SOMServiceError("stage_failed", "fixture failure", 500)

    monkeypatch.setattr(service, "_run_command", blocking_command)
    source_id = _configured_source(service)
    first = service.start_pilot(source_id=source_id, layer=1)
    assert entered.wait(timeout=3)
    second = service.start_pilot(source_id=source_id, layer=1)
    with pytest.raises(SOMServiceError) as error:
        service.start_pilot(source_id=source_id, layer=1)
    assert error.value.code == "capacity"
    release.set()
    service._futures[first["job_id"]].result(timeout=5)
    service._futures[second["job_id"]].result(timeout=5)
    service.close()


def test_service_startup_marks_orphaned_jobs_interrupted(tmp_path):
    job_id = "a" * 32
    job_dir = tmp_path / "runs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({
        "schema_version": 1,
        "job_id": job_id,
        "kind": "pilot",
        "status": "evaluating",
        "stage": "evaluate",
        "progress": {"stage_index": 3, "stage_count": 3, "current": 0, "total": 4},
        "created_at": "2026-10-26T00:00:00Z",
        "updated_at": "2026-10-26T00:01:00Z",
    }), encoding="utf-8")

    service = SOMService(run_root=tmp_path / "runs")
    job = service.get_job(job_id)
    assert job["status"] == "interrupted"
    assert job["error_code"] == "interrupted"
    assert "restarted" in job["error"]
    service.close()
