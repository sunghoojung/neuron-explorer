"""Local SOM service tests using generated fixtures and no model weights."""

from __future__ import annotations

import base64
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


@pytest.fixture(autouse=True)
def _isolate_local_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "synthetic-som-sources.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "source_roots": [],
        "native_som_roots": [],
        "external_jobs": [],
    }), encoding="utf-8")
    monkeypatch.setenv("EXPLORER_SOM_MANIFEST", str(manifest))


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
        native_som_roots=[],
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


def _native_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _native_digest(path)


def _write_native_fixture(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_root = tmp_path / "fixture-snapshot"
    run_root = tmp_path / "fixture-native-run"
    config = {
        "schema": "som-teacher-forced-pairs-v1",
        "model": "fixture-model-key",
        "layers": [1],
        "views": ["answer_mean"],
        "readouts": ["pre_answer"],
        "world_size": 1,
    }
    manifest = {"schema": "som-teacher-forced-inputs-v1", "fixture": True}
    recipe = {
        "schema": "attention-router-native-recipe-v1",
        "model": {"model_id": "org/fixture-model", "revision": "fixture-revision"},
        "tokenizer_revision": "fixture-tokenizer-revision",
    }
    tokens = {"schema": "som-teacher-token-preflight-v1", "fixture": True}
    behavior_config = {"schema": "fixture-behavior-config-v1"}
    behavior_plan = {"schema": "fixture-behavior-plan-v1"}
    inputs = {
        "configs/som_teacher_forced_v1.json": config,
        "datasets/som-teacher-forced-v1/MANIFEST.json": manifest,
        "datasets/som-teacher-forced-v1/recipe.json": recipe,
        "preflight/native-tokens.json": tokens,
        "configs/som_teacher_behavior_v1.json": behavior_config,
        "datasets/som-teacher-behavior-v1/PLAN.json": behavior_plan,
    }
    extra_snapshot_file = snapshot_root / "src" / "fixture_module.py"
    extra_snapshot_file.parent.mkdir(parents=True)
    extra_snapshot_file.write_text("FIXTURE_SENTINEL = True\n", encoding="utf-8")
    snapshot_hashes = {
        name: _native_json(snapshot_root / name, payload)
        for name, payload in inputs.items()
    }
    snapshot_hashes["src/fixture_module.py"] = _native_digest(extra_snapshot_file)
    _native_json(snapshot_root / "SNAPSHOT.json", {
        "schema": "attention-router-runtime-v1",
        "experiment_schema": "som-teacher-forced-pairs-v1",
        "exact_copied_bytes_authoritative": True,
        "source_git_sha": "1" * 40,
        "files": snapshot_hashes,
    })
    binding = {
        "bundle_sha256": snapshot_hashes["datasets/som-teacher-forced-v1/MANIFEST.json"],
        "snapshot_sha256": _native_digest(snapshot_root / "SNAPSHOT.json"),
        "token_audit_sha256": snapshot_hashes["preflight/native-tokens.json"],
    }

    map_name = "fixture-map"
    map_dir = run_root / "maps" / map_name
    map_dir.mkdir(parents=True)
    map_npz = map_dir / "map.npz"
    np.savez_compressed(
        map_npz,
        prototypes=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        cells=np.asarray([0, 0], dtype=np.int64),
        fit=np.asarray([True, False], dtype=np.bool_),
        errors=np.asarray([0.1, 0.2], dtype=np.float32),
    )
    png = map_dir / "map.png"
    png.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/aWQAAAAASUVORK5CYII="
        )
    )
    (map_dir / "worker.lock").write_bytes(b"")
    map_summary = {
        "binding": binding,
        "task": {"kind": "som", "name": map_name, "view": "answer_mean", "layer": 1, "side": 1, "seed": 7},
        "causal_steering_tested": False,
        "direction_sign": "good minus bad",
        "states": 2,
        "fit_states": 1,
        "candidate_direction_geometry": {
            "vectors": 1,
            "energy_fractions": [1.0],
            "rank90": 1,
            "participation_rank": 1.0,
        },
        "condition_geometry": {
            "vectors": 1,
            "energy_fractions": [1.0],
            "rank90": 1,
            "participation_rank": 1.0,
        },
        "associations": {
            "fixture_source/MAP_FIT": {
                "states": 1,
                "occupied_cells": 1,
                "quantization_error": 0.1,
                "condition_nmi": 0.5,
            },
        },
        "fit_history": [1.0, 0.5],
    }
    summary_hash = _native_json(map_dir / "summary.json", map_summary)
    map_files = {
        "map.npz": _native_digest(map_npz),
        "map.png": _native_digest(png),
        "summary.json": summary_hash,
        "worker.lock": _native_digest(map_dir / "worker.lock"),
    }
    _native_json(map_dir / "complete.json", {"binding": binding, "task": map_summary["task"], "files": map_files})

    (run_root / "maps").mkdir(parents=True, exist_ok=True)
    report_path = run_root / "report.html"
    report_path.write_text(
        f'<html><body><img src="maps/{map_name}/map.png"></body></html>',
        encoding="utf-8",
    )
    stability_hash = _native_json(run_root / "stability.json", [
        {"left": "fixture-a", "right": "fixture-b", "project_adjusted_rand": 0.5},
    ])
    geometry = {
        "binding": binding,
        "pairs": 1,
        "answer_sequences": 2,
        "maps": 1,
        "capture_workers": 1,
        "conditions": {"fixture_condition": 1},
        "mock": False,
        "teacher_forced": True,
        "base_weights_updated": False,
        "scientific_success_claim": False,
        "report_sha256": _native_digest(report_path),
        "stability_sha256": stability_hash,
    }
    geometry_hash = _native_json(run_root / "geometry-complete.json", geometry)

    banks_dir = run_root / "behavior" / "banks"
    bank_reports = {}
    bank_files = {}
    for name, count in (("global", 1), ("kmeans", 1), ("som", 1)):
        bank_path = banks_dir / f"{name}.npz"
        banks_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bank_path,
            center=np.zeros(4, dtype=np.float32),
            scale=np.ones(4, dtype=np.float32),
            pca_mean=np.zeros(4, dtype=np.float32),
            pca_components=np.ones((1, 4), dtype=np.float32),
            coef=np.ones((count, 1), dtype=np.float32),
            bias=np.zeros(count, dtype=np.float32),
            vectors=np.ones((count, 4), dtype=np.float32),
            activation_rms=np.asarray(1.0, dtype=np.float32),
        )
        bank_files[f"{name}.npz"] = _native_digest(bank_path)
        report = {
            "method": name,
            "direction": "good minus bad at shared suffix",
            "routing_input": "independent prompt-only pass before either answer",
            "training_families": 1,
            "training_pairs": 1,
            "global_direction_norm": 1.0,
            "mean_individual_delta_norm": 1.0,
        }
        if name != "global":
            report.update({
                "cells_with_local_estimates": 1,
                "fit_route_accuracy": 0.5,
                "fit_majority_route_accuracy": 0.5,
                "project_route_accuracy": 0.5,
                "project_majority_route_accuracy": 0.5,
                "project_by_condition": {
                    "fixture_condition": {"n": 1, "route_matches": 0, "predicted_occupancy": [1]},
                },
            })
        bank_reports[name] = report
    bank_hash = _native_json(banks_dir / "complete.json", {
        "binding": binding,
        "config_sha256": snapshot_hashes["configs/som_teacher_behavior_v1.json"],
        "plan_sha256": snapshot_hashes["datasets/som-teacher-behavior-v1/PLAN.json"],
        "files": bank_files,
        "bank_reports": bank_reports,
    })

    behavior_dir = run_root / "behavior"
    behavior_files = {
        "blinded-review.json": {"fixture": True},
        "paired-screens.json": {
            "condition_case_counts": {"fixture_condition": 1},
            "records": [{"case_id": "fixture-case", "semantic_grade": None}],
        },
        "strict-screen.json": {
            "scope": "conservative lexical screen only",
            "cells": {
                "EVALUATE/fixture_condition/base": {
                    "n": 1,
                    "strict_matches": 0,
                    "needs_semantic_review": 1,
                    "capped": 0,
                },
            },
        },
        "unblind.json": {"fixture": True},
    }
    behavior_file_hashes = {
        name: _native_json(behavior_dir / name, payload)
        for name, payload in behavior_files.items()
    }
    behavior = {
        "binding": {
            **binding,
            "banks_sha256": bank_hash,
            "behavior_plan_sha256": snapshot_hashes["datasets/som-teacher-behavior-v1/PLAN.json"],
        },
        "cases": 1,
        "arms": 1,
        "answers": 1,
        "mock": False,
        "semantic_review": "pending",
        "mitigation_established": False,
        "hypothesis_confirmed": False,
        "all_strengths_prespecified": True,
        "files": behavior_file_hashes,
    }
    behavior_hash = _native_json(behavior_dir / "complete.json", behavior)
    complete = {
        "binding": binding,
        "teacher_forced": True,
        "maps": 1,
        "free_generation_answers": 1,
        "scientific_success_claim": False,
        "mock": False,
        "geometry_sha256": geometry_hash,
        "behavior_sha256": behavior_hash,
    }
    complete_hash = _native_json(run_root / "complete.json", complete)

    capture_root = run_root / "capture" / "rank0"
    capture_root.mkdir(parents=True)
    chunk_path = capture_root / "chunk-000000.npz"
    np.savez_compressed(chunk_path, fixture=np.asarray([1], dtype=np.int32))
    identity_hash = _native_json(capture_root / "identity.json", {
        **binding,
        "rank": 0,
        "world_size": 1,
        "teacher_forced": True,
        "zero_logit_equivalence": True,
        "layers": [1],
        "readouts": ["pre_answer"],
        "capture_policy": "fixture prompt-only state",
        "provenance": {
            "model_id": "org/fixture-model",
            "revision": "fixture-revision",
            "tokenizer_revision": "fixture-tokenizer-revision",
            "checkpoint_sha256": {"fixture.safetensors": "a" * 64},
            "implementation_sha256": {"fixture_capture.py": "b" * 64},
            "metadata_sha256": {"config.json": "c" * 64},
            "loading_info": {
                "error_msgs": [],
                "mismatched_keys": [],
                "missing_keys": [],
                "unexpected_keys": [],
            },
            "layout": [{"layer": 1, "projection_module": "model.layers.1.self_attn.o_proj"}],
        },
    })
    qualification_hash = _native_json(capture_root / "native-qualification.json", {
        "binding": binding,
        "causality_tolerance": 0.01,
        "read_only_logits_exact": True,
        "records": [{"fixture": True}],
        "router_state": "fixture prompt only; no supplied continuation",
    })
    _native_json(capture_root / "complete.json", {
        "binding": binding,
        "rank": 0,
        "weights_unchanged": True,
        "pairs": 1,
        "files": {
            "chunk-000000.npz": _native_digest(chunk_path),
            "identity.json": identity_hash,
            "native-qualification.json": qualification_hash,
        },
    })

    behavior_rank_dir = run_root / "behavior" / "rank0"
    behavior_rank_dir.mkdir(parents=True, exist_ok=True)
    answers_path = behavior_rank_dir / "answers.jsonl"
    answers_path.write_text('{"fixture":true}\n', encoding="utf-8")
    qualification_path = behavior_rank_dir / "qualification.json"
    qualification_path.write_text('{"fixture":true}\n', encoding="utf-8")
    behavior_rank_hash = _native_json(behavior_rank_dir / "complete.json", {
        "binding": binding,
        "rank": 0,
        "weights_unchanged": True,
        "answers": 1,
        "files": {
            "answers.jsonl": _native_digest(answers_path),
            "qualification.json": _native_digest(qualification_path),
        },
    })
    capture_receipt_path = capture_root / "complete.json"
    map_receipt_path = map_dir / "complete.json"

    stage_specials = {
        "capture": {},
        "analyze": {
            "../../behavior/banks/complete.json": bank_hash,
            "../../geometry-complete.json": geometry_hash,
        },
        "generate": {
            "../../behavior/complete.json": behavior_hash,
            "../../complete.json": complete_hash,
            "../../report.html": _native_digest(report_path),
        },
    }
    for stage, specials in stage_specials.items():
        payloads = {
            "capture": {"capture/rank0/complete.json": _native_digest(capture_receipt_path)},
            "analyze": {f"maps/{map_name}/complete.json": _native_digest(map_receipt_path)},
            "generate": {"behavior/rank0/complete.json": behavior_rank_hash},
        }[stage]
        rank_record = {
            "binding": binding,
            "phase": stage,
            "rank": 0,
            "world_size": 1,
            "mock": False,
            "payloads": payloads,
        }
        rank_hash = _native_json(run_root / "stages" / stage / "rank0.json", rank_record)
        stage_files = {"rank0.json": rank_hash, **specials}
        _native_json(run_root / "stages" / stage / "complete.json", {
            "binding": binding,
            "phase": stage,
            "world_size": 1,
            "mock": False,
            "files": stage_files,
        })
    return run_root, snapshot_root


def test_bundled_source_is_labeled_and_artifact_roots_are_not_exposed(tmp_path):
    service = SOMService(run_root=tmp_path / "runs", native_som_roots=[])
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
    service = SOMService(run_root=tmp_path / "runs", native_som_roots=[])

    job = service.get_job("123456_0")
    assert job["status"] == "unverified"
    assert job["recorded_status"] == "PENDING"
    assert job["recorded_as_of"] == "2026-10-24T22:17:00Z"
    assert job["artifact_status"] == "not_available"
    assert job["pipeline_compatible"] is False
    assert "snapshot_root" not in job
    assert "/must/not/be-returned" not in json.dumps(service.status())
    service.close()


def test_default_local_manifest_persists_source_roots_but_explicit_empty_roots_win(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    monkeypatch.delenv("EXPLORER_SOM_MANIFEST", raising=False)
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

    configured = SOMService(repo_root=repo_root, run_root=tmp_path / "configured-runs", native_som_roots=[])
    assert any(source["dataset_kind"] == "configured_local_input" for source in configured.list_sources()["sources"])
    configured.close()

    explicit_empty = SOMService(
        repo_root=repo_root,
        run_root=tmp_path / "empty-runs",
        source_roots=[],
        native_som_roots=[],
    )
    assert not any(source["dataset_kind"] == "configured_local_input" for source in explicit_empty.list_sources()["sources"])
    explicit_empty.close()


def test_invalid_local_manifest_is_reported_without_exposing_its_path(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    monkeypatch.delenv("EXPLORER_SOM_MANIFEST", raising=False)
    manifest_dir = repo_root / "explorer"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "som_sources.local.json").write_text("not-json", encoding="utf-8")
    service = SOMService(repo_root=repo_root, run_root=tmp_path / "runs", native_som_roots=[])

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
        native_som_roots=[],
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

    service = SOMService(run_root=tmp_path / "runs", native_som_roots=[])
    job = service.get_job(job_id)
    assert job["status"] == "interrupted"
    assert job["error_code"] == "interrupted"
    assert "restarted" in job["error"]
    service.close()


def test_native_som_report_exposes_receipt_bound_summaries_and_safe_report_assets(tmp_path):
    output_root, snapshot_root = _write_native_fixture(tmp_path)
    service = SOMService(
        run_root=tmp_path / "local-runs",
        source_roots=[],
        native_som_roots=[{"output_root": str(output_root), "snapshot_root": str(snapshot_root)}],
        default_model="org/fixture-model",
        default_revision="fixture-revision",
    )

    catalog = service.list_native_reports()["native_reports"]
    assert len(catalog) == 1
    assert catalog[0]["status"] == "completed"
    assert catalog[0]["integrity_status"] == "verified"
    assert catalog[0]["capture_provenance_status"] == "verified"
    assert catalog[0]["map_count"] == 1
    assert catalog[0]["report_available"] is True
    assert "output_root" not in json.dumps(catalog)
    report = service.get_native_report(catalog[0]["native_id"])
    assert report["provenance"]["model_id"] == "org/fixture-model"
    assert report["hook_template"] == "model.layers.{layer}.self_attn.o_proj"
    assert report["feature_space"] == "attention_projection_output"
    assert report["model_revision_hook_match"] is False
    assert report["reuse_compatibility"] == "not_assessed"
    assert report["maps"][0]["map_name"] == "fixture-map"
    assert report["maps"][0]["map_shape"] == [1, 1, 4]
    assert report["behavior"]["semantic_grade_count"] == 0
    assert report["behavior"]["banks"][0]["vector_count"] == 1
    serialized = json.dumps(report)
    assert "prototypes" not in serialized
    assert "scored_records" not in serialized
    assert str(output_root) not in serialized
    assert service.get_native_report_file(catalog[0]["native_id"]) == output_root / "report.html"
    assert service.get_native_report_asset(catalog[0]["native_id"], "maps/fixture-map/map.png") == output_root / "maps/fixture-map/map.png"
    assert service.get_native_report_asset(catalog[0]["native_id"], "maps/../report.html") is None
    assert service.status()["native_reports"][0]["native_id"] == catalog[0]["native_id"]
    service.close()


def test_native_som_tamper_is_invalid_and_missing_capture_is_only_partial(tmp_path):
    output_root, snapshot_root = _write_native_fixture(tmp_path)
    chunk = output_root / "capture" / "rank0" / "chunk-000000.npz"
    chunk.unlink()
    service = SOMService(
        run_root=tmp_path / "local-runs",
        source_roots=[],
        native_som_roots=[{"output_root": str(output_root), "snapshot_root": str(snapshot_root)}],
    )
    native_id = service.list_native_reports()["native_reports"][0]["native_id"]
    partial = service.get_native_report(native_id)
    assert partial["status"] == "completed"
    assert partial["integrity_status"] == "outputs_verified"
    assert partial["capture_provenance_status"] == "pending"
    assert partial["report_available"] is True
    service.close()

    (output_root / "maps" / "fixture-map" / "map.png").write_bytes(b"modified")
    invalid_service = SOMService(
        run_root=tmp_path / "invalid-local-runs",
        source_roots=[],
        native_som_roots=[{"output_root": str(output_root), "snapshot_root": str(snapshot_root)}],
    )
    invalid = invalid_service.list_native_reports()["native_reports"][0]
    assert invalid["status"] == "invalid"
    assert invalid["integrity_status"] == "invalid"
    assert invalid.get("report_url") is None
    invalid_service.close()


def test_native_snapshot_hashes_every_indexed_file(tmp_path):
    output_root, snapshot_root = _write_native_fixture(tmp_path)
    (snapshot_root / "src" / "fixture_module.py").write_text("FIXTURE_SENTINEL = False\n", encoding="utf-8")
    service = SOMService(
        run_root=tmp_path / "local-runs",
        source_roots=[],
        native_som_roots=[{"output_root": str(output_root), "snapshot_root": str(snapshot_root)}],
    )

    report = service.list_native_reports()["native_reports"][0]
    assert report["status"] == "invalid"
    assert report["integrity_status"] == "invalid"
    service.close()


def test_native_stage_payload_digest_must_match_the_referenced_map_receipt(tmp_path):
    output_root, snapshot_root = _write_native_fixture(tmp_path)
    analyze_rank_path = output_root / "stages" / "analyze" / "rank0.json"
    analyze_rank = json.loads(analyze_rank_path.read_text(encoding="utf-8"))
    analyze_rank["payloads"]["maps/fixture-map/complete.json"] = "0" * 64
    analyze_rank_digest = _native_json(analyze_rank_path, analyze_rank)
    stage_complete_path = output_root / "stages" / "analyze" / "complete.json"
    stage_complete = json.loads(stage_complete_path.read_text(encoding="utf-8"))
    stage_complete["files"]["rank0.json"] = analyze_rank_digest
    _native_json(stage_complete_path, stage_complete)
    service = SOMService(
        run_root=tmp_path / "local-runs",
        source_roots=[],
        native_som_roots=[{"output_root": str(output_root), "snapshot_root": str(snapshot_root)}],
    )

    report = service.list_native_reports()["native_reports"][0]
    assert report["status"] == "invalid"
    assert report["integrity_status"] == "invalid"
    service.close()


def test_native_html_allows_only_local_report_assets_and_safe_image_data(tmp_path):
    png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/aWQAAAAASUVORK5CYII="
    safe_path = tmp_path / "safe.html"
    safe_path.write_text(
        '<style>body { color: #222; }</style>'
        '<img src="maps/fixture-map/map.png">'
        '<a href="behavior/strict-screen.json">summary</a>'
        f'<img src="data:image/png;base64,{png_data}">',
        encoding="utf-8",
    )
    assert SOMService._native_html_is_safe(safe_path, {"fixture-map"}) is True

    unsafe_html = (
        '<style>@import url("https://example.test/style.css");</style>',
        '<style>body { background: url(https://example.test/image.png); }</style>',
        '<a href="data:image/png;base64,' + png_data + '">remote data anchor</a>',
        '<a href="https://example.test/">external link</a>',
        '<img src="https://example.test/image.png">',
    )
    for index, contents in enumerate(unsafe_html):
        path = tmp_path / f"unsafe-{index}.html"
        path.write_text(contents, encoding="utf-8")
        assert SOMService._native_html_is_safe(path, {"fixture-map"}) is False
