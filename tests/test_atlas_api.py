"""HTTP boundaries for read-only, receipt-verified atlas artifacts."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from explorer.atlas_api import create_atlas_router

pytestmark = pytest.mark.asyncio


def _atlas(atlas_id: str = "fixture-atlas") -> dict:
    groups = [{
        "id": "false_factual_agreement",
        "label": "False factual agreement",
        "referent": "verifiable_position",
        "explicitness": "explicit",
        "is_control": False,
        "description": "Accepting an objectively incorrect user claim without evidence.",
        "color": "#d97878",
    }]
    cells = []
    for y in range(6):
        for x in range(6):
            cells.append({
                "id": f"{x}-{y}", "x": x - 0.5 * (y % 2), "y": y * 0.8660254037844386,
                "train_count": 1 if (x, y) == (0, 0) else 0,
                "eval_count": 0,
                "train_groups": {groups[0]["id"]: 1} if (x, y) == (0, 0) else {},
                "eval_groups": {},
                "mean_distance": 0.0,
            })
    return {
        "schema": "taxonomy-som-atlas-v1",
        "id": atlas_id,
        "metadata": {"model": "fixture/model", "revision": "fixture-revision", "source_status": "TFonly"},
        "groups": groups,
        "maps": [{
            "id": "L6-answer_mean-seed0", "layer": 6, "view": "answer_mean", "seed": 0,
            "rows": 6, "cols": 6, "cells": cells,
            "points": [{
                "id": "sequence-1", "group_id": groups[0]["id"], "family_id": "family-1",
                "split": "train", "cell": "0-0", "x": 0, "y": 0,
                "stance": "approval_seeking_script", "pair_id": "pair-1",
            }],
        }],
        "examples": {
            "sequence-1": {
                "pair_id": "pair-1", "group_id": groups[0]["id"], "family_id": "family-1",
                "split": "train", "stance": "approval_seeking", "stance_role": "approval_seeking_script",
                "prompt_messages": [{"role": "user", "content": "A fixture question."}],
                "target_text": "Fixture response.", "semantic_status": "synthetic fixture",
            },
        },
        "coverage": {groups[0]["id"]: {"train": 1, "eval": 0}},
        "caveats": ["Fixture only."],
        "source_urls": [],
    }


def _transfer_atlas(atlas_id: str = "transfer-fixture") -> dict:
    groups = [
        {
            "id": "explicit_position",
            "label": "Explicit position",
            "is_control": False,
            "is_source_cohort": False,
            "semantic_status": "synthetic_unreviewed",
        },
        {
            "id": "legitimate_correction",
            "label": "Legitimate correction",
            "is_control": True,
            "is_source_cohort": False,
            "semantic_status": "synthetic_unreviewed",
        },
    ]
    map_views = {
        "prompt_routing": ["pre_answer_prompt"],
        "output_contrast": [
            "answer_mean_independent_minus_candidate",
            "shared_suffix_independent_minus_candidate",
        ],
    }
    coordinates = [
        {"cell_id": index, "x": (index % 4) - 0.5 * ((index // 4) % 2), "y": (index // 4) * 0.8660254037844386}
        for index in range(16)
    ]
    maps = []
    for layer in (3, 6, 9, 13, 16, 20, 23, 26):
        for map_kind, views in map_views.items():
            for view in views:
                route = map_kind == "prompt_routing"
                cells_by_split = {
                    split: [3, 0, *([0] * 14)]
                    for split in ("train", "validation", "heldout")
                }
                synthetic_by_split = {
                    split: [2, *([0] * 15)]
                    for split in ("train", "validation", "heldout")
                }
                source_by_split = {
                    split: [1, *([0] * 15)]
                    for split in ("train", "validation", "heldout")
                }
                maps.append({
                    "id": f"layer_{layer}_{map_kind}_{view}",
                    "layer": layer,
                    "map_kind": map_kind,
                    "view": view,
                    "rows": 4,
                    "columns": 4,
                    "topology": "hexagonal",
                    "fit_split": "train",
                    "used_for_routing": route,
                    "codebook_ref": {"artifact_id": "som_artifacts", "array_name": f"layer_{layer}_prompt_weights"},
                    "bmu_ref": {"artifact_id": "map_points", "field_name": "cell_id"},
                    "cell_coordinates": coordinates,
                    "cell_summaries": cells_by_split,
                    "synthetic_cell_summaries": synthetic_by_split,
                    "source_cell_summaries": source_by_split,
                    "group_counts": {
                        split: [
                            {"group_id": "explicit_position", "counts": [1, *([0] * 15)]},
                            {"group_id": "legitimate_correction", "counts": [1, *([0] * 15)]},
                        ]
                        for split in ("train", "validation", "heldout")
                    },
                    "source_group_counts": {
                        split: [
                            {"source_collection": dataset, "counts": [1 if dataset == "NQ" else 0, *([0] * 15)]}
                            for dataset in ("NQ", "QASC", "SQuAD", "VitaminC")
                        ]
                        for split in ("train", "validation", "heldout")
                    },
                    "routing_eligible_cells": [route, *([False] * 15)],
                    "routing_eligible_counts": {
                        split: [1 if route else 0, *([0] * 15)]
                        for split in ("train", "validation", "heldout")
                    },
                })

    group_changes = {
        "explicit_position": {"mean_margin_change": 0.1, "family_count": 1},
        "legitimate_correction": {"mean_margin_change": 0.01, "family_count": 1},
    }
    control_changes = {
        "legitimate_correction": {"mean_margin_change": 0.01, "family_count": 1},
    }
    source_changes = {
        dataset: {
            "mean_margin_change": 0.02 if dataset == "NQ" else None,
            "component_count": 1 if dataset == "NQ" else 0,
        }
        for dataset in ("NQ", "QASC", "SQuAD", "VitaminC")
    }
    validation_grid = []
    for layer, alpha in ((3, 0.25), (3, 0.5), (6, 0.25), (6, 0.5)):
        validation_grid.append({
            "layer": layer,
            "alpha": alpha,
            "macro_delta": 0.1,
            "control_max_abs_mean_change": 0.01,
            "passes_control_gate": True,
            "selection_eligible": True,
            "synthetic_group_changes": group_changes,
            "control_group_changes": control_changes,
            "eligible_pair_count": 2,
            "abstention_pair_count": 0,
        })
    return {
        "schema": "sycophancy-som-transfer-atlas-v1",
        "id": atlas_id,
        "title": "Fixture transfer run",
        "metadata": {
            "model": "fixture/model",
            "revision": "fixture-revision",
            "device": "cpu",
            "source_status": "synthetic_unreviewed",
            "claim_ceiling": (
                "A held-out shift in the conditional likelihood margin of the frozen authored response scripts "
                "under a prompt-time activation perturbation."
            ),
            "not_established": [
                "This teacher-forced screen does not establish free-generation behavior.",
                "This screen does not establish causal efficacy.",
            ],
            "capture_layers": [3, 6, 9, 13, 16, 20, 23, 26],
            "map_layout": {"rows": 4, "columns": 4, "topology": "hexagonal", "cell_order": "row_major"},
            "selected_layer": 6,
            "selected_alpha": 0.5,
            "selected_method": "local_pooled",
            "status": "complete",
        },
        "groups": groups,
        "coverage": {
            **{
                split: {"pairs": 2, "rows": 2, "families": 2}
                for split in ("train", "validation", "heldout")
            },
            "source_cohort": {
                **{
                    split: {"components": 1, "pairs": 1}
                    for split in ("train", "validation", "heldout")
                },
                "by_collection": {
                    dataset: {split: 1 if dataset == "NQ" else 0 for split in ("train", "validation", "heldout")}
                    for dataset in ("NQ", "QASC", "SQuAD", "VitaminC")
                },
            },
            "synthetic_worlds": {
                split: {"families": 2, "pairs": 2}
                for split in ("train", "validation", "heldout")
            },
        },
        "maps": maps,
        "likelihood_summary": {
            "label": "Teacher-forced likelihood steering screen",
            "surrogate": True,
            "metric": "mean token logprob(independent) - mean token logprob(candidate)",
            "selection": {
                "layer": 6,
                "alpha": 0.5,
                "reason": "Fixture selection reason.",
                "validation_macro_delta": 0.1,
                "control_gate_limit_nat_per_token": 0.02,
                "control_gate_values": {"legitimate_correction": 0.01},
            },
            "validation_grid": validation_grid,
            "heldout_arms": {
                arm: {
                    "macro_delta": 0.1,
                    "eligible_pairs": 2,
                    "abstentions": 0,
                    "group_changes": group_changes,
                    "control_changes": control_changes,
                    "source_collection_changes": source_changes,
                    "family_bootstrap_interval": [-0.1, 0.2],
                }
                for arm in (
                    "baseline", "zero", "local_pooled", "global_matched_gate", "shuffled_cell",
                    "random_norm_matched", "reverse_sign", "fixed_prompt",
                )
            },
        },
    }


def _write_installed_atlas(repo_root, atlas_id: str = "fixture-atlas"):
    directory = repo_root / "research_outputs" / "atlases" / atlas_id
    directory.mkdir(parents=True)
    raw = json.dumps(_atlas(atlas_id), sort_keys=True, separators=(",", ":")).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt = {
        "schema": "taxonomy-som-atlas-receipt-v1",
        "id": atlas_id,
        "atlas_sha256": hashlib.sha256(raw).hexdigest(),
        "atlas_bytes": len(raw),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return directory


def _write_installed_transfer_atlas(repo_root, atlas_id: str = "transfer-fixture"):
    directory = repo_root / "research_outputs" / "atlases" / atlas_id
    directory.mkdir(parents=True)
    raw = json.dumps(_transfer_atlas(atlas_id), sort_keys=True, separators=(",", ":")).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt = {
        "schema": "taxonomy-som-atlas-receipt-v1",
        "id": atlas_id,
        "atlas_sha256": hashlib.sha256(raw).hexdigest(),
        "atlas_bytes": len(raw),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return directory


def _rewrite_transfer_atlas(directory, atlas: dict) -> None:
    raw = json.dumps(atlas, sort_keys=True, separators=(",", ":")).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt = {
        "schema": "taxonomy-som-atlas-receipt-v1",
        "id": atlas["id"],
        "atlas_sha256": hashlib.sha256(raw).hexdigest(),
        "atlas_bytes": len(raw),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest_asyncio.fixture
async def client(tmp_path):
    app = FastAPI()
    app.include_router(create_atlas_router(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as session:
        yield session, tmp_path


async def test_only_receipt_verified_atlases_are_listed_and_returned(client):
    session, repo_root = client
    _write_installed_atlas(repo_root)

    listing = await session.get("/api/atlases")
    assert listing.status_code == 200
    body = listing.json()
    assert [row["id"] for row in body["atlases"]] == ["fixture-atlas"]
    assert body["atlases"][0]["source_status"] == "TFonly"
    assert body["rejected"] == []

    response = await session.get("/api/atlases/fixture-atlas")
    assert response.status_code == 200
    assert response.json()["schema"] == "taxonomy-som-atlas-v1"
    assert response.json()["maps"][0]["cells"][0]["id"] == "0-0"


async def test_transfer_atlas_uses_compatible_receipt_and_exposes_named_aggregate_run(client):
    session, repo_root = client
    _write_installed_transfer_atlas(repo_root)

    listing = (await session.get("/api/atlases")).json()
    assert len(listing["atlases"]) == 1
    row = listing["atlases"][0]
    assert row["id"] == "transfer-fixture"
    assert row["title"] == "Fixture transfer run"
    assert row["run_label"] == "Fixture transfer run · transfer-fixture"
    assert row["schema"] == "sycophancy-som-transfer-atlas-v1"
    assert row["source_status"] == "synthetic_unreviewed"

    response = await session.get("/api/atlases/transfer-fixture")
    assert response.status_code == 200
    atlas = response.json()
    assert len(atlas["maps"]) == 24
    assert atlas["maps"][0]["rows"] == atlas["maps"][0]["columns"] == 4
    assert atlas["maps"][0]["routing_eligible_cells"] == [True, *([False] * 15)]
    assert atlas["maps"][1]["used_for_routing"] is False
    assert "examples" not in atlas
    assert all("points" not in map_record for map_record in atlas["maps"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda atlas: atlas["maps"][0].update(rows=6),
        lambda atlas: atlas["maps"][0]["cell_summaries"]["heldout"].__setitem__(0, -1),
        lambda atlas: atlas["maps"][0]["group_counts"]["train"][0]["counts"].__setitem__(0, 2),
        lambda atlas: atlas["maps"][1].update(used_for_routing=True),
        lambda atlas: atlas["maps"][0].update(map_kind=[]),
        lambda atlas: atlas["maps"][0]["source_group_counts"]["train"][0].update(source_collection="/private/data"),
        lambda atlas: atlas["metadata"].update(not_established=[]),
        lambda atlas: atlas.update(examples={"raw": "must not be served"}),
    ],
)
async def test_transfer_atlas_rejects_bad_dimensions_counts_routing_and_raw_payloads(client, mutate):
    session, repo_root = client
    directory = _write_installed_transfer_atlas(repo_root)
    atlas = _transfer_atlas()
    mutate(atlas)
    _rewrite_transfer_atlas(directory, atlas)

    response = await session.get("/api/atlases/transfer-fixture")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_atlas"


async def test_transfer_atlas_rejects_unbalanced_source_and_synthetic_cell_counts(client):
    session, repo_root = client
    directory = _write_installed_transfer_atlas(repo_root)
    atlas = _transfer_atlas()
    atlas["maps"][0]["source_cell_summaries"]["heldout"][0] = 2
    _rewrite_transfer_atlas(directory, atlas)

    response = await session.get("/api/atlases/transfer-fixture")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_atlas"


async def test_transfer_atlas_requires_selected_setting_to_pass_validation_gates(client):
    session, repo_root = client
    directory = _write_installed_transfer_atlas(repo_root)
    atlas = _transfer_atlas()
    selected = next(
        row for row in atlas["likelihood_summary"]["validation_grid"]
        if row["layer"] == 6 and row["alpha"] == 0.5
    )
    selected["passes_control_gate"] = False
    _rewrite_transfer_atlas(directory, atlas)

    response = await session.get("/api/atlases/transfer-fixture")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_atlas"


async def test_hash_tampering_is_rejected_and_excluded_from_listing(client):
    session, repo_root = client
    directory = _write_installed_atlas(repo_root)
    with (directory / "atlas.json").open("ab") as atlas_file:
        atlas_file.write(b" ")

    response = await session.get("/api/atlases/fixture-atlas")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_atlas"
    listing = (await session.get("/api/atlases")).json()
    assert listing["atlases"] == []
    assert listing["rejected"] == [{"id": "fixture-atlas", "code": "invalid_atlas"}]


@pytest.mark.parametrize("atlas_id", ["..", "bad.id", "bad/id", "", "a" * 65])
async def test_unsafe_or_unrecognized_ids_do_not_select_files(client, atlas_id):
    session, repo_root = client
    _write_installed_atlas(repo_root)
    response = await session.get(f"/api/atlases/{atlas_id}")
    assert response.status_code in {307, 404, 405}


async def test_wrong_schema_and_receipt_id_are_rejected(client):
    session, repo_root = client
    directory = _write_installed_atlas(repo_root)
    atlas = _atlas()
    atlas["schema"] = "unexpected"
    raw = json.dumps(atlas).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt = {
        "schema": "taxonomy-som-atlas-receipt-v1", "id": "other-atlas",
        "atlas_sha256": hashlib.sha256(raw).hexdigest(), "atlas_bytes": len(raw),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    response = await session.get("/api/atlases/fixture-atlas")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_atlas"


async def test_coordinate_pair_cells_are_supported_and_bad_nested_values_are_rejected(client):
    session, repo_root = client
    directory = _write_installed_atlas(repo_root)
    atlas = _atlas()
    atlas["maps"][0]["points"][0]["cell"] = [0, 0]
    atlas["metadata"]["source_status"] = "teacher_forced_synthetic_unreviewed"
    raw = json.dumps(atlas).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt = {
        "schema": "taxonomy-som-atlas-receipt-v1", "id": "fixture-atlas",
        "atlas_sha256": hashlib.sha256(raw).hexdigest(), "atlas_bytes": len(raw),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert (await session.get("/api/atlases/fixture-atlas")).status_code == 200

    atlas["maps"][0]["view"] = []
    raw = json.dumps(atlas).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt["atlas_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["atlas_bytes"] = len(raw)
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    response = await session.get("/api/atlases/fixture-atlas")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_atlas"


async def test_symlinked_installation_is_ignored(client, tmp_path):
    session, repo_root = client
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_installed_atlas(outside, "target-atlas")
    link = repo_root / "research_outputs" / "atlases" / "linked-atlas"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside / "research_outputs" / "atlases" / "target-atlas", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    listing = (await session.get("/api/atlases")).json()
    assert listing["atlases"] == []
    assert (await session.get("/api/atlases/linked-atlas")).status_code == 404


async def test_application_mounts_atlas_api_without_starting_lifespan():
    from explorer import server

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url=f"http://127.0.0.1:{server.PORT}",
    ) as session:
        response = await session.get("/api/atlases")
    assert response.status_code == 200
    assert set(response.json()) == {"atlases", "rejected"}


async def test_transfer_atlas_accepts_valid_zero_fallback(client):
    session, repo_root = client
    directory = _write_installed_transfer_atlas(repo_root)
    atlas = _transfer_atlas()
    atlas["metadata"].update(selected_alpha=0.0, selected_method="zero")
    selection = atlas["likelihood_summary"]["selection"]
    selection.update(alpha=0.0, reason="No positive candidate passed; no intervention.", validation_macro_delta=0.0)
    for row in atlas["likelihood_summary"]["validation_grid"]:
        row["selection_eligible"] = False
        if row["alpha"] == 0.25:
            row.update(alpha=0.0, macro_delta=0.0, control_max_abs_mean_change=0.0)
    _rewrite_transfer_atlas(directory, atlas)
    response = await session.get("/api/atlases/transfer-fixture")
    assert response.status_code == 200
