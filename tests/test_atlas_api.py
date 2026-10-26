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
