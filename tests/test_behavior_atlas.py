"""Validation boundaries for naturally generated behavior atlases."""

import copy
import hashlib
import json

import pytest

from explorer.atlas_service import AtlasService, AtlasServiceError


def fixture_atlas(measured=True):
    cells = [
        {
            "id": f"cell{i}",
            "x": float(i % 4),
            "y": float(i // 4),
            "count": int(i == 0),
            "family_count": int(i == 0),
            "fit_count": int(i == 0),
            "fit_family_count": int(i == 0),
            "label_counts": {"flip": int(i == 0)},
            "scenario_counts": {"unsupported_pressure": int(i == 0)},
            "source_counts": {"arc": int(i == 0)},
        }
        for i in range(16)
    ]
    maps = (
        [
            {
                "id": "layer6_seed0",
                "layer": 6,
                "seed": 0,
                "rows": 4,
                "columns": 4,
                "fit_count": 1,
                "fit_family_count": 1,
                "reference_count": 1,
                "reference_family_count": 1,
                "quantization_error": 0.5,
                "topographic_error": 0.0,
                "cells": cells,
            }
        ]
        if measured
        else []
    )
    return {
        "schema": "sycophancy-som-behavior-atlas-v1",
        "id": "behavior-test",
        "title": "Behavior fixture",
        "metadata": {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
            "device": "cpu",
            "source_status": "model_assisted_provisional",
            "status": "discovery_measured" if measured else "not_yet_measured",
            "not_established": [
                "No final confirmation or non-inferiority has been established."
            ],
        },
        "provenance": {
            k: "a" * 64 if measured else None
            for k in ("data_sha256", "labels_sha256", "capture_sha256", "code_sha256")
        },
        "coverage": {
            "planned": 1 if measured else 0,
            "generated": 1 if measured else 0,
            "graded": 1 if measured else 0,
            "missing": 0,
            "ungraded": 0,
            "families": 1 if measured else 0,
        },
        "groups": [
            {"id": "flip", "label": "Correct-to-wrong flip", "color": "#abcdef"}
        ],
        "maps": maps,
        "behavior": [],
    }


def install(root, atlas, receipt_schema="sycophancy-som-behavior-atlas-receipt-v1"):
    directory = root / "research_outputs/atlases/behavior-test"
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(atlas).encode()
    (directory / "atlas.json").write_bytes(raw)
    receipt = {
        "schema": receipt_schema,
        "id": "behavior-test",
        "atlas_sha256": hashlib.sha256(raw).hexdigest(),
        "atlas_bytes": len(raw),
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))


def test_empty_and_actual_map_atlases_are_accepted(tmp_path):
    service = AtlasService(tmp_path)
    for measured in (False, True):
        atlas = fixture_atlas(measured)
        install(tmp_path, atlas)
        assert service.get_atlas("behavior-test") == atlas
        assert service.list_atlases()["atlases"][0]["map_count"] == int(measured)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["maps"][0]["cells"][0].update(count=2),
        lambda x: x["maps"][0]["cells"][0].update(family_count=2),
        lambda x: x["maps"][0]["cells"][0].update(fit_count=0),
        lambda x: x["maps"][0].update(quantization_error=float("nan")),
        lambda x: x["maps"][0].update(reference_count=0, reference_family_count=0),
        lambda x: x["maps"][0]["cells"][0].update(family_count=0),
        lambda x: x["maps"][0]["cells"][0].update(fit_family_count=0),
        lambda x: x["metadata"].update(revision="unknown"),
        lambda x: x["maps"][0].update(topographic_error=1.1),
        lambda x: x["coverage"].update(missing=2),
        lambda x: x["provenance"].update(capture_sha256=None),
        lambda x: x["metadata"].update(status="not_yet_measured"),
        lambda x: x["maps"][0]["cells"][0].update(raw_prompt="not allowed"),
    ],
)
def test_invalid_counts_provenance_and_extra_fields_are_rejected(tmp_path, mutation):
    atlas = fixture_atlas()
    mutation(atlas)
    install(tmp_path, atlas)
    with pytest.raises(AtlasServiceError):
        AtlasService(tmp_path).get_atlas("behavior-test")


def test_two_arm_family_weighted_behavior_and_denominators(tmp_path):
    atlas = fixture_atlas()
    atlas["metadata"]["status"] = "behavior_measured"
    arm = {"eligible": 1, "success": 1, "other": 0, "ungraded": 0, "missing": 0}
    atlas["behavior"] = [
        {
            "scenario": "unsupported_pressure",
            "metric": "resisted_pressure",
            "baseline": copy.deepcopy(arm),
            "som": copy.deepcopy(arm),
            "paired": {
                "eligible": 1,
                "complete": 1,
                "families": 1,
                "estimand": "equal_family_mean_branch_difference",
                "delta": 0.0,
                "ci_low": None,
                "ci_high": None,
            },
        }
    ]
    install(tmp_path, atlas)
    assert (
        AtlasService(tmp_path).get_atlas("behavior-test")["behavior"]
        == atlas["behavior"]
    )
    paired = atlas["behavior"][0]["paired"]
    paired.update(baseline_rate=1.0, som_rate=1.0, planned_families=2, conservative_bounds=[-0.5, 0.5])
    install(tmp_path, atlas)
    assert AtlasService(tmp_path).get_atlas("behavior-test")["behavior"] == atlas["behavior"]
    for field in ("complete", "families"):
        original = paired[field]
        paired[field] = "invalid"
        install(tmp_path, atlas)
        with pytest.raises(AtlasServiceError):
            AtlasService(tmp_path).get_atlas("behavior-test")
        paired[field] = original
    paired["planned_families"] = 0
    install(tmp_path, atlas)
    with pytest.raises(AtlasServiceError):
        AtlasService(tmp_path).get_atlas("behavior-test")
    paired["planned_families"] = 2
    paired["conservative_bounds"] = [0.5, -0.5]
    install(tmp_path, atlas)
    with pytest.raises(AtlasServiceError):
        AtlasService(tmp_path).get_atlas("behavior-test")
    paired["conservative_bounds"] = [-0.5, 0.5]
    atlas["behavior"][0]["som"]["success"] = 2
    install(tmp_path, atlas)
    with pytest.raises(AtlasServiceError):
        AtlasService(tmp_path).get_atlas("behavior-test")


def test_receipt_must_match_schema(tmp_path):
    install(tmp_path, fixture_atlas(), "taxonomy-som-atlas-receipt-v1")
    with pytest.raises(AtlasServiceError):
        AtlasService(tmp_path).get_atlas("behavior-test")
