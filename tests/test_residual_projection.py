"""Projection and raw-geometry tests using saved arrays only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from explorer.residual_projection import pca_project_2d
from explorer.residual_service import ResidualService, ResidualServiceError


def _capture_fixture() -> dict:
    records = []
    vectors_by_id = {}
    cases = [
        ("train", "pair-train-1", "family-train-1", (0.0, 0.0), (1.0, 0.0)),
        ("train", "pair-train-2", "family-train-2", (0.0, 1.0), (1.0, 1.0)),
        ("validation", "pair-validation", "family-validation", (1000.0, 1000.0), (999.0, 1000.0)),
        ("test", "pair-test", "family-test", (3.0, 4.0), (1.0, 0.0)),
    ]
    for split, pair_id, family_id, vector_a, vector_b in cases:
        for group, vector in (("neutral", vector_a), ("pressure", vector_b)):
            sample_id = f"{pair_id}-{group}"
            sample = {
                "id": sample_id,
                "pair_id": pair_id,
                "family_id": family_id,
                "split": split,
                "group": group,
                "messages": [{"role": "user", "content": f"Question for {pair_id} in {group}."}],
                "target": " supported answer",
                "foil": " alternative answer",
            }
            records.append(sample)
            vectors_by_id[sample_id] = vector
    vectors = np.zeros((len(records), 2, 1, 1, 2), dtype=np.float32)
    for index, sample in enumerate(records):
        vectors[index, 0, 0, 0] = vectors_by_id[sample["id"]]
        vector = vectors_by_id[sample["id"]]
        if sample["split"] == "test":
            layer_three = (1.0, 0.0) if sample["group"] == "neutral" else (0.0, 2.0)
        else:
            layer_three = (vector[0] + 0.5, vector[1] - 0.25)
        vectors[index, 1, 0, 0] = layer_three
    return {
        "dataset": {"name": "projection-fixture", "samples": records},
        "arrays": {
            "vectors": vectors,
            "token_ids": np.arange(len(records), dtype=np.int64).reshape(-1, 1),
            "positions": np.zeros((len(records), 1), dtype=np.int64),
            "valid": np.ones((len(records), 1), dtype=bool),
        },
        "tokens": [["fixture"] for _ in records],
        "metadata": {
            "model_id": "org/fixture-model", "revision": "fixture-revision", "family": "llama",
            "device": "cpu", "hidden_size": 2, "layers": [2, 3], "points": ["resid_post"],
            "window": 1, "template_sha256": "a" * 64, "tokenization": "fixture",
            "token_position": "final prompt token", "causal_context": "prompt only",
            "capture_dtype": "float32", "model_activation_dtypes": ["float32"],
        },
        "manifest": {"capture_sha256": "b" * 64},
        "job": {"integrity_status": "verified"},
        "directory": Path("."),
    }


class _FakeRuntime:
    model_id = "org/fixture-model"
    revision = "fixture-revision"
    cache_namespace = "residual-projection-fixture"
    adapter = SimpleNamespace(model_id=model_id, revision=revision, family="llama", device="cpu")
    model = object()


def test_pca_rank_zero_and_rank_one_return_finite_axes():
    zero = pca_project_2d(np.ones((3, 4), dtype=np.float32), np.asarray([[2, 3, 4, 5]], dtype=np.float32))
    assert zero["rank"] == 0
    assert np.array_equal(zero["axes"], np.zeros((2, 4)))
    assert np.array_equal(zero["coordinates"], np.zeros((1, 2)))
    assert zero["explained_variance_ratio"].tolist() == [0.0, 0.0]

    line = pca_project_2d(
        np.asarray([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.asarray([[3.0, 0.0]], dtype=np.float32),
    )
    assert line["rank"] == 1
    assert np.allclose(line["explained_variance_ratio"], [1.0, 0.0])
    assert np.allclose(line["axes"][1], 0.0)
    assert line["axes"][0, np.argmax(np.abs(line["axes"][0]))] > 0
    assert np.isfinite(line["coordinates"]).all()


def test_heldout_outlier_does_not_change_train_fitted_axes():
    training = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    ordinary = np.asarray([[0.25, 0.75, 0.5]], dtype=np.float32)
    first = pca_project_2d(training, ordinary)
    with_outlier = pca_project_2d(training, np.vstack([ordinary, [1e9, -1e9, 5e8]]))
    assert np.array_equal(first["mean"], with_outlier["mean"])
    assert np.array_equal(first["axes"], with_outlier["axes"])
    assert np.array_equal(first["coordinates"][0], with_outlier["coordinates"][0])


def test_map_uses_train_only_pca_and_reports_exact_raw_pair_geometry(tmp_path):
    service = ResidualService(_FakeRuntime(), tmp_path)
    capture = _capture_fixture()
    service._load_capture = lambda _capture_id: capture
    try:
        result = service.get_capture_map(
            "a" * 32, layer=2, point="resid_post", group_a="neutral", group_b="pressure", split="test"
        )
        projection = result["projection"]
        assert result["selected"] == {
            "layer": 2, "point": "resid_post", "group_a": "neutral", "group_b": "pressure", "split": "test"
        }
        assert projection["fit_split"] == "train"
        assert projection["training_samples"] == 4
        assert projection["rank"] == 2
        assert len(projection["points"]) == 2
        assert {point["split"] for point in projection["points"]} == {"test"}
        assert projection["pairs"][0]["pair_id"] == "pair-test"
        assert projection["pairs"][0]["distance_l2"] == pytest.approx(np.sqrt(20.0))
        metrics_by_layer = {row["layer"]: row for row in result["layers"]}
        assert metrics_by_layer[2]["n_pairs"] == 1
        assert metrics_by_layer[2]["mean_delta_l2"] == pytest.approx(np.sqrt(20.0))
        assert metrics_by_layer[2]["mean_delta_rms"] == pytest.approx(np.sqrt(10.0))
        assert metrics_by_layer[2]["centroid_cosine"] == pytest.approx(0.6)
        assert metrics_by_layer[3]["mean_delta_l2"] == pytest.approx(np.sqrt(5.0))
        assert metrics_by_layer[3]["centroid_cosine"] == pytest.approx(0.0)
    finally:
        service.close()


def test_sample_detail_returns_only_requested_saved_row(tmp_path):
    service = ResidualService(_FakeRuntime(), tmp_path)
    capture = _capture_fixture()
    service._load_capture = lambda _capture_id: capture
    try:
        sample_id = "pair-test-neutral"
        result = service.get_capture_sample("a" * 32, sample_id)
        assert result["sample"]["id"] == sample_id
        assert result["sample"]["split"] == "test"
        assert result["sample"]["group"] == "neutral"
        assert result["sample"]["target"] == " supported answer"
        assert result["sample"]["messages"][0]["content"] == "Question for pair-test in neutral."
        with pytest.raises(ResidualServiceError) as error:
            service.get_capture_sample("a" * 32, "missing")
        assert error.value.http_status == 404
    finally:
        service.close()
