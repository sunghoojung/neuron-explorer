"""Deterministic NumPy SOM fitting, projection helpers, and artifact I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactError(ValueError):
    """A saved SOM artifact is invalid or incompatible."""


@dataclass(frozen=True)
class SOMArtifact:
    prototypes: np.ndarray
    basis: np.ndarray
    metadata: dict[str, Any]


def train_som(
    samples: np.ndarray,
    *,
    rows: int = 4,
    cols: int = 4,
    epochs: int = 100,
    learning_rate: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Train a small online SOM with deterministic initialization and order."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("SOM samples must have shape [n_samples, hidden_size] with nonzero dimensions")
    if not np.isfinite(samples).all():
        raise ValueError("SOM samples must be finite")
    if rows < 1 or cols < 1 or epochs < 1:
        raise ValueError("SOM rows, columns, and epochs must be positive")
    if not 0 < learning_rate <= 1:
        raise ValueError("SOM learning_rate must be in (0, 1]")

    rng = np.random.default_rng(seed)
    unit_count = rows * cols
    chosen = rng.choice(len(samples), size=unit_count, replace=unit_count > len(samples))
    weights = samples[chosen].copy()
    grid = np.asarray([(r, c) for r in range(rows) for c in range(cols)], dtype=np.float64)
    distances = np.sum((grid[:, None, :] - grid[None, :, :]) ** 2, axis=-1)
    initial_sigma = max(rows, cols) / 2.0

    for epoch in range(epochs):
        fraction = epoch / max(epochs - 1, 1)
        sigma = max(0.5, initial_sigma * (1.0 - fraction))
        rate = learning_rate * (1.0 - 0.8 * fraction)
        neighborhood = np.exp(-distances / (2.0 * sigma * sigma))
        for sample_index in rng.permutation(len(samples)):
            sample = samples[sample_index]
            bmu = int(np.argmin(np.sum((weights - sample) ** 2, axis=1)))
            weights += (rate * neighborhood[bmu, :, None]) * (sample - weights)

    return np.asarray(weights, dtype=np.float32)


def orthonormal_basis(vectors: np.ndarray, *, max_rank: int | None = None) -> np.ndarray:
    """Return a rank-safe orthonormal basis for vectors stored as columns."""
    source = np.asarray(vectors)
    if source.ndim != 2:
        raise ValueError("basis input must be a two-dimensional matrix")
    if not np.issubdtype(source.dtype, np.number) or not np.isfinite(source).all():
        raise ValueError("basis input must be finite")
    source_epsilon = (
        np.finfo(source.dtype).eps if np.issubdtype(source.dtype, np.floating) else np.finfo(np.float64).eps
    )
    matrix = np.asarray(source, dtype=np.float64)
    if max_rank is not None and max_rank < 0:
        raise ValueError("max_rank cannot be negative")
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float32)

    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float32)
    tolerance = max(matrix.shape) * source_epsilon * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if max_rank is not None:
        rank = min(rank, max_rank)
    return np.asarray(left[:, :rank], dtype=np.float32)


def project(values: np.ndarray, basis: np.ndarray, *, mode: str = "remove") -> np.ndarray:
    """Keep or remove the component in a column-oriented orthonormal basis."""
    values = np.asarray(values)
    basis = np.asarray(basis, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("values must have a hidden dimension")
    if basis.ndim != 2 or values.shape[-1] != basis.shape[0]:
        raise ValueError("basis must have shape [hidden_size, rank] matching values[-1]")
    if mode not in {"remove", "keep"}:
        raise ValueError("projection mode must be 'remove' or 'keep'")
    if not np.isfinite(values).all() or not np.isfinite(basis).all():
        raise ValueError("projection inputs must be finite")
    x = np.asarray(values, dtype=np.float64)
    if basis.shape[1] == 0:
        projected = np.zeros_like(x) if mode == "keep" else x.copy()
    else:
        component = (x @ basis) @ basis.T
        projected = component if mode == "keep" else x - component
    return np.asarray(projected, dtype=values.dtype)


def random_basis(hidden_size: int, rank: int, *, seed: int = 0) -> np.ndarray:
    if hidden_size < 1 or rank < 0 or rank > hidden_size:
        raise ValueError("random basis requires 0 <= rank <= hidden_size and hidden_size > 0")
    if rank == 0:
        return np.empty((hidden_size, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    return orthonormal_basis(rng.normal(size=(hidden_size, rank)))[:, :rank]


def fit_control_bases(train_deltas: np.ndarray, som_basis: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    """Build fixed global-mean, centered PCA, and same-rank random controls."""
    train_deltas = np.asarray(train_deltas)
    som_basis = np.asarray(som_basis, dtype=np.float32)
    if train_deltas.ndim != 2 or not np.isfinite(train_deltas).all():
        raise ValueError("training deltas must be a finite two-dimensional matrix")
    if som_basis.ndim != 2 or som_basis.shape[0] != train_deltas.shape[1]:
        raise ValueError("SOM basis hidden size must match training deltas")

    mean = np.mean(train_deltas, axis=0, keepdims=True, dtype=np.float64).astype(train_deltas.dtype)
    global_mean = orthonormal_basis(mean.T, max_rank=1)
    pca = orthonormal_basis((train_deltas - mean).T, max_rank=som_basis.shape[1])
    random = random_basis(train_deltas.shape[1], som_basis.shape[1], seed=seed)
    return {"global_mean": global_mean, "pca": pca, "random": random}


def make_artifact(
    prototypes: np.ndarray,
    *,
    model_id: str,
    revision: str,
    layer: int,
    hook: str,
    grid: tuple[int, int],
    seed: int,
    question_count: int,
    training_question_ids: list[str],
    activation_sha256: str,
    model_activation_dtype: str = "unknown",
) -> SOMArtifact:
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if prototypes.ndim != 2 or not np.isfinite(prototypes).all():
        raise ArtifactError("prototypes must be a finite [n_prototypes, hidden_size] matrix")
    if question_count < 1 or len(training_question_ids) != question_count or len(set(training_question_ids)) != question_count:
        raise ArtifactError("training question ids must be unique and match question_count")
    basis = orthonormal_basis(prototypes.T)
    if basis.shape[1] == 0:
        raise ArtifactError("SOM prototypes have zero rank; check the pressure-neutral activation deltas")
    metadata = {
        "schema_version": 1,
        "method": "som",
        "model_id": model_id,
        "revision": revision,
        "layer": int(layer),
        "hook": hook,
        "token_position": "last non-padding prompt token for collection; all sequence positions for intervention",
        "tokenization": "model adapter chat template with assistant generation prefix",
        "model_activation_dtype": model_activation_dtype,
        "projection_compute_dtype": "float32 on decoder residual device; output cast back to activation dtype",
        "hidden_size": int(prototypes.shape[1]),
        "grid": [int(grid[0]), int(grid[1])],
        "seed": int(seed),
        "training_split": "train",
        "training_conditions": ["false_pressure - neutral"],
        "training_question_count": int(question_count),
        "training_question_id_hashes": sorted(
            hashlib.sha256(qid.encode("utf-8")).hexdigest() for qid in training_question_ids
        ),
        "prototype_count": len(prototypes),
        "basis_rank": int(basis.shape[1]),
        "activation_sha256": activation_sha256,
    }
    artifact = SOMArtifact(prototypes, basis, metadata)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: SOMArtifact) -> None:
    prototypes = np.asarray(artifact.prototypes)
    basis = np.asarray(artifact.basis)
    metadata = artifact.metadata
    if not isinstance(metadata, dict):
        raise ArtifactError("artifact metadata must be a JSON object")
    if metadata.get("schema_version") != 1 or metadata.get("method") != "som":
        raise ArtifactError("unsupported artifact schema or method")
    required = (
        "model_id", "revision", "layer", "hook", "token_position", "hidden_size", "basis_rank",
        "training_question_count", "training_question_id_hashes", "activation_sha256",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ArtifactError(f"artifact metadata missing fields: {', '.join(missing)}")
    if prototypes.ndim != 2 or prototypes.shape[0] < 1 or prototypes.shape[1] < 1:
        raise ArtifactError("prototypes must have shape [n_prototypes, hidden_size]")
    if basis.ndim != 2 or basis.shape[0] != prototypes.shape[1] or basis.shape[1] < 1:
        raise ArtifactError("basis must have shape [hidden_size, rank] with positive rank")
    if not np.isfinite(prototypes).all() or not np.isfinite(basis).all():
        raise ArtifactError("artifact arrays must be finite")
    if metadata["hidden_size"] != prototypes.shape[1] or metadata["basis_rank"] != basis.shape[1]:
        raise ArtifactError("artifact metadata dimensions do not match saved arrays")
    if not isinstance(metadata["layer"], int) or metadata["layer"] < 0:
        raise ArtifactError("artifact layer must be a non-negative integer")
    if not isinstance(metadata["model_id"], str) or not metadata["model_id"]:
        raise ArtifactError("artifact model_id must be a non-empty string")
    if not isinstance(metadata["revision"], str) or not metadata["revision"]:
        raise ArtifactError("artifact revision must be a non-empty string")
    hashes = metadata["training_question_id_hashes"]
    if not isinstance(hashes, list) or any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ArtifactError("artifact training question hashes are invalid")
    if len(hashes) != metadata["training_question_count"] or len(set(hashes)) != len(hashes):
        raise ArtifactError("artifact training question hashes are invalid")
    checksum = metadata["activation_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ArtifactError("artifact activation checksum is invalid")
    gram = basis.astype(np.float64).T @ basis.astype(np.float64)
    if not np.allclose(gram, np.eye(basis.shape[1]), atol=2e-4, rtol=2e-4):
        raise ArtifactError("artifact basis columns are not orthonormal")


def save_artifact(path: str | Path, artifact: SOMArtifact) -> None:
    validate_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            prototypes=np.asarray(artifact.prototypes, dtype=np.float32),
            basis=np.asarray(artifact.basis, dtype=np.float32),
            metadata=np.asarray(json.dumps(artifact.metadata, sort_keys=True)),
        )
    temporary.replace(path)


def load_artifact(path: str | Path) -> SOMArtifact:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"prototypes", "basis", "metadata"}
            missing = required - set(archive.files)
            if missing:
                raise ArtifactError(f"artifact is missing arrays: {', '.join(sorted(missing))}")
            artifact = SOMArtifact(
                prototypes=np.asarray(archive["prototypes"], dtype=np.float32),
                basis=np.asarray(archive["basis"], dtype=np.float32),
                metadata=json.loads(str(archive["metadata"].item())),
            )
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read SOM artifact {path}: {exc}") from None
    validate_artifact(artifact)
    return artifact
