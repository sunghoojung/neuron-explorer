"""Deterministic, train-only two-dimensional projections for residual maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def pca_project_2d(training_vectors: Any, vectors: Any) -> dict[str, Any]:
    """Fit centered PCA axes on training rows and project an arbitrary split.

    The returned axes have shape [2, hidden_size]. Degenerate axes are zero,
    so rank-zero and rank-one data produce finite two-dimensional coordinates.
    SVD component signs are canonicalized by making each component's largest
    absolute loading positive.
    """
    raw_training = np.asarray(training_vectors)
    raw_values = np.asarray(vectors)
    if raw_training.ndim != 2 or raw_training.shape[0] < 2 or raw_training.shape[1] < 1:
        raise ValueError("PCA training vectors must contain at least two rows and one feature")
    if raw_values.ndim != 2 or raw_values.shape[1] != raw_training.shape[1]:
        raise ValueError("projection vectors must match the training hidden dimension")
    if not np.issubdtype(raw_training.dtype, np.number) or not np.issubdtype(raw_values.dtype, np.number):
        raise ValueError("PCA vectors must be numeric")
    if not np.isfinite(raw_training).all() or not np.isfinite(raw_values).all():
        raise ValueError("PCA vectors must be finite")

    source_epsilon = (
        np.finfo(raw_training.dtype).eps
        if np.issubdtype(raw_training.dtype, np.floating)
        else np.finfo(np.float64).eps
    )
    training = np.asarray(raw_training, dtype=np.float64)
    values = np.asarray(raw_values, dtype=np.float64)
    mean = training.mean(axis=0, dtype=np.float64)
    centered = training - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    axes = np.zeros((2, training.shape[1]), dtype=np.float64)
    if singular_values.size and singular_values[0] > 0:
        tolerance = max(centered.shape) * source_epsilon * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        components = min(2, rank)
        for index in range(components):
            component = right[index].copy()
            pivot = int(np.argmax(np.abs(component)))
            if component[pivot] < 0:
                component *= -1.0
            axes[index] = component
    else:
        rank = 0
    total_energy = float(np.sum(singular_values**2, dtype=np.float64))
    ratios = np.zeros(2, dtype=np.float64)
    if total_energy > 0:
        count = min(2, singular_values.size)
        ratios[:count] = singular_values[:count] ** 2 / total_energy
    coordinates = (values - mean) @ axes.T
    if not np.isfinite(coordinates).all() or not np.isfinite(ratios).all():
        raise ValueError("PCA projection produced non-finite values")
    return {
        "mean": mean,
        "axes": axes,
        "rank": rank,
        "explained_variance_ratio": ratios,
        "coordinates": coordinates,
    }
