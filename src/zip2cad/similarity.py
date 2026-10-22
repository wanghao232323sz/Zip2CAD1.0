"""Similarity-transform helpers used for multi-block reconstruction alignment."""

from __future__ import annotations

import numpy as np


def estimate_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation and translation mapping source onto target.

    Uses the least-squares Umeyama solution without reflection.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = np.sum(source_centered**2) / len(source)
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("source correspondences have negligible variance")

    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def transform_points(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply x_target = scale * rotation @ x_source + translation."""

    points = np.asarray(points)
    return scale * np.einsum("ij,...j->...i", rotation, points) + translation

