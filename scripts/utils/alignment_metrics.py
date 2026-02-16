from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_translation_stats(translations: Iterable[np.ndarray]) -> dict:
    """Compute translation statistics and pairwise distances."""
    translations_arr = np.stack(list(translations), axis=0)
    mean = translations_arr.mean(axis=0)
    std = translations_arr.std(axis=0)

    diffs = translations_arr[:, None, :] - translations_arr[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]

    stats = {
        "mean": mean,
        "std": std,
        "max_distance": float(upper.max()) if upper.size else 0.0,
        "median_distance": float(np.median(upper)) if upper.size else 0.0,
    }
    return stats


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute geodesic distance between rotations in radians."""
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def compute_rotation_stats(rotations: Iterable[np.ndarray]) -> dict:
    """Compute rotation alignment statistics in degrees."""
    rotations_list = list(rotations)
    num = len(rotations_list)
    if num < 2:
        return {
            "mean_angle_deg": 0.0,
            "std_angle_deg": 0.0,
            "max_angle_deg": 0.0,
        }

    angles = []
    for i in range(num):
        for j in range(i + 1, num):
            angles.append(rotation_angle(rotations_list[i], rotations_list[j]))

    angles_arr = np.array(angles, dtype=np.float64)
    angles_deg = np.degrees(angles_arr)
    return {
        "mean_angle_deg": float(angles_deg.mean()),
        "std_angle_deg": float(angles_deg.std()),
        "max_angle_deg": float(angles_deg.max()),
    }


def detect_outliers(
    translations: Iterable[np.ndarray],
    rotations: Iterable[np.ndarray],
    run_ids: Iterable[str],
    translation_threshold: float,
    rotation_threshold_deg: float,
) -> list[dict]:
    """Identify outliers using threshold comparison to reference pose."""
    translations_list = list(translations)
    rotations_list = list(rotations)
    run_ids_list = list(run_ids)

    if not translations_list:
        return []

    reference_translation = np.median(np.stack(translations_list), axis=0)
    reference_rotation = rotations_list[0]

    outliers = []
    for run_id, translation, rotation in zip(
        run_ids_list, translations_list, rotations_list
    ):
        trans_error = float(np.linalg.norm(translation - reference_translation))
        rot_error_rad = rotation_angle(reference_rotation, rotation)
        rot_error_deg = float(np.degrees(rot_error_rad))

        if trans_error > translation_threshold or rot_error_deg > rotation_threshold_deg:
            reason = []
            if trans_error > translation_threshold:
                reason.append("translation")
            if rot_error_deg > rotation_threshold_deg:
                reason.append("rotation")

            outliers.append(
                {
                    "run_id": run_id,
                    "translation_error": trans_error,
                    "rotation_error_deg": rot_error_deg,
                    "reason": " and ".join(reason),
                }
            )

    return outliers
