"""Optimization metrics and reporting utilities.

This module provides functions to compute trajectory metrics and build
optimization reports for pose graph optimization results.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


def compute_error_reduction(initial_error: float, final_error: float) -> float:
    """Compute percentage error reduction.

    Args:
        initial_error: Initial graph error.
        final_error: Final graph error after optimization.

    Returns:
        Percentage error reduction (0-100).
    """
    if initial_error <= 0:
        logger.warning("Initial error is non-positive: %.6f", initial_error)
        return 0.0

    if final_error < 0:
        logger.warning("Final error is negative: %.6f", final_error)
        return 0.0

    reduction = (initial_error - final_error) / initial_error * 100.0
    return float(max(0.0, min(100.0, reduction)))  # Clamp to [0, 100]


def compute_trajectory_smoothness(
    poses: list[np.ndarray],
    timestamps: list[float],
) -> float:
    """Compute trajectory smoothness as mean acceleration magnitude.

    Lower values indicate smoother trajectories. Computed using
    second-order finite differences of position.

    Args:
        poses: List of 4x4 transformation matrices.
        timestamps: List of timestamps corresponding to poses.

    Returns:
        Mean acceleration magnitude (m/s²), or 0.0 if cannot compute.
    """
    if len(poses) != len(timestamps):
        logger.warning(
            "Mismatched poses (%d) and timestamps (%d)",
            len(poses),
            len(timestamps),
        )
        return 0.0

    if len(poses) < 3:
        logger.debug("Need at least 3 poses to compute smoothness")
        return 0.0

    try:
        positions = [np.array(pose[:3, 3], dtype=np.float64) for pose in poses]
    except (IndexError, TypeError) as exc:
        logger.warning("Failed to extract positions: %s", exc)
        return 0.0

    accelerations = []
    for idx in range(1, len(positions) - 1):
        dt = timestamps[idx + 1] - timestamps[idx - 1]
        if dt <= 0:
            logger.debug("Non-positive time delta at index %d", idx)
            continue

        # Second-order finite difference: a = (p[i+1] - 2*p[i] + p[i-1]) / (dt/2)^2
        acc = (positions[idx + 1] - 2 * positions[idx] + positions[idx - 1]) / ((dt / 2.0) ** 2)
        acc_norm = float(np.linalg.norm(acc))

        if np.isfinite(acc_norm):
            accelerations.append(acc_norm)

    if not accelerations:
        logger.warning("No valid acceleration samples computed")
        return 0.0

    mean_acc = float(np.mean(accelerations))
    logger.debug("Computed smoothness from %d samples: %.4f m/s²", len(accelerations), mean_acc)
    return mean_acc


def compute_pose_changes(
    poses_raw: list[np.ndarray],
    poses_opt: list[np.ndarray],
) -> dict[str, float]:
    """Compute statistics on pose changes from optimization.

    Args:
        poses_raw: List of original 4x4 transformation matrices.
        poses_opt: List of optimized 4x4 transformation matrices.

    Returns:
        Dictionary with max and mean frame changes in meters.
    """
    default = {"max_frame_change_m": 0.0, "mean_frame_change_m": 0.0}

    if not poses_raw or not poses_opt:
        logger.warning("Empty pose list provided")
        return default

    if len(poses_raw) != len(poses_opt):
        logger.warning(
            "Pose count mismatch: raw=%d, optimized=%d",
            len(poses_raw),
            len(poses_opt),
        )
        return default

    changes = []
    for idx, (raw_pose, opt_pose) in enumerate(zip(poses_raw, poses_opt)):
        try:
            raw_trans = raw_pose[:3, 3]
            opt_trans = opt_pose[:3, 3]
            change = float(np.linalg.norm(opt_trans - raw_trans))

            if np.isfinite(change):
                changes.append(change)
            else:
                logger.warning("Non-finite change at pose %d", idx)

        except (IndexError, TypeError) as exc:
            logger.warning("Failed to compute change for pose %d: %s", idx, exc)

    if not changes:
        return default

    return {
        "max_frame_change_m": float(np.max(changes)),
        "mean_frame_change_m": float(np.mean(changes)),
    }


def compute_trajectory_stats(
    poses: list[np.ndarray],
    timestamps: list[float],
) -> dict[str, float]:
    """Compute trajectory statistics.

    Args:
        poses: List of 4x4 transformation matrices.
        timestamps: List of timestamps corresponding to poses.

    Returns:
        Dictionary with total distance, duration, and average speed.
    """
    default = {"total_distance_m": 0.0, "duration_s": 0.0, "avg_speed_m_s": 0.0}

    if len(poses) != len(timestamps):
        logger.warning(
            "Mismatched poses (%d) and timestamps (%d)",
            len(poses),
            len(timestamps),
        )
        return default

    if len(poses) < 2:
        logger.debug("Need at least 2 poses for trajectory stats")
        return default

    try:
        positions = [np.array(pose[:3, 3], dtype=np.float64) for pose in poses]
    except (IndexError, TypeError) as exc:
        logger.warning("Failed to extract positions: %s", exc)
        return default

    distances = []
    for idx in range(len(positions) - 1):
        dist = float(np.linalg.norm(positions[idx + 1] - positions[idx]))
        if np.isfinite(dist):
            distances.append(dist)

    if not distances:
        return default

    total_distance = float(np.sum(distances))
    duration = float(timestamps[-1] - timestamps[0])
    avg_speed = total_distance / duration if duration > 0 else 0.0

    return {
        "total_distance_m": total_distance,
        "duration_s": max(0.0, duration),
        "avg_speed_m_s": max(0.0, avg_speed),
    }


def build_optimization_report(
    run_id: str,
    num_poses: int,
    initial_error: float,
    final_error: float,
    iterations: int,
    max_iterations: int,
    smoothness: float,
    pose_change_metrics: dict[str, float],
    trajectory_stats: dict[str, float],
) -> dict:
    """Build a comprehensive optimization report.

    Args:
        run_id: Run identifier.
        num_poses: Number of poses optimized.
        initial_error: Initial graph error.
        final_error: Final graph error.
        iterations: Actual iterations used.
        max_iterations: Maximum iterations allowed.
        smoothness: Trajectory smoothness metric.
        pose_change_metrics: Pose change statistics.
        trajectory_stats: Trajectory statistics.

    Returns:
        Report dictionary suitable for JSON serialization.
    """
    converged = iterations < max_iterations
    error_reduction = compute_error_reduction(initial_error, final_error)

    report = {
        "run_id": str(run_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_poses": int(num_poses),
        "optimization": {
            "converged": bool(converged),
            "iterations": int(iterations),
            "max_iterations": int(max_iterations),
            "initial_error": float(initial_error),
            "final_error": float(final_error),
            "error_reduction_percent": float(error_reduction),
        },
        "metrics": {
            "smoothness": float(smoothness),
            **{k: float(v) for k, v in pose_change_metrics.items()},
        },
        "trajectory": {k: float(v) for k, v in trajectory_stats.items()},
    }

    logger.debug("Built optimization report for %s", run_id)
    return report
