"""Pose optimization module.

This module provides pose graph optimization for camera trajectories using GTSAM.
It processes raw poses, optimizes them to reduce drift and smooth trajectories,
and outputs both optimized poses and detailed optimization reports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.utils.calibration_utils import load_calibration
from scripts.utils.config import ensure_within_root
from scripts.utils.gtsam_helpers import (
    GTSAMError,
    OptimizationError,
    run_pose_optimization,
)
from scripts.utils.io_utils import write_json
from scripts.utils.optimization_metrics import (
    build_optimization_report,
    compute_pose_changes,
    compute_trajectory_smoothness,
    compute_trajectory_stats,
)
from scripts.utils.pose_utils import PoseFileError, PoseValidationError, load_pose_matrix, save_pose_matrix
from scripts.utils.rgbd_constraints import RgbdFrame, build_rgbd_constraints

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PoseOptimizationError(Exception):
    """Raised when pose optimization fails for a run."""


@dataclass
class PoseData:
    """Container for loaded pose data."""

    timestamps: list[float]
    names: list[str]
    file_paths: list[Path]
    matrices: list[np.ndarray]


def get_run_dirs(
    data_dir: Path,
    run_ids: list[str],
    num_runs_to_process: int | None,
) -> list[Path]:
    """Get run directories from data directory.

    Args:
        data_dir: Base data directory containing runs.
        run_ids: Specific run IDs to process, or empty for all.

    Returns:
        List of run directory paths.
    """
    if run_ids:
        dirs = [data_dir / run_id for run_id in run_ids]
        logger.debug("Using %d specified run IDs", len(dirs))
        return dirs

    dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if num_runs_to_process is None or num_runs_to_process <= 0:
        logger.debug("Found %d run directories", len(dirs))
        return dirs

    limited = dirs[:num_runs_to_process]
    logger.debug("Using first %d run directories", len(limited))
    return limited


def load_pose_series(poses_dir: Path) -> PoseData:
    """Load all poses from a directory, sorted by timestamp.

    Args:
        poses_dir: Directory containing pose JSON files named by timestamp.

    Returns:
        PoseData with timestamps, names, file paths, and matrices.

    Raises:
        PoseFileError: If no poses found or files cannot be read.
        PoseValidationError: If any pose matrix is invalid.
    """
    if not poses_dir.exists():
        raise PoseFileError(f"Poses directory does not exist: {poses_dir}")

    pose_files = list(poses_dir.glob("*.json"))
    if not pose_files:
        raise PoseFileError(f"No pose files found in: {poses_dir}")

    logger.debug("Found %d pose files in %s", len(pose_files), poses_dir)

    # Parse timestamps and sort
    entries: list[tuple[float, str, Path]] = []
    for pose_file in pose_files:
        try:
            timestamp = float(pose_file.stem)
            entries.append((timestamp, pose_file.stem, pose_file))
        except ValueError as exc:
            raise PoseFileError(f"Pose filename is not a valid timestamp: {pose_file.name}") from exc

    entries.sort(key=lambda item: item[0])

    # Validate temporal ordering
    timestamps = [e[0] for e in entries]
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            raise PoseFileError(
                f"Timestamps not strictly increasing: {timestamps[i - 1]} >= {timestamps[i]}"
            )

    # Load poses
    matrices: list[np.ndarray] = []
    for idx, (_, name, filepath) in enumerate(entries):
        try:
            matrix = load_pose_matrix(filepath)
            matrices.append(matrix)
        except (PoseFileError, PoseValidationError) as exc:
            raise PoseFileError(f"Failed to load pose {idx} ({name}): {exc}") from exc

    logger.info(
        "Loaded %d poses (timestamps %.3f to %.3f)",
        len(matrices),
        timestamps[0],
        timestamps[-1],
    )

    return PoseData(
        timestamps=timestamps,
        names=[e[1] for e in entries],
        file_paths=[e[2] for e in entries],
        matrices=matrices,
    )


def validate_run_directory(run_dir: Path) -> None:
    """Validate that a run directory has required data.

    Args:
        run_dir: Path to run directory.

    Raises:
        PoseOptimizationError: If required data is missing.
    """
    if not run_dir.exists():
        raise PoseOptimizationError(f"Run directory does not exist: {run_dir}")

    poses_dir = run_dir / "poses"
    if not poses_dir.exists():
        raise PoseOptimizationError(f"Poses directory missing: {poses_dir}")

    # Check for at least some pose files
    pose_files = list(poses_dir.glob("*.json"))
    if len(pose_files) < 2:
        raise PoseOptimizationError(
            f"Need at least 2 poses for optimization, found {len(pose_files)} in {poses_dir}"
        )


def save_optimized_poses(
    pose_names: list[str],
    optimized_poses: list[np.ndarray],
    output_dir: Path,
) -> None:
    """Save optimized poses to output directory.

    Args:
        pose_names: List of pose names (used as filenames).
        optimized_poses: List of optimized pose matrices.
        output_dir: Output directory path.

    Raises:
        PoseOptimizationError: If saving fails.
    """
    if len(pose_names) != len(optimized_poses):
        raise PoseOptimizationError(
            f"Mismatch: {len(pose_names)} names but {len(optimized_poses)} poses"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Saving %d optimized poses to %s", len(optimized_poses), output_dir)

    for pose_name, pose in zip(pose_names, optimized_poses):
        output_file = output_dir / f"{pose_name}.json"
        try:
            save_pose_matrix(pose, output_file)
        except (PoseValidationError, OSError) as exc:
            raise PoseOptimizationError(f"Failed to save pose {pose_name}: {exc}") from exc


def run_for_run(cfg: DictConfig, run_dir: Path) -> dict:
    """Run pose optimization for a single run.

    Args:
        cfg: Configuration object.
        run_dir: Path to run directory.

    Returns:
        Optimization report dictionary.

    Raises:
        PoseOptimizationError: If optimization fails.
    """
    run_id = run_dir.name
    logger.info("=" * 60)
    logger.info("Starting pose optimization for run: %s", run_id)
    logger.info("=" * 60)

    # Validate configuration paths
    output_root = Path(cfg.optimization.output_root)
    report_dir = Path(cfg.optimization.report_dir)

    try:
        ensure_within_root(output_root, Path(cfg.paths.clean_data_dir), "optimization.output_root")
        ensure_within_root(report_dir, Path(cfg.paths.clean_data_dir), "optimization.report_dir")
    except ValueError as exc:
        raise PoseOptimizationError(f"Invalid configuration: {exc}") from exc

    sentinel_path = output_root / run_id / str(cfg.optimization.sentinel_filename)
    if bool(cfg.optimization.get("skip_if_sentinel_exists", False)) and sentinel_path.exists():
        logger.info(
            "Skipping optimization for %s; sentinel exists at %s",
            run_id,
            sentinel_path,
        )
        return {"run_id": run_id, "skipped": True}

    # Validate run directory
    validate_run_directory(run_dir)

    # Load poses
    poses_dir = run_dir / "poses"
    try:
        pose_data = load_pose_series(poses_dir)
    except (PoseFileError, PoseValidationError) as exc:
        raise PoseOptimizationError(f"Failed to load poses for {run_id}: {exc}") from exc

    num_poses = len(pose_data.matrices)
    logger.info("Loaded %d poses for optimization", num_poses)

    rgbd_constraints = []
    rgbd_stats = {}
    if cfg.optimization.rgbd.enabled:
        logger.info("Building RGB-D constraints for %s", run_id)
        if not cfg.optimization.rgbd.loop_closure.enabled:
            logger.info("RGB-D loop closure disabled; using sequential RGB-D constraints only")
        frames: list[RgbdFrame] = []
        for idx, timestamp in enumerate(pose_data.names):
            rgb_file = run_dir / "rgb" / f"{timestamp}.jpg"
            depth_file = run_dir / "depth" / f"{timestamp}.png"
            calib_file = run_dir / "calib" / f"{timestamp}.yaml"

            if not rgb_file.exists():
                raise PoseOptimizationError(f"Missing RGB file: {rgb_file}")
            if not depth_file.exists():
                raise PoseOptimizationError(f"Missing depth file: {depth_file}")
            if not calib_file.exists():
                raise PoseOptimizationError(f"Missing calibration file: {calib_file}")

            calib = load_calibration(calib_file)
            frames.append(
                RgbdFrame(
                    index=idx,
                    timestamp=timestamp,
                    rgb_path=rgb_file,
                    depth_path=depth_file,
                    calib=calib,
                )
            )

        rgbd_constraints, rgbd_stats = build_rgbd_constraints(
            frames=frames,
            poses=pose_data.matrices,
            cfg=cfg.optimization.rgbd,
        )
    else:
        logger.info("RGB-D constraints disabled; using odometry-only pose optimization")

    # Run optimization
    try:
        opt_result = run_pose_optimization(
            poses=pose_data.matrices,
            prior_sigma_rot=cfg.optimization.prior_sigma_rot,
            prior_sigma_trans=cfg.optimization.prior_sigma_trans,
            odom_sigma_rot=cfg.optimization.odometry_sigma_rot,
            odom_sigma_trans=cfg.optimization.odometry_sigma_trans,
            extra_between=rgbd_constraints,
            max_iterations=cfg.optimization.max_iterations,
            relative_error_tol=cfg.optimization.relative_error_tol,
            absolute_error_tol=cfg.optimization.absolute_error_tol,
            verbosity=cfg.optimization.verbosity,
        )
    except (GTSAMError, OptimizationError) as exc:
        raise PoseOptimizationError(f"Optimization failed for {run_id}: {exc}") from exc

    logger.info(
        "Optimization complete: %d iterations, error %.6f -> %.6f (converged: %s)",
        opt_result.iterations,
        opt_result.initial_error,
        opt_result.final_error,
        opt_result.converged,
    )

    # Save optimized poses
    output_dir = output_root / run_id / cfg.optimization.poses_subdir
    try:
        save_optimized_poses(pose_data.names, opt_result.poses, output_dir)
        logger.info("Saved optimized poses to %s", output_dir)
    except PoseOptimizationError:
        raise
    except Exception as exc:
        raise PoseOptimizationError(f"Failed to save optimized poses: {exc}") from exc

    # Compute metrics
    try:
        smoothness = compute_trajectory_smoothness(opt_result.poses, pose_data.timestamps)
        pose_changes = compute_pose_changes(pose_data.matrices, opt_result.poses)
        trajectory_stats = compute_trajectory_stats(opt_result.poses, pose_data.timestamps)
    except Exception as exc:
        logger.warning("Failed to compute some metrics: %s", exc)
        smoothness = 0.0
        pose_changes = {"max_frame_change_m": 0.0, "mean_frame_change_m": 0.0}
        trajectory_stats = {"total_distance_m": 0.0, "duration_s": 0.0, "avg_speed_m_s": 0.0}

    # Build and save report
    report = build_optimization_report(
        run_id=run_id,
        num_poses=num_poses,
        initial_error=opt_result.initial_error,
        final_error=opt_result.final_error,
        iterations=opt_result.iterations,
        max_iterations=cfg.optimization.max_iterations,
        smoothness=smoothness,
        pose_change_metrics=pose_changes,
        trajectory_stats=trajectory_stats,
    )
    if rgbd_stats:
        report["rgbd_constraints"] = rgbd_stats

    report_path = report_dir / f"{run_id}_optimization_report.json"
    try:
        write_json(report_path, report)
        logger.info("Saved optimization report to %s", report_path)
    except Exception as exc:
        logger.warning("Failed to save report: %s", exc)

    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"completed_at={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        logger.debug("Wrote optimization sentinel to %s", sentinel_path)
    except OSError as exc:
        logger.warning("Failed to write optimization sentinel: %s", exc)

    logger.info(
        "Pose optimization completed for %s (poses: %d, converged: %s)",
        run_id,
        num_poses,
        opt_result.converged,
    )

    return report


def run(cfg: DictConfig) -> None:
    """Run pose optimization for all configured runs.

    Args:
        cfg: Configuration object.

    Raises:
        ValueError: If no runs found.
        PoseOptimizationError: If optimization fails.
    """
    # Configure logging if not already configured
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(cfg.paths.raw_data_dir)
    run_dirs = get_run_dirs(data_dir, list(cfg.optimization.run_ids))

    if not run_dirs:
        raise ValueError(f"No runs found in {data_dir}")

    logger.info("Processing %d runs for pose optimization", len(run_dirs))

    failed_runs: list[tuple[str, str]] = []
    successful_runs: list[str] = []

    for run_dir in run_dirs:
        try:
            run_for_run(cfg, run_dir)
            successful_runs.append(run_dir.name)
        except PoseOptimizationError as exc:
            logger.error("Failed to optimize %s: %s", run_dir.name, exc)
            failed_runs.append((run_dir.name, str(exc)))

    # Summary
    logger.info("=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Successful: %d runs", len(successful_runs))
    logger.info("Failed: %d runs", len(failed_runs))

    if failed_runs:
        logger.warning("Failed runs:")
        for run_id, error in failed_runs:
            logger.warning("  - %s: %s", run_id, error)

    if failed_runs and not successful_runs:
        raise PoseOptimizationError("All runs failed optimization")
