"""GTSAM-based pose graph optimization helpers.

This module provides utilities for building and optimizing pose graphs using GTSAM.
All operations include comprehensive error handling and logging to prevent silent failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from scripts.utils.pose_utils import PoseValidationError, validate_pose_matrix

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class GTSAMError(Exception):
    """Raised when GTSAM operations fail."""


class OptimizationError(Exception):
    """Raised when pose graph optimization fails."""


@dataclass
class OptimizationResult:
    """Results from pose graph optimization."""

    poses: list[np.ndarray]
    initial_error: float
    final_error: float
    iterations: int
    converged: bool


@dataclass
class BetweenFactorSpec:
    """Specification for an additional between factor."""

    i: int
    j: int
    relative_pose: np.ndarray
    sigma_rot: float
    sigma_trans: float
    robust: bool = True
    huber_k: float = 1.345
    label: str = "extra"


def _require_gtsam() -> tuple[Any, Any]:
    """Import and return GTSAM module and symbol shorthand.

    Returns:
        Tuple of (gtsam module, X symbol function)

    Raises:
        GTSAMError: If GTSAM is not installed or cannot be imported.
    """
    try:
        import gtsam
        from gtsam.symbol_shorthand import X

        logger.debug("GTSAM successfully imported (version: %s)", getattr(gtsam, "__version__", "unknown"))
        return gtsam, X
    except ImportError as exc:
        raise GTSAMError(
            "GTSAM is required for pose optimization. "
            "Install with 'conda install -c conda-forge gtsam' or 'pip install gtsam'."
        ) from exc
    except Exception as exc:
        raise GTSAMError(f"Failed to initialize GTSAM: {exc}") from exc


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method which is numerically stable.

    Args:
        rotation: 3x3 rotation matrix.

    Returns:
        Tuple (w, x, y, z) representing a unit quaternion.

    Raises:
        GTSAMError: If conversion fails.
    """
    try:
        # Shepperd's method for robust conversion
        m = rotation
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        # Normalize the quaternion
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-10:
            raise GTSAMError("Quaternion norm is near zero")

        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        return float(w), float(x), float(y), float(z)

    except GTSAMError:
        raise
    except Exception as exc:
        raise GTSAMError(f"Failed to convert rotation matrix to quaternion: {exc}") from exc


def _pose3_from_matrix(gtsam: Any, pose: np.ndarray, pose_index: int | None = None) -> Any:
    """Convert a 4x4 transformation matrix to GTSAM Pose3.

    Uses quaternion conversion to avoid GTSAM crashes from non-orthonormal matrices.

    Args:
        gtsam: The GTSAM module.
        pose: 4x4 homogeneous transformation matrix.
        pose_index: Optional index for error reporting.

    Returns:
        gtsam.Pose3 object.

    Raises:
        GTSAMError: If conversion fails.
    """
    context = f" (pose {pose_index})" if pose_index is not None else ""

    try:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]

        # Check for non-finite values first
        if not np.isfinite(rotation).all():
            raise GTSAMError(f"Rotation matrix contains non-finite values{context}")
        if not np.isfinite(translation).all():
            raise GTSAMError(f"Translation vector contains non-finite values{context}")

        # Check if rotation is roughly valid
        det = np.linalg.det(rotation)
        if not np.isfinite(det) or abs(det) < 0.1:
            raise GTSAMError(
                f"Rotation matrix is degenerate (det={det:.6f}){context}"
            )

        # Convert to quaternion - this is more robust than passing matrix directly
        w, x, y, z = _rotation_matrix_to_quaternion(rotation)

        # GTSAM Rot3.Quaternion takes (w, x, y, z) order
        rot3 = gtsam.Rot3.Quaternion(w, x, y, z)

        # Construct Pose3 from Rot3 + Point3 directly.
        # IMPORTANT: Do NOT use gtsam.Pose3(matrix) — it silently crashes
        # (segfault) in many GTSAM builds because the C++ Pose3(Matrix)
        # constructor feeds raw rotation elements into Rot3, which can fail
        # on non-perfectly-orthonormal matrices (especially in quaternion-mode
        # builds). The (Rot3, Point3) constructor is what all GTSAM tests use.
        point3 = gtsam.Point3(float(translation[0]), float(translation[1]), float(translation[2]))
        return gtsam.Pose3(rot3, point3)

    except GTSAMError:
        raise
    except Exception as exc:
        raise GTSAMError(f"Failed to create Pose3 from matrix{context}: {exc}") from exc


def _create_noise_model(gtsam: Any, sigma_rot: float, sigma_trans: float, label: str) -> Any:
    """Create a diagonal noise model for pose factors.

    Args:
        gtsam: The GTSAM module.
        sigma_rot: Standard deviation for rotation (radians).
        sigma_trans: Standard deviation for translation (meters).
        label: Label for error messages.

    Returns:
        gtsam.noiseModel.Diagonal object.

    Raises:
        GTSAMError: If noise model creation fails.
    """
    if sigma_rot <= 0:
        raise GTSAMError(f"{label}: rotation sigma must be positive, got {sigma_rot}")
    if sigma_trans <= 0:
        raise GTSAMError(f"{label}: translation sigma must be positive, got {sigma_trans}")

    try:
        sigmas = np.array(
            [sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans],
            dtype=np.float64,
        )
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    except Exception as exc:
        raise GTSAMError(f"Failed to create {label} noise model: {exc}") from exc


def _create_robust_noise_model(
    gtsam: Any,
    sigma_rot: float,
    sigma_trans: float,
    huber_k: float,
    label: str,
) -> Any:
    """Create a robust diagonal noise model using a Huber loss."""
    if huber_k <= 0:
        raise GTSAMError(f"{label}: huber_k must be positive, got {huber_k}")

    base = _create_noise_model(gtsam, sigma_rot, sigma_trans, label)
    try:
        huber = gtsam.noiseModel.mEstimator.Huber(huber_k)
        return gtsam.noiseModel.Robust.Create(huber, base)
    except Exception as exc:
        raise GTSAMError(f"Failed to create robust {label} noise model: {exc}") from exc


def _validate_and_prepare_poses(poses: list[np.ndarray]) -> list[np.ndarray]:
    """Validate and prepare poses for GTSAM processing.

    Args:
        poses: List of 4x4 transformation matrices.

    Returns:
        List of validated, contiguous pose matrices.

    Raises:
        PoseValidationError: If any pose is invalid.
        ValueError: If pose list is empty or too short.
    """
    if not poses:
        raise ValueError("Empty pose list provided")

    if len(poses) < 2:
        raise ValueError(f"Need at least 2 poses for optimization, got {len(poses)}")

    validated_poses = []
    for idx, pose in enumerate(poses):
        try:
            pose_array = np.asarray(pose, dtype=np.float64)
            validate_pose_matrix(pose_array)

            # Ensure C-contiguous for GTSAM
            if not pose_array.flags["C_CONTIGUOUS"]:
                pose_array = np.ascontiguousarray(pose_array)

            validated_poses.append(pose_array)

        except PoseValidationError as exc:
            raise PoseValidationError(f"Pose {idx} is invalid: {exc}") from exc
        except Exception as exc:
            raise PoseValidationError(f"Failed to process pose {idx}: {exc}") from exc

    logger.debug("Validated %d poses", len(validated_poses))
    return validated_poses


def build_pose_graph(
    poses: list[np.ndarray],
    prior_sigma_rot: float,
    prior_sigma_trans: float,
    odom_sigma_rot: float,
    odom_sigma_trans: float,
    extra_between: list[BetweenFactorSpec] | None = None,
) -> tuple[Any, Any]:
    """Build a pose graph with prior and odometry factors.

    Creates a factor graph with:
    - A prior factor on the first pose
    - Between factors connecting consecutive poses

    Args:
        poses: List of 4x4 transformation matrices.
        prior_sigma_rot: Prior rotation uncertainty (radians).
        prior_sigma_trans: Prior translation uncertainty (meters).
        odom_sigma_rot: Odometry rotation uncertainty (radians).
        odom_sigma_trans: Odometry translation uncertainty (meters).

    Returns:
        Tuple of (NonlinearFactorGraph, Values) for optimization.

    Raises:
        GTSAMError: If graph construction fails.
        PoseValidationError: If any pose is invalid.
        ValueError: If poses list is empty or parameters are invalid.
    """
    logger.info("Building pose graph with %d poses", len(poses) if poses else 0)

    gtsam, X = _require_gtsam()
    poses_list = _validate_and_prepare_poses(list(poses))
    num_poses = len(poses_list)

    logger.debug(
        "Noise parameters: prior(rot=%.4f, trans=%.4f), odom(rot=%.4f, trans=%.4f)",
        prior_sigma_rot,
        prior_sigma_trans,
        odom_sigma_rot,
        odom_sigma_trans,
    )

    try:
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
    except Exception as exc:
        raise GTSAMError(f"Failed to initialize factor graph: {exc}") from exc

    # Add all poses to initial estimate
    for idx, pose in enumerate(poses_list):
        try:
            pose3 = _pose3_from_matrix(gtsam, pose, idx)
            initial_estimate.insert(X(idx), pose3)
        except GTSAMError:
            raise
        except Exception as exc:
            raise GTSAMError(f"Failed to add pose {idx} to initial estimate: {exc}") from exc

    logger.debug("Added %d poses to initial estimate", num_poses)

    # Add prior factor on first pose
    prior_noise = _create_noise_model(gtsam, prior_sigma_rot, prior_sigma_trans, "prior")
    try:
        first_pose = _pose3_from_matrix(gtsam, poses_list[0], 0)
        prior_factor = gtsam.PriorFactorPose3(X(0), first_pose, prior_noise)
        graph.add(prior_factor)
        logger.debug("Added prior factor on pose 0")
    except GTSAMError:
        raise
    except Exception as exc:
        raise GTSAMError(f"Failed to add prior factor: {exc}") from exc

    # Add between factors for consecutive poses
    odom_noise = _create_noise_model(gtsam, odom_sigma_rot, odom_sigma_trans, "odometry")
    between_factors_added = 0

    for idx in range(num_poses - 1):
        try:
            pose_i = _pose3_from_matrix(gtsam, poses_list[idx], idx)
            pose_j = _pose3_from_matrix(gtsam, poses_list[idx + 1], idx + 1)
            relative = pose_i.between(pose_j)

            between_factor = gtsam.BetweenFactorPose3(X(idx), X(idx + 1), relative, odom_noise)
            graph.add(between_factor)
            between_factors_added += 1

        except GTSAMError:
            raise
        except Exception as exc:
            raise GTSAMError(f"Failed to add between factor for poses {idx}->{idx + 1}: {exc}") from exc

    # Add extra constraints (e.g., RGB-D, loop closures)
    extra_between = extra_between or []
    extra_factors_added = 0
    extra_counts: dict[str, int] = {}

    for spec in extra_between:
        if spec.i < 0 or spec.j < 0 or spec.i >= num_poses or spec.j >= num_poses:
            logger.warning("Skipping extra factor with out-of-range indices: %s", spec)
            continue
        if spec.i == spec.j:
            logger.warning("Skipping extra factor with identical indices: %s", spec)
            continue

        try:
            relative_pose = np.asarray(spec.relative_pose, dtype=np.float64)
            validate_pose_matrix(relative_pose)
        except (PoseValidationError, ValueError) as exc:
            logger.warning("Skipping extra factor with invalid pose (%s): %s", spec.label, exc)
            continue

        try:
            relative_pose3 = _pose3_from_matrix(gtsam, relative_pose)
            if spec.robust:
                extra_noise = _create_robust_noise_model(
                    gtsam,
                    spec.sigma_rot,
                    spec.sigma_trans,
                    spec.huber_k,
                    spec.label,
                )
            else:
                extra_noise = _create_noise_model(
                    gtsam,
                    spec.sigma_rot,
                    spec.sigma_trans,
                    spec.label,
                )

            graph.add(gtsam.BetweenFactorPose3(X(spec.i), X(spec.j), relative_pose3, extra_noise))
            extra_factors_added += 1
            extra_counts[spec.label] = extra_counts.get(spec.label, 0) + 1
        except GTSAMError:
            raise
        except Exception as exc:
            raise GTSAMError(f"Failed to add extra factor {spec.label} ({spec.i}->{spec.j}): {exc}") from exc

    logger.info(
        "Pose graph built: %d poses, 1 prior factor, %d odometry factors, %d extra factors",
        num_poses,
        between_factors_added,
        extra_factors_added,
    )
    if extra_counts:
        logger.info("Extra factor breakdown: %s", extra_counts)

    return graph, initial_estimate


def optimize_graph(
    graph: Any,
    initial_estimate: Any,
    max_iterations: int,
    relative_error_tol: float,
    absolute_error_tol: float,
    verbosity: str = "SILENT",
) -> tuple[Any, Any, float]:
    """Optimize the pose graph using Levenberg-Marquardt.

    Args:
        graph: GTSAM NonlinearFactorGraph.
        initial_estimate: GTSAM Values with initial pose estimates.
        max_iterations: Maximum optimization iterations.
        relative_error_tol: Relative error convergence tolerance.
        absolute_error_tol: Absolute error convergence tolerance.
        verbosity: GTSAM verbosity level ("SILENT", "SUMMARY", "TERMINATION", "LAMBDA", "TRYLAMBDA", "TRYCONFIG", "DAMPED", "TRYDELTA").

    Returns:
        Tuple of (optimized Values, optimizer, initial_error).

    Raises:
        OptimizationError: If optimization fails.
        GTSAMError: If GTSAM operations fail.
    """
    logger.info(
        "Starting optimization (max_iter=%d, rel_tol=%.2e, abs_tol=%.2e)",
        max_iterations,
        relative_error_tol,
        absolute_error_tol,
    )

    gtsam, _ = _require_gtsam()

    # Validate parameters
    if max_iterations <= 0:
        raise OptimizationError(f"max_iterations must be positive, got {max_iterations}")
    if relative_error_tol <= 0:
        raise OptimizationError(f"relative_error_tol must be positive, got {relative_error_tol}")
    if absolute_error_tol <= 0:
        raise OptimizationError(f"absolute_error_tol must be positive, got {absolute_error_tol}")

    # Compute initial error before optimization
    try:
        initial_error = graph.error(initial_estimate)
        logger.debug("Initial graph error: %.6f", initial_error)
    except Exception as exc:
        raise OptimizationError(f"Failed to compute initial error: {exc}") from exc

    if not np.isfinite(initial_error):
        raise OptimizationError(f"Initial error is not finite: {initial_error}")

    # Configure optimizer
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        params.setRelativeErrorTol(relative_error_tol)
        params.setAbsoluteErrorTol(absolute_error_tol)

        # Set verbosity safely
        valid_verbosities = {"SILENT", "SUMMARY", "TERMINATION", "LAMBDA", "TRYLAMBDA", "TRYCONFIG", "DAMPED", "TRYDELTA"}
        if verbosity.upper() not in valid_verbosities:
            logger.warning("Unknown verbosity '%s', using SILENT", verbosity)
            verbosity = "SILENT"
        params.setVerbosityLM(verbosity.upper())

    except Exception as exc:
        raise OptimizationError(f"Failed to configure optimizer: {exc}") from exc

    # Run optimization
    try:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
    except Exception as exc:
        raise OptimizationError(f"Optimization failed: {exc}") from exc

    # Validate result
    try:
        final_error = graph.error(result)
        iterations = optimizer.iterations()
    except Exception as exc:
        raise OptimizationError(f"Failed to extract optimization results: {exc}") from exc

    if not np.isfinite(final_error):
        raise OptimizationError(f"Final error is not finite: {final_error}")

    converged = iterations < max_iterations
    error_reduction = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0

    logger.info(
        "Optimization complete: %d iterations, error %.6f -> %.6f (%.1f%% reduction)%s",
        iterations,
        initial_error,
        final_error,
        error_reduction,
        "" if converged else " [DID NOT CONVERGE]",
    )

    return result, optimizer, initial_error


def extract_poses(result: Any, num_poses: int) -> list[np.ndarray]:
    """Extract optimized poses from GTSAM result.

    Args:
        result: GTSAM Values containing optimized poses.
        num_poses: Number of poses to extract.

    Returns:
        List of 4x4 transformation matrices.

    Raises:
        GTSAMError: If pose extraction fails.
    """
    if num_poses <= 0:
        raise GTSAMError(f"num_poses must be positive, got {num_poses}")

    _, X = _require_gtsam()

    poses = []
    for idx in range(num_poses):
        try:
            pose3 = result.atPose3(X(idx))
            matrix = np.array(pose3.matrix(), dtype=np.float64)

            # Validate extracted pose
            if not np.isfinite(matrix).all():
                raise GTSAMError(f"Extracted pose {idx} contains non-finite values")

            poses.append(matrix)

        except GTSAMError:
            raise
        except Exception as exc:
            raise GTSAMError(f"Failed to extract pose {idx}: {exc}") from exc

    logger.debug("Extracted %d optimized poses", len(poses))
    return poses


def run_pose_optimization(
    poses: list[np.ndarray],
    prior_sigma_rot: float,
    prior_sigma_trans: float,
    odom_sigma_rot: float,
    odom_sigma_trans: float,
    extra_between: list[BetweenFactorSpec] | None = None,
    max_iterations: int = 100,
    relative_error_tol: float = 1e-5,
    absolute_error_tol: float = 1e-5,
    verbosity: str = "SILENT",
) -> OptimizationResult:
    """Run complete pose graph optimization pipeline.

    This is the main entry point that combines graph building, optimization,
    and pose extraction into a single call with comprehensive error handling.

    Args:
        poses: List of 4x4 transformation matrices.
        prior_sigma_rot: Prior rotation uncertainty (radians).
        prior_sigma_trans: Prior translation uncertainty (meters).
        odom_sigma_rot: Odometry rotation uncertainty (radians).
        odom_sigma_trans: Odometry translation uncertainty (meters).
        max_iterations: Maximum optimization iterations.
        relative_error_tol: Relative error convergence tolerance.
        absolute_error_tol: Absolute error convergence tolerance.
        verbosity: GTSAM verbosity level.

    Returns:
        OptimizationResult containing optimized poses and metrics.

    Raises:
        GTSAMError: If GTSAM operations fail.
        OptimizationError: If optimization fails.
        PoseValidationError: If input poses are invalid.
    """
    logger.info("Starting pose optimization pipeline with %d poses", len(poses))

    # Build the pose graph
    graph, initial_estimate = build_pose_graph(
        poses,
        prior_sigma_rot,
        prior_sigma_trans,
        odom_sigma_rot,
        odom_sigma_trans,
        extra_between=extra_between,
    )

    # Optimize
    result, optimizer, initial_error = optimize_graph(
        graph,
        initial_estimate,
        max_iterations,
        relative_error_tol,
        absolute_error_tol,
        verbosity,
    )

    # Extract results
    optimized_poses = extract_poses(result, len(poses))
    final_error = graph.error(result)
    iterations = optimizer.iterations()
    converged = iterations < max_iterations

    return OptimizationResult(
        poses=optimized_poses,
        initial_error=initial_error,
        final_error=final_error,
        iterations=iterations,
        converged=converged,
    )
