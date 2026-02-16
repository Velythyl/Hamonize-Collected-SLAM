import json
from pathlib import Path
from typing import Tuple

import numpy as np


class PoseFileError(Exception):
    """Raised when pose file is invalid or missing."""


class PoseValidationError(Exception):
    """Raised when pose matrix fails validation."""


def load_pose_matrix(filepath: Path) -> np.ndarray:
    """Load a 4x4 pose matrix from a JSON file."""
    if not filepath.exists():
        raise PoseFileError(f"Pose file not found: {filepath}")

    try:
        with filepath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseFileError(f"Failed to read pose file: {filepath}") from exc

    pose = np.array(data, dtype=np.float64)
    validate_pose_matrix(pose, filepath)
    return pose


def save_pose_matrix(pose: np.ndarray, filepath: Path) -> None:
    """Save a 4x4 pose matrix to a JSON file."""
    validate_pose_matrix(pose, filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as handle:
        json.dump(pose.tolist(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_pose_matrix(pose: np.ndarray, filepath: Path | None = None) -> None:
    """Validate pose matrix structure and rotation properties."""
    if pose.shape != (4, 4):
        location = f" in {filepath}" if filepath else ""
        raise PoseValidationError(f"Pose matrix is not 4x4{location}")

    if not np.isfinite(pose).all():
        location = f" in {filepath}" if filepath else ""
        raise PoseValidationError(f"Pose matrix contains NaN or Inf{location}")

    bottom_row = pose[3, :]
    if not np.allclose(bottom_row, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        location = f" in {filepath}" if filepath else ""
        raise PoseValidationError(f"Pose matrix bottom row invalid{location}")

    rotation = pose[:3, :3]
    identity = np.eye(3)
    if not np.allclose(rotation @ rotation.T, identity, atol=1e-3):
        location = f" in {filepath}" if filepath else ""
        raise PoseValidationError(f"Rotation matrix is not orthonormal{location}")

    det = np.linalg.det(rotation)
    if not np.isclose(det, 1.0, atol=1e-3):
        location = f" in {filepath}" if filepath else ""
        raise PoseValidationError(f"Rotation determinant not 1 (det={det}){location}")


def decompose_pose(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract translation vector and rotation matrix from pose."""
    translation = pose[:3, 3]
    rotation = pose[:3, :3]
    return translation, rotation


def get_first_pose(run_dir: Path) -> Tuple[float, np.ndarray]:
    """Return earliest timestamp and pose matrix for a run."""
    poses_dir = run_dir / "poses"
    if not poses_dir.exists():
        raise PoseFileError(f"Poses directory not found: {poses_dir}")

    pose_files = list(poses_dir.glob("*.json"))
    if not pose_files:
        raise PoseFileError(f"No pose files found in: {poses_dir}")

    timestamps = []
    for pose_file in pose_files:
        try:
            timestamps.append((float(pose_file.stem), pose_file))
        except ValueError as exc:
            raise PoseFileError(
                f"Pose filename is not a timestamp: {pose_file.name}"
            ) from exc

    timestamps.sort(key=lambda item: item[0])
    first_timestamp, first_file = timestamps[0]
    pose = load_pose_matrix(first_file)
    return first_timestamp, pose
