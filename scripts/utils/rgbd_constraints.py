"""RGB-D constraint generation for pose graph optimization."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scripts.utils.gtsam_helpers import BetweenFactorSpec
from scripts.utils.pointcloud_utils import filter_depth, load_depth_image, load_rgb_image

logger = logging.getLogger(__name__)


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for RGB-D constraints. Install with 'pip install open3d'."
        ) from exc
    return o3d


@dataclass
class RgbdFrame:
    """Container for RGB-D frame metadata."""

    index: int
    timestamp: str
    rgb_path: Path
    depth_path: Path
    calib: dict


@dataclass
class RgbdConstraintStats:
    """Statistics for RGB-D constraint generation."""

    sequential_attempted: int = 0
    sequential_added: int = 0
    loop_attempted: int = 0
    loop_added: int = 0

    def as_dict(self) -> dict:
        return {
            "sequential_attempted": int(self.sequential_attempted),
            "sequential_added": int(self.sequential_added),
            "loop_attempted": int(self.loop_attempted),
            "loop_added": int(self.loop_added),
        }


def _rotation_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    r_rel = r1.T @ r2
    trace = np.trace(r_rel)
    value = (trace - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def _create_intrinsic(o3d, calib: dict) -> object:
    return o3d.camera.PinholeCameraIntrinsic(
        int(calib["width"]),
        int(calib["height"]),
        float(calib["K"][0, 0]),
        float(calib["K"][1, 1]),
        float(calib["K"][0, 2]),
        float(calib["K"][1, 2]),
    )


def _build_rgbd_image(o3d, frame: RgbdFrame, cfg) -> object:
    rgb = load_rgb_image(frame.rgb_path)
    depth = load_depth_image(frame.depth_path, cfg.depth_scale)
    depth = filter_depth(depth, cfg.min_depth, cfg.max_depth)

    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=float(cfg.max_depth),
        convert_rgb_to_intensity=False,
    )


def _build_point_cloud(o3d, rgbd: object, intrinsic: object, voxel_size: float) -> object:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals()
    return pcd


def _register_pair(
    o3d,
    source_frame: RgbdFrame,
    target_frame: RgbdFrame,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    cfg,
    cache: dict,
    max_corr: float,
) -> tuple[bool, np.ndarray, float, float]:
    intrinsic = _create_intrinsic(o3d, source_frame.calib)
    if source_frame.index not in cache:
        cache[source_frame.index] = _build_rgbd_image(o3d, source_frame, cfg)
    if target_frame.index not in cache:
        cache[target_frame.index] = _build_rgbd_image(o3d, target_frame, cfg)

    rgbd_source = cache[source_frame.index]
    rgbd_target = cache[target_frame.index]

    t_init = np.linalg.inv(source_pose) @ target_pose
    transformation = t_init

    if cfg.odometry.enabled:
        option = o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=float(cfg.odometry.max_depth_diff),
        )
        if hasattr(cfg.odometry, "iteration_pyramid"):
            option.iteration_number_per_pyramid_level = o3d.utility.IntVector(
                list(cfg.odometry.iteration_pyramid)
            )
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        success, t_odom, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd_source,
            rgbd_target,
            intrinsic,
            transformation,
            jacobian,
            option,
        )
        if not success:
            return False, transformation, 0.0, float("inf")
        transformation = t_odom

    fitness = 1.0
    rmse = 0.0
    if cfg.icp.enabled:
        pcd_source = _build_point_cloud(o3d, rgbd_source, intrinsic, cfg.icp.voxel_size)
        pcd_target = _build_point_cloud(o3d, rgbd_target, intrinsic, cfg.icp.voxel_size)
        if len(pcd_source.points) == 0 or len(pcd_target.points) == 0:
            return False, transformation, 0.0, float("inf")

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(cfg.icp.max_iter)
        )
        reg = o3d.pipelines.registration.registration_icp(
            pcd_source,
            pcd_target,
            float(max_corr),
            transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        transformation = reg.transformation
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)

    return True, transformation, fitness, rmse


def _compute_sigma(cfg, rmse: float) -> tuple[float, float]:
    sigma_trans = max(float(cfg.noise.sigma_trans), float(cfg.noise.rmse_scale) * rmse)
    sigma_rot = max(float(cfg.noise.sigma_rot), float(cfg.noise.rmse_scale) * rmse)
    return sigma_rot, sigma_trans


def _normalize_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    r_norm = u @ vt
    if np.linalg.det(r_norm) < 0:
        u[:, -1] *= -1
        r_norm = u @ vt
    normalized = np.array(transform, dtype=np.float64, copy=True)
    normalized[:3, :3] = r_norm
    normalized[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return normalized


def build_rgbd_constraints(
    frames: list[RgbdFrame],
    poses: list[np.ndarray],
    cfg,
) -> tuple[list[BetweenFactorSpec], dict]:
    if not frames or not poses:
        return [], {}
    if len(frames) != len(poses):
        raise ValueError("RGB-D frames and pose counts do not match")
    if not cfg.enabled:
        return [], {}

    o3d = _require_open3d()
    constraints: list[BetweenFactorSpec] = []
    stats = RgbdConstraintStats()
    cache: dict[int, object] = {}
    thread_cache: dict[int, dict[int, object]] = {}
    thread_cache_lock = threading.Lock()

    def _get_thread_cache() -> dict[int, object]:
        thread_id = threading.get_ident()
        with thread_cache_lock:
            if thread_id not in thread_cache:
                thread_cache[thread_id] = {}
            return thread_cache[thread_id]

    def _get_max_workers() -> int:
        if hasattr(cfg, "concurrency") and hasattr(cfg.concurrency, "max_workers"):
            return int(cfg.concurrency.max_workers)
        if hasattr(cfg, "max_workers"):
            return int(cfg.max_workers)
        return int(os.cpu_count() or 1)

    # Sequential constraints
    if cfg.sequential.enabled:
        stride = max(1, int(cfg.sequential.stride))
        def _process_sequential_pair(i: int) -> tuple[int, bool, bool, BetweenFactorSpec | None]:
            j = i + 1
            local_cache = _get_thread_cache()
            success, transform, fitness, rmse = _register_pair(
                o3d,
                frames[i],
                frames[j],
                poses[i],
                poses[j],
                cfg,
                local_cache,
                float(cfg.icp.max_corr),
            )
            if not success:
                return i, True, False, None
            if fitness < float(cfg.sequential.min_fitness) or rmse > float(cfg.sequential.max_rmse):
                return i, True, False, None
            sigma_rot, sigma_trans = _compute_sigma(cfg, rmse)
            transform = _normalize_transform(transform)
            spec = BetweenFactorSpec(
                i=i,
                j=j,
                relative_pose=np.asarray(transform, dtype=np.float64),
                sigma_rot=sigma_rot,
                sigma_trans=sigma_trans,
                robust=True,
                huber_k=float(cfg.noise.huber_k),
                label="rgbd_sequential",
            )
            return i, True, True, spec

        indices = list(range(0, len(frames) - 1, stride))
        max_workers = _get_max_workers()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_sequential_pair, i) for i in indices]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="RGB-D sequential constraints",
                leave=False,
            ):
                i, attempted, added, spec = future.result()
                if attempted:
                    stats.sequential_attempted += 1
                if added and spec is not None:
                    constraints.append(spec)
                    stats.sequential_added += 1

        with thread_cache_lock:
            for local_cache in thread_cache.values():
                for key, value in local_cache.items():
                    if key not in cache:
                        cache[key] = value

    # Loop closures
    if cfg.loop_closure.enabled:
        stride = max(1, int(cfg.loop_closure.stride))
        min_sep = max(1, int(cfg.loop_closure.min_separation))
        max_candidates = max(1, int(cfg.loop_closure.max_candidates))
        def _process_loop_pair(pair: tuple[int, int]) -> tuple[bool, bool, BetweenFactorSpec | None]:
            i, j = pair
            local_cache = _get_thread_cache()
            success, transform, fitness, rmse = _register_pair(
                o3d,
                frames[i],
                frames[j],
                poses[i],
                poses[j],
                cfg,
                local_cache,
                float(cfg.loop_closure.max_corr),
            )
            if not success:
                return True, False, None
            if fitness < float(cfg.loop_closure.min_fitness) or rmse > float(
                cfg.loop_closure.max_rmse
            ):
                return True, False, None
            sigma_rot, sigma_trans = _compute_sigma(cfg, rmse)
            transform = _normalize_transform(transform)
            spec = BetweenFactorSpec(
                i=i,
                j=j,
                relative_pose=np.asarray(transform, dtype=np.float64),
                sigma_rot=sigma_rot,
                sigma_trans=sigma_trans,
                robust=True,
                huber_k=float(cfg.noise.huber_k),
                label="rgbd_loop",
            )
            return True, True, spec

        candidate_pairs: list[tuple[int, int]] = []
        for i in range(0, len(frames) - min_sep, stride):
            t_i = poses[i][:3, 3]
            r_i = poses[i][:3, :3]
            candidates = []
            for j in range(i + min_sep, len(frames), stride):
                t_j = poses[j][:3, 3]
                r_j = poses[j][:3, :3]
                dist = float(np.linalg.norm(t_j - t_i))
                angle = _rotation_angle_deg(r_i, r_j)
                if dist <= float(cfg.loop_closure.max_distance_m) and angle <= float(
                    cfg.loop_closure.max_angle_deg
                ):
                    candidates.append((dist, j))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            for _, j in candidates[:max_candidates]:
                candidate_pairs.append((i, j))

        max_workers = _get_max_workers()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_loop_pair, pair) for pair in candidate_pairs]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="RGB-D loop closures",
                leave=False,
            ):
                attempted, added, spec = future.result()
                if attempted:
                    stats.loop_attempted += 1
                if added and spec is not None:
                    constraints.append(spec)
                    stats.loop_added += 1

        with thread_cache_lock:
            for local_cache in thread_cache.values():
                for key, value in local_cache.items():
                    if key not in cache:
                        cache[key] = value

    logger.info(
        "RGB-D constraints: sequential %d/%d, loop %d/%d",
        stats.sequential_added,
        stats.sequential_attempted,
        stats.loop_added,
        stats.loop_attempted,
    )

    return constraints, stats.as_dict()
