from __future__ import annotations

import copy
import logging
import shutil
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.utils.config import ensure_within_root
from scripts.utils.io_utils import write_json
from scripts.utils.pointcloud_utils import (
    _require_open3d,
    render_colored_point_cloud_comparison_views,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)
_SCREENSHOT_VIEW_NAMES = (
    "top_down",
    "bottom_up",
    "side_pos_x",
    "side_neg_x",
    "side_pos_y",
    "side_neg_y",
)


class PointCloudAlignmentError(Exception):
    """Raised when point cloud alignment fails."""


def get_run_dirs(data_dir: Path, run_ids: list[str]) -> list[Path]:
    if run_ids:
        return [data_dir / run_id for run_id in run_ids]

    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def select_reference_run(
    cfg: DictConfig,
    run_ids: list[str],
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}

    pointcloud_root = Path(cfg.pointcloud.output_root)
    input_cloud_name = str(cfg.pointcloud_alignment.input_cloud_name)
    centers: dict[str, np.ndarray] = {}

    for run_id in run_ids:
        cloud_path = pointcloud_root / run_id / input_cloud_name
        if not cloud_path.exists():
            raise PointCloudAlignmentError(
                f"Missing point cloud for reference selection: {cloud_path}"
            )

        cloud = _load_point_cloud(cloud_path)
        oobb = cloud.get_oriented_bounding_box()
        centers[run_id] = np.asarray(oobb.get_center(), dtype=float)

    for run_id in run_ids:
        scores[run_id] = 0.0

    for i, run_i in enumerate(run_ids):
        for j, run_j in enumerate(run_ids):
            if i == j:
                continue
            scores[run_i] += float(np.linalg.norm(centers[run_i] - centers[run_j]))

    reference_run = min(scores, key=scores.get)
    return reference_run, scores


def _select_reference_run(
    cfg: DictConfig,
    run_ids: list[str],
) -> tuple[str, dict[str, float]]:
    return select_reference_run(cfg, run_ids)


def _load_point_cloud(path: Path):
    o3d = _require_open3d()
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise PointCloudAlignmentError(f"Point cloud is empty: {path}")
    return pcd


def _prepare_downsampled(pcd, voxel_size: float):
    o3d = _require_open3d()
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    if pcd_down.is_empty():
        return pcd_down, None
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, fpfh


def _global_registration(cfg: DictConfig, source, target):
    o3d = _require_open3d()
    voxel_size = float(cfg.pointcloud_alignment.global_registration.voxel_size)
    source_down, source_fpfh = _prepare_downsampled(source, voxel_size)
    target_down, target_fpfh = _prepare_downsampled(target, voxel_size)

    if source_down.is_empty() or target_down.is_empty():
        raise PointCloudAlignmentError("Downsampled point clouds are empty")

    distance_threshold = voxel_size * float(
        cfg.pointcloud_alignment.global_registration.distance_threshold_factor
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        bool(cfg.pointcloud_alignment.global_registration.mutual_filter),
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        int(cfg.pointcloud_alignment.global_registration.ransac_n),
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                float(cfg.pointcloud_alignment.global_registration.edge_length_threshold)
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(cfg.pointcloud_alignment.global_registration.max_iterations),
            float(cfg.pointcloud_alignment.global_registration.confidence),
        ),
    )

    return result


def _refine_icp(cfg: DictConfig, source, target, init_transform: np.ndarray):
    o3d = _require_open3d()

    voxel_size = float(cfg.pointcloud_alignment.icp.voxel_size)
    source_down = source.voxel_down_sample(voxel_size=voxel_size)
    target_down = target.voxel_down_sample(voxel_size=voxel_size)

    if source_down.is_empty() or target_down.is_empty():
        raise PointCloudAlignmentError("ICP downsampled point clouds are empty")

    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(cfg.pointcloud_alignment.icp.max_iter)
    )

    method = str(cfg.pointcloud_alignment.icp.method).lower()
    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        float(cfg.pointcloud_alignment.icp.max_corr),
        init_transform,
        estimation,
        criteria,
    )

    return result


def _write_alignment_report(report_path: Path, payload: dict) -> None:
    write_json(report_path, payload)


def _visualize_alignment(reference_cloud, aligned_cloud, run_id: str, reference_run: str) -> None:
    o3d = _require_open3d()
    reference_vis = copy.deepcopy(reference_cloud)
    aligned_vis = copy.deepcopy(aligned_cloud)
    reference_vis.paint_uniform_color([0.0, 1.0, 0.0])
    aligned_vis.paint_uniform_color([1.0, 0.0, 0.0])
    window_name = f"Point cloud alignment: {run_id} vs {reference_run}"
    o3d.visualization.draw_geometries([reference_vis, aligned_vis], window_name=window_name)


def _has_all_alignment_screenshots(screenshots_dir: Path, prefix: str) -> bool:
    return all(
        (screenshots_dir / f"{prefix}_{view_name}.png").exists()
        for view_name in _SCREENSHOT_VIEW_NAMES
    )


def _render_alignment_screenshots(
    *,
    screenshots_cfg,
    clean_dir: Path,
    output_dir: Path,
    reference_cloud,
    aligned_cloud,
    run_id: str,
) -> None:
    screenshots_subdir = str(
        screenshots_cfg.get("output_subdir", "alignment_comparison_screenshots")
    )
    screenshots_dir = output_dir / screenshots_subdir
    ensure_within_root(
        screenshots_dir,
        clean_dir,
        "pointcloud_alignment.screenshots.output_subdir",
    )
    render_colored_point_cloud_comparison_views(
        reference_cloud,
        aligned_cloud,
        screenshots_dir,
        "reference_vs_aligned",
        width=int(screenshots_cfg.get("width", 1280)),
        height=int(screenshots_cfg.get("height", 720)),
        point_size=float(screenshots_cfg.get("point_size", 1.0)),
        zoom=float(screenshots_cfg.get("zoom", 0.7)),
        background_color=tuple(screenshots_cfg.get("background_color", (1.0, 1.0, 1.0))),
        reference_color=(0.0, 1.0, 0.0),
        aligned_color=(1.0, 0.0, 0.0),
    )
    logger.info("Alignment comparison screenshots saved for %s", run_id)


def run_for_run(
    cfg: DictConfig,
    run_id: str,
    reference_run: str,
    reference_cloud=None,
) -> dict:
    clean_dir = Path(cfg.paths.clean_data_dir)
    output_root = Path(cfg.pointcloud_alignment.output_root)
    report_dir = Path(cfg.pointcloud_alignment.report_dir)
    output_subdir = str(cfg.pointcloud_alignment.output_subdir)
    output_filtered_name = str(cfg.pointcloud_alignment.output_filtered_name)
    output_dense_name = str(cfg.pointcloud_alignment.output_dense_name)
    input_cloud_name = str(cfg.pointcloud_alignment.input_cloud_name)
    input_dense_name = str(cfg.pointcloud_alignment.input_dense_name)

    ensure_within_root(output_root, clean_dir, "pointcloud_alignment.output_root")
    ensure_within_root(report_dir, clean_dir, "pointcloud_alignment.report_dir")

    debug_vis_cfg = cfg.pointcloud_alignment.get("debug_visualization", {})
    debug_vis_enabled = bool(debug_vis_cfg.get("enabled", False))
    screenshots_cfg = cfg.pointcloud_alignment.get("screenshots", {})
    screenshots_enabled = bool(screenshots_cfg.get("enabled", False))

    reference_cloud_local = reference_cloud
    if reference_cloud_local is None:
        reference_cloud_path = (
            Path(cfg.pointcloud.output_root) / reference_run / input_cloud_name
        )
        reference_cloud_local = _load_point_cloud(reference_cloud_path)

    output_dir = output_root / output_subdir / run_id
    sentinel_path = output_dir / str(cfg.pointcloud_alignment.sentinel_filename)

    if run_id == reference_run:
        logger.info(
            "Reference run %s: copying source point cloud into aligned output",
            run_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        aligned_filtered_path = output_dir / output_filtered_name
        o3d = _require_open3d()
        o3d.io.write_point_cloud(
            str(aligned_filtered_path),
            reference_cloud_local,
            write_ascii=False,
        )

        reference_dense_path = Path(cfg.pointcloud.output_root) / run_id / input_dense_name
        aligned_dense_path = output_dir / output_dense_name
        if reference_dense_path.exists():
            shutil.copy2(reference_dense_path, aligned_dense_path)

        return {
            "run_id": run_id,
            "reference_run": reference_run,
            "skipped": True,
            "reason": "reference_run",
        }

    if bool(cfg.pointcloud_alignment.get("skip_if_sentinel_exists", False)) and sentinel_path.exists():
        optim_sentinel = (
            Path(cfg.optimization.output_root)
            / run_id
            / str(cfg.optimization.sentinel_filename)
        )
        pointcloud_sentinel = (
            Path(cfg.pointcloud.output_root)
            / run_id
            / str(cfg.pointcloud.sentinel_filename)
        )
        if not optim_sentinel.exists() or not pointcloud_sentinel.exists():
            try:
                sentinel_path.unlink()
                logger.info(
                    "Alignment sentinel is out of date for %s; deleted %s",
                    run_id,
                    sentinel_path,
                )
            except OSError as exc:
                logger.warning(
                    "Failed to delete stale alignment sentinel for %s: %s",
                    run_id,
                    exc,
                )
        else:
            if screenshots_enabled:
                screenshots_subdir = str(
                    screenshots_cfg.get("output_subdir", "alignment_comparison_screenshots")
                )
                screenshots_dir = output_dir / screenshots_subdir
                if not _has_all_alignment_screenshots(
                    screenshots_dir,
                    "reference_vs_aligned",
                ):
                    aligned_filtered_path = output_dir / output_filtered_name
                    if not aligned_filtered_path.exists():
                        logger.warning(
                            "Skipping screenshot backfill for %s; missing aligned cloud at %s",
                            run_id,
                            aligned_filtered_path,
                        )
                    else:
                        try:
                            aligned_cloud = _load_point_cloud(aligned_filtered_path)
                            _render_alignment_screenshots(
                                screenshots_cfg=screenshots_cfg,
                                clean_dir=clean_dir,
                                output_dir=output_dir,
                                reference_cloud=reference_cloud_local,
                                aligned_cloud=aligned_cloud,
                                run_id=run_id,
                            )
                        except Exception as exc:
                            fail_on_error = bool(screenshots_cfg.get("fail_on_error", True))
                            msg = (
                                "Failed to backfill alignment comparison screenshots for "
                                f"{run_id}: {exc}"
                            )
                            if fail_on_error:
                                raise RuntimeError(msg) from exc
                            logger.warning(msg)

            logger.info(
                "Skipping point cloud alignment for %s; sentinel exists at %s",
                run_id,
                sentinel_path,
            )
            return {
                "run_id": run_id,
                "reference_run": reference_run,
                "skipped": True,
            }

    source_cloud_path = Path(cfg.pointcloud.output_root) / run_id / input_cloud_name
    source_dense_path = Path(cfg.pointcloud.output_root) / run_id / input_dense_name

    if not source_cloud_path.exists():
        raise PointCloudAlignmentError(f"Missing point cloud: {source_cloud_path}")

    source_cloud = _load_point_cloud(source_cloud_path)

    global_result = None
    transform = np.eye(4)
    if cfg.pointcloud_alignment.global_registration.enabled:
        global_result = _global_registration(cfg, source_cloud, reference_cloud_local)
        transform = global_result.transformation

    icp_result = None
    if cfg.pointcloud_alignment.icp.enabled:
        icp_result = _refine_icp(cfg, source_cloud, reference_cloud_local, transform)
        transform = icp_result.transformation

    source_cloud.transform(transform)
    aligned_cloud = source_cloud

    if debug_vis_enabled:
        _visualize_alignment(reference_cloud_local, aligned_cloud, run_id, reference_run)

    aligned_filtered_path = output_dir / output_filtered_name
    aligned_dense_path = output_dir / output_dense_name
    output_dir.mkdir(parents=True, exist_ok=True)

    o3d = _require_open3d()
    o3d.io.write_point_cloud(str(aligned_filtered_path), aligned_cloud, write_ascii=False)

    if source_dense_path.exists():
        dense_cloud = _load_point_cloud(source_dense_path)
        dense_cloud.transform(transform)
        o3d.io.write_point_cloud(str(aligned_dense_path), dense_cloud, write_ascii=False)

    if screenshots_enabled:
        try:
            _render_alignment_screenshots(
                screenshots_cfg=screenshots_cfg,
                clean_dir=clean_dir,
                output_dir=output_dir,
                reference_cloud=reference_cloud_local,
                aligned_cloud=aligned_cloud,
                run_id=run_id,
            )
        except Exception as exc:
            fail_on_error = bool(screenshots_cfg.get("fail_on_error", True))
            msg = (
                "Failed to render alignment comparison screenshots for "
                f"{run_id}: {exc}"
            )
            if fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning(msg)

    result_payload = {
        "run_id": run_id,
        "reference_run": reference_run,
        "transformation": transform.tolist(),
    }

    if global_result is not None:
        result_payload["global_registration"] = {
            "fitness": float(global_result.fitness),
            "inlier_rmse": float(global_result.inlier_rmse),
        }

    if icp_result is not None:
        result_payload["icp"] = {
            "fitness": float(icp_result.fitness),
            "inlier_rmse": float(icp_result.inlier_rmse),
        }

    report_path = report_dir / f"{run_id}_pointcloud_alignment.json"
    _write_alignment_report(report_path, result_payload)

    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"completed_at={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to write point cloud alignment sentinel: %s", exc)

    return result_payload


def run(cfg: DictConfig, run_dirs: list[Path] | None = None) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(cfg.paths.raw_data_dir)
    clean_dir = Path(cfg.paths.clean_data_dir)
    output_root = Path(cfg.pointcloud_alignment.output_root)
    report_dir = Path(cfg.pointcloud_alignment.report_dir)

    ensure_within_root(output_root, clean_dir, "pointcloud_alignment.output_root")
    ensure_within_root(report_dir, clean_dir, "pointcloud_alignment.report_dir")

    selected_run_dirs = run_dirs
    if selected_run_dirs is None:
        selected_run_dirs = get_run_dirs(data_dir, list(cfg.pointcloud_alignment.run_ids))

    if not selected_run_dirs:
        raise PointCloudAlignmentError(f"No runs found in {data_dir}")

    run_ids = [run_dir.name for run_dir in selected_run_dirs]
    reference_run, scores = select_reference_run(cfg, run_ids)
    logger.info("Reference run selected: %s", reference_run)

    input_cloud_name = str(cfg.pointcloud_alignment.input_cloud_name)

    reference_cloud_path = (
        Path(cfg.pointcloud.output_root) / reference_run / input_cloud_name
    )
    reference_cloud = _load_point_cloud(reference_cloud_path)

    results = {
        "reference_run": reference_run,
        "scores": scores,
        "runs": {},
    }

    stage_start = perf_counter()
    total_runs = len(run_ids)
    aligned_count = 0
    skipped_count = 0
    for idx, run_id in enumerate(run_ids, start=1):
        run_start = perf_counter()
        logger.info("Point cloud alignment started for %s (%d/%d)", run_id, idx, total_runs)

        result_payload = run_for_run(
            cfg,
            run_id,
            reference_run,
            reference_cloud=reference_cloud,
        )
        results["runs"][run_id] = result_payload
        if bool(result_payload.get("skipped", False)):
            skipped_count += 1
        else:
            aligned_count += 1

        elapsed = perf_counter() - run_start
        status = "skipped" if bool(result_payload.get("skipped", False)) else "aligned"
        logger.info(
            "Point cloud alignment finished for %s (%d/%d, %s, %.2fs)",
            run_id,
            idx,
            total_runs,
            status,
            elapsed,
        )

    summary_path = report_dir / "pointcloud_alignment_summary.json"
    _write_alignment_report(summary_path, results)

    total_elapsed = perf_counter() - stage_start
    logger.info(
        "Point cloud alignment summary: total=%d, aligned=%d, skipped=%d, elapsed=%.2fs",
        total_runs,
        aligned_count,
        skipped_count,
        total_elapsed,
    )

    return results
