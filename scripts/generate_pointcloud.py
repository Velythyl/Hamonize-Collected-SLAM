from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig

from scripts.utils.calibration_utils import load_calibration
from scripts.utils.config import ensure_within_root
from scripts.utils.io_utils import write_json
from scripts.utils.pointcloud_utils import (
    create_point_cloud_from_rgbd,
    filter_depth,
    filter_point_cloud,
    load_depth_image,
    load_rgb_image,
    render_point_cloud_views,
    save_point_cloud,
)
from scripts.utils.pose_utils import load_pose_matrix


def get_run_dirs(data_dir: Path, run_ids: list[str]) -> list[Path]:
    if run_ids:
        return [data_dir / run_id for run_id in run_ids]

    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def load_pose_files(poses_dir: Path) -> list[Path]:
    pose_files = list(poses_dir.glob("*.json"))
    if not pose_files:
        raise ValueError(f"No optimized poses found in: {poses_dir}")

    timestamps = []
    for pose_file in pose_files:
        try:
            timestamps.append((float(pose_file.stem), pose_file))
        except ValueError as exc:
            raise ValueError(
                f"Pose filename is not a timestamp: {pose_file.name}"
            ) from exc

    timestamps.sort(key=lambda item: item[0])
    return [item[1] for item in timestamps]


def run_for_run(cfg: DictConfig, run_dir: Path) -> None:
    output_root = Path(cfg.pointcloud.output_root)
    report_dir = Path(cfg.pointcloud.report_dir)

    ensure_within_root(output_root, Path(cfg.paths.clean_data_dir), "pointcloud.output_root")
    ensure_within_root(report_dir, Path(cfg.paths.clean_data_dir), "pointcloud.report_dir")

    sentinel_path = output_root / run_dir.name / str(cfg.pointcloud.sentinel_filename)
    optim_sentinel = (
        Path(cfg.optimization.output_root)
        / run_dir.name
        / str(cfg.optimization.sentinel_filename)
    )
    if bool(cfg.pointcloud.get("skip_if_sentinel_exists", False)) and sentinel_path.exists():
        if not optim_sentinel.exists():
            try:
                sentinel_path.unlink()
                print(
                    "Point cloud sentinel is out of date (missing optimization sentinel); "
                    f"deleted {sentinel_path}"
                )
            except OSError as exc:
                print(f"Warning: failed to delete stale point cloud sentinel: {exc}")
        else:
            print(
                "Skipping point cloud generation for "
                f"{run_dir.name}; sentinel exists at {sentinel_path}"
            )
            return

    poses_dir = output_root / run_dir.name / cfg.pointcloud.poses_subdir
    pose_files = load_pose_files(poses_dir)
    use_every_n_frame = int(cfg.pointcloud.get("use_every_n_frame", 1))
    if use_every_n_frame < 1:
        raise ValueError("pointcloud.use_every_n_frame must be >= 1")
    pose_files = pose_files[::use_every_n_frame]

    dense_cloud = None
    frame_count = 0
    start_time = time.perf_counter()

    for pose_file in pose_files:
        timestamp = pose_file.stem
        rgb_file = run_dir / "rgb" / f"{timestamp}.jpg"
        depth_file = run_dir / "depth" / f"{timestamp}.png"
        calib_file = run_dir / "calib" / f"{timestamp}.yaml"

        if not rgb_file.exists():
            raise FileNotFoundError(f"Missing RGB file: {rgb_file}")
        if not depth_file.exists():
            raise FileNotFoundError(f"Missing depth file: {depth_file}")
        if not calib_file.exists():
            raise FileNotFoundError(f"Missing calibration file: {calib_file}")

        rgb = load_rgb_image(rgb_file)
        depth = load_depth_image(depth_file, cfg.pointcloud.depth_scale)
        depth = filter_depth(depth, cfg.pointcloud.min_depth, cfg.pointcloud.max_depth)
        calib = load_calibration(calib_file)
        pose = load_pose_matrix(pose_file)

        pcd = create_point_cloud_from_rgbd(rgb, depth, calib["K"], pose)
        if dense_cloud is None:
            dense_cloud = pcd
        else:
            dense_cloud += pcd

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{len(pose_files)} frames for {run_dir.name}")

    if dense_cloud is None:
        raise ValueError(f"No point cloud data generated for {run_dir.name}")

    filtered_cloud = filter_point_cloud(
        dense_cloud,
        cfg.pointcloud.voxel_size,
        cfg.pointcloud.nb_neighbors,
        cfg.pointcloud.std_ratio,
    )

    dense_center = dense_cloud.get_center()
    centering_translation = [
        -float(dense_center[0]),
        -float(dense_center[1]),
        -float(dense_center[2]),
    ]
    dense_cloud.translate(centering_translation)
    filtered_cloud.translate(centering_translation)

    dense_center_after = [float(v) for v in dense_cloud.get_center()]
    filtered_center_after = [float(v) for v in filtered_cloud.get_center()]

    output_dir = output_root / run_dir.name
    dense_path = output_dir / cfg.pointcloud.output_dense_name
    filtered_path = output_dir / cfg.pointcloud.output_filtered_name

    save_point_cloud(dense_cloud, dense_path)
    save_point_cloud(filtered_cloud, filtered_path)

    screenshots_cfg = cfg.pointcloud.get("screenshots", {})
    if bool(screenshots_cfg.get("enabled", False)):
        screenshots_subdir = str(screenshots_cfg.get("output_subdir", "pointcloud_screenshots"))
        screenshots_dir = output_dir / screenshots_subdir
        ensure_within_root(
            screenshots_dir, Path(cfg.paths.clean_data_dir), "pointcloud.screenshots.output_subdir"
        )
        try:
            render_width = int(screenshots_cfg.get("width", 1280))
            render_height = int(screenshots_cfg.get("height", 720))
            render_point_size = float(screenshots_cfg.get("point_size", 1.0))
            render_zoom = float(screenshots_cfg.get("zoom", 0.7))
            render_background = tuple(screenshots_cfg.get("background_color", (1.0, 1.0, 1.0)))

            for cloud, prefix in ((dense_cloud, "dense"), (filtered_cloud, "filtered")):
                render_point_cloud_views(
                    cloud,
                    screenshots_dir,
                    prefix,
                    width=render_width,
                    height=render_height,
                    point_size=render_point_size,
                    zoom=render_zoom,
                    background_color=render_background,
                )
        except Exception as exc:
            fail_on_error = bool(screenshots_cfg.get("fail_on_error", True))
            msg = f"Failed to render point cloud screenshots for {run_dir.name}: {exc}"
            if fail_on_error:
                raise RuntimeError(msg) from exc
            print(f"Warning: {msg}")

    elapsed = time.perf_counter() - start_time
    bbox = dense_cloud.get_axis_aligned_bounding_box()
    bbox_min = bbox.get_min_bound().tolist()
    bbox_max = bbox.get_max_bound().tolist()

    report = {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_frames": frame_count,
        "point_cloud": {
            "num_points_raw": int(len(dense_cloud.points)),
            "num_points_filtered": int(len(filtered_cloud.points)),
            "centering_translation": centering_translation,
            "center_after_centering": {
                "dense": dense_center_after,
                "filtered": filtered_center_after,
            },
            "voxel_size": cfg.pointcloud.voxel_size,
            "outlier_removal": {
                "method": "statistical",
                "nb_neighbors": cfg.pointcloud.nb_neighbors,
                "std_ratio": cfg.pointcloud.std_ratio,
            },
        },
        "bounding_box": {"min": bbox_min, "max": bbox_max},
        "processing_time_s": float(elapsed),
        "parameters": {
            "depth_scale": cfg.pointcloud.depth_scale,
            "min_depth": cfg.pointcloud.min_depth,
            "max_depth": cfg.pointcloud.max_depth,
            "use_every_n_frame": use_every_n_frame,
        },
    }

    report_path = report_dir / f"{run_dir.name}_pointcloud_report.json"
    write_json(report_path, report)

    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            f"completed_at={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"Warning: failed to write point cloud sentinel: {exc}")

    print(
        "Point cloud generation completed for",
        run_dir.name,
        f"(frames: {frame_count})",
    )


def run(cfg: DictConfig) -> None:
    data_dir = Path(cfg.paths.raw_data_dir)
    run_dirs = get_run_dirs(data_dir, list(cfg.pointcloud.run_ids))
    if not run_dirs:
        raise ValueError(f"No runs found in {data_dir}")

    for run_dir in run_dirs:
        run_for_run(cfg, run_dir)


def _load_serialized_cfg(cfg_pickle: str) -> DictConfig:
    try:
        with open(cfg_pickle, "rb") as fh:
            serialized = pickle.load(fh)
        return DictConfig(serialized)
    except Exception as exc:
        raise ValueError("Invalid serialized config file for point cloud subprocess") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate point cloud for one run")
    parser.add_argument("--worker-run", type=str, default=None)
    parser.add_argument("--cfg-pickle", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.worker_run or not args.cfg_pickle:
        raise ValueError("--worker-run and --cfg-pickle are required")

    cfg = _load_serialized_cfg(args.cfg_pickle)
    run_for_run(cfg, Path(args.worker_run))
