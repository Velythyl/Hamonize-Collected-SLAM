from __future__ import annotations

import copy
from pathlib import Path
import os
import ctypes.util

import numpy as np


def _require_open3d():
    try:
        if os.environ.get("DISPLAY") in (None, "") and os.environ.get("WAYLAND_DISPLAY") in (
            None,
            "",
        ):
            os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for point cloud generation. Install with 'pip install open3d'."
        ) from exc
    return o3d


def _has_egl_runtime() -> bool:
    return ctypes.util.find_library("EGL") is not None


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for point cloud generation. Install with 'pip install opencv-python'."
        ) from exc
    return cv2


def load_rgb_image(filepath: Path) -> np.ndarray:
    cv2 = _require_cv2()
    image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load RGB image: {filepath}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth_image(filepath: Path, depth_scale: float) -> np.ndarray:
    cv2 = _require_cv2()
    depth = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to load depth image: {filepath}")

    depth_m = depth.astype(np.float32) / depth_scale
    depth_m[depth == 0] = 0.0
    return depth_m


def filter_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    depth_filtered = depth.copy()
    depth_filtered[(depth < min_depth) | (depth > max_depth)] = 0.0
    return depth_filtered


def create_point_cloud_from_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
):
    o3d = _require_open3d()

    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=float(np.max(depth) if np.max(depth) > 0 else 1.0),
        convert_rgb_to_intensity=False,
    )

    height, width = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform(pose)
    return pcd


def filter_point_cloud(
    pcd,
    voxel_size: float,
    nb_neighbors: int,
    std_ratio: float,
):
    o3d = _require_open3d()
    if len(pcd.points) == 0:
        return pcd

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd_down.points) == 0:
        return pcd_down

    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    if not isinstance(pcd_clean, o3d.geometry.PointCloud):
        return pcd_down
    return pcd_clean


def save_point_cloud(pcd, output_file: Path) -> None:
    o3d = _require_open3d()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(output_file), pcd, write_ascii=False)
    if not success:
        raise IOError(f"Failed to save point cloud: {output_file}")


def render_colored_point_cloud_comparison_views(
    reference_cloud,
    aligned_cloud,
    output_dir: Path,
    prefix: str,
    width: int = 1280,
    height: int = 720,
    point_size: float = 1.0,
    zoom: float = 0.7,
    background_color: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    reference_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    aligned_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    allow_fallback: bool = False,
) -> None:
    if reference_cloud.is_empty() or aligned_cloud.is_empty():
        return

    reference_vis = copy.deepcopy(reference_cloud)
    aligned_vis = copy.deepcopy(aligned_cloud)
    reference_vis.paint_uniform_color(list(reference_color))
    aligned_vis.paint_uniform_color(list(aligned_color))

    comparison_cloud = reference_vis + aligned_vis
    render_point_cloud_views(
        comparison_cloud,
        output_dir,
        prefix,
        width=width,
        height=height,
        point_size=point_size,
        zoom=zoom,
        background_color=background_color,
        allow_fallback=allow_fallback,
    )


def render_point_cloud_views(
    pcd,
    output_dir: Path,
    prefix: str,
    width: int = 1280,
    height: int = 720,
    point_size: float = 1.0,
    zoom: float = 0.7,
    background_color: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    allow_fallback: bool = False,
) -> None:
    o3d = _require_open3d()
    if pcd.is_empty():
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent)) if np.any(extent) else 1.0
    distance = max_extent / max(float(zoom), 1e-6)

    views = {
        "top_down": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
        "bottom_up": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        "side_pos_x": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "side_neg_x": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "side_pos_y": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "side_neg_y": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    }

    def _render_offscreen() -> bool:
        if os.environ.get("DISPLAY") in (None, "") and os.environ.get("WAYLAND_DISPLAY") in (
            None,
            "",
        ):
            if not _has_egl_runtime():
                return False

        try:
            rendering = o3d.visualization.rendering
        except AttributeError:
            return False

        renderer = None
        try:
            renderer = rendering.OffscreenRenderer(int(width), int(height))
            scene = renderer.scene
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = float(point_size)
            if background_color is not None:
                scene.set_background([*background_color, 1.0])

            scene.add_geometry("pcd", pcd, material)

            bounds = scene.bounding_box
            render_center = bounds.get_center()
            render_extent = bounds.get_extent()
            render_max_extent = float(np.max(render_extent)) if np.any(render_extent) else 1.0
            render_distance = render_max_extent / max(float(zoom), 1e-6)
            if render_distance <= 0:
                render_distance = 1.0

            for view_name, (front, up) in views.items():
                eye = render_center - front * render_distance
                scene.camera.look_at(render_center, eye, up)
                image = renderer.render_to_image()
                output_path = output_dir / f"{prefix}_{view_name}.png"
                if not o3d.io.write_image(str(output_path), image):
                    raise IOError(f"Failed to save point cloud screenshot: {output_path}")

            return True
        except Exception:
            return False
        finally:
            if renderer is not None:
                try:
                    renderer.release_resources()
                except Exception:
                    pass

    def _render_numpy_fallback() -> bool:
        points = np.asarray(pcd.points)
        if points.size == 0:
            return False

        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        if colors is None or len(colors) != len(points):
            colors = np.tile(np.array([[0.2, 0.2, 0.2]], dtype=np.float32), (len(points), 1))
        colors = np.clip(colors, 0.0, 1.0)

        bg = np.array(background_color if background_color is not None else (1.0, 1.0, 1.0))
        bg_uint8 = (np.clip(bg, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Keep fallback rendering bounded for very large clouds.
        max_points = 500_000
        if len(points) > max_points:
            idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[idx]
            colors = colors[idx]

        saved_images = 0
        for view_name, (front, up) in views.items():
            forward = front / (np.linalg.norm(front) + 1e-12)
            up_vec = up / (np.linalg.norm(up) + 1e-12)
            right = np.cross(forward, up_vec)
            right = right / (np.linalg.norm(right) + 1e-12)
            cam_up = np.cross(right, forward)
            cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

            eye = center - forward * distance
            rel = points - eye

            x_cam = rel @ right
            y_cam = rel @ cam_up
            z_cam = rel @ forward

            valid = z_cam > 0
            if not np.any(valid):
                continue

            x_cam = x_cam[valid]
            y_cam = y_cam[valid]
            z_cam = z_cam[valid]
            cols = colors[valid]

            scale_x = np.percentile(np.abs(x_cam), 99.5)
            scale_y = np.percentile(np.abs(y_cam), 99.5)
            scale = max(float(scale_x), float(scale_y), 1e-6)

            x_norm = np.clip((x_cam / scale + 1.0) * 0.5, 0.0, 1.0)
            y_norm = np.clip((y_cam / scale + 1.0) * 0.5, 0.0, 1.0)
            u = (x_norm * (width - 1)).astype(np.int32)
            v = ((1.0 - y_norm) * (height - 1)).astype(np.int32)

            image = np.empty((int(height), int(width), 3), dtype=np.uint8)
            image[:, :] = bg_uint8

            order = np.argsort(z_cam)
            u = u[order]
            v = v[order]
            cols_uint8 = (cols[order] * 255.0).astype(np.uint8)
            image[v, u] = cols_uint8

            if point_size > 1.0:
                cv2 = _require_cv2()
                kernel_size = max(1, int(round(point_size)))
                if kernel_size > 1:
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    image = cv2.dilate(image, kernel, iterations=1)

            output_path = output_dir / f"{prefix}_{view_name}.png"
            cv2 = _require_cv2()
            if not cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise IOError(f"Failed to save point cloud screenshot: {output_path}")
            saved_images += 1

        return saved_images > 0

    if _render_offscreen():
        return

    if _render_numpy_fallback():
        return

    if not allow_fallback:
        raise RuntimeError(
            "Open3D offscreen renderer is unavailable and software fallback failed."
        )

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=int(width), height=int(height), visible=False):
        raise RuntimeError("Failed to create offscreen window for point cloud screenshots")

    try:
        vis.add_geometry(pcd)

        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.point_size = float(point_size)
            if background_color is not None:
                render_option.background_color = np.array(background_color, dtype=float)

        view_control = vis.get_view_control()

        for view_name, (front, up) in views.items():
            view_control.set_lookat(center)
            view_control.set_front(front)
            view_control.set_up(up)
            view_control.set_zoom(float(zoom))
            vis.poll_events()
            vis.update_renderer()
            output_path = output_dir / f"{prefix}_{view_name}.png"
            vis.capture_screen_image(str(output_path), do_render=True)
            if not output_path.exists():
                raise IOError(f"Failed to save point cloud screenshot: {output_path}")
    finally:
        vis.destroy_window()
