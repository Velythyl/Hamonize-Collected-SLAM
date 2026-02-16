from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from scripts.utils.alignment_metrics import (
    compute_rotation_stats,
    compute_translation_stats,
    detect_outliers,
)
from scripts.utils.config import ensure_within_root
from scripts.utils.io_utils import write_json
from scripts.utils.pose_utils import decompose_pose, get_first_pose


def get_run_dirs(
    data_dir: Path,
    run_ids: list[str],
    num_runs_to_process: int | None,
) -> list[Path]:
    if run_ids:
        return [data_dir / run_id for run_id in run_ids]

    all_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if num_runs_to_process is None or num_runs_to_process <= 0:
        return all_dirs

    return all_dirs[:num_runs_to_process]


def run(cfg: DictConfig) -> dict:
    data_dir = Path(cfg.paths.raw_data_dir)
    clean_dir = Path(cfg.paths.clean_data_dir)
    report_path = Path(cfg.alignment.output_report)

    ensure_within_root(data_dir, Path(cfg.paths.raw_data_dir), "paths.raw_data_dir")
    ensure_within_root(report_path, clean_dir, "alignment.output_report")

    run_dirs = get_run_dirs(
        data_dir,
        list(cfg.alignment.run_ids),
        cfg.num_runs_to_process,
    )

    if not run_dirs:
        raise ValueError(f"No runs found in {data_dir}")

    run_ids = []
    timestamps = []
    translations = []
    rotations = []

    for run_dir in run_dirs:
        timestamp, pose = get_first_pose(run_dir)
        translation, rotation = decompose_pose(pose)

        run_ids.append(run_dir.name)
        timestamps.append(timestamp)
        translations.append(translation)
        rotations.append(rotation)

    translation_stats = compute_translation_stats(translations)
    rotation_stats = compute_rotation_stats(rotations)

    outliers = detect_outliers(
        translations,
        rotations,
        run_ids,
        cfg.alignment.translation_threshold,
        cfg.alignment.rotation_threshold_deg,
    )

    translation_std = np.array(translation_stats["std"], dtype=np.float64)
    translation_passed = bool(
        np.all(translation_std < cfg.alignment.translation_std_threshold)
    )
    rotation_passed = bool(
        rotation_stats["std_angle_deg"] < cfg.alignment.rotation_std_threshold_deg
    )

    overall_passed = bool(
        translation_passed and rotation_passed and len(outliers) == 0
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(run_ids),
        "runs": run_ids,
        "statistics": {
            "translation": {
                "mean": translation_stats["mean"].tolist(),
                "std": translation_stats["std"].tolist(),
                "max_distance": translation_stats["max_distance"],
                "median_distance": translation_stats["median_distance"],
            },
            "rotation": rotation_stats,
        },
        "outliers": outliers,
        "success_criteria": {
            "translation_std_threshold": cfg.alignment.translation_std_threshold,
            "rotation_std_threshold_deg": cfg.alignment.rotation_std_threshold_deg,
            "translation_passed": translation_passed,
            "rotation_passed": rotation_passed,
            "overall_passed": overall_passed,
        },
    }

    write_json(report_path, report)

    print("=== Initial Pose Alignment Report ===")
    print(f"Total Runs: {len(run_ids)}")
    print("Translation Stats:")
    print(f"  Mean Position: {translation_stats['mean']}")
    print(f"  Std Dev: {translation_stats['std']}")
    print(f"  Max Distance: {translation_stats['max_distance']:.4f} m")
    print("Rotation Stats:")
    print(f"  Mean Angle: {rotation_stats['mean_angle_deg']:.2f} deg")
    print(f"  Std Dev: {rotation_stats['std_angle_deg']:.2f} deg")
    print(f"  Max Angle: {rotation_stats['max_angle_deg']:.2f} deg")
    print(f"Outliers: {len(outliers)} runs")

    if not overall_passed:
        raise ValueError(
            "Initial pose alignment failed: "
            f"translation_passed={translation_passed}, "
            f"rotation_passed={rotation_passed}, "
            f"outliers={len(outliers)}"
        )

    return report
