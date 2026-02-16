from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.align_pointclouds import run_for_run as align_pointcloud_for_run
from scripts.align_pointclouds import select_reference_run
from scripts.generate_pointcloud import run_for_run as generate_pointcloud_for_run
from scripts.optimize_poses import run_for_run as optimize_pose_for_run

logger = logging.getLogger(__name__)


def _load_cfg(config_path: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Failed to load config as DictConfig: {config_path}")
    return cfg


def _get_run_dirs(
    data_dir: Path,
    run_ids: list[str],
    num_runs_to_process: int | None,
) -> list[Path]:
    if run_ids:
        return [data_dir / run_id for run_id in run_ids]

    all_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    if num_runs_to_process is None or int(num_runs_to_process) <= 0:
        return all_dirs

    return all_dirs[: int(num_runs_to_process)]


def _load_run_ids(run_ids_file: Path) -> list[str]:
    if not run_ids_file.exists():
        raise FileNotFoundError(f"Run IDs file not found: {run_ids_file}")

    run_ids = [line.strip() for line in run_ids_file.read_text(encoding="utf-8").splitlines()]
    run_ids = [run_id for run_id in run_ids if run_id]
    if not run_ids:
        raise ValueError(f"Run IDs file is empty: {run_ids_file}")
    return run_ids


def cmd_prepare_run_list(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    run_dirs = _get_run_dirs(
        Path(cfg.paths.raw_data_dir),
        list(cfg.optimization.run_ids),
        cfg.num_runs_to_process,
    )
    if not run_dirs:
        raise ValueError("No run directories found")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(run_dir.name for run_dir in run_dirs) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(run_dirs)} run IDs to {output_file}")


def cmd_optimize_run(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    run_dir = Path(cfg.paths.raw_data_dir) / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg.optimization.rgbd.enabled = False
    cfg.optimization.rgbd.loop_closure.enabled = False

    optimize_pose_for_run(cfg, run_dir)
    generate_pointcloud_for_run(cfg, run_dir)

    print(
        "Completed optimization + pointcloud generation for "
        f"{args.run_id} with RGB-D constraints and loop closure enabled"
    )


def cmd_select_reference(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    run_ids = _load_run_ids(Path(args.run_ids_file))
    reference_id, scores = select_reference_run(cfg, run_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(reference_id + "\n", encoding="utf-8")

    scores_path = output_path.with_suffix(output_path.suffix + ".scores.txt")
    score_lines = [f"{run_id}\t{scores[run_id]:.8f}" for run_id in sorted(scores)]
    scores_path.write_text("\n".join(score_lines) + "\n", encoding="utf-8")

    print(f"Selected reference run: {reference_id}")
    print(f"Wrote reference run ID to {output_path}")
    print(f"Wrote reference scores to {scores_path}")


def cmd_align_run(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    reference_id = Path(args.reference_file).read_text(encoding="utf-8").strip()
    if not reference_id:
        raise ValueError(f"Reference file is empty: {args.reference_file}")

    result = align_pointcloud_for_run(cfg, args.run_id, reference_id)
    print(f"Alignment result for {args.run_id}: {result}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-run Slurm task entrypoints for rw-pp-cleanup",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_run_list = subparsers.add_parser("prepare-run-list")
    prepare_run_list.add_argument("--config", required=True, help="Path to config.yaml")
    prepare_run_list.add_argument("--output", required=True, help="Output file containing run IDs")
    prepare_run_list.set_defaults(func=cmd_prepare_run_list)

    optimize_run = subparsers.add_parser("optimize-run")
    optimize_run.add_argument("--config", required=True, help="Path to config.yaml")
    optimize_run.add_argument("--run-id", required=True, help="Run ID to process")
    optimize_run.set_defaults(func=cmd_optimize_run)

    select_reference = subparsers.add_parser("select-reference")
    select_reference.add_argument("--config", required=True, help="Path to config.yaml")
    select_reference.add_argument("--run-ids-file", required=True, help="Path to run IDs file")
    select_reference.add_argument("--output", required=True, help="Path to output reference ID file")
    select_reference.set_defaults(func=cmd_select_reference)

    align_run = subparsers.add_parser("align-run")
    align_run.add_argument("--config", required=True, help="Path to config.yaml")
    align_run.add_argument("--run-id", required=True, help="Run ID to align")
    align_run.add_argument(
        "--reference-file",
        required=True,
        help="Path to file containing selected reference run ID",
    )
    align_run.set_defaults(func=cmd_align_run)

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
