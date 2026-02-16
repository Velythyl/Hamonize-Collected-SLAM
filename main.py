from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.verify_alignment import run as run_alignment
from scripts.optimize_poses import run_for_run as run_optimization_for_run
from scripts.align_pointclouds import run as run_pointcloud_alignment


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


def process_run(serialized_cfg: dict, run_dir: Path) -> None:
	cfg = OmegaConf.create(serialized_cfg)
	try:
		run_optimization_for_run(cfg, run_dir)
		run_pointcloud_for_run_subprocess(serialized_cfg, run_dir)
		validate_pointcloud_outputs_exist(cfg, run_dir)
	except Exception as exc:
		raise RuntimeError(f"Run failed for {run_dir}") from exc


def run_pointcloud_for_run_subprocess(serialized_cfg: dict, run_dir: Path) -> None:
	script_path = Path(__file__).resolve().parent / "scripts" / "generate_pointcloud.py"
	with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as cfg_file:
		pickle.dump(serialized_cfg, cfg_file)
		cfg_pickle_path = Path(cfg_file.name)

	try:
		proc = subprocess.Popen(
			[
				sys.executable,
				str(script_path),
				"--worker-run",
				str(run_dir),
				"--cfg-pickle",
				str(cfg_pickle_path),
			],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
		stdout, stderr = proc.communicate()
	finally:
		try:
			cfg_pickle_path.unlink(missing_ok=True)
		except OSError:
			pass

	if proc.returncode != 0:
		extra = ""
		if proc.returncode < 0:
			extra = (
				f" (terminated by signal {-proc.returncode}; potential OOM if signal=9)"
			)
		raise RuntimeError(
			f"Point cloud subprocess failed for {run_dir} with exit code {proc.returncode}"
			f"{extra}\nstdout:\n{stdout}\nstderr:\n{stderr}"
		)


def validate_pointcloud_outputs_exist(cfg: DictConfig, run_dir: Path) -> None:
	output_dir = Path(cfg.pointcloud.output_root) / run_dir.name
	expected_paths = [
		output_dir / str(cfg.pointcloud.output_dense_name),
		output_dir / str(cfg.pointcloud.output_filtered_name),
	]

	screenshots_cfg = cfg.pointcloud.get("screenshots", {})
	if bool(screenshots_cfg.get("enabled", False)):
		screenshots_subdir = str(screenshots_cfg.get("output_subdir", "pointcloud_screenshots"))
		screenshots_dir = output_dir / screenshots_subdir
		view_names = [
			"top_down",
			"bottom_up",
			"side_pos_x",
			"side_neg_x",
			"side_pos_y",
			"side_neg_y",
		]
		for prefix in ("dense", "filtered"):
			for view_name in view_names:
				expected_paths.append(screenshots_dir / f"{prefix}_{view_name}.png")

	missing_paths = [path for path in expected_paths if not path.exists()]
	if missing_paths:
		missing_text = "\n".join(str(path) for path in missing_paths)
		raise FileNotFoundError(
			f"Missing expected point cloud outputs for {run_dir.name}:\n{missing_text}"
		)


@hydra_main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	if not cfg.alignment.enabled:
		raise ValueError(
			"alignment.enabled must be true because initial pose verification is required"
		)

	if not cfg.optimization.enabled:
		raise ValueError(
			"optimization.enabled must be true because pose optimization is required"
		)

	if not cfg.pointcloud.enabled:
		raise ValueError(
			"pointcloud.enabled must be true because point cloud generation is required"
		)

	if not cfg.pointcloud_alignment.enabled:
		raise ValueError(
			"pointcloud_alignment.enabled must be true because point cloud alignment is required"
		)

	alignment_ids = list(cfg.alignment.run_ids)
	optimization_ids = list(cfg.optimization.run_ids)
	pointcloud_ids = list(cfg.pointcloud.run_ids)
	pointcloud_alignment_ids = list(cfg.pointcloud_alignment.run_ids)
	non_empty = [
		ids
		for ids in (
			alignment_ids,
			optimization_ids,
			pointcloud_ids,
			pointcloud_alignment_ids,
		)
		if ids
	]
	if non_empty:
		if not (
			alignment_ids
			and optimization_ids
			and pointcloud_ids
			and pointcloud_alignment_ids
		):
			raise ValueError(
				"alignment.run_ids, optimization.run_ids, pointcloud.run_ids, and "
				"pointcloud_alignment.run_ids must all be set to the same list, or all "
				"left empty"
			)
		if (
			alignment_ids != optimization_ids
			or alignment_ids != pointcloud_ids
			or alignment_ids != pointcloud_alignment_ids
		):
			raise ValueError(
				"Run IDs must match across alignment, optimization, pointcloud, "
				"and pointcloud_alignment stages"
			)

	data_dir = Path(cfg.paths.raw_data_dir)
	run_dirs = get_run_dirs(
		data_dir,
		list(cfg.optimization.run_ids),
		cfg.num_runs_to_process,
	)
	if not run_dirs:
		raise ValueError(f"No runs found in {data_dir}")

	# downstream stages always read from raw_data_dir and write to clean_data_dir
    
	serialized_cfg = OmegaConf.to_container(cfg, resolve=True)
	if cfg.USE_MULTIPROCESSING:
		with ProcessPoolExecutor(
			max_workers=cfg.concurrency.max_workers,
			mp_context=mp.get_context("spawn"),
		) as executor:
			futures = [executor.submit(process_run, serialized_cfg, run_dir) for run_dir in run_dirs]
			for future in tqdm(
				as_completed(futures), total=len(futures), desc="Processing runs"
			):
				future.result()
	else:
		for run_dir in tqdm(run_dirs, desc="Processing runs"):
			process_run(serialized_cfg, run_dir)

	run_pointcloud_alignment(cfg, run_dirs)


if __name__ == "__main__":
	main()
