# Hamonize-Collected-SLAM
Defines a pipeline using GTSAM and Open3D to perform pose estimation, RGBD constaints, loop closure, etc.; between different robot runs. Then, aligns the point clouds to some reference trajectory.

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/b351deb1-1f24-4e76-963b-418fdac0885d" />

# Installation

```bash
uv sync
```

# Usage

Tune the `configs/config.yaml`. 
You can define the input/output directories using `configs/config.yaml`. We assume that the input and output dirs are both found in some local directory: resp. `./<main dir>/<raw_data_dir>` and `./<main dir>/<clean_data_dir>`. If you are on a compute cluster, we recommend that you use `ln -s` to link `<main dir>` to `$SCRATCH`.

Run individual runs using `uv run main.py`

Run a full pipeline on all runs with `./scripts/slurm/submit_full_processing.sh configs/config.yaml`. This will:

1. Launch `N` sbatch runs that will optimize the poses and build point clouds using 8 CPUs, 32GB RAM, with a time limit of 8 hours.
2. Launch `1` sbatch run that will load all the point clouds and identify some median run (right now just looks at the initial pose; assuming all runs start at roughly the same initial pose)
3. Launch `N` sbatch runs that will perform global and local point registration to align the clouds to the reference point cloud.
4. All this uses `slurm --dependency` to manage the runs and trigger them to run after the previous step has concluded. 

# License

We use several libraries and their dependencies as shown in `pyproject.toml`. Please see each of these libraries for their respective licenses.
