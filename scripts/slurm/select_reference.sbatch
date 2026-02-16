#!/bin/bash
#SBATCH --job-name=rwpp-ref
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --error=slurm_logs/%x-%A.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch select_reference.sbatch <config_path> <run_ids_file> <reference_output_file>"
  exit 1
fi

CONFIG_PATH="$1"
RUN_IDS_FILE="$2"
REFERENCE_OUTPUT_FILE="$3"

mkdir -p slurm_logs

echo "[reference] selecting reference from run IDs in $RUN_IDS_FILE"
uv run python scripts/slurm_tasks.py select-reference \
  --config "$CONFIG_PATH" \
  --run-ids-file "$RUN_IDS_FILE" \
  --output "$REFERENCE_OUTPUT_FILE"
