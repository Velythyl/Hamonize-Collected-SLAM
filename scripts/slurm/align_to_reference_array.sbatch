#!/bin/bash
#SBATCH --job-name=rwpp-align
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch align_to_reference_array.sbatch <config_path> <run_ids_file> <reference_file>"
  exit 1
fi

CONFIG_PATH="$1"
RUN_IDS_FILE="$2"
REFERENCE_FILE="$3"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set"
  exit 1
fi

RUN_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUN_IDS_FILE")
if [[ -z "$RUN_ID" ]]; then
  echo "No run ID found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $RUN_IDS_FILE"
  exit 1
fi

mkdir -p slurm_logs

echo "[align] task_id=${SLURM_ARRAY_TASK_ID} run_id=${RUN_ID}"
uv run python scripts/slurm_tasks.py align-run \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --reference-file "$REFERENCE_FILE"
