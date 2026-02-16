#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/slurm/submit_full_processing.sh <config_path>"
  exit 1
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

mkdir -p slurm_logs
mkdir -p rw-pp/slurm_state

RUN_IDS_FILE="rw-pp/slurm_state/run_ids.txt"
REFERENCE_FILE="rw-pp/slurm_state/reference_run_id.txt"

uv run python scripts/slurm_tasks.py prepare-run-list \
  --config "$CONFIG_PATH" \
  --output "$RUN_IDS_FILE"

NUM_RUNS=$(wc -l < "$RUN_IDS_FILE" | tr -d ' ')
if [[ "$NUM_RUNS" -le 0 ]]; then
  echo "No runs found in $RUN_IDS_FILE"
  exit 1
fi

ARRAY_RANGE="0-$((NUM_RUNS - 1))"
echo "Submitting workflow for $NUM_RUNS runs (array range: $ARRAY_RANGE)"

OPT_JOB_ID=$(sbatch --parsable --array="$ARRAY_RANGE" \
  scripts/slurm/optimize_pointcloud_array.sbatch \
  "$CONFIG_PATH" "$RUN_IDS_FILE")
echo "Submitted optimization array job: $OPT_JOB_ID"

REF_JOB_ID=$(sbatch --parsable --dependency="afterok:${OPT_JOB_ID}" \
  scripts/slurm/select_reference.sbatch \
  "$CONFIG_PATH" "$RUN_IDS_FILE" "$REFERENCE_FILE")
echo "Submitted reference selection job: $REF_JOB_ID (afterok:$OPT_JOB_ID)"

ALIGN_JOB_ID=$(sbatch --parsable --dependency="afterok:${REF_JOB_ID}" --array="$ARRAY_RANGE" \
  scripts/slurm/align_to_reference_array.sbatch \
  "$CONFIG_PATH" "$RUN_IDS_FILE" "$REFERENCE_FILE")
echo "Submitted alignment array job: $ALIGN_JOB_ID (afterok:$REF_JOB_ID)"

echo "Workflow submitted successfully."
echo "  run_ids:   $RUN_IDS_FILE"
echo "  reference: $REFERENCE_FILE"
