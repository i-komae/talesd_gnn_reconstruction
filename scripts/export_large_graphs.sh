#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_EVENTS_PER_FILE="${MAX_EVENTS_PER_FILE:-0}"
ENERGY_SAMPLE_PER_BIN="${ENERGY_SAMPLE_PER_BIN:-200000}"
ENERGY_SAMPLE_STRATIFY_PARTICLE="${ENERGY_SAMPLE_STRATIFY_PARTICLE:-1}"
ENERGY_BIN_WIDTH="${ENERGY_BIN_WIDTH:-0.1}"
ENERGY_OVERSAMPLE_FACTOR="${ENERGY_OVERSAMPLE_FACTOR:-1.0}"
EXPORT_WORKERS="${EXPORT_WORKERS:-6}"
WORKER_MAX_FILES="${WORKER_MAX_FILES:-0}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
OPEN_RETRIES="${OPEN_RETRIES:-3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-energyflat${ENERGY_SAMPLE_PER_BIN}_gapped_export_${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${RUN_DIR}/graphs}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}/${RUN_NAME}.h5}"
LOG_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
SUMMARY_DIR="${RUN_DIR}/summaries"
READ_DONE_MARKER="${RUN_DIR}/DST_READ_COMPLETE.txt"
LOG_PATH="${LOG_DIR}/export.log"

INPUT_DIRS=(
  "/Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313"
  "/Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313"
  "/Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313"
  "/Volumes/TALE/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316"
  "/Volumes/TALE/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316"
  "/Volumes/TALE/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316"
)

mkdir -p "${GRAPH_RUN_DIR}" "${LOG_DIR}" "${CONFIG_DIR}" "${SUMMARY_DIR}"
touch "${LOG_PATH}"

log() {
  printf "%s\n" "$*" | tee -a "${LOG_PATH}"
}

{
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "GRAPH_OUTPUT=${GRAPH_OUTPUT}"
  echo "MAX_EVENTS_PER_FILE=${MAX_EVENTS_PER_FILE}"
  echo "ENERGY_SAMPLE_PER_BIN=${ENERGY_SAMPLE_PER_BIN}"
  echo "ENERGY_SAMPLE_STRATIFY_PARTICLE=${ENERGY_SAMPLE_STRATIFY_PARTICLE}"
  echo "ENERGY_BIN_WIDTH=${ENERGY_BIN_WIDTH}"
  echo "ENERGY_OVERSAMPLE_FACTOR=${ENERGY_OVERSAMPLE_FACTOR}"
  echo "EXPORT_WORKERS=${EXPORT_WORKERS}"
  echo "WORKER_MAX_FILES=${WORKER_MAX_FILES}"
  echo "SHARD_SIZE=${SHARD_SIZE}"
  echo "OPEN_RETRIES=${OPEN_RETRIES}"
  printf "INPUT_DIRS=%s\n" "${INPUT_DIRS[*]}"
} > "${CONFIG_DIR}/export.env"

cat <<EOF | tee -a "${LOG_PATH}"
======================================================================
LARGE GRAPH EXPORT READY

run_dir:
  ${RUN_DIR}

graph_output:
  ${GRAPH_OUTPUT}

max_events_per_file=${MAX_EVENTS_PER_FILE}
energy_sample_per_bin=${ENERGY_SAMPLE_PER_BIN}
energy_sample_stratify_particle=${ENERGY_SAMPLE_STRATIFY_PARTICLE}
energy_bin_width=${ENERGY_BIN_WIDTH}
energy_oversample_factor=${ENERGY_OVERSAMPLE_FACTOR}
export_workers=${EXPORT_WORKERS}
shard_size=${SHARD_SIZE}

This step reads DST files from /Volumes/TALE.
After the DST FILE READING COMPLETE marker, training can continue from local HDF5 only.
$(date)
======================================================================
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  log "DRY_RUN=1: export command will not be executed."
  exit 0
fi

./build_extensions.sh 2>&1 | tee -a "${LOG_PATH}"

cmd=(.venv/bin/talesd-gnn export)
for input_dir in "${INPUT_DIRS[@]}"; do
  cmd+=(--input-dir "${input_dir}")
done
cmd+=(
  -o "${GRAPH_OUTPUT}"
  --kind mc
  --max-events-per-file "${MAX_EVENTS_PER_FILE}"
  --energy-sample-per-bin "${ENERGY_SAMPLE_PER_BIN}"
  --energy-bin-width "${ENERGY_BIN_WIDTH}"
  --energy-oversample-factor "${ENERGY_OVERSAMPLE_FACTOR}"
  --workers "${EXPORT_WORKERS}"
  --worker-max-files "${WORKER_MAX_FILES}"
  --shard-size "${SHARD_SIZE}"
  --open-retries "${OPEN_RETRIES}"
  --skip-errors
)
if [[ "${ENERGY_SAMPLE_STRATIFY_PARTICLE}" == "1" ]]; then
  cmd+=(--energy-sample-stratify-particle)
fi
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"

cat <<EOF | tee "${READ_DONE_MARKER}" | tee -a "${LOG_PATH}"
======================================================================
DST FILE READING COMPLETE

Network DST input under /Volumes/TALE is no longer needed by this run.
The remaining training and diagnostic steps should read local HDF5 graph shards:
  ${GRAPH_OUTPUT}

You can disconnect the network volume after this line if needed.
$(date)
======================================================================
EOF

printf "%s\n" "${GRAPH_OUTPUT}" > "${CONFIG_DIR}/graph_input.txt"
.venv/bin/python scripts/summarize_graph_shards.py "${GRAPH_OUTPUT}" -o "${SUMMARY_DIR}/graph_summary.json" 2>&1 | tee -a "${LOG_PATH}"

cat > "${RUN_DIR}/README.txt" <<EOF
Run: ${RUN_NAME}
Created: $(date)
Purpose: large local graph export for multi-day mass-free reconstruction training.

Important files:
  config/export.env
  config/graph_input.txt
  logs/export.log
  summaries/graph_summary.json
  DST_READ_COMPLETE.txt

Graph input for training:
  ${GRAPH_OUTPUT}
EOF

log "run_dir=${RUN_DIR}"
log "graph_input=${GRAPH_OUTPUT}"
log "graph_summary=${SUMMARY_DIR}/graph_summary.json"
log "read_done_marker=${READ_DONE_MARKER}"
date | tee -a "${LOG_PATH}"
