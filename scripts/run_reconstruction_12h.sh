#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_EVENTS_PER_FILE="${MAX_EVENTS_PER_FILE:-64}"
EXPORT_WORKERS="${EXPORT_WORKERS:-6}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
COLLATE_THREADS="${COLLATE_THREADS:-0}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_DIR="${GRAPH_DIR:-${HOME}/TALE/gnn/outputs/graphs}"
TAG="${TAG:-reco_12h_${MAX_EVENTS_PER_FILE}perfile_${TRAIN_EPOCHS}epoch}"
GRAPH_INPUT="${GRAPH_INPUT:-}"

MODEL_DIR="${OUTPUT_ROOT}/models"
LOG_DIR="${OUTPUT_ROOT}/logs"
GRAPH_OUTPUT="${GRAPH_DIR}/${TAG}.h5"
CHECKPOINT="${MODEL_DIR}/${TAG}.pt"
READ_DONE_MARKER="${OUTPUT_ROOT}/${TAG}.DST_READ_COMPLETE.txt"
LOG_PATH="${LOG_DIR}/${TAG}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${GRAPH_DIR}" "${MODEL_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ -z "${GRAPH_INPUT}" ]]; then
  EXISTING_MASS_GRAPH="${GRAPH_DIR}/mass_12h_${MAX_EVENTS_PER_FILE}perfile_6epoch.h5"
  if compgen -G "${EXISTING_MASS_GRAPH%.h5}_*.h5" > /dev/null; then
    GRAPH_INPUT="${EXISTING_MASS_GRAPH}"
  fi
fi
TRAIN_GRAPHS="${GRAPH_INPUT:-${GRAPH_OUTPUT}}"

echo "tag=${TAG}"
echo "output_root=${OUTPUT_ROOT}"
echo "graph_output=${GRAPH_OUTPUT}"
echo "graph_input=${GRAPH_INPUT:-}"
echo "train_graphs=${TRAIN_GRAPHS}"
echo "checkpoint=${CHECKPOINT}"
echo "log_path=${LOG_PATH}"
echo "max_events_per_file=${MAX_EVENTS_PER_FILE}"
echo "export_workers=${EXPORT_WORKERS}"
echo "train_epochs=${TRAIN_EPOCHS}"
echo "train_workers=${TRAIN_WORKERS}"
echo "batch_size=${BATCH_SIZE}"
echo "detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}"
echo "val_fraction=${VAL_FRACTION}"
echo "test_fraction=${TEST_FRACTION}"
date

./build_extensions.sh

if [[ -n "${GRAPH_INPUT}" ]]; then
  echo "GRAPH_INPUT is set: skipping DST export and training from existing graph shards:"
  echo "  ${GRAPH_INPUT}"
elif [[ "${REUSE_GRAPHS:-0}" == "1" ]]; then
  echo "REUSE_GRAPHS=1: skipping DST export and reusing existing graph shards for ${GRAPH_OUTPUT}"
else
  .venv/bin/talesd-gnn export \
    --input-dir /Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313 \
    --input-dir /Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313 \
    --input-dir /Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313 \
    --input-dir /Volumes/TALE/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316 \
    --input-dir /Volumes/TALE/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316 \
    --input-dir /Volumes/TALE/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316 \
    -o "${GRAPH_OUTPUT}" \
    --kind mc \
    --max-events-per-file "${MAX_EVENTS_PER_FILE}" \
    --workers "${EXPORT_WORKERS}" \
    --worker-max-files 200 \
    --shard-size 50000 \
    --open-retries 3 \
    --skip-errors
fi

cat <<EOF | tee "${READ_DONE_MARKER}"
======================================================================
LOCAL GRAPH INPUT READY

This run will train from local HDF5 graph shards:
  ${TRAIN_GRAPHS}

If the script skipped export, no DST files were read in this run.
If export ran above, DST file reading is now complete.

You can disconnect the network volume after this line if needed.
$(date)
======================================================================
EOF

.venv/bin/talesd-gnn train \
  --graphs "${TRAIN_GRAPHS}" \
  -o "${CHECKPOINT}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device cpu \
  --num-workers "${TRAIN_WORKERS}" \
  --prefetch-factor 2 \
  --collate-backend cpp \
  --collate-threads "${COLLATE_THREADS}" \
  --detector-embedding-dim "${DETECTOR_EMBEDDING_DIM}" \
  --split-mode source-stratified \
  --test-fraction "${TEST_FRACTION}" \
  --val-fraction "${VAL_FRACTION}" \
  --diagnostic-energy-bin-width 0.1 \
  --diagnostic-min-bin-count 20

DIAGNOSTICS_DIR="${CHECKPOINT}.diagnostics"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${CHECKPOINT}.metrics.json"
echo "diagnostics_dir=${DIAGNOSTICS_DIR}"
echo "learning_curve=${DIAGNOSTICS_DIR}/learning_curve.pdf"
echo "validation_dir=${DIAGNOSTICS_DIR}/validation"
echo "test_dir=${DIAGNOSTICS_DIR}/test"
echo "log_path=${LOG_PATH}"
date
