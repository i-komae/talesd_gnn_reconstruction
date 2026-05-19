#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_EVENTS_PER_FILE="${MAX_EVENTS_PER_FILE:-16}"
EXPORT_WORKERS="${EXPORT_WORKERS:-6}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
COLLATE_THREADS="${COLLATE_THREADS:-0}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
TAG="${TAG:-mass_4h_${MAX_EVENTS_PER_FILE}perfile_${TRAIN_EPOCHS}epoch}"

GRAPH_OUTPUT="outputs/graphs/${TAG}.h5"
CHECKPOINT="outputs/${TAG}.pt"

mkdir -p outputs/graphs

echo "tag=${TAG}"
echo "max_events_per_file=${MAX_EVENTS_PER_FILE}"
echo "export_workers=${EXPORT_WORKERS}"
echo "train_epochs=${TRAIN_EPOCHS}"
echo "train_workers=${TRAIN_WORKERS}"
echo "batch_size=${BATCH_SIZE}"
echo "detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}"
echo "val_fraction=${VAL_FRACTION}"
echo "test_fraction=${TEST_FRACTION}"

./build_extensions.sh

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

.venv/bin/talesd-gnn train \
  --graphs "${GRAPH_OUTPUT}" \
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
  --mass-classification \
  --mass-loss-weight 0.1 \
  --diagnostic-energy-bin-width 0.1 \
  --diagnostic-min-bin-count 20

echo "checkpoint=${CHECKPOINT}"
echo "metrics=${CHECKPOINT}.metrics.json"
DIAGNOSTICS_DIR="${CHECKPOINT}.diagnostics"
echo "diagnostics_dir=${DIAGNOSTICS_DIR}"
echo "learning_curve=${DIAGNOSTICS_DIR}/learning_curve.pdf"
echo "validation_dir=${DIAGNOSTICS_DIR}/validation"
echo "test_dir=${DIAGNOSTICS_DIR}/test"
