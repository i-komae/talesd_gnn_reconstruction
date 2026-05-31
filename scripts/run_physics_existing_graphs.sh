#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_DIR="${GRAPH_DIR:-${HOME}/TALE/gnn/outputs/graphs}"
GRAPH_INPUT="${GRAPH_INPUT:-${GRAPH_DIR}/mass_12h_64perfile_6epoch.h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
TAG="${TAG:-physics_reco_existing_4epoch}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
COLLATE_THREADS="${COLLATE_THREADS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-160}"
LAYERS="${LAYERS:-5}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
LR="${LR:-5e-4}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.2}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.4}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.05}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
MODEL_DIR="${OUTPUT_ROOT}/models"
LOG_DIR="${OUTPUT_ROOT}/logs"
CHECKPOINT="${MODEL_DIR}/${TAG}.pt"
LOG_PATH="${LOG_DIR}/${TAG}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "tag=${TAG}"
echo "graph_input=${GRAPH_INPUT}"
echo "checkpoint=${CHECKPOINT}"
echo "log_path=${LOG_PATH}"
echo "train_epochs=${TRAIN_EPOCHS}"
echo "hidden_dim=${HIDDEN_DIM}"
echo "layers=${LAYERS}"
echo "detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}"
echo "lr=${LR}"
echo "loss_mode=physics"
echo "energy_weight=${ENERGY_WEIGHT}"
echo "core_weight=${CORE_WEIGHT}"
echo "direction_weight=${DIRECTION_WEIGHT}"
echo "core_scale_km=${CORE_SCALE_KM}"
echo "This script does not read DST files."
date

./build_extensions.sh

.venv/bin/talesd-gnn train \
  --graphs "${GRAPH_INPUT}" \
  -o "${CHECKPOINT}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --layers "${LAYERS}" \
  --dropout 0.04 \
  --model-architecture physics \
  --readout-heads 4 \
  --detector-embedding-dim "${DETECTOR_EMBEDDING_DIM}" \
  --loss-mode physics \
  --energy-loss-weight "${ENERGY_WEIGHT}" \
  --core-loss-weight "${CORE_WEIGHT}" \
  --direction-loss-weight "${DIRECTION_WEIGHT}" \
  --core-loss-scale-km "${CORE_SCALE_KM}" \
  --device cpu \
  --num-workers "${TRAIN_WORKERS}" \
  --prefetch-factor 2 \
  --collate-backend cpp \
  --collate-threads "${COLLATE_THREADS}" \
  --split-mode source-stratified \
  --test-fraction "${TEST_FRACTION}" \
  --val-fraction "${VAL_FRACTION}" \
  --diagnostic-energy-bin-width 0.1 \
  --diagnostic-min-bin-count 20 \
  ${MAX_GRAPHS:+--max-graphs "${MAX_GRAPHS}"}

DIAGNOSTICS_DIR="${CHECKPOINT}.diagnostics"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${CHECKPOINT}.metrics.json"
echo "diagnostics_dir=${DIAGNOSTICS_DIR}"
echo "learning_curve=${DIAGNOSTICS_DIR}/learning_curve.pdf"
echo "validation_dir=${DIAGNOSTICS_DIR}/validation"
echo "test_dir=${DIAGNOSTICS_DIR}/test"
echo "log_path=${LOG_PATH}"
date
