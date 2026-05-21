#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_INPUT="${GRAPH_INPUT:-}"
if [[ -z "${GRAPH_INPUT}" && -n "${EXPORT_RUN_DIR:-}" && -f "${EXPORT_RUN_DIR}/config/graph_input.txt" ]]; then
  GRAPH_INPUT="$(< "${EXPORT_RUN_DIR}/config/graph_input.txt")"
fi
if [[ -z "${GRAPH_INPUT}" ]]; then
  echo "GRAPH_INPUT is required, or set EXPORT_RUN_DIR to a run directory containing config/graph_input.txt" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-large_train_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
CONFIG_DIR="${RUN_DIR}/config"

MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-baseline}"
HIDDEN_DIM="${HIDDEN_DIM:-160}"
LAYERS="${LAYERS:-4}"
DROPOUT="${DROPOUT:-0.05}"
READOUT_HEADS="${READOUT_HEADS:-4}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-none}"
WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
WAVEFORM_TRANSFORMER_HEADS="${WAVEFORM_TRANSFORMER_HEADS:-4}"
WAVEFORM_TRANSFORMER_LAYERS="${WAVEFORM_TRANSFORMER_LAYERS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-reduce-on-plateau}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LR_PATIENCE="${LR_PATIENCE:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
LOSS_MODE="${LOSS_MODE:-scaled-mse}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.0}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.12}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TRAINING_TASK="${TRAINING_TASK:-reconstruction}"
MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.20}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
DEVICE="${DEVICE:-cpu}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
PRECISION_MIN_BIN_COUNT="${PRECISION_MIN_BIN_COUNT:-1000}"
MAX_GRAPHS="${MAX_GRAPHS:-}"

CONFIG_NAME="${CONFIG_NAME:-${TRAINING_TASK}_${MODEL_ARCHITECTURE}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${PARTICLE_FILTER}_${TRAIN_EPOCHS}epoch}"
CHECKPOINT="${CHECKPOINT_DIR}/${CONFIG_NAME}.pt"
LOG_PATH="${LOG_DIR}/${CONFIG_NAME}.log"
METRICS_PATH="${CHECKPOINT}.metrics.json"
PRECISION_REPORT="${SUMMARY_DIR}/${CONFIG_NAME}_precision_targets.txt"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}" "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/train.env" <<EOF
RUN_ID=${RUN_ID}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GRAPH_INPUT=${GRAPH_INPUT}
OUTPUT_ROOT=${OUTPUT_ROOT}
CONFIG_NAME=${CONFIG_NAME}
MODEL_ARCHITECTURE=${MODEL_ARCHITECTURE}
HIDDEN_DIM=${HIDDEN_DIM}
LAYERS=${LAYERS}
DROPOUT=${DROPOUT}
READOUT_HEADS=${READOUT_HEADS}
DETECTOR_EMBEDDING_DIM=${DETECTOR_EMBEDDING_DIM}
WAVEFORM_ENCODER=${WAVEFORM_ENCODER}
WAVEFORM_EMBEDDING_DIM=${WAVEFORM_EMBEDDING_DIM}
WAVEFORM_TRANSFORMER_HEADS=${WAVEFORM_TRANSFORMER_HEADS}
WAVEFORM_TRANSFORMER_LAYERS=${WAVEFORM_TRANSFORMER_LAYERS}
TRAIN_EPOCHS=${TRAIN_EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
LR=${LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
LR_SCHEDULER=${LR_SCHEDULER}
LR_FACTOR=${LR_FACTOR}
LR_PATIENCE=${LR_PATIENCE}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}
LOSS_MODE=${LOSS_MODE}
ENERGY_WEIGHT=${ENERGY_WEIGHT}
CORE_WEIGHT=${CORE_WEIGHT}
DIRECTION_WEIGHT=${DIRECTION_WEIGHT}
CORE_SCALE_KM=${CORE_SCALE_KM}
TRAIN_WORKERS=${TRAIN_WORKERS}
PREPROCESS_WORKERS=${PREPROCESS_WORKERS}
COLLATE_THREADS=${COLLATE_THREADS}
PREFETCH_FACTOR=${PREFETCH_FACTOR}
TRAINING_TASK=${TRAINING_TASK}
MASS_CLASSIFICATION=${MASS_CLASSIFICATION}
VAL_FRACTION=${VAL_FRACTION}
TEST_FRACTION=${TEST_FRACTION}
SPLIT_MODE=${SPLIT_MODE}
PARTICLE_FILTER=${PARTICLE_FILTER}
DEVICE=${DEVICE}
DIAGNOSTIC_MIN_BIN_COUNT=${DIAGNOSTIC_MIN_BIN_COUNT}
MAX_GRAPHS=${MAX_GRAPHS}
EOF

cat <<EOF
======================================================================
LARGE TRAINING READY

run_dir:
  ${RUN_DIR}

checkpoint:
  ${CHECKPOINT}

graph_input:
  ${GRAPH_INPUT}

This script does not read DST files.
Network storage is not needed after this line if the graph shards are local.
$(date)
======================================================================
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: training command will not be executed."
  exit 0
fi

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  echo "SKIP_BUILD=1: using existing extension build"
else
  ./build_extensions.sh
fi

{
  echo "run_dir=${RUN_DIR}"
  echo "config=${CONFIG_NAME}"
  echo "graph_input=${GRAPH_INPUT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "This job does not read DST files."
  date

  cmd=(.venv/bin/talesd-gnn train \
    --graphs "${GRAPH_INPUT}" \
    -o "${CHECKPOINT}" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --layers "${LAYERS}" \
    --dropout "${DROPOUT}" \
    --lr-scheduler "${LR_SCHEDULER}" \
    --lr-factor "${LR_FACTOR}" \
    --lr-patience "${LR_PATIENCE}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --model-architecture "${MODEL_ARCHITECTURE}" \
    --readout-heads "${READOUT_HEADS}" \
    --detector-embedding-dim "${DETECTOR_EMBEDDING_DIM}" \
    --waveform-encoder "${WAVEFORM_ENCODER}" \
    --waveform-embedding-dim "${WAVEFORM_EMBEDDING_DIM}" \
    --waveform-transformer-heads "${WAVEFORM_TRANSFORMER_HEADS}" \
    --waveform-transformer-layers "${WAVEFORM_TRANSFORMER_LAYERS}" \
    --loss-mode "${LOSS_MODE}" \
    --energy-loss-weight "${ENERGY_WEIGHT}" \
    --core-loss-weight "${CORE_WEIGHT}" \
    --direction-loss-weight "${DIRECTION_WEIGHT}" \
    --core-loss-scale-km "${CORE_SCALE_KM}" \
    --particle-filter "${PARTICLE_FILTER}" \
    --device "${DEVICE}" \
    --num-workers "${TRAIN_WORKERS}" \
    --preprocess-workers "${PREPROCESS_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --collate-backend cpp \
    --collate-threads "${COLLATE_THREADS}" \
    --training-task "${TRAINING_TASK}" \
    --split-mode "${SPLIT_MODE}" \
    --test-fraction "${TEST_FRACTION}" \
    --val-fraction "${VAL_FRACTION}" \
    --diagnostic-energy-bin-width 0.1 \
    --diagnostic-min-bin-count "${DIAGNOSTIC_MIN_BIN_COUNT}")

  if [[ -n "${MAX_GRAPHS}" ]]; then
    cmd+=(--max-graphs "${MAX_GRAPHS}")
  fi
  if [[ "${MASS_CLASSIFICATION}" == "1" || "${TRAINING_TASK}" == "mass" ]]; then
    cmd+=(--mass-classification)
  fi

  "${cmd[@]}"

  echo "checkpoint=${CHECKPOINT}"
  echo "metrics=${METRICS_PATH}"
  echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
  date
} 2>&1 | tee "${LOG_PATH}"

.venv/bin/python scripts/summarize_metrics.py "${METRICS_PATH}" -o "${SUMMARY_DIR}/metrics_summary.csv"
if [[ "${TRAINING_TASK}" == "mass" ]]; then
  precision_status="N/A"
  cat > "${PRECISION_REPORT}" <<EOF
Precision gate is not applied to mass-only training.

Use:
  checkpoints/${CONFIG_NAME}.pt.metrics.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_classification.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_classification.pdf
EOF
else
  if .venv/bin/python scripts/check_precision_targets.py "${METRICS_PATH}" --min-bin-count "${PRECISION_MIN_BIN_COUNT}" -o "${PRECISION_REPORT}"; then
    precision_status="PASS"
  else
    precision_status="FAIL"
  fi
fi

cat > "${RUN_DIR}/README.txt" <<EOF
Run: ${RUN_NAME}
Created: $(date)
Purpose: large ${TRAINING_TASK} training from existing local graph shards.

Important files:
  config/train.env
  logs/${CONFIG_NAME}.log
  checkpoints/${CONFIG_NAME}.pt
  checkpoints/${CONFIG_NAME}.pt.metrics.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/
  summaries/metrics_summary.csv
  summaries/${CONFIG_NAME}_precision_targets.txt
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_classification.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_classification.pdf

Graph input:
  ${GRAPH_INPUT}

Precision gate:
  ${precision_status}
EOF

echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${METRICS_PATH}"
echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
echo "summary_csv=${SUMMARY_DIR}/metrics_summary.csv"
echo "precision_report=${PRECISION_REPORT}"
echo "precision_status=${precision_status}"
echo "log_path=${LOG_PATH}"
date
