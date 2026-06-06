#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_INPUT="${GRAPH_INPUT:-}"
if [[ -z "${GRAPH_INPUT}" ]]; then
  echo "GRAPH_INPUT is required for hetero training" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-hetero_train_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
CONFIG_DIR="${RUN_DIR}/config"

HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-5}"
DROPOUT="${DROPOUT:-0.08}"
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"
WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
WAVEFORM_LENGTH="${WAVEFORM_LENGTH:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
LOSS_MODE="${LOSS_MODE:-physics}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.0}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.05}"
ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-1.0}"
ENERGY_BIAS_WEIGHT="${ENERGY_BIAS_WEIGHT:-1.0}"
ENERGY_PARTICLE_BIAS_WEIGHT="${ENERGY_PARTICLE_BIAS_WEIGHT:-1.0}"
ENERGY_BIAS_BIN_WIDTH="${ENERGY_BIAS_BIN_WIDTH:-0.1}"
ENERGY_BIAS_MIN_BIN_COUNT="${ENERGY_BIAS_MIN_BIN_COUNT:-8}"
MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-0}"
MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT:-0.15}"
MASS_LOSS_MODE="${MASS_LOSS_MODE:-bce}"
MASS_FOCAL_GAMMA="${MASS_FOCAL_GAMMA:-2.0}"
MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0.5}"
MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}"
QUALITY_PREDICTION="${QUALITY_PREDICTION:-0}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.2}"
QUALITY_ANGULAR_SCALE_DEG="${QUALITY_ANGULAR_SCALE_DEG:-1.0}"
QUALITY_CORE_SCALE_KM="${QUALITY_CORE_SCALE_KM:-0.05}"
QUALITY_ENERGY_SCALE="${QUALITY_ENERGY_SCALE:-0.10}"
ERROR_PREDICTION="${ERROR_PREDICTION:-0}"
ERROR_WEIGHT="${ERROR_WEIGHT:-0.2}"
ERROR_ANGULAR_SCALE_DEG="${ERROR_ANGULAR_SCALE_DEG:-1.0}"
ERROR_CORE_SCALE_KM="${ERROR_CORE_SCALE_KM:-0.05}"
ERROR_ENERGY_SCALE="${ERROR_ENERGY_SCALE:-0.10}"
NLL_WEIGHT="${NLL_WEIGHT:-0.2}"
NLL_SIGMA_ENERGY_FLOOR="${NLL_SIGMA_ENERGY_FLOOR:-0.01}"
NLL_SIGMA_ANGLE_FLOOR_DEG="${NLL_SIGMA_ANGLE_FLOOR_DEG:-0.05}"
NLL_SIGMA_CORE_FLOOR_KM="${NLL_SIGMA_CORE_FLOOR_KM:-0.005}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.20}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
DEVICE="${DEVICE:-cpu}"
DIAGNOSTICS="${DIAGNOSTICS:-1}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-0}"
SEED="${SEED:-12345}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ "${LOSS_MODE}" == "physics-nll" || "${LOSS_MODE}" == "nll" ]]; then
  ERROR_PREDICTION=1
fi

if [[ "${FEATURE_IMPORTANCE}" == "1" ]]; then
  echo "hetero feature importance is not implemented yet; set FEATURE_IMPORTANCE=0" >&2
  exit 2
fi

if [[ -z "${CONFIG_NAME:-}" ]]; then
  AUX_HEAD_TAG=""
  if [[ "${QUALITY_PREDICTION}" == "1" && "${ERROR_PREDICTION}" == "1" ]]; then
    AUX_HEAD_TAG="_quality_error"
  elif [[ "${QUALITY_PREDICTION}" == "1" ]]; then
    AUX_HEAD_TAG="_quality"
  elif [[ "${ERROR_PREDICTION}" == "1" ]]; then
    AUX_HEAD_TAG="_error"
  fi
  if [[ "${MASS_CLASSIFICATION}" == "1" ]]; then
    CONFIG_NAME="hetero_reconstruction_mass${AUX_HEAD_TAG}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${MASS_LOSS_MODE}_${TRAIN_EPOCHS}epoch"
  else
    CONFIG_NAME="hetero_reconstruction${AUX_HEAD_TAG}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${TRAIN_EPOCHS}epoch"
  fi
fi

CHECKPOINT="${CHECKPOINT_DIR}/${CONFIG_NAME}.pt"
LOG_PATH="${LOG_DIR}/${CONFIG_NAME}.log"
METRICS_PATH="${CHECKPOINT}.metrics.json"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}" "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/train.env" <<EOF
RUN_ID=${RUN_ID}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GRAPH_INPUT=${GRAPH_INPUT}
CONFIG_NAME=${CONFIG_NAME}
HIDDEN_DIM=${HIDDEN_DIM}
LAYERS=${LAYERS}
DROPOUT=${DROPOUT}
WAVEFORM_ENCODER=${WAVEFORM_ENCODER}
WAVEFORM_EMBEDDING_DIM=${WAVEFORM_EMBEDDING_DIM}
WAVEFORM_LENGTH=${WAVEFORM_LENGTH}
TRAIN_EPOCHS=${TRAIN_EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
LR=${LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
LOSS_MODE=${LOSS_MODE}
ENERGY_WEIGHT=${ENERGY_WEIGHT}
CORE_WEIGHT=${CORE_WEIGHT}
DIRECTION_WEIGHT=${DIRECTION_WEIGHT}
CORE_SCALE_KM=${CORE_SCALE_KM}
ANGULAR_SCALE_DEG=${ANGULAR_SCALE_DEG}
ENERGY_BIAS_WEIGHT=${ENERGY_BIAS_WEIGHT}
ENERGY_PARTICLE_BIAS_WEIGHT=${ENERGY_PARTICLE_BIAS_WEIGHT}
ENERGY_BIAS_BIN_WIDTH=${ENERGY_BIAS_BIN_WIDTH}
ENERGY_BIAS_MIN_BIN_COUNT=${ENERGY_BIAS_MIN_BIN_COUNT}
MASS_CLASSIFICATION=${MASS_CLASSIFICATION}
MASS_LOSS_WEIGHT=${MASS_LOSS_WEIGHT}
MASS_LOSS_MODE=${MASS_LOSS_MODE}
MASS_FOCAL_GAMMA=${MASS_FOCAL_GAMMA}
MASS_RANKING_WEIGHT=${MASS_RANKING_WEIGHT}
MASS_RANKING_MARGIN=${MASS_RANKING_MARGIN}
QUALITY_PREDICTION=${QUALITY_PREDICTION}
QUALITY_WEIGHT=${QUALITY_WEIGHT}
QUALITY_ANGULAR_SCALE_DEG=${QUALITY_ANGULAR_SCALE_DEG}
QUALITY_CORE_SCALE_KM=${QUALITY_CORE_SCALE_KM}
QUALITY_ENERGY_SCALE=${QUALITY_ENERGY_SCALE}
ERROR_PREDICTION=${ERROR_PREDICTION}
ERROR_WEIGHT=${ERROR_WEIGHT}
ERROR_ANGULAR_SCALE_DEG=${ERROR_ANGULAR_SCALE_DEG}
ERROR_CORE_SCALE_KM=${ERROR_CORE_SCALE_KM}
ERROR_ENERGY_SCALE=${ERROR_ENERGY_SCALE}
NLL_WEIGHT=${NLL_WEIGHT}
NLL_SIGMA_ENERGY_FLOOR=${NLL_SIGMA_ENERGY_FLOOR}
NLL_SIGMA_ANGLE_FLOOR_DEG=${NLL_SIGMA_ANGLE_FLOOR_DEG}
NLL_SIGMA_CORE_FLOOR_KM=${NLL_SIGMA_CORE_FLOOR_KM}
VAL_FRACTION=${VAL_FRACTION}
TEST_FRACTION=${TEST_FRACTION}
SOURCE_VAL_FRACTION=${SOURCE_VAL_FRACTION}
SOURCE_TEST_FRACTION=${SOURCE_TEST_FRACTION}
SPLIT_MODE=${SPLIT_MODE}
DEVICE=${DEVICE}
DIAGNOSTICS=${DIAGNOSTICS}
DIAGNOSTIC_MIN_BIN_COUNT=${DIAGNOSTIC_MIN_BIN_COUNT}
FEATURE_IMPORTANCE=${FEATURE_IMPORTANCE}
SEED=${SEED}
EOF

cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli train-hetero
  --graphs "${GRAPH_INPUT}"
  -o "${CHECKPOINT}"
  --epochs "${TRAIN_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --layers "${LAYERS}"
  --dropout "${DROPOUT}"
  --waveform-encoder "${WAVEFORM_ENCODER}"
  --waveform-embedding-dim "${WAVEFORM_EMBEDDING_DIM}"
  --loss-mode "${LOSS_MODE}"
  --energy-loss-weight "${ENERGY_WEIGHT}"
  --core-loss-weight "${CORE_WEIGHT}"
  --direction-loss-weight "${DIRECTION_WEIGHT}"
  --core-loss-scale-km "${CORE_SCALE_KM}"
  --angular-loss-scale-deg "${ANGULAR_SCALE_DEG}"
  --energy-bias-loss-weight "${ENERGY_BIAS_WEIGHT}"
  --energy-particle-bias-loss-weight "${ENERGY_PARTICLE_BIAS_WEIGHT}"
  --energy-bias-bin-width "${ENERGY_BIAS_BIN_WIDTH}"
  --energy-bias-min-bin-count "${ENERGY_BIAS_MIN_BIN_COUNT}"
  --mass-loss-weight "${MASS_LOSS_WEIGHT}"
  --mass-loss-mode "${MASS_LOSS_MODE}"
  --mass-focal-gamma "${MASS_FOCAL_GAMMA}"
  --mass-ranking-weight "${MASS_RANKING_WEIGHT}"
  --mass-ranking-margin "${MASS_RANKING_MARGIN}"
  --quality-loss-weight "${QUALITY_WEIGHT}"
  --quality-angular-scale-deg "${QUALITY_ANGULAR_SCALE_DEG}"
  --quality-core-scale-km "${QUALITY_CORE_SCALE_KM}"
  --quality-energy-scale "${QUALITY_ENERGY_SCALE}"
  --error-loss-weight "${ERROR_WEIGHT}"
  --error-angular-scale-deg "${ERROR_ANGULAR_SCALE_DEG}"
  --error-core-scale-km "${ERROR_CORE_SCALE_KM}"
  --error-energy-scale "${ERROR_ENERGY_SCALE}"
  --nll-loss-weight "${NLL_WEIGHT}"
  --nll-sigma-energy-floor "${NLL_SIGMA_ENERGY_FLOOR}"
  --nll-sigma-angle-floor-deg "${NLL_SIGMA_ANGLE_FLOOR_DEG}"
  --nll-sigma-core-floor-km "${NLL_SIGMA_CORE_FLOOR_KM}"
  --split-mode "${SPLIT_MODE}"
  --test-fraction "${TEST_FRACTION}"
  --val-fraction "${VAL_FRACTION}"
  --source-test-fraction "${SOURCE_TEST_FRACTION}"
  --source-val-fraction "${SOURCE_VAL_FRACTION}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --diagnostic-energy-bin-width 0.1
  --diagnostic-min-bin-count "${DIAGNOSTIC_MIN_BIN_COUNT}")

if [[ -n "${WAVEFORM_LENGTH}" ]]; then
  cmd+=(--waveform-length "${WAVEFORM_LENGTH}")
fi
if [[ "${MASS_CLASSIFICATION}" == "1" ]]; then
  cmd+=(--mass-classification)
fi
if [[ "${QUALITY_PREDICTION}" == "1" ]]; then
  cmd+=(--quality-prediction)
fi
if [[ "${ERROR_PREDICTION}" == "1" ]]; then
  cmd+=(--error-prediction)
fi
if [[ "${DIAGNOSTICS}" == "1" ]]; then
  cmd+=(--diagnostics)
fi

cat <<EOF
======================================================================
HETERO TRAINING READY

run_dir:
  ${RUN_DIR}

checkpoint:
  ${CHECKPOINT}

graph_input:
  ${GRAPH_INPUT}

This script trains from hetero HDF5 graphs made by talesd-gnn export-hetero.
$(date)
======================================================================
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

{
  echo "run_dir=${RUN_DIR}"
  echo "config=${CONFIG_NAME}"
  echo "graph_input=${GRAPH_INPUT}"
  echo "checkpoint=${CHECKPOINT}"
  date
  echo "stage=start talesd_gnn_train_hetero date=$(date)"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "ERROR: training command finished but checkpoint was not written: ${CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -s "${METRICS_PATH}" ]]; then
    echo "ERROR: training command finished but metrics were not written: ${METRICS_PATH}" >&2
    exit 1
  fi
  echo "stage=done talesd_gnn_train_hetero date=$(date)"
  echo "checkpoint=${CHECKPOINT}"
  echo "metrics=${METRICS_PATH}"
  echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
  date
} 2>&1 | tee "${LOG_PATH}"

"${PYTHON_BIN}" scripts/summarize_metrics.py "${METRICS_PATH}" -o "${SUMMARY_DIR}/metrics_summary.csv"

cat > "${RUN_DIR}/README.txt" <<EOF
Run: ${RUN_NAME}
Created: $(date)
Purpose: hetero training from existing dstio.tale.graph HDF5 graphs.

Important files:
  config/train.env
  logs/${CONFIG_NAME}.log
  checkpoints/${CONFIG_NAME}.pt
  checkpoints/${CONFIG_NAME}.pt.metrics.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/
  summaries/metrics_summary.csv

Graph input:
  ${GRAPH_INPUT}

Not yet integrated:
  hetero feature importance
  full HGT/HeteroConv production architecture
  large-scale server training confirmation
EOF

echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${METRICS_PATH}"
echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
echo "summary_csv=${SUMMARY_DIR}/metrics_summary.csv"
echo "log_path=${LOG_PATH}"
date
