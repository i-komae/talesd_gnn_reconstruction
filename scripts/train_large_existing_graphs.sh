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
CLASSIFICATION_ARCH="${CLASSIFICATION_ARCH:-enhanced}"
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
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-0}"
LOSS_MODE="${LOSS_MODE:-scaled-mse}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.0}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.05}"
ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-1.0}"
ENERGY_BIAS_WEIGHT="${ENERGY_BIAS_WEIGHT:-0.0}"
ENERGY_PARTICLE_BIAS_WEIGHT="${ENERGY_PARTICLE_BIAS_WEIGHT:-0.0}"
ENERGY_BIAS_BIN_WIDTH="${ENERGY_BIAS_BIN_WIDTH:-0.1}"
ENERGY_BIAS_MIN_BIN_COUNT="${ENERGY_BIAS_MIN_BIN_COUNT:-8}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
H5_MAX_OPEN_FILES="${H5_MAX_OPEN_FILES:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"
TRAINING_TASK="${TRAINING_TASK:-reconstruction}"
MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-0}"
MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT:-0.1}"
MASS_LOSS_MODE="${MASS_LOSS_MODE:-focal}"
MASS_FOCAL_GAMMA="${MASS_FOCAL_GAMMA:-2.0}"
MASS_POS_WEIGHT_MODE="${MASS_POS_WEIGHT_MODE:-none}"
MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0}"
MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}"
MASS_COLLAPSE_PATIENCE="${MASS_COLLAPSE_PATIENCE:-3}"
MASS_COLLAPSE_SCORE_STD="${MASS_COLLAPSE_SCORE_STD:-1e-3}"
MASS_COLLAPSE_BALANCED_ACCURACY="${MASS_COLLAPSE_BALANCED_ACCURACY:-0.505}"
QUALITY_PREDICTION="${QUALITY_PREDICTION:-0}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.2}"
QUALITY_ANGULAR_SCALE_DEG="${QUALITY_ANGULAR_SCALE_DEG:-1.0}"
QUALITY_CORE_SCALE_KM="${QUALITY_CORE_SCALE_KM:-0.05}"
QUALITY_ENERGY_SCALE="${QUALITY_ENERGY_SCALE:-0.10}"
ERROR_PREDICTION="${ERROR_PREDICTION:-0}"
ERROR_WEIGHT="${ERROR_WEIGHT:-0.2}"
ERROR_ANGULAR_SCALE_DEG="${ERROR_ANGULAR_SCALE_DEG:-${QUALITY_ANGULAR_SCALE_DEG}}"
ERROR_CORE_SCALE_KM="${ERROR_CORE_SCALE_KM:-${QUALITY_CORE_SCALE_KM}}"
ERROR_ENERGY_SCALE="${ERROR_ENERGY_SCALE:-${QUALITY_ENERGY_SCALE}}"
NLL_WEIGHT="${NLL_WEIGHT:-0.2}"
NLL_SIGMA_ENERGY_FLOOR="${NLL_SIGMA_ENERGY_FLOOR:-0.01}"
NLL_SIGMA_ANGLE_FLOOR_DEG="${NLL_SIGMA_ANGLE_FLOOR_DEG:-0.05}"
NLL_SIGMA_CORE_FLOOR_KM="${NLL_SIGMA_CORE_FLOOR_KM:-0.005}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.20}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
DEVICE="${DEVICE:-cpu}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
PRECISION_MIN_BIN_COUNT="${PRECISION_MIN_BIN_COUNT:-1000}"
EPOCH_LEARNING_CURVE="${EPOCH_LEARNING_CURVE:-1}"
BEST_DIAGNOSTICS="${BEST_DIAGNOSTICS:-1}"
BEST_DIAGNOSTIC_MAX_GRAPHS="${BEST_DIAGNOSTIC_MAX_GRAPHS:-20000}"
FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-1}"
FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation}"
FEATURE_IMPORTANCE_MAX_GRAPHS="${FEATURE_IMPORTANCE_MAX_GRAPHS:-50000}"
FEATURE_IMPORTANCE_BATCH_SIZE="${FEATURE_IMPORTANCE_BATCH_SIZE:-256}"
FEATURE_IMPORTANCE_DEVICE="${FEATURE_IMPORTANCE_DEVICE:-${DEVICE}}"
MAX_GRAPHS="${MAX_GRAPHS:-}"
SEED="${SEED:-12345}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ "${TRAINING_TASK}" != "mass" && ( "${LOSS_MODE}" == "physics-nll" || "${LOSS_MODE}" == "nll" ) ]]; then
  ERROR_PREDICTION=1
fi

export TALESD_GNN_H5_MAX_OPEN_FILES="${H5_MAX_OPEN_FILES}"

if [[ -z "${CONFIG_NAME:-}" ]]; then
  if [[ "${TRAINING_TASK}" == "mass" ]]; then
    CONFIG_NAME="mass_${MODEL_ARCHITECTURE}_clf${CLASSIFICATION_ARCH}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${MASS_LOSS_MODE}_${PARTICLE_FILTER}_${TRAIN_EPOCHS}epoch"
  elif [[ "${MASS_CLASSIFICATION}" == "1" ]]; then
    CONFIG_NAME="${TRAINING_TASK}_mass_${MODEL_ARCHITECTURE}_clf${CLASSIFICATION_ARCH}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${MASS_LOSS_MODE}_${PARTICLE_FILTER}_${TRAIN_EPOCHS}epoch"
  else
    CONFIG_NAME="${TRAINING_TASK}_${MODEL_ARCHITECTURE}_clf${CLASSIFICATION_ARCH}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${PARTICLE_FILTER}_${TRAIN_EPOCHS}epoch"
  fi
fi
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
CLASSIFICATION_ARCH=${CLASSIFICATION_ARCH}
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
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS}
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
TRAIN_WORKERS=${TRAIN_WORKERS}
PREPROCESS_WORKERS=${PREPROCESS_WORKERS}
COLLATE_THREADS=${COLLATE_THREADS}
PREFETCH_FACTOR=${PREFETCH_FACTOR}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS}
H5_MAX_OPEN_FILES=${H5_MAX_OPEN_FILES}
PIN_MEMORY=${PIN_MEMORY}
TRAINING_TASK=${TRAINING_TASK}
MASS_CLASSIFICATION=${MASS_CLASSIFICATION}
MASS_LOSS_WEIGHT=${MASS_LOSS_WEIGHT}
MASS_LOSS_MODE=${MASS_LOSS_MODE}
MASS_FOCAL_GAMMA=${MASS_FOCAL_GAMMA}
MASS_POS_WEIGHT_MODE=${MASS_POS_WEIGHT_MODE}
MASS_RANKING_WEIGHT=${MASS_RANKING_WEIGHT}
MASS_RANKING_MARGIN=${MASS_RANKING_MARGIN}
MASS_COLLAPSE_PATIENCE=${MASS_COLLAPSE_PATIENCE}
MASS_COLLAPSE_SCORE_STD=${MASS_COLLAPSE_SCORE_STD}
MASS_COLLAPSE_BALANCED_ACCURACY=${MASS_COLLAPSE_BALANCED_ACCURACY}
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
PARTICLE_FILTER=${PARTICLE_FILTER}
DEVICE=${DEVICE}
DIAGNOSTIC_MIN_BIN_COUNT=${DIAGNOSTIC_MIN_BIN_COUNT}
EPOCH_LEARNING_CURVE=${EPOCH_LEARNING_CURVE}
BEST_DIAGNOSTICS=${BEST_DIAGNOSTICS}
BEST_DIAGNOSTIC_MAX_GRAPHS=${BEST_DIAGNOSTIC_MAX_GRAPHS}
FEATURE_IMPORTANCE=${FEATURE_IMPORTANCE}
FEATURE_IMPORTANCE_SPLIT=${FEATURE_IMPORTANCE_SPLIT}
FEATURE_IMPORTANCE_MAX_GRAPHS=${FEATURE_IMPORTANCE_MAX_GRAPHS}
FEATURE_IMPORTANCE_BATCH_SIZE=${FEATURE_IMPORTANCE_BATCH_SIZE}
FEATURE_IMPORTANCE_DEVICE=${FEATURE_IMPORTANCE_DEVICE}
MAX_GRAPHS=${MAX_GRAPHS}
SEED=${SEED}
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

  cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli train \
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
    --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}" \
    --model-architecture "${MODEL_ARCHITECTURE}" \
    --readout-heads "${READOUT_HEADS}" \
    --classification-arch "${CLASSIFICATION_ARCH}" \
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
    --angular-loss-scale-deg "${ANGULAR_SCALE_DEG}" \
    --energy-bias-loss-weight "${ENERGY_BIAS_WEIGHT}" \
    --energy-particle-bias-loss-weight "${ENERGY_PARTICLE_BIAS_WEIGHT}" \
    --energy-bias-bin-width "${ENERGY_BIAS_BIN_WIDTH}" \
    --energy-bias-min-bin-count "${ENERGY_BIAS_MIN_BIN_COUNT}" \
    --nll-loss-weight "${NLL_WEIGHT}" \
    --nll-sigma-energy-floor "${NLL_SIGMA_ENERGY_FLOOR}" \
    --nll-sigma-angle-floor-deg "${NLL_SIGMA_ANGLE_FLOOR_DEG}" \
    --nll-sigma-core-floor-km "${NLL_SIGMA_CORE_FLOOR_KM}" \
    --particle-filter "${PARTICLE_FILTER}" \
    --device "${DEVICE}" \
    --num-workers "${TRAIN_WORKERS}" \
    --preprocess-workers "${PREPROCESS_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --collate-backend cpp \
    --collate-threads "${COLLATE_THREADS}" \
    --training-task "${TRAINING_TASK}" \
    --mass-loss-weight "${MASS_LOSS_WEIGHT}" \
    --mass-loss-mode "${MASS_LOSS_MODE}" \
    --mass-focal-gamma "${MASS_FOCAL_GAMMA}" \
    --mass-pos-weight-mode "${MASS_POS_WEIGHT_MODE}" \
    --mass-ranking-weight "${MASS_RANKING_WEIGHT}" \
    --mass-ranking-margin "${MASS_RANKING_MARGIN}" \
    --mass-collapse-patience "${MASS_COLLAPSE_PATIENCE}" \
    --mass-collapse-score-std "${MASS_COLLAPSE_SCORE_STD}" \
    --mass-collapse-balanced-accuracy "${MASS_COLLAPSE_BALANCED_ACCURACY}" \
    --split-mode "${SPLIT_MODE}" \
    --test-fraction "${TEST_FRACTION}" \
    --val-fraction "${VAL_FRACTION}" \
    --source-test-fraction "${SOURCE_TEST_FRACTION}" \
    --source-val-fraction "${SOURCE_VAL_FRACTION}" \
    --seed "${SEED}" \
    --diagnostic-energy-bin-width 0.1 \
    --diagnostic-min-bin-count "${DIAGNOSTIC_MIN_BIN_COUNT}" \
    --best-diagnostic-max-graphs "${BEST_DIAGNOSTIC_MAX_GRAPHS}")

  if [[ -n "${MAX_GRAPHS}" ]]; then
    cmd+=(--max-graphs "${MAX_GRAPHS}")
  fi
  if [[ "${EPOCH_LEARNING_CURVE}" == "0" ]]; then
    cmd+=(--no-epoch-learning-curve)
  fi
  if [[ "${BEST_DIAGNOSTICS}" == "0" ]]; then
    cmd+=(--no-best-diagnostics)
  fi
  if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
    cmd+=(--persistent-workers)
  fi
  if [[ "${PIN_MEMORY}" == "0" ]]; then
    cmd+=(--no-pin-memory)
  fi
  if [[ "${MASS_CLASSIFICATION}" == "1" || "${TRAINING_TASK}" == "mass" ]]; then
    cmd+=(--mass-classification)
  fi
  if [[ "${QUALITY_PREDICTION}" == "1" ]]; then
    cmd+=(--quality-prediction \
      --quality-loss-weight "${QUALITY_WEIGHT}" \
      --quality-angular-scale-deg "${QUALITY_ANGULAR_SCALE_DEG}" \
      --quality-core-scale-km "${QUALITY_CORE_SCALE_KM}" \
      --quality-energy-scale "${QUALITY_ENERGY_SCALE}")
  fi
  if [[ "${ERROR_PREDICTION}" == "1" && "${TRAINING_TASK}" != "mass" ]]; then
    cmd+=(--error-prediction \
      --error-loss-weight "${ERROR_WEIGHT}" \
      --error-angular-scale-deg "${ERROR_ANGULAR_SCALE_DEG}" \
      --error-core-scale-km "${ERROR_CORE_SCALE_KM}" \
      --error-energy-scale "${ERROR_ENERGY_SCALE}")
  fi

  echo "stage=start talesd_gnn_train date=$(date)"
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
  echo "stage=done talesd_gnn_train date=$(date)"

  echo "checkpoint=${CHECKPOINT}"
  echo "metrics=${METRICS_PATH}"
  echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
  date
} 2>&1 | tee "${LOG_PATH}"

"${PYTHON_BIN}" scripts/summarize_metrics.py "${METRICS_PATH}" -o "${SUMMARY_DIR}/metrics_summary.csv"
FEATURE_IMPORTANCE_DIR="${CHECKPOINT}.diagnostics/feature_importance/${FEATURE_IMPORTANCE_SPLIT}"
FEATURE_IMPORTANCE_SUMMARY="${FEATURE_IMPORTANCE_DIR}/feature_group_importance.json"
if [[ "${FEATURE_IMPORTANCE}" == "1" ]]; then
  {
    echo "stage=start feature_importance date=$(date)"
    feature_cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli feature-importance
      --graphs "${GRAPH_INPUT}"
      --checkpoint "${CHECKPOINT}"
      -o "${FEATURE_IMPORTANCE_DIR}"
      --split "${FEATURE_IMPORTANCE_SPLIT}"
      --max-graphs "${FEATURE_IMPORTANCE_MAX_GRAPHS}"
      --batch-size "${FEATURE_IMPORTANCE_BATCH_SIZE}"
      --device "${FEATURE_IMPORTANCE_DEVICE}"
      --seed "${SEED}")
    printf 'command:'
    printf ' %q' "${feature_cmd[@]}"
    printf '\n'
    "${feature_cmd[@]}"
    if [[ ! -s "${FEATURE_IMPORTANCE_SUMMARY}" ]]; then
      echo "ERROR: feature importance finished but summary was not written: ${FEATURE_IMPORTANCE_SUMMARY}" >&2
      exit 1
    fi
    echo "stage=done feature_importance date=$(date)"
    echo "feature_importance=${FEATURE_IMPORTANCE_SUMMARY}"
  } 2>&1 | tee -a "${LOG_PATH}"
else
  FEATURE_IMPORTANCE_SUMMARY=""
fi

if [[ "${TRAINING_TASK}" == "mass" ]]; then
  precision_status="N/A"
  cat > "${PRECISION_REPORT}" <<EOF
Precision gate is not applied to mass-only training.

Use:
  checkpoints/${CONFIG_NAME}.pt.metrics.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_confusion_matrix.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_score_distribution.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_roc.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_accuracy_by_true_energy.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_confusion_matrix.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_score_distribution.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_roc.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_accuracy_by_true_energy.pdf
EOF
else
  if "${PYTHON_BIN}" scripts/check_precision_targets.py "${METRICS_PATH}" --min-bin-count "${PRECISION_MIN_BIN_COUNT}" -o "${PRECISION_REPORT}"; then
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
  checkpoints/${CONFIG_NAME}.pt.diagnostics/prediction_cache.npz
  summaries/metrics_summary.csv
  summaries/${CONFIG_NAME}_precision_targets.txt
  checkpoints/${CONFIG_NAME}.pt.diagnostics/feature_importance/${FEATURE_IMPORTANCE_SPLIT}/feature_group_importance.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/feature_importance/${FEATURE_IMPORTANCE_SPLIT}/feature_group_importance.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_confusion_matrix.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_score_distribution.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_roc.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/validation/mass_accuracy_by_true_energy.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_confusion_matrix.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_score_distribution.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_roc.pdf
  checkpoints/${CONFIG_NAME}.pt.diagnostics/test/mass_accuracy_by_true_energy.pdf

Graph input:
  ${GRAPH_INPUT}

Precision gate:
  ${precision_status}
EOF

echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${METRICS_PATH}"
echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
echo "prediction_cache=${CHECKPOINT}.diagnostics/prediction_cache.npz"
echo "summary_csv=${SUMMARY_DIR}/metrics_summary.csv"
if [[ -n "${FEATURE_IMPORTANCE_SUMMARY}" ]]; then
  echo "feature_importance=${FEATURE_IMPORTANCE_SUMMARY}"
fi
echo "precision_report=${PRECISION_REPORT}"
echo "precision_status=${precision_status}"
echo "log_path=${LOG_PATH}"
date
