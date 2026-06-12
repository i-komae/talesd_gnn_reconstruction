#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_INPUT="${GRAPH_INPUT:-}"
if [[ -z "${GRAPH_INPUT}" ]]; then
  echo "GRAPH_INPUT is required for hetero training" >&2
  exit 2
fi
GRAPH_INPUT_ORIGINAL="${GRAPH_INPUT}"

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
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-hetero_attention}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
READOUT_HEADS="${READOUT_HEADS:-4}"
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"
WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
WAVEFORM_LENGTH="${WAVEFORM_LENGTH:-}"
WAVEFORM_TRANSFORMER_HEADS="${WAVEFORM_TRANSFORMER_HEADS:-4}"
WAVEFORM_TRANSFORMER_LAYERS="${WAVEFORM_TRANSFORMER_LAYERS:-1}"
WAVEFORM_TRANSFORMER_MAX_TOKENS="${WAVEFORM_TRANSFORMER_MAX_TOKENS:-128}"
WAVEFORM_TRANSFORMER_DOWNSAMPLE="${WAVEFORM_TRANSFORMER_DOWNSAMPLE:-adaptive_avg}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    BATCH_SIZE=32
  else
    BATCH_SIZE=128
  fi
fi
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    GRADIENT_ACCUMULATION_STEPS=4
  else
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
LOSS_MODE="${LOSS_MODE:-physics}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.0}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.05}"
ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-1.0}"
ENERGY_BIAS_WEIGHT="${ENERGY_BIAS_WEIGHT:-1.0}"
ENERGY_PARTICLE_BIAS_WEIGHT="${ENERGY_PARTICLE_BIAS_WEIGHT:-0.0}"
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
TRAIN_WORKERS="${TRAIN_WORKERS:--1}"
SPLIT_WORKERS="${SPLIT_WORKERS:-4}"
if [[ -z "${PREFETCH_FACTOR:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    PREFETCH_FACTOR=1
  else
    PREFETCH_FACTOR=2
  fi
fi
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
VALIDATE_EVERY_N_EPOCHS="${VALIDATE_EVERY_N_EPOCHS:-1}"
MAX_VAL_GRAPHS="${MAX_VAL_GRAPHS:-}"
MAX_GRAPHS="${MAX_GRAPHS:-}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-32}"
CHECKPOINT_MILESTONES="${CHECKPOINT_MILESTONES:-8,16,32,64}"
HETERO_TRAINING_DATA_FORMAT="${HETERO_TRAINING_DATA_FORMAT:-fast_tensor}"
HETERO_RELATIONS="${HETERO_RELATIONS:-all}"
DATALOADER_TIMEOUT_SEC="${DATALOADER_TIMEOUT_SEC:-300}"
DATA_WAIT_WARN_SEC="${DATA_WAIT_WARN_SEC:-30}"
PROFILE="${PROFILE:-${TALESD_GNN_PROFILE:-0}}"
if [[ -z "${PIN_MEMORY:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    PIN_MEMORY=0
  else
    PIN_MEMORY=1
  fi
fi
if [[ -z "${AMP:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" && "${DEVICE}" == cuda* ]]; then
    AMP=fp16
  else
    AMP=off
  fi
fi
TRAIN_LOADER_MEMORY_BUDGET_GIB="${TRAIN_LOADER_MEMORY_BUDGET_GIB:-}"
TRAIN_LOADER_MEMORY_ESTIMATE_SAMPLES="${TRAIN_LOADER_MEMORY_ESTIMATE_SAMPLES:-512}"
DIAGNOSTICS="${DIAGNOSTICS:-1}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-0}"
FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation}"
FEATURE_IMPORTANCE_MAX_GRAPHS="${FEATURE_IMPORTANCE_MAX_GRAPHS:-50000}"
FEATURE_IMPORTANCE_BATCH_SIZE="${FEATURE_IMPORTANCE_BATCH_SIZE:-256}"
FEATURE_IMPORTANCE_DEVICE="${FEATURE_IMPORTANCE_DEVICE:-${DEVICE}}"
ATTENTION_MAPS="${ATTENTION_MAPS:-1}"
ATTENTION_MAPS_SPLIT="${ATTENTION_MAPS_SPLIT:-validation}"
ATTENTION_MAPS_MAX_GRAPHS="${ATTENTION_MAPS_MAX_GRAPHS:-16}"
ATTENTION_MAPS_DEVICE="${ATTENTION_MAPS_DEVICE:-${DEVICE}}"
SEED="${SEED:-12345}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PREPARE_FAST_CACHE="${PREPARE_FAST_CACHE:-0}"
FAST_CACHE_COMPRESSION="${FAST_CACHE_COMPRESSION:-lzf}"

if [[ "${LOSS_MODE}" == "physics-nll" || "${LOSS_MODE}" == "nll" ]]; then
  ERROR_PREDICTION=1
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
    CONFIG_NAME="hetero_reconstruction_mass${AUX_HEAD_TAG}_${MODEL_ARCHITECTURE}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${MASS_LOSS_MODE}_${TRAIN_EPOCHS}epoch"
  else
    CONFIG_NAME="hetero_reconstruction${AUX_HEAD_TAG}_${MODEL_ARCHITECTURE}_wf${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${LOSS_MODE}_${TRAIN_EPOCHS}epoch"
  fi
fi

CHECKPOINT="${CHECKPOINT_DIR}/${CONFIG_NAME}.pt"
LOG_PATH="${LOG_DIR}/${CONFIG_NAME}.log"
METRICS_PATH="${CHECKPOINT}.metrics.json"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}" "${CONFIG_DIR}"

if [[ "${PREPARE_FAST_CACHE}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  FAST_CACHE_PATH="${RUN_DIR}/cache/${RUN_NAME}.flat.h5"
  mkdir -p "$(dirname "${FAST_CACHE_PATH}")"
  {
    echo "stage=start prepare_hetero_fast_cache date=$(date)"
    cache_cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli convert-hetero-to-flat-cache
      --input "${GRAPH_INPUT}"
      -o "${FAST_CACHE_PATH}"
      --compression "${FAST_CACHE_COMPRESSION}")
    printf 'command:'
    printf ' %q' "${cache_cmd[@]}"
    printf '\n'
    "${cache_cmd[@]}"
    echo "stage=done prepare_hetero_fast_cache date=$(date)"
    echo "fast_cache=${FAST_CACHE_PATH}"
  } 2>&1 | tee "${LOG_DIR}/${CONFIG_NAME}.fast_cache.log"
  GRAPH_INPUT="${FAST_CACHE_PATH}"
elif [[ "${PREPARE_FAST_CACHE}" == "1" ]]; then
  FAST_CACHE_PATH="${RUN_DIR}/cache/${RUN_NAME}.flat.h5"
fi

cat > "${CONFIG_DIR}/train.env" <<EOF
RUN_ID=${RUN_ID}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GRAPH_INPUT=${GRAPH_INPUT}
GRAPH_INPUT_ORIGINAL=${GRAPH_INPUT_ORIGINAL}
PREPARE_FAST_CACHE=${PREPARE_FAST_CACHE}
FAST_CACHE_COMPRESSION=${FAST_CACHE_COMPRESSION}
CONFIG_NAME=${CONFIG_NAME}
HIDDEN_DIM=${HIDDEN_DIM}
LAYERS=${LAYERS}
DROPOUT=${DROPOUT}
MODEL_ARCHITECTURE=${MODEL_ARCHITECTURE}
ATTENTION_HEADS=${ATTENTION_HEADS}
READOUT_HEADS=${READOUT_HEADS}
WAVEFORM_ENCODER=${WAVEFORM_ENCODER}
WAVEFORM_EMBEDDING_DIM=${WAVEFORM_EMBEDDING_DIM}
WAVEFORM_LENGTH=${WAVEFORM_LENGTH}
WAVEFORM_TRANSFORMER_HEADS=${WAVEFORM_TRANSFORMER_HEADS}
WAVEFORM_TRANSFORMER_LAYERS=${WAVEFORM_TRANSFORMER_LAYERS}
WAVEFORM_TRANSFORMER_MAX_TOKENS=${WAVEFORM_TRANSFORMER_MAX_TOKENS}
WAVEFORM_TRANSFORMER_DOWNSAMPLE=${WAVEFORM_TRANSFORMER_DOWNSAMPLE}
TRAIN_EPOCHS=${TRAIN_EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
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
TRAIN_WORKERS=${TRAIN_WORKERS}
SPLIT_WORKERS=${SPLIT_WORKERS}
PREFETCH_FACTOR=${PREFETCH_FACTOR}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS}
VALIDATE_EVERY_N_EPOCHS=${VALIDATE_EVERY_N_EPOCHS}
MAX_VAL_GRAPHS=${MAX_VAL_GRAPHS}
MAX_GRAPHS=${MAX_GRAPHS}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}
EARLY_STOPPING_MIN_EPOCHS=${EARLY_STOPPING_MIN_EPOCHS}
CHECKPOINT_MILESTONES=${CHECKPOINT_MILESTONES}
HETERO_TRAINING_DATA_FORMAT=${HETERO_TRAINING_DATA_FORMAT}
HETERO_RELATIONS=${HETERO_RELATIONS}
DATALOADER_TIMEOUT_SEC=${DATALOADER_TIMEOUT_SEC}
DATA_WAIT_WARN_SEC=${DATA_WAIT_WARN_SEC}
PROFILE=${PROFILE}
PIN_MEMORY=${PIN_MEMORY}
AMP=${AMP}
TRAIN_LOADER_MEMORY_BUDGET_GIB=${TRAIN_LOADER_MEMORY_BUDGET_GIB}
TRAIN_LOADER_MEMORY_ESTIMATE_SAMPLES=${TRAIN_LOADER_MEMORY_ESTIMATE_SAMPLES}
DIAGNOSTICS=${DIAGNOSTICS}
DIAGNOSTIC_MIN_BIN_COUNT=${DIAGNOSTIC_MIN_BIN_COUNT}
FEATURE_IMPORTANCE=${FEATURE_IMPORTANCE}
FEATURE_IMPORTANCE_SPLIT=${FEATURE_IMPORTANCE_SPLIT}
FEATURE_IMPORTANCE_MAX_GRAPHS=${FEATURE_IMPORTANCE_MAX_GRAPHS}
FEATURE_IMPORTANCE_BATCH_SIZE=${FEATURE_IMPORTANCE_BATCH_SIZE}
FEATURE_IMPORTANCE_DEVICE=${FEATURE_IMPORTANCE_DEVICE}
ATTENTION_MAPS=${ATTENTION_MAPS}
ATTENTION_MAPS_SPLIT=${ATTENTION_MAPS_SPLIT}
ATTENTION_MAPS_MAX_GRAPHS=${ATTENTION_MAPS_MAX_GRAPHS}
ATTENTION_MAPS_DEVICE=${ATTENTION_MAPS_DEVICE}
SEED=${SEED}
EOF

cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli train-hetero
  --graphs "${GRAPH_INPUT}"
  -o "${CHECKPOINT}"
  --epochs "${TRAIN_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --layers "${LAYERS}"
  --dropout "${DROPOUT}"
  --model-architecture "${MODEL_ARCHITECTURE}"
  --attention-heads "${ATTENTION_HEADS}"
  --readout-heads "${READOUT_HEADS}"
  --waveform-encoder "${WAVEFORM_ENCODER}"
  --waveform-embedding-dim "${WAVEFORM_EMBEDDING_DIM}"
  --waveform-transformer-heads "${WAVEFORM_TRANSFORMER_HEADS}"
  --waveform-transformer-layers "${WAVEFORM_TRANSFORMER_LAYERS}"
  --waveform-transformer-max-tokens "${WAVEFORM_TRANSFORMER_MAX_TOKENS}"
  --waveform-transformer-downsample "${WAVEFORM_TRANSFORMER_DOWNSAMPLE}"
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
  --split-workers "${SPLIT_WORKERS}"
  --num-workers "${TRAIN_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --val-num-workers "${VAL_NUM_WORKERS}"
  --validate-every-n-epochs "${VALIDATE_EVERY_N_EPOCHS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-epochs "${EARLY_STOPPING_MIN_EPOCHS}"
  --checkpoint-milestones "${CHECKPOINT_MILESTONES}"
  --amp "${AMP}"
  --loader-memory-estimate-samples "${TRAIN_LOADER_MEMORY_ESTIMATE_SAMPLES}"
  --training-data-format "${HETERO_TRAINING_DATA_FORMAT}"
  --hetero-relations "${HETERO_RELATIONS}"
  --dataloader-timeout-sec "${DATALOADER_TIMEOUT_SEC}"
  --data-wait-warn-sec "${DATA_WAIT_WARN_SEC}"
  --diagnostic-energy-bin-width 0.1
  --diagnostic-min-bin-count "${DIAGNOSTIC_MIN_BIN_COUNT}")

if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
  cmd+=(--persistent-workers)
fi
if [[ "${PIN_MEMORY}" == "0" ]]; then
  cmd+=(--no-pin-memory)
fi
if [[ -n "${TRAIN_LOADER_MEMORY_BUDGET_GIB}" ]]; then
  cmd+=(--loader-memory-budget-gib "${TRAIN_LOADER_MEMORY_BUDGET_GIB}")
fi
if [[ -n "${MAX_GRAPHS}" ]]; then
  cmd+=(--max-graphs "${MAX_GRAPHS}")
fi
if [[ -n "${MAX_VAL_GRAPHS}" ]]; then
  cmd+=(--max-val-graphs "${MAX_VAL_GRAPHS}")
fi
if [[ "${PROFILE}" == "1" ]]; then
  cmd+=(--profile)
fi
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

ATTENTION_MAPS_DIR="${CHECKPOINT}.diagnostics/attention_maps/${ATTENTION_MAPS_SPLIT}"
ATTENTION_MAPS_SUMMARY="${ATTENTION_MAPS_DIR}/attention_maps.json"
if [[ "${ATTENTION_MAPS}" == "1" ]]; then
  if [[ "${MODEL_ARCHITECTURE}" != "hetero_attention" ]]; then
    {
      echo "stage=skip attention_maps date=$(date)"
      echo "reason=model_architecture_${MODEL_ARCHITECTURE}_has_no_relation_attention"
    } 2>&1 | tee -a "${LOG_PATH}"
    ATTENTION_MAPS_SUMMARY=""
  else
    {
      echo "stage=start attention_maps date=$(date)"
      attention_cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli attention-maps
        --graphs "${GRAPH_INPUT}"
        --checkpoint "${CHECKPOINT}"
        -o "${ATTENTION_MAPS_DIR}"
        --split "${ATTENTION_MAPS_SPLIT}"
        --max-graphs "${ATTENTION_MAPS_MAX_GRAPHS}"
        --device "${ATTENTION_MAPS_DEVICE}"
        --seed "${SEED}")
      printf 'command:'
      printf ' %q' "${attention_cmd[@]}"
      printf '\n'
      "${attention_cmd[@]}"
      if [[ ! -s "${ATTENTION_MAPS_SUMMARY}" ]]; then
        echo "ERROR: attention maps finished but summary was not written: ${ATTENTION_MAPS_SUMMARY}" >&2
        exit 1
      fi
      echo "stage=done attention_maps date=$(date)"
      echo "attention_maps=${ATTENTION_MAPS_SUMMARY}"
    } 2>&1 | tee -a "${LOG_PATH}"
  fi
else
  ATTENTION_MAPS_SUMMARY=""
fi

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
  checkpoints/${CONFIG_NAME}.pt.diagnostics/feature_importance/${FEATURE_IMPORTANCE_SPLIT}/feature_group_importance.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/attention_maps/${ATTENTION_MAPS_SPLIT}/attention_maps.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/attention_maps/${ATTENTION_MAPS_SPLIT}/attention_maps.npz
  summaries/metrics_summary.csv

Graph input:
  ${GRAPH_INPUT}

Architecture:
  ${MODEL_ARCHITECTURE}
  hetero_attention uses relation-specific multi-head attention and type-wise attention readout.
  It does not use PyG HGTConv/HGSampling; each TALE event graph is trained as a full event graph.

Not yet confirmed:
  large-scale server training completion
EOF

echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${METRICS_PATH}"
echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
echo "feature_importance=${FEATURE_IMPORTANCE_SUMMARY}"
echo "attention_maps=${ATTENTION_MAPS_SUMMARY}"
echo "summary_csv=${SUMMARY_DIR}/metrics_summary.csv"
echo "log_path=${LOG_PATH}"
date
