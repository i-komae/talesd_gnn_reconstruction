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
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-minimal_hetero}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
READOUT_HEADS="${READOUT_HEADS:-4}"
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"
WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
WAVEFORM_LENGTH="${WAVEFORM_LENGTH:-}"
WAVEFORM_TRANSFORMER_HEADS="${WAVEFORM_TRANSFORMER_HEADS:-4}"
WAVEFORM_TRANSFORMER_LAYERS="${WAVEFORM_TRANSFORMER_LAYERS:-1}"
WAVEFORM_TRANSFORMER_MAX_TOKENS="${WAVEFORM_TRANSFORMER_MAX_TOKENS:-128}"
WAVEFORM_TRANSFORMER_DOWNSAMPLE="${WAVEFORM_TRANSFORMER_DOWNSAMPLE:-adaptive_avg}"
USE_PULSE_PARENT_WAVEFORM="${USE_PULSE_PARENT_WAVEFORM:-1}"
USE_PULSE_BOUNDS="${USE_PULSE_BOUNDS:-1}"
PULSE_WAVEFORM_ENCODER="${PULSE_WAVEFORM_ENCODER:-crop_cnn}"
USE_RELATIVE_POSITIONS="${USE_RELATIVE_POSITIONS:-1}"
DETECTOR_READOUT_MASK="${DETECTOR_READOUT_MASK:-signal}"
PULSE_READOUT_MASK="${PULSE_READOUT_MASK:-all}"
SPEED_BENCHMARK="${SPEED_BENCHMARK:-0}"
PREPARE_FAST_CACHE_WAS_SET=0
if [[ -n "${PREPARE_FAST_CACHE:-}" ]]; then
  PREPARE_FAST_CACHE_WAS_SET=1
fi
if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
  MAX_GRAPHS="${MAX_GRAPHS:-4096}"
  MAX_VAL_GRAPHS="${MAX_VAL_GRAPHS:-512}"
  VALIDATE_EVERY_N_EPOCHS="${VALIDATE_EVERY_N_EPOCHS:-1}"
  DIAGNOSTICS="${DIAGNOSTICS:-0}"
  ATTENTION_MAPS="${ATTENTION_MAPS:-0}"
  FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-0}"
  PROFILE="${PROFILE:-1}"
  TALESD_GNN_PROFILE="${TALESD_GNN_PROFILE:-1}"
  CHECKPOINT_MILESTONES="${CHECKPOINT_MILESTONES:-}"
  MILESTONE_EVAL_EPOCHS="${MILESTONE_EVAL_EPOCHS:-}"
  HETERO_TRAINING_DATA_FORMAT="${HETERO_TRAINING_DATA_FORMAT:-fast_tensor}"
  PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
  TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
  PIN_MEMORY="${PIN_MEMORY:-0}"
  WAVEFORM_TRANSFORMER_MAX_TOKENS="${WAVEFORM_TRANSFORMER_MAX_TOKENS:-128}"
else
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
fi
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
SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE:-explicit}"
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
CHECKPOINT_MILESTONE_FULL_EVAL="${CHECKPOINT_MILESTONE_FULL_EVAL:-0}"
ALLOW_TRAIN_LOSS_CHECKPOINT="${ALLOW_TRAIN_LOSS_CHECKPOINT:-0}"
if [[ -z "${MILESTONE_EVAL_EPOCHS+x}" ]]; then
  if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
    MILESTONE_EVAL_EPOCHS=""
  else
    MILESTONE_EVAL_EPOCHS="8,16,32,64"
  fi
fi
if [[ -z "${CHECKPOINT_MILESTONES+x}" ]]; then
  CHECKPOINT_MILESTONES="${MILESTONE_EVAL_EPOCHS}"
fi
MILESTONE_EVAL_SPLIT="${MILESTONE_EVAL_SPLIT:-validation test}"
MILESTONE_EVAL_MAX_GRAPHS="${MILESTONE_EVAL_MAX_GRAPHS:-0}"
MILESTONE_EVAL_CURRENT_MODEL="${MILESTONE_EVAL_CURRENT_MODEL:-0}"
MILESTONE_EVAL_BEST_MODEL="${MILESTONE_EVAL_BEST_MODEL:-1}"
MILESTONE_EVAL_DIAGNOSTICS="${MILESTONE_EVAL_DIAGNOSTICS:-1}"
HETERO_TRAINING_DATA_FORMAT="${HETERO_TRAINING_DATA_FORMAT:-fast_tensor}"
FINAL_EVAL_DATA_FORMAT="${FINAL_EVAL_DATA_FORMAT:-${HETERO_TRAINING_DATA_FORMAT}}"
CORE_TARGET_MODE="${CORE_TARGET_MODE:-signal_bary_relative}"
COORDINATE_FEATURE_MODE="${COORDINATE_FEATURE_MODE:-relative_only}"
HETERO_RELATIONS="${HETERO_RELATIONS:-all}"
HETERO_RELATION_PRESET="${HETERO_RELATION_PRESET:-minimal}"
DATALOADER_TIMEOUT_SEC="${DATALOADER_TIMEOUT_SEC:-120}"
DATA_WAIT_WARN_SEC="${DATA_WAIT_WARN_SEC:-30}"
TRAIN_PROGRESS_INTERVAL_SEC="${TALESD_GNN_TRAIN_PROGRESS_INTERVAL_SEC:-${TRAIN_PROGRESS_INTERVAL_SEC:-60}}"
VALIDATION_PROGRESS_INTERVAL_SEC="${TALESD_GNN_VALIDATION_PROGRESS_INTERVAL_SEC:-${VALIDATION_PROGRESS_INTERVAL_SEC:-60}}"
PREDICT_PROGRESS_INTERVAL_SEC="${TALESD_GNN_PREDICT_PROGRESS_INTERVAL_SEC:-${PREDICT_PROGRESS_INTERVAL_SEC:-60}}"
SCALER_PROGRESS_INTERVAL_SEC="${TALESD_GNN_SCALER_PROGRESS_INTERVAL_SEC:-${SCALER_PROGRESS_INTERVAL_SEC:-60}}"
FLAT_CACHE_PROGRESS_INTERVAL_SEC="${HETERO_FLAT_CACHE_PROGRESS_INTERVAL_SEC:-${FLAT_CACHE_PROGRESS_INTERVAL_SEC:-60}}"
export TALESD_GNN_TRAIN_PROGRESS_INTERVAL_SEC="${TRAIN_PROGRESS_INTERVAL_SEC}"
export TALESD_GNN_VALIDATION_PROGRESS_INTERVAL_SEC="${VALIDATION_PROGRESS_INTERVAL_SEC}"
export TALESD_GNN_PREDICT_PROGRESS_INTERVAL_SEC="${PREDICT_PROGRESS_INTERVAL_SEC}"
export TALESD_GNN_SCALER_PROGRESS_INTERVAL_SEC="${SCALER_PROGRESS_INTERVAL_SEC}"
export HETERO_FLAT_CACHE_PROGRESS_INTERVAL_SEC="${FLAT_CACHE_PROGRESS_INTERVAL_SEC}"
export TALESD_GNN_DATALOADER_TIMEOUT_SEC="${DATALOADER_TIMEOUT_SEC}"
export TALESD_GNN_DATA_WAIT_WARN_SEC="${DATA_WAIT_WARN_SEC}"
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
if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  DIAGNOSTICS="${DIAGNOSTICS:-0}"
else
  DIAGNOSTICS="${DIAGNOSTICS:-1}"
fi
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-0}"
FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation test}"
FEATURE_IMPORTANCE_MAX_GRAPHS="${FEATURE_IMPORTANCE_MAX_GRAPHS:-50000}"
FEATURE_IMPORTANCE_BATCH_SIZE="${FEATURE_IMPORTANCE_BATCH_SIZE:-256}"
FEATURE_IMPORTANCE_DEVICE="${FEATURE_IMPORTANCE_DEVICE:-${DEVICE}}"
if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  ATTENTION_MAPS="${ATTENTION_MAPS:-0}"
else
  ATTENTION_MAPS="${ATTENTION_MAPS:-1}"
fi
ATTENTION_MAPS_SPLIT="${ATTENTION_MAPS_SPLIT:-validation test}"
ATTENTION_MAPS_MAX_GRAPHS="${ATTENTION_MAPS_MAX_GRAPHS:-16}"
ATTENTION_MAPS_DEVICE="${ATTENTION_MAPS_DEVICE:-${DEVICE}}"
SEED="${SEED:-12345}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PREPARE_FAST_CACHE="${PREPARE_FAST_CACHE:-0}"
FAST_CACHE_COMPRESSION="${FAST_CACHE_COMPRESSION:-none}"
FAST_CACHE_MODE="${FAST_CACHE_MODE:-training}"
FAST_CACHE_VERIFY_SAMPLES="${FAST_CACHE_VERIFY_SAMPLES:-5}"
SCALER_CACHE="${SCALER_CACHE:-${RUN_DIR}/cache/${RUN_NAME}.scalers.json}"
REUSE_SCALER_CACHE="${REUSE_SCALER_CACHE:-1}"

detect_hetero_graph_format() {
  "${PYTHON_BIN}" -c 'import sys, pathlib, h5py
p = pathlib.Path(sys.argv[1]).expanduser()
if p.is_dir():
    files = sorted(p.rglob("*.h5"))
    if not files:
        print("none")
        raise SystemExit(0)
    p = files[0]
with h5py.File(p, "r") as handle:
    fmt = str(handle.attrs.get("format", ""))
print("flat_hdf5" if fmt == "talesd_gnn_hetero_graphs_flat" else ("grouped_hdf5" if fmt == "talesd_gnn_hetero_graphs" else (fmt or "unknown")))' "$1" 2>/dev/null || echo "unknown"
}
GRAPH_INPUT_FORMAT="$(detect_hetero_graph_format "${GRAPH_INPUT}")"

if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
  echo "hetero_speed_benchmark enabled=1 train_epochs=${TRAIN_EPOCHS} max_graphs=${MAX_GRAPHS} max_val_graphs=${MAX_VAL_GRAPHS} profile=${PROFILE} checkpoint_milestones='${CHECKPOINT_MILESTONES}' milestone_eval_epochs='${MILESTONE_EVAL_EPOCHS}' prepare_fast_cache=${PREPARE_FAST_CACHE}"
fi
if [[ "${WAVEFORM_ENCODER}" == "transformer" && "${BATCH_SIZE}" =~ ^[0-9]+$ && "${BATCH_SIZE}" -lt 16 ]]; then
  echo "WARNING: BATCH_SIZE is small for optimized transformer path; consider BATCH_SIZE=32 or 64" >&2
fi
if [[ "${GRAPH_INPUT_FORMAT}" == "grouped_hdf5" && "${PREPARE_FAST_CACHE}" != "1" ]]; then
  echo "hetero_graph_input format=grouped_hdf5 prepare_fast_cache=0 action=train_grouped_fast_tensor"
  echo "WARNING: grouped HDF5 + fast_tensor is the current standard path; monitor hetero_epoch_profile data_wait_s before enabling post-hoc flat cache." >&2
elif [[ "${GRAPH_INPUT_FORMAT}" == "flat_hdf5" ]]; then
  echo "hetero_graph_input format=flat_hdf5 prepare_fast_cache=${PREPARE_FAST_CACHE} action=use_existing_flat_cache"
else
  echo "WARNING: could not confirm hetero HDF5 format for GRAPH_INPUT=${GRAPH_INPUT}; use PREPARE_FAST_CACHE=0 unless this is an explicit converter run" >&2
fi
if [[ "${PREPARE_FAST_CACHE}" == "1" ]]; then
  echo "WARNING: PREPARE_FAST_CACHE=1 performs grouped-to-flat conversion before training. This may be slow. Prefer directly exported flat HDF5 or PREPARE_FAST_CACHE=0." >&2
fi
cat <<'EOF'
recommended_speed_benchmark:
  SPEED_BENCHMARK=1 WAVEFORM_ENCODER=transformer PREPARE_FAST_CACHE=0 DEVICE=cuda scripts/submit_server_hetero_reco_mass_quality_training.sh
recommended_production_start:
  WAVEFORM_ENCODER=cnn-gru MODEL_ARCHITECTURE=minimal_hetero PULSE_WAVEFORM_ENCODER=crop_cnn USE_PULSE_PARENT_WAVEFORM=1 USE_PULSE_BOUNDS=1 USE_RELATIVE_POSITIONS=1 DETECTOR_READOUT_MASK=signal HETERO_RELATION_PRESET=minimal BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=4 AMP=fp16 PREPARE_FAST_CACHE=0 HETERO_TRAINING_DATA_FORMAT=fast_tensor FINAL_EVAL_DATA_FORMAT=fast_tensor PERSISTENT_WORKERS=1 PREFETCH_FACTOR=1 TRAIN_WORKERS=4 PIN_MEMORY=0 FEATURE_IMPORTANCE=0 ATTENTION_MAPS=0 DIAGNOSTICS=0 scripts/submit_server_hetero_reco_mass_quality_training.sh
recommended_waveform_ablation:
  WAVEFORM_ENCODER=cnn-gru MODEL_ARCHITECTURE=minimal_hetero PULSE_WAVEFORM_ENCODER=crop_cnn USE_PULSE_PARENT_WAVEFORM=1 USE_PULSE_BOUNDS=1 USE_RELATIVE_POSITIONS=1 DETECTOR_READOUT_MASK=signal HETERO_RELATION_PRESET=minimal scripts/submit_server_hetero_reco_mass_quality_training.sh
EOF

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

if [[ "${PREPARE_FAST_CACHE}" == "1" && "${GRAPH_INPUT_FORMAT}" == "flat_hdf5" ]]; then
  FAST_CACHE_PATH="${GRAPH_INPUT}"
  echo "stage=skip prepare_hetero_fast_cache reason=input_already_flat path=${GRAPH_INPUT}"
elif [[ "${PREPARE_FAST_CACHE}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  FAST_CACHE_PATH="${RUN_DIR}/cache/${RUN_NAME}.flat.h5"
  mkdir -p "$(dirname "${FAST_CACHE_PATH}")"
  {
    echo "stage=start prepare_hetero_fast_cache date=$(date)"
    cache_cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli convert-hetero-to-flat-cache
      --input "${GRAPH_INPUT}"
      -o "${FAST_CACHE_PATH}"
      --compression "${FAST_CACHE_COMPRESSION}"
      --cache-mode "${FAST_CACHE_MODE}"
      --core-anchor-mode "${CORE_TARGET_MODE}"
      --verify-samples "${FAST_CACHE_VERIFY_SAMPLES}"
      --progress-interval-sec "${FLAT_CACHE_PROGRESS_INTERVAL_SEC}")
    if [[ -n "${MAX_GRAPHS:-}" && "${MAX_GRAPHS}" != "0" ]]; then
      cache_cmd+=(--max-graphs "${MAX_GRAPHS}")
    fi
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

if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
  echo "hetero_speed_benchmark prepare_fast_cache=${PREPARE_FAST_CACHE} graph_input_original=${GRAPH_INPUT_ORIGINAL} graph_input_effective=${GRAPH_INPUT}"
else
  echo "hetero_training_input prepare_fast_cache=${PREPARE_FAST_CACHE} graph_input_original=${GRAPH_INPUT_ORIGINAL} graph_input_effective=${GRAPH_INPUT}"
fi
mkdir -p "$(dirname "${SCALER_CACHE}")"

cat > "${CONFIG_DIR}/train.env" <<EOF
RUN_ID=${RUN_ID}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GRAPH_INPUT=${GRAPH_INPUT}
GRAPH_INPUT_ORIGINAL=${GRAPH_INPUT_ORIGINAL}
PREPARE_FAST_CACHE=${PREPARE_FAST_CACHE}
FAST_CACHE_COMPRESSION=${FAST_CACHE_COMPRESSION}
FAST_CACHE_MODE=${FAST_CACHE_MODE}
FAST_CACHE_VERIFY_SAMPLES=${FAST_CACHE_VERIFY_SAMPLES}
GRAPH_INPUT_FORMAT=${GRAPH_INPUT_FORMAT}
SCALER_CACHE=${SCALER_CACHE}
REUSE_SCALER_CACHE=${REUSE_SCALER_CACHE}
SPEED_BENCHMARK=${SPEED_BENCHMARK}
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
USE_PULSE_PARENT_WAVEFORM=${USE_PULSE_PARENT_WAVEFORM}
USE_PULSE_BOUNDS=${USE_PULSE_BOUNDS}
PULSE_WAVEFORM_ENCODER=${PULSE_WAVEFORM_ENCODER}
USE_RELATIVE_POSITIONS=${USE_RELATIVE_POSITIONS}
DETECTOR_READOUT_MASK=${DETECTOR_READOUT_MASK}
PULSE_READOUT_MASK=${PULSE_READOUT_MASK}
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
CORE_TARGET_MODE=${CORE_TARGET_MODE}
COORDINATE_FEATURE_MODE=${COORDINATE_FEATURE_MODE}
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
SOURCE_FRACTION_MODE=${SOURCE_FRACTION_MODE}
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
CHECKPOINT_MILESTONE_FULL_EVAL=${CHECKPOINT_MILESTONE_FULL_EVAL}
ALLOW_TRAIN_LOSS_CHECKPOINT=${ALLOW_TRAIN_LOSS_CHECKPOINT}
MILESTONE_EVAL_EPOCHS=${MILESTONE_EVAL_EPOCHS}
MILESTONE_EVAL_SPLIT=${MILESTONE_EVAL_SPLIT}
MILESTONE_EVAL_MAX_GRAPHS=${MILESTONE_EVAL_MAX_GRAPHS}
MILESTONE_EVAL_CURRENT_MODEL=${MILESTONE_EVAL_CURRENT_MODEL}
MILESTONE_EVAL_BEST_MODEL=${MILESTONE_EVAL_BEST_MODEL}
MILESTONE_EVAL_DIAGNOSTICS=${MILESTONE_EVAL_DIAGNOSTICS}
HETERO_TRAINING_DATA_FORMAT=${HETERO_TRAINING_DATA_FORMAT}
FINAL_EVAL_DATA_FORMAT=${FINAL_EVAL_DATA_FORMAT}
HETERO_RELATIONS=${HETERO_RELATIONS}
HETERO_RELATION_PRESET=${HETERO_RELATION_PRESET}
DATALOADER_TIMEOUT_SEC=${DATALOADER_TIMEOUT_SEC}
DATA_WAIT_WARN_SEC=${DATA_WAIT_WARN_SEC}
TRAIN_PROGRESS_INTERVAL_SEC=${TRAIN_PROGRESS_INTERVAL_SEC}
VALIDATION_PROGRESS_INTERVAL_SEC=${VALIDATION_PROGRESS_INTERVAL_SEC}
PREDICT_PROGRESS_INTERVAL_SEC=${PREDICT_PROGRESS_INTERVAL_SEC}
SCALER_PROGRESS_INTERVAL_SEC=${SCALER_PROGRESS_INTERVAL_SEC}
FLAT_CACHE_PROGRESS_INTERVAL_SEC=${FLAT_CACHE_PROGRESS_INTERVAL_SEC}
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

echo "hetero_training_script_config speed_benchmark=${SPEED_BENCHMARK} waveform_encoder=${WAVEFORM_ENCODER} pulse_waveform_encoder=${PULSE_WAVEFORM_ENCODER} use_pulse_parent_waveform=${USE_PULSE_PARENT_WAVEFORM} use_pulse_bounds=${USE_PULSE_BOUNDS} use_relative_positions=${USE_RELATIVE_POSITIONS} detector_readout_mask=${DETECTOR_READOUT_MASK} pulse_readout_mask=${PULSE_READOUT_MASK} relation_preset=${HETERO_RELATION_PRESET:-custom} batch_size=${BATCH_SIZE} gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} prepare_fast_cache=${PREPARE_FAST_CACHE} training_data_format=${HETERO_TRAINING_DATA_FORMAT} final_eval_data_format=${FINAL_EVAL_DATA_FORMAT} core_target_mode=${CORE_TARGET_MODE} coordinate_feature_mode=${COORDINATE_FEATURE_MODE} checkpoint_milestones='${CHECKPOINT_MILESTONES}' checkpoint_milestone_full_eval=${CHECKPOINT_MILESTONE_FULL_EVAL} milestone_eval_epochs='${MILESTONE_EVAL_EPOCHS}' milestone_eval_split='${MILESTONE_EVAL_SPLIT}' milestone_eval_max_graphs=${MILESTONE_EVAL_MAX_GRAPHS} milestone_eval_current_model=${MILESTONE_EVAL_CURRENT_MODEL} milestone_eval_best_model=${MILESTONE_EVAL_BEST_MODEL} milestone_eval_diagnostics=${MILESTONE_EVAL_DIAGNOSTICS} diagnostics=${DIAGNOSTICS} attention_maps=${ATTENTION_MAPS} feature_importance=${FEATURE_IMPORTANCE} profile=${PROFILE} pin_memory=${PIN_MEMORY} prefetch_factor=${PREFETCH_FACTOR} persistent_workers=${PERSISTENT_WORKERS} train_workers=${TRAIN_WORKERS} train_progress_interval_sec=${TRAIN_PROGRESS_INTERVAL_SEC} validation_progress_interval_sec=${VALIDATION_PROGRESS_INTERVAL_SEC} predict_progress_interval_sec=${PREDICT_PROGRESS_INTERVAL_SEC} scaler_progress_interval_sec=${SCALER_PROGRESS_INTERVAL_SEC} flat_cache_progress_interval_sec=${FLAT_CACHE_PROGRESS_INTERVAL_SEC} dataloader_timeout_sec=${DATALOADER_TIMEOUT_SEC} data_wait_warn_sec=${DATA_WAIT_WARN_SEC}"
echo "hetero_milestone_eval_config enabled=$([[ -n "${MILESTONE_EVAL_EPOCHS}" ]] && echo 1 || echo 0) epochs=${MILESTONE_EVAL_EPOCHS} splits=${MILESTONE_EVAL_SPLIT} current_model=${MILESTONE_EVAL_CURRENT_MODEL} best_model=${MILESTONE_EVAL_BEST_MODEL} max_graphs=${MILESTONE_EVAL_MAX_GRAPHS} diagnostics=${MILESTONE_EVAL_DIAGNOSTICS} attention_maps=0 feature_importance=0"
echo "hetero_logging_config train_progress_interval_sec=${TRAIN_PROGRESS_INTERVAL_SEC} validation_progress_interval_sec=${VALIDATION_PROGRESS_INTERVAL_SEC} predict_progress_interval_sec=${PREDICT_PROGRESS_INTERVAL_SEC} scaler_progress_interval_sec=${SCALER_PROGRESS_INTERVAL_SEC} flat_cache_progress_interval_sec=${FLAT_CACHE_PROGRESS_INTERVAL_SEC} dataloader_timeout_sec=${DATALOADER_TIMEOUT_SEC} data_wait_warn_sec=${DATA_WAIT_WARN_SEC} expected_max_silent_sec=${DATALOADER_TIMEOUT_SEC}"
echo "hetero_scaler_cache default_scope=run_local reuse_across_runs=0 path=${SCALER_CACHE}"
echo "hetero_scaler_cache recommendation='set SCALER_CACHE=/path/to/shared/hetero_scalers.json for cross-run reuse'"
echo "hetero_postprocess_config diagnostics=${DIAGNOSTICS} attention_maps=${ATTENTION_MAPS} feature_importance=${FEATURE_IMPORTANCE}"

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
  --pulse-waveform-encoder "${PULSE_WAVEFORM_ENCODER}"
  --detector-readout-mask "${DETECTOR_READOUT_MASK}"
  --pulse-readout-mask "${PULSE_READOUT_MASK}"
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
  --source-fraction-mode "${SOURCE_FRACTION_MODE}"
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
  --milestone-eval-epochs "${MILESTONE_EVAL_EPOCHS}"
  --milestone-eval-split "${MILESTONE_EVAL_SPLIT}"
  --milestone-eval-max-graphs "${MILESTONE_EVAL_MAX_GRAPHS}"
  --amp "${AMP}"
  --loader-memory-estimate-samples "${TRAIN_LOADER_MEMORY_ESTIMATE_SAMPLES}"
  --training-data-format "${HETERO_TRAINING_DATA_FORMAT}"
  --final-eval-data-format "${FINAL_EVAL_DATA_FORMAT}"
  --core-target-mode "${CORE_TARGET_MODE}"
  --coordinate-feature-mode "${COORDINATE_FEATURE_MODE}"
  --scaler-cache "${SCALER_CACHE}"
  --hetero-relations "${HETERO_RELATIONS}"
  --hetero-relation-preset "${HETERO_RELATION_PRESET:-all}"
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
if [[ "${USE_PULSE_PARENT_WAVEFORM}" == "1" ]]; then
  cmd+=(--use-pulse-parent-waveform)
else
  cmd+=(--no-use-pulse-parent-waveform)
fi
if [[ "${USE_PULSE_BOUNDS}" == "1" ]]; then
  cmd+=(--use-pulse-bounds)
else
  cmd+=(--no-use-pulse-bounds)
fi
if [[ "${USE_RELATIVE_POSITIONS}" == "1" ]]; then
  cmd+=(--use-relative-positions)
else
  cmd+=(--no-use-relative-positions)
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
if [[ "${CHECKPOINT_MILESTONE_FULL_EVAL}" == "1" ]]; then
  cmd+=(--checkpoint-milestone-full-eval)
fi
if [[ "${ALLOW_TRAIN_LOSS_CHECKPOINT}" == "1" ]]; then
  cmd+=(--allow-train-loss-checkpoint)
fi
if [[ "${MILESTONE_EVAL_CURRENT_MODEL}" == "1" ]]; then
  cmd+=(--milestone-eval-current-model)
else
  cmd+=(--no-milestone-eval-current-model)
fi
if [[ "${MILESTONE_EVAL_BEST_MODEL}" == "1" ]]; then
  cmd+=(--milestone-eval-best-model)
else
  cmd+=(--no-milestone-eval-best-model)
fi
if [[ "${MILESTONE_EVAL_DIAGNOSTICS}" == "1" ]]; then
  cmd+=(--milestone-eval-diagnostics)
else
  cmd+=(--no-milestone-eval-diagnostics)
fi
if [[ "${REUSE_SCALER_CACHE}" == "1" ]]; then
  cmd+=(--reuse-scaler-cache)
else
  cmd+=(--no-reuse-scaler-cache)
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
  if [[ "${DIAGNOSTICS}" == "1" ]]; then
    echo "stage=done diagnostics date=$(date) diagnostics_dir=${CHECKPOINT}.diagnostics"
  else
    echo "stage=skip diagnostics enabled=0 date=$(date)"
  fi
  date
} 2>&1 | tee "${LOG_PATH}"

{
  echo "stage=start metrics_summary date=$(date)"
  "${PYTHON_BIN}" scripts/summarize_metrics.py "${METRICS_PATH}" -o "${SUMMARY_DIR}/metrics_summary.csv"
  echo "stage=done metrics_summary date=$(date) summary_csv=${SUMMARY_DIR}/metrics_summary.csv"
} 2>&1 | tee -a "${LOG_PATH}"
FEATURE_IMPORTANCE_SUMMARIES=()
if [[ "${FEATURE_IMPORTANCE}" == "1" ]]; then
  for feature_split in ${FEATURE_IMPORTANCE_SPLIT}; do
    FEATURE_IMPORTANCE_DIR="${CHECKPOINT}.diagnostics/feature_importance/${feature_split}"
    FEATURE_IMPORTANCE_SUMMARY="${FEATURE_IMPORTANCE_DIR}/feature_group_importance.json"
    {
      echo "stage=start feature_importance split=${feature_split} date=$(date)"
      feature_cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli feature-importance
        --graphs "${GRAPH_INPUT}"
        --checkpoint "${CHECKPOINT}"
        -o "${FEATURE_IMPORTANCE_DIR}"
        --split "${feature_split}"
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
      echo "stage=done feature_importance split=${feature_split} date=$(date)"
      echo "feature_importance=${FEATURE_IMPORTANCE_SUMMARY}"
    } 2>&1 | tee -a "${LOG_PATH}"
    FEATURE_IMPORTANCE_SUMMARIES+=("${FEATURE_IMPORTANCE_SUMMARY}")
  done
else
  {
    echo "stage=skip feature_importance enabled=0 date=$(date)"
  } 2>&1 | tee -a "${LOG_PATH}"
fi

FEATURE_IMPORTANCE_SUMMARY="${FEATURE_IMPORTANCE_SUMMARIES[*]:-}"
ATTENTION_MAPS_SUMMARIES=()
if [[ "${ATTENTION_MAPS}" == "1" ]]; then
  if [[ "${MODEL_ARCHITECTURE}" != "hetero_attention" ]]; then
    {
      echo "stage=skip attention_maps date=$(date)"
      echo "reason=model_architecture_${MODEL_ARCHITECTURE}_has_no_relation_attention"
    } 2>&1 | tee -a "${LOG_PATH}"
  else
    for attention_split in ${ATTENTION_MAPS_SPLIT}; do
      ATTENTION_MAPS_DIR="${CHECKPOINT}.diagnostics/attention_maps/${attention_split}"
      ATTENTION_MAPS_SUMMARY="${ATTENTION_MAPS_DIR}/attention_maps.json"
      {
        echo "stage=start attention_maps split=${attention_split} date=$(date)"
        attention_cmd=("${PYTHON_BIN}" -m talesd_gnn_reconstruction.cli attention-maps
          --graphs "${GRAPH_INPUT}"
          --checkpoint "${CHECKPOINT}"
          -o "${ATTENTION_MAPS_DIR}"
          --split "${attention_split}"
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
        echo "stage=done attention_maps split=${attention_split} date=$(date)"
        echo "attention_maps=${ATTENTION_MAPS_SUMMARY}"
      } 2>&1 | tee -a "${LOG_PATH}"
      ATTENTION_MAPS_SUMMARIES+=("${ATTENTION_MAPS_SUMMARY}")
    done
  fi
else
  {
    echo "stage=skip attention_maps enabled=0 date=$(date)"
  } 2>&1 | tee -a "${LOG_PATH}"
fi
ATTENTION_MAPS_SUMMARY="${ATTENTION_MAPS_SUMMARIES[*]:-}"

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
  checkpoints/${CONFIG_NAME}.pt.diagnostics/feature_importance/<split>/feature_group_importance.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/attention_maps/<split>/attention_maps.json
  checkpoints/${CONFIG_NAME}.pt.diagnostics/attention_maps/<split>/attention_maps.npz
  checkpoints/${CONFIG_NAME}.pt.diagnostics/attention_maps/<split>/attention_maps.pdf
  summaries/metrics_summary.csv

Postprocess splits:
  feature_importance: ${FEATURE_IMPORTANCE_SPLIT}
  attention_maps: ${ATTENTION_MAPS_SPLIT}

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
