#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage:
  GRAPH_INPUT=/path/to/hetero_graph.h5 scripts/submit_server_hetero_training.sh

Submit hetero TALE-SD training from dstio.tale.graph HDF5 graphs.
GRAPH_INPUT must point to hetero HDF5 made by talesd-gnn export-hetero.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${GRAPH_INPUT:-}" ]]; then
  usage >&2
  echo "GRAPH_INPUT is required and must be a hetero HDF5 graph file or shard directory." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SPEED_BENCHMARK="${SPEED_BENCHMARK:-0}"
PREPARE_FAST_CACHE_WAS_SET=0
if [[ -n "${PREPARE_FAST_CACHE:-}" ]]; then
  PREPARE_FAST_CACHE_WAS_SET=1
fi
if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
else
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
fi
PARTITION_FOR_NAME="${PARTITION:-v100-al9_long}"
RESOURCE_TAG="${RESOURCE_TAG:-${PARTITION_FOR_NAME%%-*}}"

export RUN_ID
export SPEED_BENCHMARK
export RESOURCE_TAG
export TRAINING_BACKEND="hetero"
export TRAINING_TASK="reconstruction"
export GRAPH_INPUT
export RUN_NAME="${RUN_NAME:-server_hetero_reco_mass_${RESOURCE_TAG}_${TRAIN_EPOCHS}epoch_${RUN_ID}}"

export PARTITION="${PARTITION:-v100-al9_long}"
export GPUS="${GPUS:-1}"
export CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
export MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-192}"

export MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-hetero_attention}"
export ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
export READOUT_HEADS="${READOUT_HEADS:-4}"
export HIDDEN_DIM="${HIDDEN_DIM:-192}"
export LAYERS="${LAYERS:-5}"
export DROPOUT="${DROPOUT:-0.08}"
export WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"
export WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
export WAVEFORM_LENGTH="${WAVEFORM_LENGTH:-}"
export WAVEFORM_TRANSFORMER_HEADS="${WAVEFORM_TRANSFORMER_HEADS:-4}"
export WAVEFORM_TRANSFORMER_LAYERS="${WAVEFORM_TRANSFORMER_LAYERS:-1}"
export WAVEFORM_TRANSFORMER_MAX_TOKENS="${WAVEFORM_TRANSFORMER_MAX_TOKENS:-128}"
export WAVEFORM_TRANSFORMER_DOWNSAMPLE="${WAVEFORM_TRANSFORMER_DOWNSAMPLE:-adaptive_avg}"

export TRAIN_EPOCHS
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    export BATCH_SIZE=32
  else
    export BATCH_SIZE=128
  fi
else
  export BATCH_SIZE
fi
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    export GRADIENT_ACCUMULATION_STEPS=4
  else
    export GRADIENT_ACCUMULATION_STEPS=1
  fi
else
  export GRADIENT_ACCUMULATION_STEPS
fi
if [[ -z "${PIN_MEMORY:-}" && "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  export PIN_MEMORY=0
fi
if [[ -z "${PREFETCH_FACTOR:-}" && "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  export PREFETCH_FACTOR=1
fi
if [[ -z "${TRAIN_WORKERS:-}" && "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  export TRAIN_WORKERS=4
fi
if [[ "${PREPARE_FAST_CACHE_WAS_SET}" == "0" && ( "${SPEED_BENCHMARK}" == "1" || "${WAVEFORM_ENCODER}" == "transformer" ) ]]; then
  export PREPARE_FAST_CACHE=1
else
  export PREPARE_FAST_CACHE="${PREPARE_FAST_CACHE:-0}"
fi
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
export VALIDATE_EVERY_N_EPOCHS="${VALIDATE_EVERY_N_EPOCHS:-1}"
export HETERO_TRAINING_DATA_FORMAT="${HETERO_TRAINING_DATA_FORMAT:-fast_tensor}"
export FINAL_EVAL_DATA_FORMAT="${FINAL_EVAL_DATA_FORMAT:-${HETERO_TRAINING_DATA_FORMAT}}"
export HETERO_RELATIONS="${HETERO_RELATIONS:-all}"
export DATALOADER_TIMEOUT_SEC="${DATALOADER_TIMEOUT_SEC:-300}"
export DATA_WAIT_WARN_SEC="${DATA_WAIT_WARN_SEC:-30}"
export LR="${LR:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
export LOSS_MODE="${LOSS_MODE:-physics}"
export ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
export CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
export DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.0}"
export CORE_SCALE_KM="${CORE_SCALE_KM:-0.05}"
export ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-1.0}"
export ENERGY_BIAS_WEIGHT="${ENERGY_BIAS_WEIGHT:-1.0}"
export ENERGY_PARTICLE_BIAS_WEIGHT="${ENERGY_PARTICLE_BIAS_WEIGHT:-0.0}"
export ENERGY_BIAS_BIN_WIDTH="${ENERGY_BIAS_BIN_WIDTH:-0.1}"
export ENERGY_BIAS_MIN_BIN_COUNT="${ENERGY_BIAS_MIN_BIN_COUNT:-8}"

export MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-1}"
export MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT:-0.15}"
export MASS_LOSS_MODE="${MASS_LOSS_MODE:-bce}"
export MASS_FOCAL_GAMMA="${MASS_FOCAL_GAMMA:-2.0}"
export MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0.5}"
export MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}"

export QUALITY_PREDICTION="${QUALITY_PREDICTION:-0}"
export ERROR_PREDICTION="${ERROR_PREDICTION:-0}"
export FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-0}"
if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
  export ATTENTION_MAPS="${ATTENTION_MAPS:-0}"
  export DIAGNOSTICS="${DIAGNOSTICS:-0}"
  export CHECKPOINT_MILESTONES="${CHECKPOINT_MILESTONES:-}"
  export MAX_GRAPHS="${MAX_GRAPHS:-4096}"
  export MAX_VAL_GRAPHS="${MAX_VAL_GRAPHS:-512}"
  export VALIDATE_EVERY_N_EPOCHS="${VALIDATE_EVERY_N_EPOCHS:-1}"
else
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    export ATTENTION_MAPS="${ATTENTION_MAPS:-0}"
    export DIAGNOSTICS="${DIAGNOSTICS:-0}"
  else
    export ATTENTION_MAPS="${ATTENTION_MAPS:-1}"
    export DIAGNOSTICS="${DIAGNOSTICS:-1}"
  fi
fi
if [[ "${PREPARE_FAST_CACHE}" == "0" && "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  echo "WARNING: transformer hetero production training should use PREPARE_FAST_CACHE=1 unless GRAPH_INPUT is already flat_hdf5" >&2
fi
export ATTENTION_MAPS_SPLIT="${ATTENTION_MAPS_SPLIT:-validation}"
export ATTENTION_MAPS_MAX_GRAPHS="${ATTENTION_MAPS_MAX_GRAPHS:-16}"
export ATTENTION_MAPS_DEVICE="${ATTENTION_MAPS_DEVICE:-${DEVICE:-cuda}}"
export SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-0}"

export SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
export VAL_FRACTION="${VAL_FRACTION:-0.05}"
export TEST_FRACTION="${TEST_FRACTION:-0.10}"
export SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
export SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.20}"
export DEVICE="${DEVICE:-cuda}"
if [[ -z "${AMP:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" && "${DEVICE}" == cuda* ]]; then
    export AMP=fp16
  else
    export AMP=off
  fi
else
  export AMP
fi
export DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
export SPLIT_WORKERS="${SPLIT_WORKERS:-4}"

exec "${SCRIPT_DIR}/submit_server_waveform_full_training.sh"
