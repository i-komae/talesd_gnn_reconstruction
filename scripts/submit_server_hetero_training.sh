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
TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
PARTITION_FOR_NAME="${PARTITION:-v100-al9_long}"
RESOURCE_TAG="${RESOURCE_TAG:-${PARTITION_FOR_NAME%%-*}}"

export RUN_ID
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

export TRAIN_EPOCHS
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    export BATCH_SIZE=8
  else
    export BATCH_SIZE=128
  fi
else
  export BATCH_SIZE
fi
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ "${WAVEFORM_ENCODER}" == "transformer" ]]; then
    export GRADIENT_ACCUMULATION_STEPS=16
  else
    export GRADIENT_ACCUMULATION_STEPS=1
  fi
else
  export GRADIENT_ACCUMULATION_STEPS
fi
if [[ -z "${PIN_MEMORY:-}" && "${WAVEFORM_ENCODER}" == "transformer" ]]; then
  export PIN_MEMORY=0
fi
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
export ATTENTION_MAPS="${ATTENTION_MAPS:-1}"
export ATTENTION_MAPS_SPLIT="${ATTENTION_MAPS_SPLIT:-validation}"
export ATTENTION_MAPS_MAX_GRAPHS="${ATTENTION_MAPS_MAX_GRAPHS:-16}"
export ATTENTION_MAPS_DEVICE="${ATTENTION_MAPS_DEVICE:-${DEVICE:-cuda}}"
export DIAGNOSTICS="${DIAGNOSTICS:-1}"
export SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-0}"

export SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
export VAL_FRACTION="${VAL_FRACTION:-0.05}"
export TEST_FRACTION="${TEST_FRACTION:-0.10}"
export SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
export SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.20}"
export DEVICE="${DEVICE:-cuda}"
export DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
export SPLIT_WORKERS="${SPLIT_WORKERS:-4}"

exec "${SCRIPT_DIR}/submit_server_waveform_full_training.sh"
