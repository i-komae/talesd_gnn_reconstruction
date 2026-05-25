#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage:
  scripts/submit_server_reco_mass_training.sh

Submit joint reconstruction + mass-classification training.
By default, GRAPH_INPUT points to the current energy-flat HDF5 graph directory.
Set GRAPH_INPUT explicitly to override it.
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

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
DEFAULT_GRAPH_INPUT="${DEFAULT_GRAPH_INPUT:-/dicos_ui_home/ikomae/work/gnn/graphs/server_graph_export_energyflat200000_20260524_075508}"
PARTITION_FOR_NAME="${PARTITION:-b6000-al9_long}"
RESOURCE_TAG="${RESOURCE_TAG:-${PARTITION_FOR_NAME%%-*}}"

export RUN_ID
export RESOURCE_TAG
export GRAPH_INPUT="${GRAPH_INPUT:-${DEFAULT_GRAPH_INPUT}}"
export RUN_NAME="${RUN_NAME:-server_reco_mass_quality_${RESOURCE_TAG}_${TRAIN_EPOCHS}epoch_${RUN_ID}}"
export TRAINING_TASK="reconstruction"
export MASS_CLASSIFICATION="1"
export MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT:-0.05}"
export MASS_LOSS_MODE="${MASS_LOSS_MODE:-bce}"
export MASS_POS_WEIGHT_MODE="${MASS_POS_WEIGHT_MODE:-none}"
export MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0.5}"
export MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}"

export MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-physics}"
export HIDDEN_DIM="${HIDDEN_DIM:-192}"
export LAYERS="${LAYERS:-5}"
export DROPOUT="${DROPOUT:-0.08}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
export CLASSIFICATION_ARCH="${CLASSIFICATION_ARCH:-enhanced}"
export WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"

export TRAIN_EPOCHS
export LR_SCHEDULER="${LR_SCHEDULER:-reduce-on-plateau}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
export EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-32}"

export LOSS_MODE="${LOSS_MODE:-physics}"
export ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.2}"
export CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
export DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.4}"
export CORE_SCALE_KM="${CORE_SCALE_KM:-0.12}"
export ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-1.0}"

export QUALITY_PREDICTION="${QUALITY_PREDICTION:-1}"
export QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.2}"
export QUALITY_ANGULAR_SCALE_DEG="${QUALITY_ANGULAR_SCALE_DEG:-1.0}"
export QUALITY_CORE_SCALE_KM="${QUALITY_CORE_SCALE_KM:-0.05}"
export QUALITY_ENERGY_SCALE="${QUALITY_ENERGY_SCALE:-0.10}"
export ERROR_PREDICTION="${ERROR_PREDICTION:-0}"
export ERROR_WEIGHT="${ERROR_WEIGHT:-0.0}"
export NLL_WEIGHT="${NLL_WEIGHT:-0.0}"

exec "${SCRIPT_DIR}/submit_server_waveform_full_training.sh"
