#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage:
  scripts/submit_server_mass_only_training.sh [epochs]

Submit mass-only waveform training with the project-standard server settings.
Default: 128 max epochs with validation early stopping. The optional epochs argument overrides TRAIN_EPOCHS.
By default, GRAPH_INPUT points to the current energy-flat HDF5 graph directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  export TRAIN_EPOCHS="$1"
fi
TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
if ! [[ "${TRAIN_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_EPOCHS must be a positive integer: ${TRAIN_EPOCHS}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_GRAPH_INPUT="${DEFAULT_GRAPH_INPUT:-/dicos_ui_home/ikomae/work/gnn/graphs/server_graph_export_energyflat200000_20260524_075508}"

export RUN_ID
export GRAPH_INPUT="${GRAPH_INPUT:-${DEFAULT_GRAPH_INPUT}}"
export RUN_NAME="${RUN_NAME:-server_mass_waveform_direct_${TRAIN_EPOCHS}epoch_${RUN_ID}}"
export TRAIN_EPOCHS
export TRAINING_TASK="mass"
export MASS_CLASSIFICATION="1"
export QUALITY_PREDICTION="0"
export ERROR_PREDICTION="0"
export CLASSIFICATION_ARCH="${CLASSIFICATION_ARCH:-enhanced}"
export MASS_LOSS_MODE="${MASS_LOSS_MODE:-bce}"
export MASS_POS_WEIGHT_MODE="${MASS_POS_WEIGHT_MODE:-none}"
export MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0.5}"
export MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
export EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-32}"
export DROPOUT="${DROPOUT:-0.12}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-0}"

exec "${SCRIPT_DIR}/submit_server_waveform_full_training.sh"
