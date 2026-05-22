#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage:
  scripts/submit_server_mass_only_training.sh [epochs]

Submit mass-only waveform training with the project-standard server settings.
Default: 48 epochs. The optional epochs argument overrides TRAIN_EPOCHS.
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
TRAIN_EPOCHS="${TRAIN_EPOCHS:-48}"
if ! [[ "${TRAIN_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_EPOCHS must be a positive integer: ${TRAIN_EPOCHS}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export RUN_ID
export RUN_NAME="${RUN_NAME:-server_mass_waveform_direct_${TRAIN_EPOCHS}epoch_${RUN_ID}}"
export TRAIN_EPOCHS
export TRAINING_TASK="mass"
export MASS_CLASSIFICATION="1"
export QUALITY_PREDICTION="0"
export CLASSIFICATION_ARCH="${CLASSIFICATION_ARCH:-enhanced}"
export MASS_LOSS_MODE="${MASS_LOSS_MODE:-focal}"
export MASS_POS_WEIGHT_MODE="${MASS_POS_WEIGHT_MODE:-none}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-1}"

exec "${SCRIPT_DIR}/submit_server_waveform_full_training.sh"
