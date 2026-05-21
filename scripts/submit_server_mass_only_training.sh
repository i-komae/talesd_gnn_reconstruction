#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export RUN_ID
export RUN_NAME="${RUN_NAME:-server_mass_only_b6000_${RUN_ID}}"
export TRAINING_TASK="mass"
export MASS_CLASSIFICATION="1"
export SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-1}"

exec "${SCRIPT_DIR}/submit_server_waveform_full_training.sh"
