#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SPEED_BENCHMARK="${SPEED_BENCHMARK:-0}"
PARTITION_FOR_NAME="${PARTITION:-v100-al9_long}"
RESOURCE_TAG="${RESOURCE_TAG:-${PARTITION_FOR_NAME%%-*}}"
if [[ "${SPEED_BENCHMARK}" == "1" ]]; then
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
else
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
fi

export RUN_ID
export SPEED_BENCHMARK
export RESOURCE_TAG
export TRAIN_EPOCHS
export RUN_NAME="${RUN_NAME:-server_hetero_reco_mass_quality_${RESOURCE_TAG}_${TRAIN_EPOCHS}epoch_${RUN_ID}}"
export TRAINING_TASK="reconstruction"
export MASS_CLASSIFICATION="1"
export LOSS_MODE="${LOSS_MODE:-physics}"
export QUALITY_PREDICTION="1"
export ERROR_PREDICTION="0"
export ERROR_WEIGHT="0.0"

exec "${SCRIPT_DIR}/submit_server_hetero_training.sh"
