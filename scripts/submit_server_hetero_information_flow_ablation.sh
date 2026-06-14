#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<EOF
Usage:
  GRAPH_INPUT=/path/to/hetero_graphs scripts/submit_server_hetero_information_flow_ablation.sh

Submit six 2-epoch hetero information-flow ablations:
  A current baseline, B parent waveform, C parent waveform + bounds,
  D parent waveform + bounds + minimal relations, E waveform off + minimal,
  F old-like pulse waveform crop + minimal relations.

This submitter keeps the training task/loss/split/resource defaults from the
normal hetero reco+mass quality submitter and only changes the information-flow
switches required for the ablation.
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
  echo "GRAPH_INPUT is required." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-hetero_infoflow_$(date +%Y%m%d_%H%M%S)}"
SUBMITTER="${SUBMITTER:-${SCRIPT_DIR}/submit_server_hetero_reco_mass_quality_training.sh}"

COMMON_ENV=(
  "SPEED_BENCHMARK=1"
  "TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}"
  "MAX_GRAPHS=${MAX_GRAPHS:-4096}"
  "MAX_VAL_GRAPHS=${MAX_VAL_GRAPHS:-512}"
  "FEATURE_IMPORTANCE=${FEATURE_IMPORTANCE:-0}"
  "ATTENTION_MAPS=${ATTENTION_MAPS:-0}"
  "DIAGNOSTICS=${DIAGNOSTICS:-0}"
  "PREPARE_FAST_CACHE=${PREPARE_FAST_CACHE:-0}"
  "HETERO_TRAINING_DATA_FORMAT=${HETERO_TRAINING_DATA_FORMAT:-fast_tensor}"
  "FINAL_EVAL_DATA_FORMAT=${FINAL_EVAL_DATA_FORMAT:-fast_tensor}"
  "WAVEFORM_TRANSFORMER_MAX_TOKENS=${WAVEFORM_TRANSFORMER_MAX_TOKENS:-128}"
  "BATCH_SIZE=${BATCH_SIZE:-32}"
  "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}"
  "AMP=${AMP:-fp16}"
  "PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}"
  "PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}"
  "TRAIN_WORKERS=${TRAIN_WORKERS:-4}"
  "PIN_MEMORY=${PIN_MEMORY:-0}"
)

submit_one() {
  local tag="$1"
  local use_parent="$2"
  local use_bounds="$3"
  local pulse_encoder="$4"
  local relation_preset="$5"
  local waveform_encoder="$6"

  local run_name="server_hetero_infoflow_${tag}_${RUN_ID}"
  echo "submit infoflow tag=${tag} run_name=${run_name} waveform_encoder=${waveform_encoder} use_parent=${use_parent} use_bounds=${use_bounds} pulse_waveform_encoder=${pulse_encoder} relation_preset=${relation_preset}"
  env \
    "${COMMON_ENV[@]}" \
    "RUN_ID=${RUN_ID}" \
    "RUN_NAME=${run_name}" \
    "USE_PULSE_PARENT_WAVEFORM=${use_parent}" \
    "USE_PULSE_BOUNDS=${use_bounds}" \
    "PULSE_WAVEFORM_ENCODER=${pulse_encoder}" \
    "HETERO_RELATION_PRESET=${relation_preset}" \
    "WAVEFORM_ENCODER=${waveform_encoder}" \
    "${SUBMITTER}"
}

submit_one "a_current_baseline" "0" "0" "none" "all" "${WAVEFORM_ENCODER:-transformer}"
submit_one "b_parent_waveform" "1" "0" "none" "all" "${WAVEFORM_ENCODER:-transformer}"
submit_one "c_parent_waveform_bounds" "1" "1" "bounds" "all" "${WAVEFORM_ENCODER:-transformer}"
submit_one "d_parent_bounds_minimal" "1" "1" "bounds" "minimal" "${WAVEFORM_ENCODER:-transformer}"
submit_one "e_waveform_off_minimal" "0" "1" "bounds" "minimal" "none"
submit_one "f_pulse_crop_cnn_minimal" "1" "1" "crop_cnn" "minimal" "${WAVEFORM_ENCODER:-cnn-gru}"
