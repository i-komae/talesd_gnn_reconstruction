#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SIZES="${SIZES:-50000,20000,10000}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_NAME="${SWEEP_NAME:-hetero_balanced_size_sweep_${RUN_ID}}"
SUBMIT_EXPORTS="${SUBMIT_EXPORTS:-1}"
SUBMIT_TRAINING="${SUBMIT_TRAINING:-0}"
DRY_RUN="${DRY_RUN:-0}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
ALLOW_MISSING_GRAPH="${ALLOW_MISSING_GRAPH:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.45}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.45}"

IFS="," read -r -a size_array <<< "${SIZES}"

for size in "${size_array[@]}"; do
  size="${size// /}"
  [[ -n "${size}" ]] || continue
  dataset_name="hetero_balanced_flat${size}_${RUN_ID}"
  graph_input="${GRAPH_ROOT}/${dataset_name}/${dataset_name}.h5"
  if [[ "${SUBMIT_EXPORTS}" == "1" ]]; then
    ENERGY_SAMPLE_PER_BIN="${size}" \
    RUN_ID="${RUN_ID}" \
    RUN_NAME="${dataset_name}" \
    VAL_FRACTION="${VAL_FRACTION}" \
    TEST_FRACTION="${TEST_FRACTION}" \
    SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
    SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
    DRY_RUN="${DRY_RUN}" \
    "${SCRIPT_DIR}/submit_server_hetero_balanced_graph_export.sh"
  fi
  if [[ "${SUBMIT_TRAINING}" == "1" ]]; then
    first_shard="${graph_input%.h5}_0000.h5"
    if [[ "${DRY_RUN}" != "1" && "${ALLOW_MISSING_GRAPH}" != "1" && ! -s "${graph_input}" && ! -s "${first_shard}" ]]; then
      echo "graph input is not ready for training: ${graph_input}" >&2
      echo "Run exports first, confirm H5/shards and summaries, then rerun with SUBMIT_EXPORTS=0 SUBMIT_TRAINING=1." >&2
      exit 2
    fi
    GRAPH_INPUT="${graph_input}" \
    RUN_ID="${RUN_ID}" \
    RUN_NAME="${dataset_name}_reco_mass_quality_${RUN_ID}" \
    VAL_FRACTION="${VAL_FRACTION}" \
    TEST_FRACTION="${TEST_FRACTION}" \
    SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
    SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
    DRY_RUN="${DRY_RUN}" \
    "${SCRIPT_DIR}/submit_server_hetero_reco_mass_quality_training.sh"

    GRAPH_INPUT="${graph_input}" \
    RUN_ID="${RUN_ID}" \
    RUN_NAME="${dataset_name}_reco_mass_error_${RUN_ID}" \
    VAL_FRACTION="${VAL_FRACTION}" \
    TEST_FRACTION="${TEST_FRACTION}" \
    SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
    SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
    DRY_RUN="${DRY_RUN}" \
    "${SCRIPT_DIR}/submit_server_hetero_reco_mass_error_training.sh"
  else
    printf "training inputs for size=%s: %s\n" "${size}" "${graph_input}" >&2
  fi
done

printf "sweep_name=%s\n" "${SWEEP_NAME}" >&2
printf "split_event_fractions: train=1-val-test val=%s test=%s\n" "${VAL_FRACTION}" "${TEST_FRACTION}" >&2
printf "split_source_fractions: train=1-val-test val=%s test=%s\n" "${SOURCE_VAL_FRACTION}" "${SOURCE_TEST_FRACTION}" >&2
