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
REFILL_ATTEMPTS="${REFILL_ATTEMPTS:-2}"
REFILL_SAFETY_FACTOR="${REFILL_SAFETY_FACTOR:-1.25}"
REFILL_MIN_EFFICIENCY="${REFILL_MIN_EFFICIENCY:-0.01}"
SERIAL_RESHARDS="${SERIAL_RESHARDS:-1}"

IFS="," read -r -a size_array <<< "${SIZES}"

clean_sizes=()
max_size=0
for raw_size in "${size_array[@]}"; do
  size="${raw_size// /}"
  [[ -n "${size}" ]] || continue
  if ! [[ "${size}" =~ ^[0-9]+$ ]]; then
    echo "invalid SIZES entry: ${size}" >&2
    exit 2
  fi
  clean_sizes+=("${size}")
  if (( size > max_size )); then
    max_size="${size}"
  fi
done

if (( ${#clean_sizes[@]} == 0 )); then
  echo "SIZES is empty" >&2
  exit 2
fi

if [[ "${SUBMIT_EXPORTS}" == "1" ]]; then
  base_dataset_name="hetero_balanced_flat${max_size}_${RUN_ID}"
  base_graph_input="${GRAPH_ROOT}/${base_dataset_name}/${base_dataset_name}.h5"
  base_job_id=""
  if [[ "${DRY_RUN}" == "1" ]]; then
    ENERGY_SAMPLE_PER_BIN="${max_size}" \
    RUN_ID="${RUN_ID}" \
    RUN_NAME="${base_dataset_name}" \
    REFILL_ATTEMPTS="${REFILL_ATTEMPTS}" \
    REFILL_SAFETY_FACTOR="${REFILL_SAFETY_FACTOR}" \
    REFILL_MIN_EFFICIENCY="${REFILL_MIN_EFFICIENCY}" \
    VAL_FRACTION="${VAL_FRACTION}" \
    TEST_FRACTION="${TEST_FRACTION}" \
    SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
    SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
    DRY_RUN="${DRY_RUN}" \
    "${SCRIPT_DIR}/submit_server_hetero_balanced_graph_export.sh"
  else
    base_job_id="$(
      ENERGY_SAMPLE_PER_BIN="${max_size}" \
      RUN_ID="${RUN_ID}" \
      RUN_NAME="${base_dataset_name}" \
      REFILL_ATTEMPTS="${REFILL_ATTEMPTS}" \
      REFILL_SAFETY_FACTOR="${REFILL_SAFETY_FACTOR}" \
      REFILL_MIN_EFFICIENCY="${REFILL_MIN_EFFICIENCY}" \
      VAL_FRACTION="${VAL_FRACTION}" \
      TEST_FRACTION="${TEST_FRACTION}" \
      SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
      SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
      SBATCH_PARSABLE=1 \
      "${SCRIPT_DIR}/submit_server_hetero_balanced_graph_export.sh"
    )"
    printf "base export job for sample_per_bin=%s: %s\n" "${max_size}" "${base_job_id}" >&2
  fi

  previous_reshard_job_id=""
  for size in "${clean_sizes[@]}"; do
    if (( size == max_size )); then
      printf "derive job for size=%s: skipped; base export writes %s\n" "${size}" "${base_graph_input}" >&2
      continue
    fi
    dataset_name="hetero_balanced_flat${size}_${RUN_ID}"
    graph_input="${GRAPH_ROOT}/${dataset_name}/${dataset_name}.h5"
    dependency=""
    if [[ -n "${base_job_id}" ]]; then
      dependency="afterok:${base_job_id}"
    fi
    if [[ "${SERIAL_RESHARDS}" == "1" && -n "${previous_reshard_job_id}" ]]; then
      dependency="afterok:${previous_reshard_job_id}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      GRAPH_INPUT="${base_graph_input}" \
      ENERGY_SAMPLE_PER_BIN="${size}" \
      RUN_ID="${RUN_ID}" \
      RUN_NAME="${dataset_name}" \
      GRAPH_RUN_DIR="${GRAPH_ROOT}/${dataset_name}" \
      GRAPH_OUTPUT="${graph_input}" \
      VAL_FRACTION="${VAL_FRACTION}" \
      TEST_FRACTION="${TEST_FRACTION}" \
      SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
      SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
      DRY_RUN="${DRY_RUN}" \
      "${SCRIPT_DIR}/submit_server_hetero_reshard.sh"
    else
      reshard_job_id="$(
        GRAPH_INPUT="${base_graph_input}" \
        ENERGY_SAMPLE_PER_BIN="${size}" \
        RUN_ID="${RUN_ID}" \
        RUN_NAME="${dataset_name}" \
        GRAPH_RUN_DIR="${GRAPH_ROOT}/${dataset_name}" \
        GRAPH_OUTPUT="${graph_input}" \
        VAL_FRACTION="${VAL_FRACTION}" \
        TEST_FRACTION="${TEST_FRACTION}" \
        SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
        SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
        SBATCH_DEPENDENCY="${dependency}" \
        SBATCH_PARSABLE=1 \
        "${SCRIPT_DIR}/submit_server_hetero_reshard.sh"
      )"
      printf "derive job for size=%s: %s dependency=%s\n" "${size}" "${reshard_job_id}" "${dependency}" >&2
      if [[ "${SERIAL_RESHARDS}" == "1" ]]; then
        previous_reshard_job_id="${reshard_job_id}"
      fi
    fi
  done
fi

for size in "${clean_sizes[@]}"; do
  size="${size// /}"
  [[ -n "${size}" ]] || continue
  dataset_name="hetero_balanced_flat${size}_${RUN_ID}"
  graph_input="${GRAPH_ROOT}/${dataset_name}/${dataset_name}.h5"
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
printf "export_strategy=base-max-size-plus-reshard max_size=%s sizes=%s\n" "${max_size}" "${SIZES}" >&2
printf "export_refill: attempts=%s safety_factor=%s min_efficiency=%s\n" "${REFILL_ATTEMPTS}" "${REFILL_SAFETY_FACTOR}" "${REFILL_MIN_EFFICIENCY}" >&2
printf "serial_reshards=%s\n" "${SERIAL_RESHARDS}" >&2
printf "split_event_fractions: train=1-val-test val=%s test=%s\n" "${VAL_FRACTION}" "${TEST_FRACTION}" >&2
printf "split_source_fractions: train=1-val-test val=%s test=%s\n" "${SOURCE_VAL_FRACTION}" "${SOURCE_TEST_FRACTION}" >&2
