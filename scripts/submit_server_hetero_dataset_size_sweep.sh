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
SERIAL_EXPORTS="${SERIAL_EXPORTS:-1}"

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
  previous_export_job_id=""
  for size in "${clean_sizes[@]}"; do
    dataset_name="hetero_balanced_flat${size}_${RUN_ID}"
    dependency=""
    if [[ "${SERIAL_EXPORTS}" == "1" && -n "${previous_export_job_id}" ]]; then
      dependency="afterok:${previous_export_job_id}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      ENERGY_SAMPLE_PER_BIN="${size}" \
      RUN_ID="${RUN_ID}" \
      RUN_NAME="${dataset_name}" \
      REFILL_ATTEMPTS="${REFILL_ATTEMPTS}" \
      VAL_FRACTION="${VAL_FRACTION}" \
      TEST_FRACTION="${TEST_FRACTION}" \
      SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
      SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
      DRY_RUN="${DRY_RUN}" \
      "${SCRIPT_DIR}/submit_server_hetero_balanced_graph_export.sh"
    else
      export_job_id="$(
        ENERGY_SAMPLE_PER_BIN="${size}" \
        RUN_ID="${RUN_ID}" \
        RUN_NAME="${dataset_name}" \
        REFILL_ATTEMPTS="${REFILL_ATTEMPTS}" \
        VAL_FRACTION="${VAL_FRACTION}" \
        TEST_FRACTION="${TEST_FRACTION}" \
        SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
        SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
        SBATCH_DEPENDENCY="${dependency}" \
        SBATCH_PARSABLE=1 \
        "${SCRIPT_DIR}/submit_server_hetero_balanced_graph_export.sh"
      )"
      printf "export job for sample_per_bin=%s: %s dependency=%s\n" "${size}" "${export_job_id}" "${dependency}" >&2
      if [[ "${SERIAL_EXPORTS}" == "1" ]]; then
        previous_export_job_id="${export_job_id}"
      fi
    fi
  done
fi

for size in "${clean_sizes[@]}"; do
  size="${size// /}"
  [[ -n "${size}" ]] || continue
  dataset_name="hetero_balanced_flat${size}_${RUN_ID}"
  graph_input="${GRAPH_ROOT}/${dataset_name}"
  if [[ "${SUBMIT_TRAINING}" == "1" ]]; then
    first_shard=""
    if [[ -d "${graph_input}" ]]; then
      first_shard="$(find "${graph_input}" -type f -name '*.h5' -size +0c -print -quit)"
    fi
    if [[ "${DRY_RUN}" != "1" && "${ALLOW_MISSING_GRAPH}" != "1" && -z "${first_shard}" ]]; then
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
printf "export_strategy=dstio-balanced-per-size sizes=%s\n" "${SIZES}" >&2
printf "export_refill: attempts=%s owner=dstio.write_balanced_graph_h5\n" "${REFILL_ATTEMPTS}" >&2
printf "serial_exports=%s\n" "${SERIAL_EXPORTS}" >&2
printf "split_event_fractions: train=1-val-test val=%s test=%s\n" "${VAL_FRACTION}" "${TEST_FRACTION}" >&2
printf "split_source_fractions: train=1-val-test val=%s test=%s\n" "${SOURCE_VAL_FRACTION}" "${SOURCE_TEST_FRACTION}" >&2
