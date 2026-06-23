#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_allsrc_homogeneous_hetero_comparison.sh: starting"

RUN_ID="${RUN_ID:-allsrc_compare_$(date +%Y%m%d_%H%M%S)}"
LIGHT_TARGET="${LIGHT_TARGET:-50000}"
SOURCE_GROUPS_PER_STRATUM="${SOURCE_GROUPS_PER_STRATUM:-298}"
if ! [[ "${LIGHT_TARGET}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIGHT_TARGET must be a positive integer: ${LIGHT_TARGET}" >&2
  exit 2
fi
if ! [[ "${SOURCE_GROUPS_PER_STRATUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SOURCE_GROUPS_PER_STRATUM must be a positive integer: ${SOURCE_GROUPS_PER_STRATUM}" >&2
  exit 2
fi

GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
LOCAL_RUNTIME_CACHE="${LOCAL_RUNTIME_CACHE:-0}"
PARTITION="${PARTITION:-v100-al9_long}"
H5_PARTITION="${H5_PARTITION:-edr1-al9_large}"
H5_RUN_NAME="${H5_RUN_NAME:-hetero_light_allsrc_target${LIGHT_TARGET}_${RUN_ID}}"
GRAPH_INPUT="${GRAPH_INPUT:-${GRAPH_ROOT}/${H5_RUN_NAME}}"
GRAPHS_PER_SOURCE_GROUP="${GRAPHS_PER_SOURCE_GROUP:-$(( (LIGHT_TARGET + SOURCE_GROUPS_PER_STRATUM - 1) / SOURCE_GROUPS_PER_STRATUM ))}"
ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA:-1}"
REFILL_MIN_GRAPHS_PER_SOURCE_GROUP="${REFILL_MIN_GRAPHS_PER_SOURCE_GROUP:-${GRAPHS_PER_SOURCE_GROUP}}"
MAX_REFILL_SOURCE_GROUPS_PER_STRATUM="${MAX_REFILL_SOURCE_GROUPS_PER_STRATUM:-64}"
MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS:-0}"
SOURCE_GROUP_SELECTION="all"
PULSE_MASK="${PULSE_MASK:-ising_kept}"
HETERO_MODEL_ARCHITECTURE="${HETERO_MODEL_ARCHITECTURE:-hetero_attention}"
HETERO_RELATION_PRESET="${HETERO_RELATION_PRESET:-minimal}"
SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE:-event}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.20}"
HOMOGENEOUS_BATCH_SIZE="${HOMOGENEOUS_BATCH_SIZE:-256}"
HOMOGENEOUS_TRAIN_WORKERS="${HOMOGENEOUS_TRAIN_WORKERS:-6}"
HOMOGENEOUS_PREPROCESS_WORKERS="${HOMOGENEOUS_PREPROCESS_WORKERS:-8}"
HOMOGENEOUS_PREFETCH_FACTOR="${HOMOGENEOUS_PREFETCH_FACTOR:-2}"
HOMOGENEOUS_PERSISTENT_WORKERS="${HOMOGENEOUS_PERSISTENT_WORKERS:-0}"
HOMOGENEOUS_PIN_MEMORY="${HOMOGENEOUS_PIN_MEMORY:-1}"
HOMOGENEOUS_ENERGY_BIAS_WEIGHT="${HOMOGENEOUS_ENERGY_BIAS_WEIGHT:-0.0}"
HOMOGENEOUS_ENERGY_PARTICLE_BIAS_WEIGHT="${HOMOGENEOUS_ENERGY_PARTICLE_BIAS_WEIGHT:-0.0}"
HETERO_ENERGY_BIAS_WEIGHT="${HETERO_ENERGY_BIAS_WEIGHT:-${HOMOGENEOUS_ENERGY_BIAS_WEIGHT}}"
HETERO_ENERGY_PARTICLE_BIAS_WEIGHT="${HETERO_ENERGY_PARTICLE_BIAS_WEIGHT:-${HOMOGENEOUS_ENERGY_PARTICLE_BIAS_WEIGHT}}"
DRY_RUN="${DRY_RUN:-0}"
H5_EXPORT_WORKERS="${H5_EXPORT_WORKERS:-96}"
H5_CPUS_PER_TASK="${H5_CPUS_PER_TASK:-${H5_EXPORT_WORKERS}}"
H5_MEM="${H5_MEM:-384G}"
H5_TIME_LIMIT="${H5_TIME_LIMIT:-1-00:00:00}"
MAX_SINGLE_JOB_LIGHT_TARGET="${MAX_SINGLE_JOB_LIGHT_TARGET:-100000}"
ALLOW_LONG_SINGLE_H5_EXPORT="${ALLOW_LONG_SINGLE_H5_EXPORT:-0}"

for pair in \
  "H5_EXPORT_WORKERS:${H5_EXPORT_WORKERS}" \
  "H5_CPUS_PER_TASK:${H5_CPUS_PER_TASK}"
do
  key="${pair%%:*}"
  value="${pair#*:}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${key} must be a positive integer: ${value}" >&2
    exit 2
  fi
done
if ! [[ "${MAX_SINGLE_JOB_LIGHT_TARGET}" =~ ^[0-9]+$ ]]; then
  echo "MAX_SINGLE_JOB_LIGHT_TARGET must be a non-negative integer: ${MAX_SINGLE_JOB_LIGHT_TARGET}" >&2
  exit 2
fi
if [[ "${SOURCE_GROUP_SELECTION}" == "all" && "${LIGHT_TARGET}" -gt "${MAX_SINGLE_JOB_LIGHT_TARGET}" && "${ALLOW_LONG_SINGLE_H5_EXPORT}" != "1" ]]; then
  cat >&2 <<EOF
Refusing unsafe single-job all-source H5 export.

LIGHT_TARGET=${LIGHT_TARGET}
SOURCE_GROUP_SELECTION=${SOURCE_GROUP_SELECTION}
GRAPHS_PER_SOURCE_GROUP=${GRAPHS_PER_SOURCE_GROUP}
MAX_SINGLE_JOB_LIGHT_TARGET=${MAX_SINGLE_JOB_LIGHT_TARGET}

This configuration is above the current single-job safety limit. Increase
MAX_SINGLE_JOB_LIGHT_TARGET only after checking the H5 export worker count,
runtime limit, and previous log rate, or set ALLOW_LONG_SINGLE_H5_EXPORT=1 for
an explicitly accepted manual override.
EOF
  exit 2
fi

status "ALL-SOURCE HOMOGENEOUS/HETEROGENEOUS COMPARISON"
status "run_id: ${RUN_ID}"
status "light_target: ${LIGHT_TARGET}"
status "graphs_per_source_group: ${GRAPHS_PER_SOURCE_GROUP}"
status "source_group_selection: ${SOURCE_GROUP_SELECTION}"
status "refill_min_graphs_per_source_group: ${REFILL_MIN_GRAPHS_PER_SOURCE_GROUP}"
status "max_refill_source_groups_per_stratum: ${MAX_REFILL_SOURCE_GROUPS_PER_STRATUM}"
status "graph_input_after_export: ${GRAPH_INPUT}"
status "h5_partition: ${H5_PARTITION}"
status "h5_export_workers: ${H5_EXPORT_WORKERS}"
status "h5_cpus_per_task: ${H5_CPUS_PER_TASK}"
status "h5_mem: ${H5_MEM}"
status "h5_time_limit: ${H5_TIME_LIMIT}"
status "max_single_job_light_target: ${MAX_SINGLE_JOB_LIGHT_TARGET}"
status "allow_long_single_h5_export: ${ALLOW_LONG_SINGLE_H5_EXPORT}"
status "training_partition: ${PARTITION}"
status "run_uv_sync: ${RUN_UV_SYNC}"
status "local_runtime_cache: ${LOCAL_RUNTIME_CACHE}"
status "source_fraction_mode: ${SOURCE_FRACTION_MODE}"
status "source_val_fraction: ${SOURCE_VAL_FRACTION}"
status "source_test_fraction: ${SOURCE_TEST_FRACTION}"
status "homogeneous_defaults: batch=${HOMOGENEOUS_BATCH_SIZE} workers=${HOMOGENEOUS_TRAIN_WORKERS}/${HOMOGENEOUS_PREPROCESS_WORKERS} prefetch=${HOMOGENEOUS_PREFETCH_FACTOR} persistent=${HOMOGENEOUS_PERSISTENT_WORKERS} pin_memory=${HOMOGENEOUS_PIN_MEMORY} energy_bias=${HOMOGENEOUS_ENERGY_BIAS_WEIGHT} particle_bias=${HOMOGENEOUS_ENERGY_PARTICLE_BIAS_WEIGHT}"
status "heterogeneous_loss_defaults: energy_bias=${HETERO_ENERGY_BIAS_WEIGHT} particle_bias=${HETERO_ENERGY_PARTICLE_BIAS_WEIGHT}"
status "heterogeneous_selection_basis: provisional next setting from synced validation loss/milestones and failure modes"
status "heterogeneous_note: final model choice requires validation/test on the new all-source H5; light562 is smoke only"
status "heterogeneous_default: ${HETERO_MODEL_ARCHITECTURE} cnn-gru crop_cnn ising_kept ${HETERO_RELATION_PRESET}-relations"

if [[ "${DRY_RUN}" == "1" ]]; then
  RUN_NAME="${H5_RUN_NAME}" \
  GRAPHS_PER_SOURCE_GROUP="${GRAPHS_PER_SOURCE_GROUP}" \
  SOURCE_GROUP_SELECTION="${SOURCE_GROUP_SELECTION}" \
  ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA}" \
  REFILL_MIN_GRAPHS_PER_SOURCE_GROUP="${REFILL_MIN_GRAPHS_PER_SOURCE_GROUP}" \
  MAX_REFILL_SOURCE_GROUPS_PER_STRATUM="${MAX_REFILL_SOURCE_GROUPS_PER_STRATUM}" \
  EXPORT_WORKERS="${H5_EXPORT_WORKERS}" \
  CPUS_PER_TASK="${H5_CPUS_PER_TASK}" \
  MEM="${H5_MEM}" \
  TIME_LIMIT="${H5_TIME_LIMIT}" \
  PARTITION="${H5_PARTITION}" \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS}" \
  DRY_RUN=1 \
  "${SCRIPT_DIR}/submit_server_hetero_light_graph_export.sh"

HETERO_GRAPH_INPUT="${GRAPH_INPUT}" \
  RUN_ID="homogeneous_from_allsrc_${RUN_ID}" \
  PULSE_MASK="${PULSE_MASK}" \
  CONVERT_WORKERS="${CONVERT_WORKERS:-32}" \
  CONVERT_CPUS_PER_TASK="${CONVERT_CPUS_PER_TASK:-${CONVERT_WORKERS:-32}}" \
  DRY_RUN=1 \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  LOCAL_RUNTIME_CACHE="${LOCAL_RUNTIME_CACHE}" \
  PARTITION="${PARTITION}" \
  SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE}" \
  SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
  SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
  BATCH_SIZE="${HOMOGENEOUS_BATCH_SIZE}" \
  TRAIN_WORKERS="${HOMOGENEOUS_TRAIN_WORKERS}" \
  PREPROCESS_WORKERS="${HOMOGENEOUS_PREPROCESS_WORKERS}" \
  PREFETCH_FACTOR="${HOMOGENEOUS_PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${HOMOGENEOUS_PERSISTENT_WORKERS}" \
  PIN_MEMORY="${HOMOGENEOUS_PIN_MEMORY}" \
  ENERGY_BIAS_WEIGHT="${HOMOGENEOUS_ENERGY_BIAS_WEIGHT}" \
  ENERGY_PARTICLE_BIAS_WEIGHT="${HOMOGENEOUS_ENERGY_PARTICLE_BIAS_WEIGHT}" \
  "${SCRIPT_DIR}/submit_server_homogeneous_from_hetero_training.sh"

  GRAPH_INPUT="${GRAPH_INPUT}" \
  RUN_ID="hetero_${HETERO_MODEL_ARCHITECTURE}_allsrc_${RUN_ID}" \
  MODEL_ARCHITECTURE="${HETERO_MODEL_ARCHITECTURE}" \
  WAVEFORM_ENCODER=cnn-gru \
  PULSE_WAVEFORM_ENCODER=crop_cnn \
  USE_PULSE_PARENT_WAVEFORM=1 \
  USE_PULSE_BOUNDS=1 \
  USE_RELATIVE_POSITIONS=1 \
  DETECTOR_READOUT_MASK=ising_kept \
  PULSE_READOUT_MASK=ising_kept \
  HETERO_RELATION_PRESET="${HETERO_RELATION_PRESET}" \
  SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE}" \
  SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
  SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
  BATCH_SIZE="${HETERO_BATCH_SIZE:-64}" \
  GRADIENT_ACCUMULATION_STEPS="${HETERO_GRADIENT_ACCUMULATION_STEPS:-2}" \
  AMP=fp16 \
  TRAIN_WORKERS="${HETERO_TRAIN_WORKERS:-4}" \
  PERSISTENT_WORKERS=1 \
  PREFETCH_FACTOR=1 \
  PIN_MEMORY=0 \
  ENERGY_BIAS_WEIGHT="${HETERO_ENERGY_BIAS_WEIGHT}" \
  ENERGY_PARTICLE_BIAS_WEIGHT="${HETERO_ENERGY_PARTICLE_BIAS_WEIGHT}" \
  PREPARE_FAST_CACHE=0 \
  FEATURE_IMPORTANCE=1 \
  FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation test}" \
  ATTENTION_MAPS="${ATTENTION_MAPS:-0}" \
  DIAGNOSTICS=1 \
  DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-100}" \
  PROFILE=1 \
  PARTITION="${PARTITION}" \
  DRY_RUN=1 \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  LOCAL_RUNTIME_CACHE="${LOCAL_RUNTIME_CACHE}" \
  "${SCRIPT_DIR}/submit_server_hetero_reco_mass_quality_training.sh"
  exit 0
fi

h5_submit_output="$(
  RUN_NAME="${H5_RUN_NAME}" \
  GRAPHS_PER_SOURCE_GROUP="${GRAPHS_PER_SOURCE_GROUP}" \
  SOURCE_GROUP_SELECTION="${SOURCE_GROUP_SELECTION}" \
  ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA}" \
  REFILL_MIN_GRAPHS_PER_SOURCE_GROUP="${REFILL_MIN_GRAPHS_PER_SOURCE_GROUP}" \
  MAX_REFILL_SOURCE_GROUPS_PER_STRATUM="${MAX_REFILL_SOURCE_GROUPS_PER_STRATUM}" \
  EXPORT_WORKERS="${H5_EXPORT_WORKERS}" \
  CPUS_PER_TASK="${H5_CPUS_PER_TASK}" \
  MEM="${H5_MEM}" \
  TIME_LIMIT="${H5_TIME_LIMIT}" \
  PARTITION="${H5_PARTITION}" \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS}" \
  SBATCH_PARSABLE=1 \
  "${SCRIPT_DIR}/submit_server_hetero_light_graph_export.sh"
)"
printf "%s\n" "${h5_submit_output}"
h5_job_id="$(printf "%s\n" "${h5_submit_output}" | awk '/^[0-9]+(;|$)/ {sub(/;.*/, "", $0); value=$0} END {print value}')"
if [[ -z "${h5_job_id}" || ! "${h5_job_id}" =~ ^[0-9]+$ ]]; then
  echo "failed to parse H5 export job id from submit output" >&2
  exit 1
fi
h5_dependency="afterok:${h5_job_id}"
status "h5_export_job_id=${h5_job_id}"

homogeneous_submit_output="$(
  HETERO_GRAPH_INPUT="${GRAPH_INPUT}" \
  RUN_ID="homogeneous_from_allsrc_${RUN_ID}" \
  PULSE_MASK="${PULSE_MASK}" \
  CONVERT_WORKERS="${CONVERT_WORKERS:-32}" \
  CONVERT_CPUS_PER_TASK="${CONVERT_CPUS_PER_TASK:-${CONVERT_WORKERS:-32}}" \
  SBATCH_DEPENDENCY="${h5_dependency}" \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  LOCAL_RUNTIME_CACHE="${LOCAL_RUNTIME_CACHE}" \
  PARTITION="${PARTITION}" \
  SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE}" \
  SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
  SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}" \
  MODEL_ARCHITECTURE=physics \
  WAVEFORM_ENCODER=cnn-gru \
  BATCH_SIZE="${HOMOGENEOUS_BATCH_SIZE}" \
  TRAIN_WORKERS="${HOMOGENEOUS_TRAIN_WORKERS}" \
  PREPROCESS_WORKERS="${HOMOGENEOUS_PREPROCESS_WORKERS}" \
  PREFETCH_FACTOR="${HOMOGENEOUS_PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${HOMOGENEOUS_PERSISTENT_WORKERS}" \
  PIN_MEMORY="${HOMOGENEOUS_PIN_MEMORY}" \
  LOSS_MODE=physics \
  MASS_CLASSIFICATION=1 \
  QUALITY_PREDICTION=1 \
  ERROR_PREDICTION=0 \
  ENERGY_BIAS_WEIGHT="${HOMOGENEOUS_ENERGY_BIAS_WEIGHT}" \
  ENERGY_PARTICLE_BIAS_WEIGHT="${HOMOGENEOUS_ENERGY_PARTICLE_BIAS_WEIGHT}" \
  FEATURE_IMPORTANCE=1 \
  FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation test}" \
  BEST_DIAGNOSTICS=1 \
  DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-100}" \
  "${SCRIPT_DIR}/submit_server_homogeneous_from_hetero_training.sh"
)"
printf "%s\n" "${homogeneous_submit_output}"
homogeneous_conversion_job_id="$(printf "%s\n" "${homogeneous_submit_output}" | awk -F= '/^conversion_job_id=/ {value=$2} END {print value}')"
homogeneous_training_job_id="$(printf "%s\n" "${homogeneous_submit_output}" | awk '/^Submitted batch job / {ids[++n]=$4} END {print ids[2]}')"

heterogeneous_submit_output="$(
  GRAPH_INPUT="${GRAPH_INPUT}" \
  RUN_ID="hetero_${HETERO_MODEL_ARCHITECTURE}_allsrc_${RUN_ID}" \
  SBATCH_DEPENDENCY="${h5_dependency}" \
  PARTITION="${PARTITION}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}" \
  MODEL_ARCHITECTURE="${HETERO_MODEL_ARCHITECTURE}" \
  WAVEFORM_ENCODER=cnn-gru \
  PULSE_WAVEFORM_ENCODER=crop_cnn \
  USE_PULSE_PARENT_WAVEFORM=1 \
  USE_PULSE_BOUNDS=1 \
  USE_RELATIVE_POSITIONS=1 \
  DETECTOR_READOUT_MASK=ising_kept \
  PULSE_READOUT_MASK=ising_kept \
  HETERO_RELATION_PRESET="${HETERO_RELATION_PRESET}" \
  SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE}" \
  SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION}" \
  SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION}" \
  CORE_TARGET_MODE=signal_bary_relative \
  COORDINATE_FEATURE_MODE=relative_only \
  HETERO_TRAINING_DATA_FORMAT=fast_tensor \
  FINAL_EVAL_DATA_FORMAT=fast_tensor \
  BATCH_SIZE="${HETERO_BATCH_SIZE:-64}" \
  GRADIENT_ACCUMULATION_STEPS="${HETERO_GRADIENT_ACCUMULATION_STEPS:-2}" \
  AMP=fp16 \
  TRAIN_WORKERS="${HETERO_TRAIN_WORKERS:-4}" \
  PERSISTENT_WORKERS=1 \
  PREFETCH_FACTOR=1 \
  PIN_MEMORY=0 \
  ENERGY_BIAS_WEIGHT="${HETERO_ENERGY_BIAS_WEIGHT}" \
  ENERGY_PARTICLE_BIAS_WEIGHT="${HETERO_ENERGY_PARTICLE_BIAS_WEIGHT}" \
  PREPARE_FAST_CACHE=0 \
  FEATURE_IMPORTANCE=1 \
  FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation test}" \
  ATTENTION_MAPS="${ATTENTION_MAPS:-0}" \
  ATTENTION_MAPS_SPLIT="${ATTENTION_MAPS_SPLIT:-validation test}" \
  DIAGNOSTICS=1 \
  DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-100}" \
  PROFILE=1 \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  LOCAL_RUNTIME_CACHE="${LOCAL_RUNTIME_CACHE}" \
  "${SCRIPT_DIR}/submit_server_hetero_reco_mass_quality_training.sh"
)"
printf "%s\n" "${heterogeneous_submit_output}"
heterogeneous_training_job_id="$(printf "%s\n" "${heterogeneous_submit_output}" | awk '/^Submitted batch job / {value=$4} END {print value}')"

cat <<EOF
======================================================================
ALL-SOURCE COMPARISON SUBMITTED
h5_export_job_id: ${h5_job_id}
h5_graph_input: ${GRAPH_INPUT}
h5_partition: ${H5_PARTITION}
training_partition: ${PARTITION}
source_fraction_mode: ${SOURCE_FRACTION_MODE}
source_val_fraction: ${SOURCE_VAL_FRACTION}
source_test_fraction: ${SOURCE_TEST_FRACTION}
heterogeneous_energy_bias_weight: ${HETERO_ENERGY_BIAS_WEIGHT}
heterogeneous_energy_particle_bias_weight: ${HETERO_ENERGY_PARTICLE_BIAS_WEIGHT}
homogeneous_conversion_job_id: ${homogeneous_conversion_job_id:-unknown}
homogeneous_training_job_id: ${homogeneous_training_job_id:-unknown}
homogeneous_conversion_dependency: ${h5_dependency}
homogeneous_training_dependency: afterok:${homogeneous_conversion_job_id:-unknown}
heterogeneous_training_job_id: ${heterogeneous_training_job_id:-unknown}
heterogeneous_dependency: ${h5_dependency}
heterogeneous_selection_basis: provisional next setting from synced validation loss/milestones and failure modes; final choice requires validation/test on this new all-source H5
heterogeneous_model: ${HETERO_MODEL_ARCHITECTURE} + cnn-gru + crop_cnn + ising-kept masks + ${HETERO_RELATION_PRESET} relations
homogeneous_model: physics + cnn-gru from ising-kept homogeneous conversion
======================================================================
EOF
