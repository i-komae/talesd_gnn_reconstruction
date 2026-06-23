#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_hetero_source_balanced_size_sweep.sh: starting"

RUN_ID="${RUN_ID:-hetero_source_balanced_size_$(date +%Y%m%d_%H%M%S)}"
TARGETS="${TARGETS:-50000 100000}"
TARGET_REFERENCE_SOURCE_GROUPS="${TARGET_REFERENCE_SOURCE_GROUPS:-${SOURCE_GROUPS_PER_STRATUM:-298}}"
SOURCE_GROUP_SELECTION="${SOURCE_GROUP_SELECTION:-all}"
if [[ "${SOURCE_GROUP_SELECTION}" == "all" ]]; then
  MAX_SOURCE_GROUPS_PER_STRATUM="${MAX_SOURCE_GROUPS_PER_STRATUM:-}"
else
  MAX_SOURCE_GROUPS_PER_STRATUM="${MAX_SOURCE_GROUPS_PER_STRATUM:-}"
fi
ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA:-1}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS:-0}"

if ! [[ "${TARGET_REFERENCE_SOURCE_GROUPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_REFERENCE_SOURCE_GROUPS must be a positive integer used only to derive graphs_per_source_group: ${TARGET_REFERENCE_SOURCE_GROUPS}" >&2
  exit 2
fi
if [[ "${SOURCE_GROUP_SELECTION}" != "balanced_min" && "${SOURCE_GROUP_SELECTION}" != "all" ]]; then
  echo "SOURCE_GROUP_SELECTION must be balanced_min or all: ${SOURCE_GROUP_SELECTION}" >&2
  exit 2
fi
if [[ -n "${MAX_SOURCE_GROUPS_PER_STRATUM}" ]] && ! [[ "${MAX_SOURCE_GROUPS_PER_STRATUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SOURCE_GROUPS_PER_STRATUM must be empty or a positive integer: ${MAX_SOURCE_GROUPS_PER_STRATUM}" >&2
  exit 2
fi
if [[ "${SOURCE_GROUP_SELECTION}" == "all" && -n "${MAX_SOURCE_GROUPS_PER_STRATUM}" ]]; then
  echo "MAX_SOURCE_GROUPS_PER_STRATUM cannot be used with SOURCE_GROUP_SELECTION=all" >&2
  exit 2
fi

status "HETERO SOURCE-BALANCED SIZE SWEEP"
status "run_id: ${RUN_ID}"
status "targets: ${TARGETS}"
status "target_reference_source_groups: ${TARGET_REFERENCE_SOURCE_GROUPS}"
status "target_usage: used only to compute graphs_per_source_group; source groups are not capped unless MAX_SOURCE_GROUPS_PER_STRATUM is set"
status "source_group_selection: ${SOURCE_GROUP_SELECTION}"
status "max_source_groups_per_stratum: ${MAX_SOURCE_GROUPS_PER_STRATUM}"
status "allow_underfull_strata: ${ALLOW_UNDERFULL_STRATA}"
status "run_uv_sync: ${RUN_UV_SYNC}"
status "make_input_distributions: ${MAKE_INPUT_DISTRIBUTIONS}"

for target in ${TARGETS}; do
  if ! [[ "${target}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGETS contains a non-positive integer: ${target}" >&2
    exit 2
  fi
  graphs_per_source_group=$(( (target + TARGET_REFERENCE_SOURCE_GROUPS - 1) / TARGET_REFERENCE_SOURCE_GROUPS ))
  implied_target=$(( graphs_per_source_group * TARGET_REFERENCE_SOURCE_GROUPS ))
  if [[ "${SOURCE_GROUP_SELECTION}" == "all" ]]; then
    run_name="hetero_source_balanced_all_target${target}_${RUN_ID}"
  else
    run_name="hetero_source_balanced_min_target${target}_${RUN_ID}"
  fi
  status "submit target=${target} graphs_per_source_group=${graphs_per_source_group} implied_target_at_reference_sources=${implied_target} source_group_selection=${SOURCE_GROUP_SELECTION} run_name=${run_name}"
  RUN_NAME="${run_name}" \
  RUN_ID="${RUN_ID}" \
  GRAPHS_PER_SOURCE_GROUP="${graphs_per_source_group}" \
  SOURCE_GROUP_SELECTION="${SOURCE_GROUP_SELECTION}" \
  MAX_SOURCE_GROUPS_PER_STRATUM="${MAX_SOURCE_GROUPS_PER_STRATUM}" \
  ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA}" \
  REFILL_MIN_GRAPHS_PER_SOURCE_GROUP="${graphs_per_source_group}" \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS}" \
  "${SCRIPT_DIR}/submit_server_hetero_source_balanced_graph_export.sh"
done
