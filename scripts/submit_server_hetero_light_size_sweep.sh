#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_hetero_light_size_sweep.sh: starting"

RUN_ID="${RUN_ID:-hetero_light_size_$(date +%Y%m%d_%H%M%S)}"
LIGHT_TARGETS="${LIGHT_TARGETS:-50000 100000}"
SOURCE_GROUPS_PER_STRATUM="${SOURCE_GROUPS_PER_STRATUM:-298}"
SOURCE_GROUP_SELECTION="${SOURCE_GROUP_SELECTION:-all}"
ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA:-1}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS:-0}"

if ! [[ "${SOURCE_GROUPS_PER_STRATUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SOURCE_GROUPS_PER_STRATUM must be a positive integer used to derive graphs_per_source_group: ${SOURCE_GROUPS_PER_STRATUM}" >&2
  exit 2
fi
if [[ "${SOURCE_GROUP_SELECTION}" != "balanced_min" && "${SOURCE_GROUP_SELECTION}" != "all" ]]; then
  echo "SOURCE_GROUP_SELECTION must be balanced_min or all: ${SOURCE_GROUP_SELECTION}" >&2
  exit 2
fi

status "HETERO LIGHT SIZE SWEEP"
status "run_id: ${RUN_ID}"
status "light_targets: ${LIGHT_TARGETS}"
status "source_groups_per_stratum_for_target: ${SOURCE_GROUPS_PER_STRATUM}"
status "source_group_selection: ${SOURCE_GROUP_SELECTION}"
status "allow_underfull_strata: ${ALLOW_UNDERFULL_STRATA}"
status "run_uv_sync: ${RUN_UV_SYNC}"
status "make_input_distributions: ${MAKE_INPUT_DISTRIBUTIONS}"

for target in ${LIGHT_TARGETS}; do
  if ! [[ "${target}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIGHT_TARGETS contains a non-positive integer: ${target}" >&2
    exit 2
  fi
  graphs_per_source_group=$(( (target + SOURCE_GROUPS_PER_STRATUM - 1) / SOURCE_GROUPS_PER_STRATUM ))
  implied_target=$(( graphs_per_source_group * SOURCE_GROUPS_PER_STRATUM ))
  if [[ "${SOURCE_GROUP_SELECTION}" == "all" ]]; then
    run_name="hetero_light_allsrc_target${target}_${RUN_ID}"
  else
    run_name="hetero_light_target${target}_${RUN_ID}"
  fi
  status "submit target=${target} graphs_per_source_group=${graphs_per_source_group} implied_target_at_reference_sources=${implied_target} source_group_selection=${SOURCE_GROUP_SELECTION} run_name=${run_name}"
  RUN_NAME="${run_name}" \
  RUN_ID="${RUN_ID}" \
  GRAPHS_PER_SOURCE_GROUP="${graphs_per_source_group}" \
  SOURCE_GROUP_SELECTION="${SOURCE_GROUP_SELECTION}" \
  ALLOW_UNDERFULL_STRATA="${ALLOW_UNDERFULL_STRATA}" \
  RUN_UV_SYNC="${RUN_UV_SYNC}" \
  MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS}" \
  "${SCRIPT_DIR}/submit_server_hetero_light_graph_export.sh"
done
