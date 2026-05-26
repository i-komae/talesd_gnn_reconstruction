#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  scripts/submit_server_small_graph_dataset.sh

Submit small HDF5 graph dataset creation to Slurm.

Main environment overrides:
  GRAPH_INPUT=/path/to/large_graph_dir_or_h5
  RUN_NAME=small_energyflat2000_YYYYMMDD_HHMMSS
  GRAPH_OUTPUT=/path/to/output.h5
  PER_BIN=2000
  EXTRA_PER_BINS=500,200
  DERIVE_ONLY=1
  MAX_TOTAL=50000
  PARTICLE_FILTER=all|proton|iron
  SCAN_WORKERS=auto
  WRITE_WORKERS=auto
  OUTPUT_SHARDS=auto
  TARGET_EVENTS_PER_SHARD=20000
  PROGRESS_INTERVAL=30
  PARTITION=auto
  CPUS_PER_TASK=auto
  MEM=auto
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

q() {
  printf "%q" "$1"
}

status() {
  printf "%s\n" "$*" >&2
}

split_csv() {
  local value="$1"
  value="${value//,/ }"
  printf "%s\n" ${value}
}

is_usable_node_state() {
  local state="$1"
  case "${state}" in
    *DOWN*|*DRAIN*|*FAIL*|*MAINT*|*POWER*|*NOT_RESPONDING*|*UNKNOWN*)
      return 1
      ;;
  esac
  return 0
}

count_input_shards() {
  local count=""
  if [[ -x "${REPO}/.venv/bin/python" ]]; then
    count="$(
      cd "${REPO}" && .venv/bin/python - "${GRAPH_INPUT}" <<'PY' 2>/dev/null || true
import sys
from talesd_gnn_reconstruction.cli import _expand_h5_graph_paths

try:
    print(len(_expand_h5_graph_paths(sys.argv[1:])))
except Exception:
    print(0)
PY
    )"
  fi
  if [[ "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${count}"
    return
  fi
  if [[ -d "${GRAPH_INPUT}" ]]; then
    find "${GRAPH_INPUT}" -maxdepth 1 -type f -name '*.h5' | wc -l | tr -d ' '
  elif [[ -f "${GRAPH_INPUT}" && "${GRAPH_INPUT}" == *.h5 ]]; then
    echo 1
  else
    echo 0
  fi
}

select_cpu_resources() {
  local report_path="$1"
  local best_partition=""
  local best_node=""
  local best_free_cpu=-1
  local best_free_mem=-1
  local best_request_cpu=-1
  local best_request_mem=-1
  local part node node_info parsed cpu_alloc cpu_eff cpu_tot real_mem alloc_mem free_cpu free_mem state
  local request_cpu request_mem mem_by_cpu

  if ! command -v sinfo >/dev/null 2>&1 || ! command -v scontrol >/dev/null 2>&1; then
    echo "sinfo and scontrol are required for AUTO_RESOURCES=1." >&2
    return 2
  fi

  status "Scanning CPU resources: partitions=${PARTITION}"
  : > "${report_path}"
  printf "partition\tnode\tstate\tfree_cpu\tcpu_effective\tcpu_total\tcpu_alloc\tfree_mem_mb\treal_mem_mb\talloc_mem_mb\trequest_cpu\trequest_mem_mb\n" >> "${report_path}"

  for part in $(split_csv "${PARTITION}" | sed '/^auto$/d'); do
    [[ -n "${part}" ]] || continue
    while IFS= read -r node; do
      [[ -n "${node}" ]] || continue
      node_info="$(scontrol show node "${node}")"
      parsed="$(
        printf "%s\n" "${node_info}" | awk '
          {
            for (i = 1; i <= NF; i++) {
              split($i, f, "=")
              if (f[1] == "CPUAlloc") cpu_alloc = f[2]
              if (f[1] == "CPUEfctv") cpu_eff = f[2]
              if (f[1] == "CPUTot") cpu_tot = f[2]
              if (f[1] == "RealMemory") real_mem = f[2]
              if (f[1] == "AllocMem") alloc_mem = f[2]
              if (f[1] == "State") state = f[2]
            }
          }
          END {
            if (cpu_alloc == "") cpu_alloc = 0
            if (cpu_eff == "") cpu_eff = 0
            if (cpu_tot == "") cpu_tot = 0
            if (cpu_eff <= 0 || (cpu_tot > 0 && cpu_eff > cpu_tot)) cpu_eff = cpu_tot
            if (real_mem == "") real_mem = 0
            if (alloc_mem == "") alloc_mem = 0
            if (state == "") state = "unknown"
            print cpu_alloc, cpu_eff, cpu_tot, real_mem, alloc_mem, state
          }
        '
      )"
      read -r cpu_alloc cpu_eff cpu_tot real_mem alloc_mem state <<< "${parsed}"
      free_cpu=$((cpu_eff - cpu_alloc))
      free_mem=$((real_mem - alloc_mem))
      if (( free_cpu < 0 )); then
        free_cpu=0
      fi
      if (( free_mem < 0 )); then
        free_mem=0
      fi

      request_cpu="${free_cpu}"
      if [[ "${AUTO_CPU_FRACTION_PERCENT}" =~ ^[0-9]+$ ]] && (( AUTO_CPU_FRACTION_PERCENT > 0 && AUTO_CPU_FRACTION_PERCENT < 100 )); then
        request_cpu=$((free_cpu * AUTO_CPU_FRACTION_PERCENT / 100))
        if (( free_cpu > 0 && request_cpu < 1 )); then
          request_cpu=1
        fi
      fi
      if (( AUTO_MAX_CPUS_EFFECTIVE > 0 && request_cpu > AUTO_MAX_CPUS_EFFECTIVE )); then
        request_cpu="${AUTO_MAX_CPUS_EFFECTIVE}"
      fi
      if (( request_cpu > free_cpu )); then
        request_cpu="${free_cpu}"
      fi

      request_mem="${free_mem}"
      request_mem=$((request_mem - AUTO_MEM_RESERVE_MB))
      if (( request_mem < 0 )); then
        request_mem=0
      fi
      if (( AUTO_MEM_PER_CPU_MB > 0 && request_cpu > 0 )); then
        mem_by_cpu=$((request_cpu * AUTO_MEM_PER_CPU_MB))
        if (( request_mem > mem_by_cpu )); then
          request_mem="${mem_by_cpu}"
        fi
      fi
      if (( AUTO_MAX_MEM_MB > 0 && request_mem > AUTO_MAX_MEM_MB )); then
        request_mem="${AUTO_MAX_MEM_MB}"
      fi

      printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n" \
        "${part}" "${node}" "${state}" "${free_cpu}" "${cpu_eff}" "${cpu_tot}" "${cpu_alloc}" \
        "${free_mem}" "${real_mem}" "${alloc_mem}" "${request_cpu}" "${request_mem}" >> "${report_path}"
      status "  ${part} ${node}: state=${state} free_cpu=${free_cpu} cpu_effective=${cpu_eff} cpu_total=${cpu_tot} free_mem=${free_mem}M request_cpu=${request_cpu} request_mem=${request_mem}M"
      if ! is_usable_node_state "${state}"; then
        status "    skipped: node state is not usable for new jobs"
        continue
      fi
      if (( request_cpu <= 0 || request_mem <= 0 )); then
        status "    skipped: capped request would be empty"
        continue
      fi
      if (( request_cpu > best_request_cpu || (request_cpu == best_request_cpu && request_mem > best_request_mem) )); then
        best_partition="${part}"
        best_node="${node}"
        best_free_cpu="${free_cpu}"
        best_free_mem="${free_mem}"
        best_request_cpu="${request_cpu}"
        best_request_mem="${request_mem}"
      fi
    done < <(sinfo -h -p "${part}" -N -o "%N" 2>/dev/null | sort -u)
  done

  if [[ -z "${best_partition}" || -z "${best_node}" || "${best_request_cpu}" -le 0 || "${best_request_mem}" -le 0 ]]; then
    echo "No usable free CPU node was found in candidate partitions: ${PARTITION}" >&2
    echo "Resource scan:" >&2
    cat "${report_path}" >&2
    return 2
  fi

  PARTITION="${best_partition}"
  if [[ "${PIN_NODE}" == "1" ]]; then
    NODELIST="${best_node}"
  fi
  RESOURCE_SIZING_NODE="${best_node}"
  CPUS_PER_TASK="${best_request_cpu}"
  MEM="${best_request_mem}M"
  status "Selected CPU resource: partition=${PARTITION} node=${NODELIST:-any} cpus_per_task=${CPUS_PER_TASK} mem=${MEM} free_cpu=${best_free_cpu} free_mem=${best_free_mem}M caps=max_cpus=${AUTO_MAX_CPUS_EFFECTIVE} cpu_fraction=${AUTO_CPU_FRACTION_PERCENT}% mem_per_cpu=${AUTO_MEM_PER_CPU_MB}M max_mem=${AUTO_MAX_MEM_MB}M reserve=${AUTO_MEM_RESERVE_MB}M"
}

REPO="${REPO:-/ceph/sharedfs/work/SATORI/ikomae/src/talesd_gnn_reconstruction}"
DEFAULT_GRAPH_INPUT="${DEFAULT_GRAPH_INPUT:-/dicos_ui_home/ikomae/work/gnn/graphs/server_graph_export_energyflat200000_20260524_075508}"
GRAPH_INPUT="${GRAPH_INPUT:-${DEFAULT_GRAPH_INPUT}}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

PER_BIN="${PER_BIN:-2000}"
EXTRA_PER_BINS="${EXTRA_PER_BINS:-500,200}"
DERIVE_ONLY="${DERIVE_ONLY:-0}"
MAX_TOTAL="${MAX_TOTAL:-}"
ENERGY_BIN_WIDTH="${ENERGY_BIN_WIDTH:-0.1}"
STRATIFY_PARTICLE="${STRATIFY_PARTICLE:-1}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
SEED="${SEED:-12345}"
OVERWRITE="${OVERWRITE:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

CPU_PARTITIONS="${CPU_PARTITIONS:-edr1-al9_large,edr2-al9_large}"
AUTO_RESOURCES="${AUTO_RESOURCES:-1}"
PIN_NODE="${PIN_NODE:-1}"
PARTITION="${PARTITION:-auto}"
NODELIST="${NODELIST:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-auto}"
SCAN_WORKERS="${SCAN_WORKERS:-auto}"
WRITE_WORKERS="${WRITE_WORKERS:-auto}"
OUTPUT_SHARDS="${OUTPUT_SHARDS:-auto}"
TARGET_EVENTS_PER_SHARD="${TARGET_EVENTS_PER_SHARD:-20000}"
MEM="${MEM:-auto}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
RUN_NAME="${RUN_NAME:-small_graph_energyflat${PER_BIN}_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${GRAPH_ROOT}/${RUN_NAME}}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}/${RUN_NAME}.h5}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_MEM_RESERVE_MB="${AUTO_MEM_RESERVE_MB:-32768}"
AUTO_MAX_CPUS="${AUTO_MAX_CPUS:-0}"
AUTO_CPU_FRACTION_PERCENT="${AUTO_CPU_FRACTION_PERCENT:-75}"
AUTO_MEM_PER_CPU_MB="${AUTO_MEM_PER_CPU_MB:-4096}"
AUTO_MAX_MEM_MB="${AUTO_MAX_MEM_MB:-0}"
RESOURCE_SIZING_NODE="${RESOURCE_SIZING_NODE:-}"

case "${STRATIFY_PARTICLE}" in
  0|1) ;;
  *)
    echo "STRATIFY_PARTICLE must be 0 or 1: ${STRATIFY_PARTICLE}" >&2
    exit 2
    ;;
esac
case "${OVERWRITE}" in
  0|1) ;;
  *)
    echo "OVERWRITE must be 0 or 1: ${OVERWRITE}" >&2
    exit 2
    ;;
esac
case "${SHOW_PROGRESS}" in
  0|1) ;;
  *)
    echo "SHOW_PROGRESS must be 0 or 1: ${SHOW_PROGRESS}" >&2
    exit 2
    ;;
esac
case "${DERIVE_ONLY}" in
  0|1) ;;
  *)
    echo "DERIVE_ONLY must be 0 or 1: ${DERIVE_ONLY}" >&2
    exit 2
    ;;
esac
if ! [[ "${PROGRESS_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROGRESS_INTERVAL must be a positive integer: ${PROGRESS_INTERVAL}" >&2
  exit 2
fi
case "${PARTICLE_FILTER}" in
  all|proton|iron) ;;
  *)
    echo "PARTICLE_FILTER must be all, proton, or iron: ${PARTICLE_FILTER}" >&2
    exit 2
    ;;
esac
if ! [[ "${PER_BIN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_BIN must be a positive integer: ${PER_BIN}" >&2
  exit 2
fi
if [[ -n "${EXTRA_PER_BINS}" ]]; then
  for extra_per_bin in $(split_csv "${EXTRA_PER_BINS}"); do
    if ! [[ "${extra_per_bin}" =~ ^[1-9][0-9]*$ ]]; then
      echo "EXTRA_PER_BINS must be a comma-separated list of positive integers: ${EXTRA_PER_BINS}" >&2
      exit 2
    fi
    if (( extra_per_bin >= PER_BIN )); then
      echo "EXTRA_PER_BINS values must be smaller than PER_BIN (${extra_per_bin} >= ${PER_BIN})" >&2
      exit 2
    fi
  done
fi
case "${AUTO_RESOURCES}" in
  0|1) ;;
  *)
    echo "AUTO_RESOURCES must be 0 or 1: ${AUTO_RESOURCES}" >&2
    exit 2
    ;;
esac
case "${PIN_NODE}" in
  0|1) ;;
  *)
    echo "PIN_NODE must be 0 or 1: ${PIN_NODE}" >&2
    exit 2
    ;;
esac
for auto_value_name in AUTO_MEM_RESERVE_MB AUTO_MAX_CPUS AUTO_CPU_FRACTION_PERCENT AUTO_MEM_PER_CPU_MB AUTO_MAX_MEM_MB; do
  auto_value="${!auto_value_name}"
  if ! [[ "${auto_value}" =~ ^[0-9]+$ ]]; then
    echo "${auto_value_name} must be a non-negative integer: ${auto_value}" >&2
    exit 2
  fi
done
if [[ -n "${MAX_TOTAL}" ]] && ! [[ "${MAX_TOTAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOTAL must be empty or a positive integer: ${MAX_TOTAL}" >&2
  exit 2
fi
if ! [[ "${TARGET_EVENTS_PER_SHARD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_EVENTS_PER_SHARD must be a positive integer: ${TARGET_EVENTS_PER_SHARD}" >&2
  exit 2
fi
if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi
if [[ ! -e "${GRAPH_INPUT}" ]]; then
  echo "graph input not found: ${GRAPH_INPUT}" >&2
  echo "Set GRAPH_INPUT=/path/to/large_graph_dir_or_h5 to override the default." >&2
  exit 2
fi
if [[ "${DERIVE_ONLY}" != "1" && -e "${GRAPH_OUTPUT}" && "${OVERWRITE}" != "1" ]]; then
  echo "graph output already exists: ${GRAPH_OUTPUT}" >&2
  echo "Set OVERWRITE=1 or choose a different RUN_NAME/GRAPH_OUTPUT." >&2
  exit 2
fi

INPUT_SHARD_COUNT="${INPUT_SHARD_COUNT:-$(count_input_shards)}"
if ! [[ "${INPUT_SHARD_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "INPUT_SHARD_COUNT must be a non-negative integer: ${INPUT_SHARD_COUNT}" >&2
  exit 2
fi
if (( INPUT_SHARD_COUNT <= 0 )); then
  echo "no input HDF5 shards matched GRAPH_INPUT: ${GRAPH_INPUT}" >&2
  exit 2
fi
AUTO_MAX_CPUS_EFFECTIVE="${AUTO_MAX_CPUS}"
if (( AUTO_MAX_CPUS_EFFECTIVE == 0 || AUTO_MAX_CPUS_EFFECTIVE > INPUT_SHARD_COUNT )); then
  AUTO_MAX_CPUS_EFFECTIVE="${INPUT_SHARD_COUNT}"
fi

SBATCH_DIR="${RUN_DIR}/slurm"
SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
LOG_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
mkdir -p "${SBATCH_DIR}" "${SLURM_LOG_DIR}" "${SUMMARY_DIR}" "${LOG_DIR}" "${CONFIG_DIR}" "${GRAPH_RUN_DIR}"

RESOURCE_REPORT="${CONFIG_DIR}/resource_selection.tsv"
if [[ "${AUTO_RESOURCES}" == "1" ]]; then
  if [[ "${PARTITION}" == "auto" ]]; then
    PARTITION="${CPU_PARTITIONS}"
  fi
  status "AUTO_RESOURCES=1: selecting CPU node before writing sbatch"
  select_cpu_resources "${RESOURCE_REPORT}"
else
  if [[ "${PARTITION}" == "auto" || -z "${PARTITION}" || "${CPUS_PER_TASK}" == "auto" || "${MEM}" == "auto" ]]; then
    cat >&2 <<EOF
AUTO_RESOURCES=0 requires explicit PARTITION, CPUS_PER_TASK, and MEM.
Example:
  AUTO_RESOURCES=0 PARTITION=<partition> CPUS_PER_TASK=<cpus> MEM=<memory> scripts/submit_server_small_graph_dataset.sh
EOF
    exit 2
  fi
  {
    printf "partition\tnode\tstate\tfree_cpu\tcpu_effective\tcpu_total\tcpu_alloc\tfree_mem_mb\treal_mem_mb\talloc_mem_mb\n"
    printf "%s\t%s\tmanual\t0\t0\t0\t0\t0\t0\t0\n" "${PARTITION}" "${NODELIST:-}"
  } > "${RESOURCE_REPORT}"
fi

if [[ "${SCAN_WORKERS}" == "auto" ]]; then
  SCAN_WORKERS="${CPUS_PER_TASK}"
  if (( SCAN_WORKERS > INPUT_SHARD_COUNT )); then
    SCAN_WORKERS="${INPUT_SHARD_COUNT}"
  fi
elif ! [[ "${SCAN_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SCAN_WORKERS must be auto or a positive integer: ${SCAN_WORKERS}" >&2
  exit 2
fi
if ! [[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPUS_PER_TASK must resolve to a positive integer: ${CPUS_PER_TASK}" >&2
  exit 2
fi
if (( SCAN_WORKERS > CPUS_PER_TASK )); then
  echo "SCAN_WORKERS must be <= CPUS_PER_TASK (${SCAN_WORKERS} > ${CPUS_PER_TASK})" >&2
  exit 2
fi
if [[ "${WRITE_WORKERS}" == "auto" ]]; then
  WRITE_WORKERS="${SCAN_WORKERS}"
elif ! [[ "${WRITE_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WRITE_WORKERS must be auto or a positive integer: ${WRITE_WORKERS}" >&2
  exit 2
fi
if (( WRITE_WORKERS > CPUS_PER_TASK )); then
  echo "WRITE_WORKERS must be <= CPUS_PER_TASK (${WRITE_WORKERS} > ${CPUS_PER_TASK})" >&2
  exit 2
fi
if [[ "${OUTPUT_SHARDS}" == "auto" ]]; then
  OUTPUT_SHARDS_ARG=0
elif ! [[ "${OUTPUT_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OUTPUT_SHARDS must be auto or a positive integer: ${OUTPUT_SHARDS}" >&2
  exit 2
else
  OUTPUT_SHARDS_ARG="${OUTPUT_SHARDS}"
fi

SBATCH_FILE="${SBATCH_DIR}/${RUN_NAME}.sbatch"
SBATCH_NODELIST_LINE=""
if [[ -n "${NODELIST}" ]]; then
  SBATCH_NODELIST_LINE="#SBATCH --nodelist=${NODELIST}"
fi

cat > "${CONFIG_DIR}/small_graph_dataset.env" <<EOF
REPO=${REPO}
GRAPH_INPUT=${GRAPH_INPUT}
GRAPH_OUTPUT=${GRAPH_OUTPUT}
PER_BIN=${PER_BIN}
EXTRA_PER_BINS=${EXTRA_PER_BINS}
DERIVE_ONLY=${DERIVE_ONLY}
MAX_TOTAL=${MAX_TOTAL}
ENERGY_BIN_WIDTH=${ENERGY_BIN_WIDTH}
STRATIFY_PARTICLE=${STRATIFY_PARTICLE}
PARTICLE_FILTER=${PARTICLE_FILTER}
SEED=${SEED}
SHOW_PROGRESS=${SHOW_PROGRESS}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL}
SCAN_WORKERS=${SCAN_WORKERS}
WRITE_WORKERS=${WRITE_WORKERS}
OUTPUT_SHARDS=${OUTPUT_SHARDS}
OUTPUT_SHARDS_ARG=${OUTPUT_SHARDS_ARG}
TARGET_EVENTS_PER_SHARD=${TARGET_EVENTS_PER_SHARD}
INPUT_SHARD_COUNT=${INPUT_SHARD_COUNT}
AUTO_RESOURCES=${AUTO_RESOURCES}
AUTO_MAX_CPUS_EFFECTIVE=${AUTO_MAX_CPUS_EFFECTIVE}
RESOURCE_SIZING_NODE=${RESOURCE_SIZING_NODE}
PARTITION=${PARTITION}
NODELIST=${NODELIST}
CPUS_PER_TASK=${CPUS_PER_TASK}
MEM=${MEM}
TIME_LIMIT=${TIME_LIMIT}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
EOF

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${PARTITION}
${SBATCH_NODELIST_LINE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_LOG_DIR}/%x_%j.out
#SBATCH --error=${SLURM_LOG_DIR}/%x_%j.err

set -Eeuo pipefail

REPO=$(q "${REPO}")
GRAPH_INPUT=$(q "${GRAPH_INPUT}")
GRAPH_OUTPUT=$(q "${GRAPH_OUTPUT}")
PER_BIN=$(q "${PER_BIN}")
EXTRA_PER_BINS=$(q "${EXTRA_PER_BINS}")
DERIVE_ONLY=$(q "${DERIVE_ONLY}")
MAX_TOTAL=$(q "${MAX_TOTAL}")
ENERGY_BIN_WIDTH=$(q "${ENERGY_BIN_WIDTH}")
STRATIFY_PARTICLE=$(q "${STRATIFY_PARTICLE}")
PARTICLE_FILTER=$(q "${PARTICLE_FILTER}")
SEED=$(q "${SEED}")
OVERWRITE=$(q "${OVERWRITE}")
SHOW_PROGRESS=$(q "${SHOW_PROGRESS}")
PROGRESS_INTERVAL=$(q "${PROGRESS_INTERVAL}")
SCAN_WORKERS=$(q "${SCAN_WORKERS}")
WRITE_WORKERS=$(q "${WRITE_WORKERS}")
OUTPUT_SHARDS=$(q "${OUTPUT_SHARDS}")
OUTPUT_SHARDS_ARG=$(q "${OUTPUT_SHARDS_ARG}")
TARGET_EVENTS_PER_SHARD=$(q "${TARGET_EVENTS_PER_SHARD}")
RUN_NAME=$(q "${RUN_NAME}")
RUN_DIR=$(q "${RUN_DIR}")
LOG_DIR=$(q "${LOG_DIR}")
SUMMARY_DIR=$(q "${SUMMARY_DIR}")
CONFIG_DIR=$(q "${CONFIG_DIR}")
UV_CACHE_DIR=$(q "${UV_CACHE_DIR}")
UV_LINK_MODE=$(q "${UV_LINK_MODE}")
OMP_NUM_THREADS=$(q "${OMP_NUM_THREADS}")
export UV_CACHE_DIR UV_LINK_MODE OMP_NUM_THREADS
export TALESD_GNN_PROGRESS_INTERVAL="\${PROGRESS_INTERVAL}"

JOB_LOG_PATH="\${LOG_DIR}/\${RUN_NAME}.job.log"

extra_graph_output() {
  local per_bin="\$1"
  local graph_dir graph_file graph_stem extra_stem root_dir
  graph_dir=\$(dirname "\${GRAPH_OUTPUT}")
  graph_file=\$(basename "\${GRAPH_OUTPUT}")
  graph_stem="\${graph_file%.h5}"
  extra_stem=\$(printf "%s" "\${graph_stem}" | sed -E "s/energyflat[0-9]+/energyflat\${per_bin}/")
  if [[ "\${extra_stem}" == "\${graph_stem}" ]]; then
    extra_stem="\${graph_stem}_perbin\${per_bin}"
  fi
  if [[ "\$(basename "\${graph_dir}")" == "\${graph_stem}" ]]; then
    root_dir=\$(dirname "\${graph_dir}")
  else
    root_dir="\${graph_dir}"
  fi
  printf "%s/%s/%s.h5" "\${root_dir}" "\${extra_stem}" "\${extra_stem}"
}

{
  echo "======================================================================"
  echo "SMALL GRAPH DATASET JOB START"
  echo "date=\$(date)"
  echo "hostname=\$(hostname 2>/dev/null || true)"
  echo "slurm_job_id=\${SLURM_JOB_ID:-}"
  echo "repo=\${REPO}"
  echo "graph_input=\${GRAPH_INPUT}"
  echo "graph_output=\${GRAPH_OUTPUT}"
  echo "per_bin=\${PER_BIN}"
  echo "extra_per_bins=\${EXTRA_PER_BINS}"
  echo "derive_only=\${DERIVE_ONLY}"
  echo "max_total=\${MAX_TOTAL}"
  echo "energy_bin_width=\${ENERGY_BIN_WIDTH}"
  echo "stratify_particle=\${STRATIFY_PARTICLE}"
  echo "particle_filter=\${PARTICLE_FILTER}"
  echo "seed=\${SEED}"
  echo "show_progress=\${SHOW_PROGRESS}"
  echo "progress_interval_sec=\${PROGRESS_INTERVAL}"
  echo "scan_workers=\${SCAN_WORKERS}"
  echo "write_workers=\${WRITE_WORKERS}"
  echo "output_shards=\${OUTPUT_SHARDS}"
  echo "target_events_per_shard=\${TARGET_EVENTS_PER_SHARD}"
  echo "input_shard_count=${INPUT_SHARD_COUNT}"
  echo "auto_resources=${AUTO_RESOURCES}"
  echo "auto_max_cpus_effective=${AUTO_MAX_CPUS_EFFECTIVE}"
  echo "resource_sizing_node=${RESOURCE_SIZING_NODE:-}"
  echo "======================================================================"
} | tee "\${JOB_LOG_PATH}"

cd "\${REPO}"

cmd=(.venv/bin/python scripts/make_small_graph_dataset.py
  --graphs "\${GRAPH_INPUT}"
  -o "\${GRAPH_OUTPUT}"
  --per-bin "\${PER_BIN}"
  --energy-bin-width "\${ENERGY_BIN_WIDTH}"
  --particle-filter "\${PARTICLE_FILTER}"
  --seed "\${SEED}"
  --scan-workers "\${SCAN_WORKERS}"
  --write-workers "\${WRITE_WORKERS}"
  --output-shards "\${OUTPUT_SHARDS_ARG}"
  --target-events-per-shard "\${TARGET_EVENTS_PER_SHARD}"
)
if [[ -n "\${EXTRA_PER_BINS}" ]]; then
  IFS=',' read -r -a extra_per_bin_values <<< "\${EXTRA_PER_BINS}"
  cmd+=(--also-per-bin "\${extra_per_bin_values[@]}")
fi
if [[ "\${DERIVE_ONLY}" == "1" ]]; then
  cmd+=(--derive-only)
fi
if [[ -n "\${MAX_TOTAL}" ]]; then
  cmd+=(--max-total "\${MAX_TOTAL}")
fi
if [[ "\${STRATIFY_PARTICLE}" == "1" ]]; then
  cmd+=(--stratify-particle)
else
  cmd+=(--no-stratify-particle)
fi
if [[ "\${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "\${SHOW_PROGRESS}" != "1" ]]; then
  cmd+=(--no-progress)
fi

printf "command:" | tee -a "\${JOB_LOG_PATH}"
printf " %q" "\${cmd[@]}" | tee -a "\${JOB_LOG_PATH}"
printf "\\n" | tee -a "\${JOB_LOG_PATH}"

"\${cmd[@]}" 2>&1 | tee -a "\${JOB_LOG_PATH}"

GRAPH_INPUTS_LIST="\${CONFIG_DIR}/graph_inputs.txt"
: > "\${GRAPH_INPUTS_LIST}"
if [[ "\${DERIVE_ONLY}" != "1" ]]; then
  SUMMARY_JSON="\${SUMMARY_DIR}/\${RUN_NAME}.graph_summary.json"
  .venv/bin/python scripts/summarize_graph_shards.py "\${GRAPH_OUTPUT}" -o "\${SUMMARY_JSON}" 2>&1 | tee -a "\${JOB_LOG_PATH}"
  printf "%s\\n" "\${GRAPH_OUTPUT}" >> "\${GRAPH_INPUTS_LIST}"
  printf "%s\\n" "\${GRAPH_OUTPUT}" > "\${CONFIG_DIR}/graph_input.txt"
else
  SUMMARY_JSON=""
fi

if [[ -n "\${EXTRA_PER_BINS}" ]]; then
  IFS=',' read -r -a extra_per_bin_values <<< "\${EXTRA_PER_BINS}"
  for extra_per_bin in "\${extra_per_bin_values[@]}"; do
    extra_output=\$(extra_graph_output "\${extra_per_bin}")
    extra_summary="\${SUMMARY_DIR}/\${RUN_NAME}-perbin\${extra_per_bin}.graph_summary.json"
    .venv/bin/python scripts/summarize_graph_shards.py "\${extra_output}" -o "\${extra_summary}" 2>&1 | tee -a "\${JOB_LOG_PATH}"
    printf "%s\\n" "\${extra_output}" >> "\${GRAPH_INPUTS_LIST}"
  done
  if [[ "\${DERIVE_ONLY}" == "1" ]]; then
    cp "\${GRAPH_INPUTS_LIST}" "\${CONFIG_DIR}/graph_input.txt"
  fi
fi

{
  echo "======================================================================"
  echo "SMALL GRAPH DATASET COMPLETE"
  echo
  echo "Use these HDF5 graphs for small tuning:"
  sed 's/^/  /' "\${GRAPH_INPUTS_LIST}"
  echo
  echo "summaries:"
  ls -1 "\${SUMMARY_DIR}"/\${RUN_NAME}*.graph_summary.json 2>/dev/null | sed 's/^/  /' || true
  echo
  echo "date=\$(date)"
  echo "======================================================================"
} | tee -a "\${JOB_LOG_PATH}"
EOF

cat <<EOF
======================================================================
SMALL GRAPH DATASET SBATCH READY

sbatch_file:
  ${SBATCH_FILE}

run_dir:
  ${RUN_DIR}

graph_output:
  ${GRAPH_OUTPUT}

job_log:
  ${LOG_DIR}/${RUN_NAME}.job.log

partition=${PARTITION}
nodelist=${NODELIST:-}
time_limit=${TIME_LIMIT}
cpus_per_task=${CPUS_PER_TASK}
scan_workers=${SCAN_WORKERS}
write_workers=${WRITE_WORKERS}
output_shards=${OUTPUT_SHARDS}
target_events_per_shard=${TARGET_EVENTS_PER_SHARD}
input_shard_count=${INPUT_SHARD_COUNT}
auto_resources=${AUTO_RESOURCES}
auto_max_cpus_effective=${AUTO_MAX_CPUS_EFFECTIVE}
resource_sizing_node=${RESOURCE_SIZING_NODE:-}
mem=${MEM}
per_bin=${PER_BIN}
extra_per_bins=${EXTRA_PER_BINS}
derive_only=${DERIVE_ONLY}
max_total=${MAX_TOTAL}
particle_filter=${PARTICLE_FILTER}
progress_interval_sec=${PROGRESS_INTERVAL}
input:
  ${GRAPH_INPUT}
Resource scan:
  ${RESOURCE_REPORT}
======================================================================
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting"
  exit 0
fi

sbatch "${SBATCH_FILE}"
