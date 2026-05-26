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
  MAX_TOTAL=50000
  PARTICLE_FILTER=all|proton|iron
  PROGRESS_INTERVAL=30
  PARTITION=edr2-al9_moderate_serial
  CPUS_PER_TASK=1
  MEM=64G
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

REPO="${REPO:-/ceph/sharedfs/work/SATORI/ikomae/src/talesd_gnn_reconstruction}"
DEFAULT_GRAPH_INPUT="${DEFAULT_GRAPH_INPUT:-/dicos_ui_home/ikomae/work/gnn/graphs/server_graph_export_energyflat200000_20260524_075508}"
GRAPH_INPUT="${GRAPH_INPUT:-${DEFAULT_GRAPH_INPUT}}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

PER_BIN="${PER_BIN:-2000}"
MAX_TOTAL="${MAX_TOTAL:-}"
ENERGY_BIN_WIDTH="${ENERGY_BIN_WIDTH:-0.1}"
STRATIFY_PARTICLE="${STRATIFY_PARTICLE:-1}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
SEED="${SEED:-12345}"
OVERWRITE="${OVERWRITE:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

PARTITION="${PARTITION:-edr2-al9_moderate_serial}"
NODELIST="${NODELIST:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
RUN_NAME="${RUN_NAME:-small_graph_energyflat${PER_BIN}_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${GRAPH_ROOT}/${RUN_NAME}}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}/${RUN_NAME}.h5}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
DRY_RUN="${DRY_RUN:-0}"

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
if [[ -n "${MAX_TOTAL}" ]] && ! [[ "${MAX_TOTAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOTAL must be empty or a positive integer: ${MAX_TOTAL}" >&2
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
if [[ -e "${GRAPH_OUTPUT}" && "${OVERWRITE}" != "1" ]]; then
  echo "graph output already exists: ${GRAPH_OUTPUT}" >&2
  echo "Set OVERWRITE=1 or choose a different RUN_NAME/GRAPH_OUTPUT." >&2
  exit 2
fi

SBATCH_DIR="${RUN_DIR}/slurm"
SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
LOG_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
mkdir -p "${SBATCH_DIR}" "${SLURM_LOG_DIR}" "${SUMMARY_DIR}" "${LOG_DIR}" "${CONFIG_DIR}" "${GRAPH_RUN_DIR}"

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
MAX_TOTAL=${MAX_TOTAL}
ENERGY_BIN_WIDTH=${ENERGY_BIN_WIDTH}
STRATIFY_PARTICLE=${STRATIFY_PARTICLE}
PARTICLE_FILTER=${PARTICLE_FILTER}
SEED=${SEED}
SHOW_PROGRESS=${SHOW_PROGRESS}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL}
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
MAX_TOTAL=$(q "${MAX_TOTAL}")
ENERGY_BIN_WIDTH=$(q "${ENERGY_BIN_WIDTH}")
STRATIFY_PARTICLE=$(q "${STRATIFY_PARTICLE}")
PARTICLE_FILTER=$(q "${PARTICLE_FILTER}")
SEED=$(q "${SEED}")
OVERWRITE=$(q "${OVERWRITE}")
SHOW_PROGRESS=$(q "${SHOW_PROGRESS}")
PROGRESS_INTERVAL=$(q "${PROGRESS_INTERVAL}")
RUN_NAME=$(q "${RUN_NAME}")
RUN_DIR=$(q "${RUN_DIR}")
LOG_DIR=$(q "${LOG_DIR}")
SUMMARY_DIR=$(q "${SUMMARY_DIR}")
CONFIG_DIR=$(q "${CONFIG_DIR}")
UV_CACHE_DIR=$(q "${UV_CACHE_DIR}")
UV_LINK_MODE=$(q "${UV_LINK_MODE}")
export UV_CACHE_DIR UV_LINK_MODE
export TALESD_GNN_PROGRESS_INTERVAL="\${PROGRESS_INTERVAL}"

JOB_LOG_PATH="\${LOG_DIR}/\${RUN_NAME}.job.log"

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
  echo "max_total=\${MAX_TOTAL}"
  echo "energy_bin_width=\${ENERGY_BIN_WIDTH}"
  echo "stratify_particle=\${STRATIFY_PARTICLE}"
  echo "particle_filter=\${PARTICLE_FILTER}"
  echo "seed=\${SEED}"
  echo "show_progress=\${SHOW_PROGRESS}"
  echo "progress_interval_sec=\${PROGRESS_INTERVAL}"
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
)
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

SUMMARY_JSON="\${SUMMARY_DIR}/\${RUN_NAME}.graph_summary.json"
.venv/bin/python scripts/summarize_graph_shards.py "\${GRAPH_OUTPUT}" -o "\${SUMMARY_JSON}" 2>&1 | tee -a "\${JOB_LOG_PATH}"
printf "%s\\n" "\${GRAPH_OUTPUT}" > "\${CONFIG_DIR}/graph_input.txt"

{
  echo "======================================================================"
  echo "SMALL GRAPH DATASET COMPLETE"
  echo
  echo "Use this HDF5 graph for small tuning:"
  echo "  \${GRAPH_OUTPUT}"
  echo
  echo "summary:"
  echo "  \${SUMMARY_JSON}"
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
mem=${MEM}
per_bin=${PER_BIN}
max_total=${MAX_TOTAL}
particle_filter=${PARTICLE_FILTER}
progress_interval_sec=${PROGRESS_INTERVAL}
input:
  ${GRAPH_INPUT}
======================================================================
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting"
  exit 0
fi

sbatch "${SBATCH_FILE}"
