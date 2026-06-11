#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_hetero_light_graph_export.sh: starting"

REPO="${REPO:-${DEFAULT_REPO}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GRAPHS_PER_SOURCE_GROUP="${GRAPHS_PER_SOURCE_GROUP:-10}"
SOURCE_GROUP_OVERDRAW_FACTOR="${SOURCE_GROUP_OVERDRAW_FACTOR:-10}"
RUN_NAME="${RUN_NAME:-hetero_light_${GRAPHS_PER_SOURCE_GROUP}per_shower_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${GRAPH_ROOT}/${RUN_NAME}}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}}"
SELECTION_SUMMARY="${SELECTION_SUMMARY:-${GRAPH_RUN_DIR}/summaries/hetero_light_selection_summary.json}"

DEFAULT_INPUT_DIRS="/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313:/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313:/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316"
DEFAULT_MC_CALIB_DIR="/dicos_ui_home/ikomae/work/taleMC/calib"
INPUT_DIRS="${INPUT_DIRS:-${DEFAULT_INPUT_DIRS}}"
INPUT_LISTS="${INPUT_LISTS:-}"
INPUT_FILES="${INPUT_FILES:-}"

MC_CALIB_DIR="${MC_CALIB_DIR:-${TALE_MC_CALIB_DIR:-}}"
if [[ -z "${MC_CALIB_DIR}" && -d "${DEFAULT_MC_CALIB_DIR}" ]]; then
  MC_CALIB_DIR="${DEFAULT_MC_CALIB_DIR}"
fi
CONST_DST="${CONST_DST:-${TALESD_CONST_DST:-}}"
if [[ -z "${CONST_DST}" && -n "${MC_CALIB_DIR}" && -f "${MC_CALIB_DIR%/}/talesdconst_pass2.dst" ]]; then
  CONST_DST="${MC_CALIB_DIR%/}/talesdconst_pass2.dst"
fi
if [[ -z "${CONST_DST}" && -n "${MC_CALIB_DIR}" && -f "${MC_CALIB_DIR%/}/talesdconst_pass2.dst.gz" ]]; then
  CONST_DST="${MC_CALIB_DIR%/}/talesdconst_pass2.dst.gz"
fi

PARTITION="${PARTITION:-edr1-al9_large}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
EXPORT_WORKERS="${EXPORT_WORKERS:-32}"
MAX_SOURCE_GROUPS_PER_STRATUM="${MAX_SOURCE_GROUPS_PER_STRATUM:-}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
MIN_EVENT_DATE="${MIN_EVENT_DATE:-191002}"
OPEN_RETRIES="${OPEN_RETRIES:-3}"
OPEN_RETRY_DELAY="${OPEN_RETRY_DELAY:-1.0}"
REQUIRE_REFERENCE_CORE="${REQUIRE_REFERENCE_CORE:-1}"
SPLIT_WORKERS="${SPLIT_WORKERS:-8}"
SUMMARY_WORKERS="${SUMMARY_WORKERS:-${SPLIT_WORKERS}}"
MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS:-0}"
INPUT_DISTRIBUTION_MAX_GRAPHS="${INPUT_DISTRIBUTION_MAX_GRAPHS:-100000}"
INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE="${INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE:-200000}"
SEED="${SEED:-12345}"
H5_PROGRESS_INTERVAL_SEC="${H5_PROGRESS_INTERVAL_SEC:-30}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_PARSABLE="${SBATCH_PARSABLE:-0}"

if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi
if [[ -z "${CONST_DST}" ]]; then
  echo "CONST_DST is required for MC hetero light export." >&2
  exit 2
fi
if [[ -z "${MC_CALIB_DIR}" ]]; then
  echo "MC_CALIB_DIR is required for MC hetero light export." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/slurm" "${RUN_DIR}/summaries" "${GRAPH_RUN_DIR}/summaries"
SBATCH_FILE="${RUN_DIR}/slurm/${RUN_NAME}.sbatch"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

input_args=()
if [[ -n "${INPUT_DIRS}" ]]; then
  old_ifs="${IFS}"
  IFS=":"
  for item in ${INPUT_DIRS}; do
    [[ -n "${item}" ]] && input_args+=(--input-dir "${item}")
  done
  IFS="${old_ifs}"
fi
if [[ -n "${INPUT_LISTS}" ]]; then
  old_ifs="${IFS}"
  IFS=":"
  for item in ${INPUT_LISTS}; do
    [[ -n "${item}" ]] && input_args+=(--input-list "${item}")
  done
  IFS="${old_ifs}"
fi
if [[ -n "${INPUT_FILES}" ]]; then
  old_ifs="${IFS}"
  IFS=":"
  for item in ${INPUT_FILES}; do
    [[ -n "${item}" ]] && input_args+=("${item}")
  done
  IFS="${old_ifs}"
fi

max_groups_line=""
if [[ -n "${MAX_SOURCE_GROUPS_PER_STRATUM}" ]]; then
  printf -v max_groups_line '  --max-source-groups-per-stratum "%s" \\\n' "${MAX_SOURCE_GROUPS_PER_STRATUM}"
fi
core_line=""
if [[ "${REQUIRE_REFERENCE_CORE}" == "1" ]]; then
  printf -v core_line '  --require-reference-core \\\n'
fi
input_lines=""
if (( ${#input_args[@]} > 0 )); then
  for item in "${input_args[@]}"; do
    printf -v quoted_item "%q" "${item}"
    input_lines+="  ${quoted_item} \\
"
  done
fi

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_LOG_DIR}/%x_%j.log
#SBATCH --error=${SLURM_LOG_DIR}/%x_%j.log

set -euo pipefail
JOB_LOG_PATH="${LOG_DIR}/${RUN_NAME}.job.log"
mkdir -p "${LOG_DIR}" "${SLURM_LOG_DIR}" "${GRAPH_RUN_DIR}/summaries"
exec > >(tee -a "\${JOB_LOG_PATH}") 2>&1

echo "======================================================================"
echo "HETERO LIGHT GRAPH EXPORT"
echo "date=\$(date)"
echo "hostname=\$(hostname 2>/dev/null || true)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "job_log=\${JOB_LOG_PATH}"
echo "slurm_log=${SLURM_LOG_DIR}/%x_%j.log"
echo "run_name=${RUN_NAME}"
echo "graph_output=${GRAPH_OUTPUT}"
echo "graphs_per_source_group=${GRAPHS_PER_SOURCE_GROUP}"
echo "source_group_overdraw_factor=${SOURCE_GROUP_OVERDRAW_FACTOR}"
echo "max_source_groups_per_stratum=${MAX_SOURCE_GROUPS_PER_STRATUM}"
echo "export_workers=${EXPORT_WORKERS}"
echo "run_uv_sync=${RUN_UV_SYNC}"
echo "make_input_distributions=${MAKE_INPUT_DISTRIBUTIONS}"
echo "selection_strategy=filename_source_group_light_v1"
echo "======================================================================"

cd "${REPO}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
export OMP_NUM_THREADS=1

if [[ "${RUN_UV_SYNC}" == "1" ]]; then
  env UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --frozen
fi

.venv/bin/python -m talesd_gnn_reconstruction.cli export-hetero-light \\
  --kind "mc" \\
  --const-dst "${CONST_DST}" \\
  --mc-calib-dir "${MC_CALIB_DIR}" \\
  --graphs-per-source-group "${GRAPHS_PER_SOURCE_GROUP}" \\
  --source-group-overdraw-factor "${SOURCE_GROUP_OVERDRAW_FACTOR}" \\
${max_groups_line}\
  --seed "${SEED}" \\
  --workers "${EXPORT_WORKERS}" \\
  --h5-progress-interval-sec "${H5_PROGRESS_INTERVAL_SEC}" \\
  --selection-summary "${SELECTION_SUMMARY}" \\
  --min-event-date "${MIN_EVENT_DATE}" \\
  --shard-size "${SHARD_SIZE}" \\
  --open-retries "${OPEN_RETRIES}" \\
  --open-retry-delay "${OPEN_RETRY_DELAY}" \\
  --skip-errors \\
  --skip-missing-mc-calibration \\
${core_line}${input_lines}  -o "${GRAPH_OUTPUT}"

.venv/bin/python scripts/summarize_graph_shards.py "${GRAPH_RUN_DIR}" \\
  --workers "${SUMMARY_WORKERS}" \\
  -o "${GRAPH_RUN_DIR}/summaries/graph_summary.json"
cp -f "${GRAPH_RUN_DIR}/summaries/graph_summary.json" "${RUN_DIR}/summaries/graph_summary.json"

.venv/bin/python scripts/summarize_split_distributions.py "${GRAPH_RUN_DIR}" \\
  -o "${GRAPH_RUN_DIR}/summaries/split_distribution_summary.json" \\
  --plot-dir "${GRAPH_RUN_DIR}/summaries/split_distributions" \\
  --split-workers "${SPLIT_WORKERS}"
cp -f "${GRAPH_RUN_DIR}/summaries/split_distribution_summary.json" "${RUN_DIR}/summaries/split_distribution_summary.json"

if [[ "${MAKE_INPUT_DISTRIBUTIONS}" == "1" ]]; then
  .venv/bin/talesd-gnn input-distributions \\
    --graphs "${GRAPH_RUN_DIR}" \\
    --max-graphs "${INPUT_DISTRIBUTION_MAX_GRAPHS}" \\
    --max-values-per-feature "${INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE}" \\
    --seed "${SEED}" \\
    -o "${GRAPH_RUN_DIR}/summaries/input_distributions"
fi
EOF

cat >&2 <<EOF
HETERO LIGHT GRAPH EXPORT SBATCH READY
run_name: ${RUN_NAME}
graph_output: ${GRAPH_OUTPUT}
selection_summary: ${SELECTION_SUMMARY}
graphs_per_source_group: ${GRAPHS_PER_SOURCE_GROUP}
source_group_overdraw_factor: ${SOURCE_GROUP_OVERDRAW_FACTOR}
max_source_groups_per_stratum: ${MAX_SOURCE_GROUPS_PER_STRATUM}
seed: ${SEED}
export_workers: ${EXPORT_WORKERS}
run_uv_sync: ${RUN_UV_SYNC}
make_input_distributions: ${MAKE_INPUT_DISTRIBUTIONS}
sbatch_file: ${SBATCH_FILE}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

sbatch_args=()
if [[ "${SBATCH_PARSABLE}" == "1" ]]; then
  sbatch_args+=(--parsable)
fi
sbatch "${sbatch_args[@]}" "${SBATCH_FILE}"
