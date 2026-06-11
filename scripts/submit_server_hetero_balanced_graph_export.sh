#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_hetero_balanced_graph_export.sh: starting"

REPO="${REPO:-${DEFAULT_REPO}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ENERGY_SAMPLE_PER_BIN="${ENERGY_SAMPLE_PER_BIN:-50000}"
RUN_NAME="${RUN_NAME:-hetero_balanced_flat${ENERGY_SAMPLE_PER_BIN}_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${GRAPH_ROOT}/${RUN_NAME}}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}}"
SELECTION_SUMMARY="${SELECTION_SUMMARY:-${GRAPH_RUN_DIR}/summaries/hetero_selection_summary.json}"

DEFAULT_INPUT_DIRS="/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313:/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313:/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316"
DEFAULT_MC_CALIB_DIR="/dicos_ui_home/ikomae/work/taleMC/calib"
INPUT_DIRS="${INPUT_DIRS:-${DEFAULT_INPUT_DIRS}}"
INPUT_LISTS="${INPUT_LISTS:-}"
INPUT_FILES="${INPUT_FILES:-}"

KIND="${KIND:-mc}"
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
MEM="${MEM:-192G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
EXPORT_WORKERS="${EXPORT_WORKERS:-32}"
SCAN_WORKERS="${SCAN_WORKERS:-${EXPORT_WORKERS}}"
SELECTION_WORKERS="${SELECTION_WORKERS:-1}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
MIN_EVENT_DATE="${MIN_EVENT_DATE:-191002}"
OPEN_RETRIES="${OPEN_RETRIES:-3}"
OPEN_RETRY_DELAY="${OPEN_RETRY_DELAY:-1.0}"
ENERGY_SAMPLE_STRATIFY_PARTICLE="${ENERGY_SAMPLE_STRATIFY_PARTICLE:-1}"
REFILL_ATTEMPTS="${REFILL_ATTEMPTS:-2}"
BALANCE_ZENITH_BIN_WIDTH_DEG="${BALANCE_ZENITH_BIN_WIDTH_DEG:-10}"
BALANCE_AZIMUTH_BIN_WIDTH_DEG="${BALANCE_AZIMUTH_BIN_WIDTH_DEG:-30}"
BALANCE_CORE_BIN_WIDTH_KM="${BALANCE_CORE_BIN_WIDTH_KM:-0.5}"
BALANCE_TIME_BIN_WIDTH_SEC="${BALANCE_TIME_BIN_WIDTH_SEC:-3600}"
REQUIRE_REFERENCE_CORE="${REQUIRE_REFERENCE_CORE:-1}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.45}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.45}"
SPLIT_WORKERS="${SPLIT_WORKERS:-8}"
SUMMARY_WORKERS="${SUMMARY_WORKERS:-${SPLIT_WORKERS}}"
MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS:-1}"
INPUT_DISTRIBUTION_MAX_GRAPHS="${INPUT_DISTRIBUTION_MAX_GRAPHS:-100000}"
INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE="${INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE:-200000}"
SEED="${SEED:-12345}"
WRITE_BLOCK_SIZE="${WRITE_BLOCK_SIZE:-2048}"
H5_BACKEND="${H5_BACKEND:-auto}"
H5_PROGRESS_INTERVAL_SEC="${H5_PROGRESS_INTERVAL_SEC:-30}"
SOURCE_SCAN_PROGRESS_INTERVAL_EVENTS="${SOURCE_SCAN_PROGRESS_INTERVAL_EVENTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_SELECTION="${DRY_RUN_SELECTION:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
SBATCH_PARSABLE="${SBATCH_PARSABLE:-0}"

if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi
if [[ "${KIND}" == "mc" && -z "${CONST_DST}" ]]; then
  echo "CONST_DST is required for MC hetero export." >&2
  exit 2
fi
if [[ "${KIND}" == "mc" && -z "${MC_CALIB_DIR}" ]]; then
  echo "MC_CALIB_DIR is required for MC hetero export." >&2
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

particle_line=""
if [[ "${ENERGY_SAMPLE_STRATIFY_PARTICLE}" == "1" ]]; then
  printf -v particle_line '  --energy-sample-stratify-particle \\\n'
fi
core_line=""
if [[ "${REQUIRE_REFERENCE_CORE}" == "1" ]]; then
  printf -v core_line '  --require-reference-core \\\n'
fi
dry_selection_line=""
if [[ "${DRY_RUN_SELECTION}" == "1" ]]; then
  printf -v dry_selection_line '  --dry-run-selection \\\n'
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
echo "HETERO BALANCED GRAPH EXPORT"
echo "date=\$(date)"
echo "hostname=\$(hostname 2>/dev/null || true)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "job_log=\${JOB_LOG_PATH}"
echo "slurm_log=${SLURM_LOG_DIR}/%x_%j.log"
echo "run_name=${RUN_NAME}"
echo "graph_output=${GRAPH_OUTPUT}"
echo "energy_sample_per_bin=${ENERGY_SAMPLE_PER_BIN}"
echo "refill_attempts=${REFILL_ATTEMPTS}"
echo "export_workers=${EXPORT_WORKERS}"
echo "scan_workers=${SCAN_WORKERS}"
echo "selection_workers=${SELECTION_WORKERS}"
echo "h5_backend=${H5_BACKEND}"
echo "write_block_size=${WRITE_BLOCK_SIZE}"
echo "summary_workers=${SUMMARY_WORKERS}"
echo "make_input_distributions=${MAKE_INPUT_DISTRIBUTIONS}"
echo "selection_strategy=source_group_manifest_filename_energy_v1"
echo "======================================================================"

cd "${REPO}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
export OMP_NUM_THREADS=1

env UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --frozen

.venv/bin/python -m talesd_gnn_reconstruction.cli export-hetero \\
  --kind "${KIND}" \\
  --const-dst "${CONST_DST}" \\
  --mc-calib-dir "${MC_CALIB_DIR}" \\
  --energy-sample-per-bin "${ENERGY_SAMPLE_PER_BIN}" \\
${particle_line}\
  --refill-attempts "${REFILL_ATTEMPTS}" \\
  --seed "${SEED}" \\
  --write-block-size "${WRITE_BLOCK_SIZE}" \\
  --h5-backend "${H5_BACKEND}" \\
  --h5-progress-interval-sec "${H5_PROGRESS_INTERVAL_SEC}" \\
  --source-scan-progress-interval-events "${SOURCE_SCAN_PROGRESS_INTERVAL_EVENTS}" \\
  --workers "${EXPORT_WORKERS}" \\
  --scan-workers "${SCAN_WORKERS}" \\
  --selection-workers "${SELECTION_WORKERS}" \\
  --balance-zenith-bin-width-deg "${BALANCE_ZENITH_BIN_WIDTH_DEG}" \\
  --balance-azimuth-bin-width-deg "${BALANCE_AZIMUTH_BIN_WIDTH_DEG}" \\
  --balance-core-bin-width-km "${BALANCE_CORE_BIN_WIDTH_KM}" \\
  --balance-time-bin-width-sec "${BALANCE_TIME_BIN_WIDTH_SEC}" \\
  --selection-summary "${SELECTION_SUMMARY}" \\
  --min-event-date "${MIN_EVENT_DATE}" \\
  --shard-size "${SHARD_SIZE}" \\
  --open-retries "${OPEN_RETRIES}" \\
  --open-retry-delay "${OPEN_RETRY_DELAY}" \\
  --skip-errors \\
  --skip-missing-mc-calibration \\
${core_line}${dry_selection_line}${input_lines}  -o "${GRAPH_OUTPUT}"

if [[ "${DRY_RUN_SELECTION}" != "1" ]]; then
  .venv/bin/python scripts/summarize_graph_shards.py "${GRAPH_RUN_DIR}" \\
    --workers "${SUMMARY_WORKERS}" \\
    -o "${GRAPH_RUN_DIR}/summaries/graph_summary.json"
  cp -f "${GRAPH_RUN_DIR}/summaries/graph_summary.json" "${RUN_DIR}/summaries/graph_summary.json"

  .venv/bin/python scripts/summarize_split_distributions.py "${GRAPH_RUN_DIR}" \\
    -o "${GRAPH_RUN_DIR}/summaries/split_distribution_summary.json" \\
    --plot-dir "${GRAPH_RUN_DIR}/summaries/split_distributions" \\
    --val-fraction "${VAL_FRACTION}" \\
    --test-fraction "${TEST_FRACTION}" \\
    --source-val-fraction "${SOURCE_VAL_FRACTION}" \\
    --source-test-fraction "${SOURCE_TEST_FRACTION}" \\
    --seed "${SEED}" \\
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
fi
EOF

cat >&2 <<EOF
HETERO BALANCED GRAPH EXPORT SBATCH READY
run_name: ${RUN_NAME}
graph_output: ${GRAPH_OUTPUT}
selection_summary: ${SELECTION_SUMMARY}
energy_sample_per_bin: ${ENERGY_SAMPLE_PER_BIN}
seed: ${SEED}
export_workers: ${EXPORT_WORKERS}
scan_workers: ${SCAN_WORKERS}
selection_workers: ${SELECTION_WORKERS}
h5_backend: ${H5_BACKEND}
write_block_size: ${WRITE_BLOCK_SIZE}
split_event_fractions: train=1-val-test, val=${VAL_FRACTION}, test=${TEST_FRACTION}
split_source_fractions: train=1-val-test, val=${SOURCE_VAL_FRACTION}, test=${SOURCE_TEST_FRACTION}
sbatch_file: ${SBATCH_FILE}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

sbatch_args=()
if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
  sbatch_args+=(--dependency="${SBATCH_DEPENDENCY}")
fi
if [[ "${SBATCH_PARSABLE}" == "1" ]]; then
  sbatch_args+=(--parsable)
fi
sbatch "${sbatch_args[@]}" "${SBATCH_FILE}"
