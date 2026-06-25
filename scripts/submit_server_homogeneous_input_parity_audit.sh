#!/usr/bin/env bash
set -euo pipefail

echo "submit_server_homogeneous_input_parity_audit.sh: starting"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

RUN_ID="${RUN_ID:-input_parity_$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-edr1-al9_large}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
DRY_RUN="${DRY_RUN:-0}"

REFERENCE_GRAPH="${REFERENCE_GRAPH:-/dicos_ui_home/ikomae/work/gnn/graphs/flat50000}"
CANDIDATE_GRAPH="${CANDIDATE_GRAPH:-/dicos_ui_home/ikomae/work/gnn/graphs/hetero_to_homogeneous_ising_kept_homogeneous_from_allsrc_samescale_sourcebalanced_50000_fix_20260624_042552}"
MATCH_KEY="${MATCH_KEY:-source_group_index}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SEED="${SEED:-12345}"
SKIP_WAVEFORMS="${SKIP_WAVEFORMS:-0}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-30}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
RUN_NAME="homogeneous_input_parity_${RUN_ID}"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
SLURM_DIR="${RUN_DIR}/slurm"
SUMMARY_JSON="${RUN_DIR}/input_parity_summary.json"
EXAMPLES_JSONL="${RUN_DIR}/input_parity_examples.jsonl"
JOB_LOG="${LOG_DIR}/${RUN_NAME}.job.log"
SBATCH_FILE="${SLURM_DIR}/${RUN_NAME}.sbatch"

mkdir -p "${LOG_DIR}" "${SLURM_DIR}"

SKIP_WAVEFORMS_ARG=""
if [[ "${SKIP_WAVEFORMS}" == "1" ]]; then
  SKIP_WAVEFORMS_ARG="--skip-waveforms"
fi

cat >"${SBATCH_FILE}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=h5_input_parity
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/%x_%j.log
#SBATCH --error=${LOG_DIR}/%x_%j.log

set -euo pipefail

exec > >(tee -a "${JOB_LOG}") 2>&1

echo "======================================================================"
echo "HOMOGENEOUS H5 INPUT PARITY AUDIT"
echo "date=\$(date)"
echo "hostname=\$(hostname)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "reference_graph=${REFERENCE_GRAPH}"
echo "candidate_graph=${CANDIDATE_GRAPH}"
echo "match_key=${MATCH_KEY}"
echo "sample_size=${SAMPLE_SIZE}"
echo "skip_waveforms=${SKIP_WAVEFORMS}"
echo "summary_json=${SUMMARY_JSON}"
echo "examples_jsonl=${EXAMPLES_JSONL}"
echo "======================================================================"

cd "${REPO}"
if [[ "${RUN_UV_SYNC}" == "1" ]]; then
  env UV_CACHE_DIR=/dicos_ui_home/ikomae/work/uv-cache uv sync --frozen
fi

.venv/bin/python scripts/audit_homogeneous_input_parity.py \\
  --reference "${REFERENCE_GRAPH}" \\
  --candidate "${CANDIDATE_GRAPH}" \\
  -o "${SUMMARY_JSON}" \\
  --examples-output "${EXAMPLES_JSONL}" \\
  --match-key "${MATCH_KEY}" \\
  --sample-size "${SAMPLE_SIZE}" \\
  --seed "${SEED}" \\
  --progress-interval-sec "${PROGRESS_INTERVAL_SEC}" \\
  ${SKIP_WAVEFORMS_ARG}

echo "stage=done homogeneous_input_parity summary=${SUMMARY_JSON}"
SBATCH

echo "HOMOGENEOUS INPUT PARITY SBATCH READY"
echo "run_name: ${RUN_NAME}"
echo "run_dir: ${RUN_DIR}"
echo "reference_graph: ${REFERENCE_GRAPH}"
echo "candidate_graph: ${CANDIDATE_GRAPH}"
echo "match_key: ${MATCH_KEY}"
echo "sample_size: ${SAMPLE_SIZE}"
echo "skip_waveforms: ${SKIP_WAVEFORMS}"
echo "partition: ${PARTITION}"
echo "cpus_per_task: ${CPUS_PER_TASK}"
echo "mem: ${MEM}"
echo "time_limit: ${TIME_LIMIT}"
echo "sbatch_file: ${SBATCH_FILE}"
echo "job_log: ${JOB_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

sbatch "${SBATCH_FILE}"
