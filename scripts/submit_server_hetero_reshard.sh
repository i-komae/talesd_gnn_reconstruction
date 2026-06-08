#!/usr/bin/env bash
set -euo pipefail

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_hetero_reshard.sh: starting"

REPO="${REPO:-/dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GRAPH_INPUT="${GRAPH_INPUT:-}"
if [[ -z "${GRAPH_INPUT}" ]]; then
  echo "GRAPH_INPUT is required. Pass an existing hetero HDF5 shard base or directory." >&2
  exit 2
fi

INPUT_BASE="$(basename "${GRAPH_INPUT}")"
INPUT_BASE="${INPUT_BASE%.h5}"
RUN_NAME="${RUN_NAME:-${INPUT_BASE}_reshuffled_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${GRAPH_ROOT}/${RUN_NAME}}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}/${RUN_NAME}.h5}"

PARTITION="${PARTITION:-edr1-al9_large}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
SEED="${SEED:-12345}"
OUTPUT_ORDER="${OUTPUT_ORDER:-interleaved}"
OUTPUT_LOCALITY_RUN_SIZE="${OUTPUT_LOCALITY_RUN_SIZE:-32}"
ENERGY_SAMPLE_STRATIFY_PARTICLE="${ENERGY_SAMPLE_STRATIFY_PARTICLE:-1}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.45}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}"
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.45}"
SPLIT_WORKERS="${SPLIT_WORKERS:-8}"
SUMMARY_WORKERS="${SUMMARY_WORKERS:-${SPLIT_WORKERS}}"
MAKE_INPUT_DISTRIBUTIONS="${MAKE_INPUT_DISTRIBUTIONS:-1}"
INPUT_DISTRIBUTION_MAX_GRAPHS="${INPUT_DISTRIBUTION_MAX_GRAPHS:-100000}"
INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE="${INPUT_DISTRIBUTION_MAX_VALUES_PER_FEATURE:-200000}"
DRY_RUN="${DRY_RUN:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/slurm" "${RUN_DIR}/summaries" "${GRAPH_RUN_DIR}/summaries"
SBATCH_FILE="${RUN_DIR}/slurm/${RUN_NAME}.sbatch"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

particle_line=""
if [[ "${ENERGY_SAMPLE_STRATIFY_PARTICLE}" == "1" ]]; then
  printf -v particle_line '  --energy-sample-stratify-particle \\\n'
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
echo "HETERO HDF5 RESHARD"
echo "date=\$(date)"
echo "hostname=\$(hostname 2>/dev/null || true)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "job_log=\${JOB_LOG_PATH}"
echo "slurm_log=${SLURM_LOG_DIR}/%x_%j.log"
echo "run_name=${RUN_NAME}"
echo "graph_input=${GRAPH_INPUT}"
echo "graph_output=${GRAPH_OUTPUT}"
echo "output_order=${OUTPUT_ORDER}"
echo "shard_size=${SHARD_SIZE}"
echo "seed=${SEED}"
echo "======================================================================"

cd "${REPO}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
export OMP_NUM_THREADS=1

env UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --frozen

.venv/bin/python -m talesd_gnn_reconstruction.cli reshard-hetero \\
  --graphs "${GRAPH_INPUT}" \\
  --output-order "${OUTPUT_ORDER}" \\
  --output-locality-run-size "${OUTPUT_LOCALITY_RUN_SIZE}" \\
  --seed "${SEED}" \\
  --shard-size "${SHARD_SIZE}" \\
${particle_line}\
  -o "${GRAPH_OUTPUT}"

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
EOF

cat >&2 <<EOF
HETERO HDF5 RESHARD SBATCH READY
run_name: ${RUN_NAME}
graph_input: ${GRAPH_INPUT}
graph_output: ${GRAPH_OUTPUT}
output_order: ${OUTPUT_ORDER}
seed: ${SEED}
split_event_fractions: train=1-val-test, val=${VAL_FRACTION}, test=${TEST_FRACTION}
split_source_fractions: train=1-val-test, val=${SOURCE_VAL_FRACTION}, test=${SOURCE_TEST_FRACTION}
sbatch_file: ${SBATCH_FILE}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

sbatch "${SBATCH_FILE}"
