#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_homogeneous_from_hetero_training.sh: starting"

REPO="${REPO:-${DEFAULT_REPO}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
HETERO_GRAPH_INPUT="${HETERO_GRAPH_INPUT:-${GRAPH_INPUT:-}}"
if [[ -z "${HETERO_GRAPH_INPUT}" ]]; then
  echo "HETERO_GRAPH_INPUT or GRAPH_INPUT is required." >&2
  exit 2
fi

PULSE_MASK="${PULSE_MASK:-ising_kept}"
CONVERT_RUN_NAME="${CONVERT_RUN_NAME:-hetero_to_homogeneous_${PULSE_MASK}_${RUN_ID}}"
CONVERT_RUN_DIR="${CONVERT_RUN_DIR:-${OUTPUT_ROOT}/runs/${CONVERT_RUN_NAME}}"
HOMOGENEOUS_GRAPH_OUTPUT="${HOMOGENEOUS_GRAPH_OUTPUT:-${GRAPH_ROOT}/${CONVERT_RUN_NAME}}"

CONVERT_PARTITION="${CONVERT_PARTITION:-edr1-al9_large}"
CONVERT_SBATCH_DEPENDENCY="${CONVERT_SBATCH_DEPENDENCY:-${SBATCH_DEPENDENCY:-}}"
CONVERT_SBATCH_DEPENDENCY_LINE=""
if [[ -n "${CONVERT_SBATCH_DEPENDENCY}" ]]; then
  CONVERT_SBATCH_DEPENDENCY_LINE="#SBATCH --dependency=${CONVERT_SBATCH_DEPENDENCY}"
fi
if [[ -d "${HETERO_GRAPH_INPUT}" ]]; then
  CONVERT_INPUT_SHARDS="$(find "${HETERO_GRAPH_INPUT}" -type f \( -name '*.h5' -o -name '*.hdf5' \) | wc -l | tr -d ' ')"
elif [[ -f "${HETERO_GRAPH_INPUT}" ]]; then
  CONVERT_INPUT_SHARDS="1"
else
  CONVERT_INPUT_SHARDS="${CONVERT_INPUT_SHARDS:-1}"
fi
if [[ -z "${CONVERT_INPUT_SHARDS}" || "${CONVERT_INPUT_SHARDS}" == "0" ]]; then
  CONVERT_INPUT_SHARDS="1"
fi
CONVERT_WORKERS="${CONVERT_WORKERS:-${CONVERT_INPUT_SHARDS}}"
CONVERT_CPUS_PER_TASK="${CONVERT_CPUS_PER_TASK:-${CONVERT_WORKERS}}"
CONVERT_MEM="${CONVERT_MEM:-64G}"
CONVERT_TIME_LIMIT="${CONVERT_TIME_LIMIT:-12:00:00}"
CONVERT_SHARD_SIZE="${CONVERT_SHARD_SIZE:-100000}"
CONVERT_MAX_EVENTS="${CONVERT_MAX_EVENTS:-}"
CONVERT_PROGRESS_INTERVAL_SEC="${CONVERT_PROGRESS_INTERVAL_SEC:-30}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SUBMIT_TRAINING="${SUBMIT_TRAINING:-1}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
PARTITION="${PARTITION:-v100-al9_long}"
RESOURCE_TAG="${RESOURCE_TAG:-${PARTITION%%-*}}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-server_homogeneous_from_hetero_reco_mass_quality_${RESOURCE_TAG}_${TRAIN_EPOCHS}epoch_${RUN_ID}}"

mkdir -p "${CONVERT_RUN_DIR}/slurm" "${CONVERT_RUN_DIR}/logs" "${CONVERT_RUN_DIR}/summaries"
CONVERT_SBATCH_FILE="${CONVERT_RUN_DIR}/slurm/${CONVERT_RUN_NAME}.sbatch"
CONVERT_LOG_DIR="${CONVERT_RUN_DIR}/logs"
CONVERT_SLURM_LOG_DIR="${CONVERT_RUN_DIR}/slurm"

max_events_line=""
if [[ -n "${CONVERT_MAX_EVENTS}" ]]; then
  printf -v max_events_line '  --max-events "%s" \\\n' "${CONVERT_MAX_EVENTS}"
fi

cat > "${CONVERT_SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${CONVERT_RUN_NAME}
#SBATCH --partition=${CONVERT_PARTITION}
${CONVERT_SBATCH_DEPENDENCY_LINE}
#SBATCH --cpus-per-task=${CONVERT_CPUS_PER_TASK}
#SBATCH --mem=${CONVERT_MEM}
#SBATCH --time=${CONVERT_TIME_LIMIT}
#SBATCH --output=${CONVERT_SLURM_LOG_DIR}/%x_%j.log
#SBATCH --error=${CONVERT_SLURM_LOG_DIR}/%x_%j.log

set -euo pipefail
JOB_LOG_PATH="${CONVERT_LOG_DIR}/${CONVERT_RUN_NAME}.job.log"
mkdir -p "${CONVERT_LOG_DIR}" "${CONVERT_SLURM_LOG_DIR}" "${HOMOGENEOUS_GRAPH_OUTPUT}"
exec > >(tee -a "\${JOB_LOG_PATH}") 2>&1

echo "======================================================================"
echo "HETERO TO HOMOGENEOUS CONVERSION"
echo "date=\$(date)"
echo "hostname=\$(hostname 2>/dev/null || true)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "job_log=\${JOB_LOG_PATH}"
echo "run_name=${CONVERT_RUN_NAME}"
echo "hetero_graph_input=${HETERO_GRAPH_INPUT}"
echo "homogeneous_graph_output=${HOMOGENEOUS_GRAPH_OUTPUT}"
echo "pulse_mask=${PULSE_MASK}"
echo "shard_size=${CONVERT_SHARD_SIZE}"
echo "input_shards=${CONVERT_INPUT_SHARDS}"
echo "workers=${CONVERT_WORKERS}"
echo "sbatch_dependency=${CONVERT_SBATCH_DEPENDENCY:-none}"
echo "======================================================================"

cd "${REPO}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
export OMP_NUM_THREADS=1

if [[ "${RUN_UV_SYNC}" == "1" ]]; then
  env UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --frozen
fi

.venv/bin/python -m talesd_gnn_reconstruction.cli convert-hetero-to-homogeneous \
  --graphs "${HETERO_GRAPH_INPUT}" \
  -o "${HOMOGENEOUS_GRAPH_OUTPUT}" \
  --pulse-mask "${PULSE_MASK}" \
  --shard-size "${CONVERT_SHARD_SIZE}" \
  --workers "${CONVERT_WORKERS}" \
${max_events_line}  --progress-interval-sec "${CONVERT_PROGRESS_INTERVAL_SEC}" \
  --overwrite

echo "stage=done hetero_to_homogeneous date=\$(date)"
EOF

cat <<EOF
======================================================================
HETERO TO HOMOGENEOUS SBATCH READY
convert_sbatch_file: ${CONVERT_SBATCH_FILE}
convert_run_dir: ${CONVERT_RUN_DIR}
convert_job_log: ${CONVERT_LOG_DIR}/${CONVERT_RUN_NAME}.job.log
hetero_graph_input: ${HETERO_GRAPH_INPUT}
homogeneous_graph_output: ${HOMOGENEOUS_GRAPH_OUTPUT}
pulse_mask: ${PULSE_MASK}
convert_workers: ${CONVERT_WORKERS}
convert_sbatch_dependency: ${CONVERT_SBATCH_DEPENDENCY:-none}
submit_training: ${SUBMIT_TRAINING}
training_run_name: ${TRAIN_RUN_NAME}
training_partition: ${PARTITION}
source_fraction_mode: ${SOURCE_FRACTION_MODE:-explicit}
source_val_fraction: ${SOURCE_VAL_FRACTION:-0.10}
source_test_fraction: ${SOURCE_TEST_FRACTION:-0.20}
======================================================================
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  exit 0
fi

submit_output="$(sbatch "${CONVERT_SBATCH_FILE}")"
echo "${submit_output}"
convert_job_id="${submit_output##* }"

if [[ "${SUBMIT_TRAINING}" != "1" ]]; then
  echo "conversion_job_id=${convert_job_id}"
  exit 0
fi

GRAPH_INPUT="${HOMOGENEOUS_GRAPH_OUTPUT}" \
ALLOW_MISSING_GRAPH_INPUT=1 \
SBATCH_DEPENDENCY="afterok:${convert_job_id}" \
TRAINING_BACKEND=homogeneous \
RUN_ID="${RUN_ID}" \
RUN_NAME="${TRAIN_RUN_NAME}" \
PARTITION="${PARTITION}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-physics}" \
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}" \
BATCH_SIZE="${BATCH_SIZE:-256}" \
TRAIN_WORKERS="${TRAIN_WORKERS:-6}" \
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}" \
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}" \
PIN_MEMORY="${PIN_MEMORY:-1}" \
LOSS_MODE="${LOSS_MODE:-physics}" \
MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-1}" \
MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT:-0.15}" \
MASS_LOSS_MODE="${MASS_LOSS_MODE:-bce}" \
MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0.5}" \
MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}" \
QUALITY_PREDICTION="${QUALITY_PREDICTION:-1}" \
ERROR_PREDICTION="${ERROR_PREDICTION:-0}" \
ENERGY_BIAS_WEIGHT="${ENERGY_BIAS_WEIGHT:-0.0}" \
ENERGY_PARTICLE_BIAS_WEIGHT="${ENERGY_PARTICLE_BIAS_WEIGHT:-0.0}" \
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-100}" \
BEST_DIAGNOSTICS="${BEST_DIAGNOSTICS:-1}" \
FEATURE_IMPORTANCE="${FEATURE_IMPORTANCE:-1}" \
FEATURE_IMPORTANCE_SPLIT="${FEATURE_IMPORTANCE_SPLIT:-validation test}" \
DEVICE="${DEVICE:-cuda}" \
SOURCE_FRACTION_MODE="${SOURCE_FRACTION_MODE:-explicit}" \
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.10}" \
SOURCE_TEST_FRACTION="${SOURCE_TEST_FRACTION:-0.20}" \
RUN_UV_SYNC="${RUN_UV_SYNC}" \
"${SCRIPT_DIR}/submit_server_reco_mass_training.sh"

echo "conversion_job_id=${convert_job_id}"
