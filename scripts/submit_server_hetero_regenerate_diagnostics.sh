#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

status() {
  printf "%s\n" "$*" >&2
}

status "submit_server_hetero_regenerate_diagnostics.sh: starting"

REPO="${REPO:-${DEFAULT_REPO}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_INPUT="${GRAPH_INPUT:-/dicos_ui_home/ikomae/work/gnn/graphs/hetero_light_balanced562_hetero_light_refill_20260611_092828}"
DEFAULT_RUN_DIRS="${OUTPUT_ROOT}/runs/server_hetero_reco_mass_quality_v100_128epoch_light562_coreanchor_eval_20260612_103003:${OUTPUT_ROOT}/runs/server_hetero_reco_mass_error_v100_128epoch_light562_coreanchor_eval_20260612_103003"
RUN_DIRS="${RUN_DIRS:-${DEFAULT_RUN_DIRS}}"

RUN_ID="${RUN_ID:-regenerate_hetero_diagnostics_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-hetero_regenerate_diagnostics_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"

PARTITION="${PARTITION:-v100-al9_long}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
DEVICE="${DEVICE:-cuda}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HETERO_DATA_FORMAT="${HETERO_DATA_FORMAT:-fast_tensor}"
DIAGNOSTIC_ENERGY_BIN_WIDTH="${DIAGNOSTIC_ENERGY_BIN_WIDTH:-0.1}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
FEATURE_IMPORTANCE_MAX_GRAPHS="${FEATURE_IMPORTANCE_MAX_GRAPHS:-50000}"
ATTENTION_MAPS_MAX_GRAPHS="${ATTENTION_MAPS_MAX_GRAPHS:-16}"
SPLITS="${SPLITS:-validation test}"
REFRESH_PREDICTION_CACHE="${REFRESH_PREDICTION_CACHE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_PARSABLE="${SBATCH_PARSABLE:-0}"

if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi
if [[ "${DEVICE}" == "cuda" && "${GPUS}" == "0" ]]; then
  echo "DEVICE=cuda requires GPUS > 0" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/slurm" "${RUN_DIR}/logs"
SBATCH_FILE="${RUN_DIR}/slurm/${RUN_NAME}.sbatch"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
LOG_DIR="${RUN_DIR}/logs"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_LOG_DIR}/%x_%j.log
#SBATCH --error=${SLURM_LOG_DIR}/%x_%j.log

set -euo pipefail

JOB_LOG_PATH="${LOG_DIR}/${RUN_NAME}.job.log"
mkdir -p "${LOG_DIR}" "${SLURM_LOG_DIR}"
exec > >(tee -a "\${JOB_LOG_PATH}") 2>&1

echo "======================================================================"
echo "HETERO DIAGNOSTICS REGENERATION"
echo "date=\$(date)"
echo "hostname=\$(hostname 2>/dev/null || true)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "job_log=\${JOB_LOG_PATH}"
echo "slurm_log=${SLURM_LOG_DIR}/%x_%j.log"
echo "repo=${REPO}"
echo "graph_input=${GRAPH_INPUT}"
echo "run_dirs=${RUN_DIRS}"
echo "device=${DEVICE}"
echo "batch_size=${BATCH_SIZE}"
echo "num_workers=${NUM_WORKERS}"
echo "hetero_data_format=${HETERO_DATA_FORMAT}"
echo "feature_importance_max_graphs=${FEATURE_IMPORTANCE_MAX_GRAPHS}"
echo "attention_maps_max_graphs=${ATTENTION_MAPS_MAX_GRAPHS}"
echo "splits=${SPLITS}"
echo "refresh_prediction_cache=${REFRESH_PREDICTION_CACHE}"
echo "======================================================================"

cd "${REPO}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
export OMP_NUM_THREADS=1

if [[ "${RUN_UV_SYNC}" == "1" ]]; then
  env UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --frozen
fi

if [[ ! -e "${GRAPH_INPUT}" ]]; then
  echo "ERROR: GRAPH_INPUT not found: ${GRAPH_INPUT}" >&2
  exit 1
fi

IFS=":" read -r -a run_dirs <<< "${RUN_DIRS}"
for TARGET_RUN_DIR in "\${run_dirs[@]}"; do
  if [[ -z "\${TARGET_RUN_DIR}" ]]; then
    continue
  fi
  if [[ ! -d "\${TARGET_RUN_DIR}/checkpoints" ]]; then
    echo "ERROR: checkpoint directory not found: \${TARGET_RUN_DIR}/checkpoints" >&2
    exit 1
  fi

  CHECKPOINT=\$(find "\${TARGET_RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.pt' ! -name '*.best_through_epoch*.pt' | sort | tail -n 1)
  if [[ -z "\${CHECKPOINT}" || ! -s "\${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found in \${TARGET_RUN_DIR}/checkpoints" >&2
    find "\${TARGET_RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.pt' -print >&2
    exit 1
  fi

  echo "--------------------------------------------------------------------"
  echo "target_run_dir=\${TARGET_RUN_DIR}"
  echo "checkpoint=\${CHECKPOINT}"
  echo "graph_input=${GRAPH_INPUT}"

  refresh_args=()
  if [[ "${REFRESH_PREDICTION_CACHE}" == "1" ]]; then
    refresh_args+=(--refresh-prediction-cache)
  fi

  diagnostic_cmd=(.venv/bin/python scripts/generate_diagnostics_from_checkpoint.py
    --checkpoint "\${CHECKPOINT}"
    --graphs "${GRAPH_INPUT}"
    "\${refresh_args[@]}"
    --hetero-data-format "${HETERO_DATA_FORMAT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --diagnostic-energy-bin-width "${DIAGNOSTIC_ENERGY_BIN_WIDTH}"
    --diagnostic-min-bin-count "${DIAGNOSTIC_MIN_BIN_COUNT}")
  printf 'command:'
  printf ' %q' "\${diagnostic_cmd[@]}"
  printf '\\n'
  "\${diagnostic_cmd[@]}"

  for SPLIT in ${SPLITS}; do
    feature_dir="\${CHECKPOINT}.diagnostics/feature_importance/\${SPLIT}"
    feature_summary="\${feature_dir}/feature_group_importance.json"
    feature_cmd=(.venv/bin/python -m talesd_gnn_reconstruction.cli feature-importance
      --graphs "${GRAPH_INPUT}"
      --checkpoint "\${CHECKPOINT}"
      -o "\${feature_dir}"
      --split "\${SPLIT}"
      --max-graphs "${FEATURE_IMPORTANCE_MAX_GRAPHS}"
      --batch-size "${BATCH_SIZE}"
      --device "${DEVICE}")
    printf 'command:'
    printf ' %q' "\${feature_cmd[@]}"
    printf '\\n'
    "\${feature_cmd[@]}"
    if [[ ! -s "\${feature_summary}" ]]; then
      echo "ERROR: feature importance summary was not written: \${feature_summary}" >&2
      exit 1
    fi

    attention_dir="\${CHECKPOINT}.diagnostics/attention_maps/\${SPLIT}"
    attention_summary="\${attention_dir}/attention_maps.json"
    attention_cmd=(.venv/bin/python -m talesd_gnn_reconstruction.cli attention-maps
      --graphs "${GRAPH_INPUT}"
      --checkpoint "\${CHECKPOINT}"
      -o "\${attention_dir}"
      --split "\${SPLIT}"
      --max-graphs "${ATTENTION_MAPS_MAX_GRAPHS}"
      --device "${DEVICE}")
    printf 'command:'
    printf ' %q' "\${attention_cmd[@]}"
    printf '\\n'
    "\${attention_cmd[@]}"
    if [[ ! -s "\${attention_summary}" ]]; then
      echo "ERROR: attention maps summary was not written: \${attention_summary}" >&2
      exit 1
    fi
  done
done

echo "stage=done hetero_diagnostics_regeneration date=\$(date)"
EOF

status "HETERO DIAGNOSTICS REGENERATION SBATCH READY"
status "run_name: ${RUN_NAME}"
status "run_dir: ${RUN_DIR}"
status "graph_input: ${GRAPH_INPUT}"
status "run_dirs: ${RUN_DIRS}"
status "partition: ${PARTITION}"
status "gpus: ${GPUS}"
status "sbatch_file: ${SBATCH_FILE}"
status "job_log: ${LOG_DIR}/${RUN_NAME}.job.log"

if [[ "${DRY_RUN}" == "1" ]]; then
  status "DRY_RUN=1: not submitting"
  exit 0
fi

sbatch_args=()
if [[ "${SBATCH_PARSABLE}" == "1" ]]; then
  sbatch_args+=(--parsable)
fi
sbatch "${sbatch_args[@]}" "${SBATCH_FILE}"
