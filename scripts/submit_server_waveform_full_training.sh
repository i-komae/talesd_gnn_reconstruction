#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction}"
DEFAULT_GRAPH_INPUT="${DEFAULT_GRAPH_INPUT:-/dicos_ui_home/ikomae/work/gnn/graphs/server_graph_export_energyflat200000_20260524_075508}"
GRAPH_INPUT="${GRAPH_INPUT:-${DEFAULT_GRAPH_INPUT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

PARTITION="${PARTITION:-b6000-al9_long}"
NODELIST="${NODELIST:-}"
RESOURCE_TAG="${RESOURCE_TAG:-${PARTITION%%-*}}"
GPUS="${GPUS:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-256}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((GPUS * CPUS_PER_GPU))}"
MEM="${MEM:-$((GPUS * MEM_PER_GPU_GB))G}"
if [[ -z "${TIME_LIMIT:-}" ]]; then
  case "${PARTITION}" in
    *_long*|*long-*)
      TIME_LIMIT="7-00:00:00"
      ;;
    *_short*|*short-*)
      TIME_LIMIT="6:00:00"
      ;;
    a100_devel-al9)
      TIME_LIMIT="20:00"
      ;;
    *)
      TIME_LIMIT="5-00:00:00"
      ;;
  esac
fi

TRAIN_EPOCHS="${TRAIN_EPOCHS:-128}"
RUN_NAME="${RUN_NAME:-server_reco_quality_${RESOURCE_TAG}_${TRAIN_EPOCHS}epoch_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-6}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-${CPUS_PER_TASK}}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
H5_MAX_OPEN_FILES="${H5_MAX_OPEN_FILES:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"
TRAINING_TASK="${TRAINING_TASK:-reconstruction}"
MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-0}"
MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT:-0.1}"
MASS_LOSS_MODE="${MASS_LOSS_MODE:-focal}"
MASS_FOCAL_GAMMA="${MASS_FOCAL_GAMMA:-2.0}"
MASS_POS_WEIGHT_MODE="${MASS_POS_WEIGHT_MODE:-none}"
MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT:-0}"
MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN:-1.0}"
MASS_COLLAPSE_PATIENCE="${MASS_COLLAPSE_PATIENCE:-3}"
MASS_COLLAPSE_SCORE_STD="${MASS_COLLAPSE_SCORE_STD:-1e-3}"
MASS_COLLAPSE_BALANCED_ACCURACY="${MASS_COLLAPSE_BALANCED_ACCURACY:-0.505}"
DEVICE="${DEVICE:-cuda}"

MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-physics}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-5}"
DROPOUT="${DROPOUT:-0.05}"
READOUT_HEADS="${READOUT_HEADS:-4}"
CLASSIFICATION_ARCH="${CLASSIFICATION_ARCH:-enhanced}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"
WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
WAVEFORM_TRANSFORMER_HEADS="${WAVEFORM_TRANSFORMER_HEADS:-4}"
WAVEFORM_TRANSFORMER_LAYERS="${WAVEFORM_TRANSFORMER_LAYERS:-1}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-reduce-on-plateau}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LR_PATIENCE="${LR_PATIENCE:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-32}"
LOSS_MODE="${LOSS_MODE:-physics}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.2}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.4}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.12}"
ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-1.0}"
QUALITY_PREDICTION="${QUALITY_PREDICTION:-1}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.2}"
QUALITY_ANGULAR_SCALE_DEG="${QUALITY_ANGULAR_SCALE_DEG:-1.0}"
QUALITY_CORE_SCALE_KM="${QUALITY_CORE_SCALE_KM:-0.05}"
QUALITY_ENERGY_SCALE="${QUALITY_ENERGY_SCALE:-0.10}"
ERROR_PREDICTION="${ERROR_PREDICTION:-0}"
ERROR_WEIGHT="${ERROR_WEIGHT:-0.0}"
ERROR_ANGULAR_SCALE_DEG="${ERROR_ANGULAR_SCALE_DEG:-${QUALITY_ANGULAR_SCALE_DEG}}"
ERROR_CORE_SCALE_KM="${ERROR_CORE_SCALE_KM:-${QUALITY_CORE_SCALE_KM}}"
ERROR_ENERGY_SCALE="${ERROR_ENERGY_SCALE:-${QUALITY_ENERGY_SCALE}}"
NLL_WEIGHT="${NLL_WEIGHT:-0.0}"
NLL_SIGMA_ENERGY_FLOOR="${NLL_SIGMA_ENERGY_FLOOR:-0.01}"
NLL_SIGMA_ANGLE_FLOOR_DEG="${NLL_SIGMA_ANGLE_FLOOR_DEG:-0.05}"
NLL_SIGMA_CORE_FLOOR_KM="${NLL_SIGMA_CORE_FLOOR_KM:-0.005}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
PRECISION_MIN_BIN_COUNT="${PRECISION_MIN_BIN_COUNT:-1000}"
MAX_GRAPHS="${MAX_GRAPHS:-}"

RUN_BUILD="${RUN_BUILD:-0}"
SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-0}"
LOCAL_GRAPH_CACHE="${LOCAL_GRAPH_CACHE:-auto}"
LOCAL_GRAPH_ROOT="${LOCAL_GRAPH_ROOT:-auto}"
LOCAL_GRAPH_ROOT_CANDIDATES="${LOCAL_GRAPH_ROOT_CANDIDATES:-/ssd/${USER:-ikomae}/talesd_gnn:/tmp/${USER:-ikomae}/talesd_gnn}"
LOCAL_GRAPH_FALLBACK_ROOTS="${LOCAL_GRAPH_FALLBACK_ROOTS:-/tmp/${USER:-ikomae}/talesd_gnn}"
LOCAL_GRAPH_CACHE_SCOPE="${LOCAL_GRAPH_CACHE_SCOPE:-shared}"
LOCAL_GRAPH_CLEANUP="${LOCAL_GRAPH_CLEANUP:-1}"
LOCAL_GRAPH_COPY_TOOL="${LOCAL_GRAPH_COPY_TOOL:-auto}"
LOCAL_GRAPH_WAIT_TIMEOUT_SEC="${LOCAL_GRAPH_WAIT_TIMEOUT_SEC:-21600}"
LOCAL_GRAPH_STALE_LOCK_SEC="${LOCAL_GRAPH_STALE_LOCK_SEC:-60}"
LOCAL_RUNTIME_CACHE="${LOCAL_RUNTIME_CACHE:-auto}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-auto}"
LOCAL_RUNTIME_ROOT_CANDIDATES="${LOCAL_RUNTIME_ROOT_CANDIDATES:-${LOCAL_GRAPH_ROOT_CANDIDATES}}"
LOCAL_RUNTIME_FALLBACK_ROOTS="${LOCAL_RUNTIME_FALLBACK_ROOTS:-${LOCAL_GRAPH_FALLBACK_ROOTS}}"
LOCAL_RUNTIME_CLEANUP="${LOCAL_RUNTIME_CLEANUP:-1}"
LOCAL_RUNTIME_COPY_TOOL="${LOCAL_RUNTIME_COPY_TOOL:-auto}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-1}"
EPOCH_LEARNING_CURVE="${EPOCH_LEARNING_CURVE:-1}"
BEST_DIAGNOSTICS="${BEST_DIAGNOSTICS:-1}"
BEST_DIAGNOSTIC_MAX_GRAPHS="${BEST_DIAGNOSTIC_MAX_GRAPHS:-20000}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
TALESD_GNN_H5_MAX_OPEN_FILES="${TALESD_GNN_H5_MAX_OPEN_FILES:-${H5_MAX_OPEN_FILES}}"
DRY_RUN="${DRY_RUN:-0}"

case "${LOCAL_GRAPH_CACHE}" in
  0|1|auto)
    ;;
  *)
    echo "LOCAL_GRAPH_CACHE must be 0, 1, or auto: ${LOCAL_GRAPH_CACHE}" >&2
    exit 2
    ;;
esac
case "${LOCAL_GRAPH_CLEANUP}" in
  0|1)
    ;;
  *)
    echo "LOCAL_GRAPH_CLEANUP must be 0 or 1: ${LOCAL_GRAPH_CLEANUP}" >&2
    exit 2
    ;;
esac
case "${LOCAL_GRAPH_COPY_TOOL}" in
  auto|rsync|cp)
    ;;
  *)
    echo "LOCAL_GRAPH_COPY_TOOL must be auto, rsync, or cp: ${LOCAL_GRAPH_COPY_TOOL}" >&2
    exit 2
    ;;
esac
case "${LOCAL_RUNTIME_CACHE}" in
  0|1|auto)
    ;;
  *)
    echo "LOCAL_RUNTIME_CACHE must be 0, 1, or auto: ${LOCAL_RUNTIME_CACHE}" >&2
    exit 2
    ;;
esac
case "${LOCAL_RUNTIME_CLEANUP}" in
  0|1)
    ;;
  *)
    echo "LOCAL_RUNTIME_CLEANUP must be 0 or 1: ${LOCAL_RUNTIME_CLEANUP}" >&2
    exit 2
    ;;
esac
case "${LOCAL_RUNTIME_COPY_TOOL}" in
  auto|rsync|cp)
    ;;
  *)
    echo "LOCAL_RUNTIME_COPY_TOOL must be auto, rsync, or cp: ${LOCAL_RUNTIME_COPY_TOOL}" >&2
    exit 2
    ;;
esac
case "${LOCAL_GRAPH_CACHE_SCOPE}" in
  shared|job)
    ;;
  *)
    echo "LOCAL_GRAPH_CACHE_SCOPE must be shared or job: ${LOCAL_GRAPH_CACHE_SCOPE}" >&2
    exit 2
    ;;
esac
if ! [[ "${LOCAL_GRAPH_WAIT_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_GRAPH_WAIT_TIMEOUT_SEC must be a non-negative integer: ${LOCAL_GRAPH_WAIT_TIMEOUT_SEC}" >&2
  exit 2
fi
if ! [[ "${LOCAL_GRAPH_STALE_LOCK_SEC}" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_GRAPH_STALE_LOCK_SEC must be a non-negative integer: ${LOCAL_GRAPH_STALE_LOCK_SEC}" >&2
  exit 2
fi

if [[ "${PARTITION}" == a100* && "${ALLOW_A100:-0}" != "1" ]]; then
  cat >&2 <<EOF
Refusing to submit to A100 partition by default: ${PARTITION}

Use B6000 or V100:
  PARTITION=b6000-al9_long scripts/submit_server_waveform_full_training.sh
  PARTITION=v100-al9_long scripts/submit_server_waveform_full_training.sh

If A100 is explicitly required, set ALLOW_A100=1.
EOF
  exit 2
fi

if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi
if [[ ! -e "${GRAPH_INPUT}" ]]; then
  echo "graph input not found: ${GRAPH_INPUT}" >&2
  echo "Set GRAPH_INPUT=/path/to/graph_directory_or_h5 to override the default." >&2
  exit 2
fi

SBATCH_DIR="${RUN_DIR}/slurm"
SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${SBATCH_DIR}" "${SLURM_LOG_DIR}" "${SUMMARY_DIR}" "${LOG_DIR}"

# Runtime cache is prepared inside the allocated Slurm job only.
# Do not copy the repo or venv before sbatch submission.

SBATCH_FILE="${SBATCH_DIR}/${RUN_NAME}.sbatch"
SBATCH_NODELIST_LINE=""
if [[ -n "${NODELIST}" ]]; then
  SBATCH_NODELIST_LINE="#SBATCH --nodelist=${NODELIST}"
fi

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${PARTITION}
${SBATCH_NODELIST_LINE}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_LOG_DIR}/%x_%j.out
#SBATCH --error=${SLURM_LOG_DIR}/%x_%j.err

set -Eeuo pipefail

JOB_LOG_PATH="${LOG_DIR}/${RUN_NAME}.job.log"
EARLY_LOG_PATH="${LOG_DIR}/${RUN_NAME}.early.log"

{
  echo "======================================================================"
  echo "EARLY JOB START"
  echo "date=\$(date)"
  echo "hostname=\$(hostname 2>/dev/null || true)"
  echo "slurm_job_id=\${SLURM_JOB_ID:-}"
  echo "job_log=\${JOB_LOG_PATH}"
  echo "early_log=\${EARLY_LOG_PATH}"
  echo "slurm_stdout=${SLURM_LOG_DIR}/%x_%j.out"
  echo "slurm_stderr=${SLURM_LOG_DIR}/%x_%j.err"
  echo "======================================================================"
} >> "\${EARLY_LOG_PATH}" 2>&1 || true

trap 'status=\$?; line=\${LINENO}; command=\${BASH_COMMAND}; echo "ERROR status=\${status} line=\${line} command=\${command}" >&2; echo "ERROR status=\${status} line=\${line} command=\${command}" >> "\${EARLY_LOG_PATH}" 2>/dev/null || true' ERR

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${SLURM_LOG_DIR}"
exec > >(tee -a "\${JOB_LOG_PATH}" "\${EARLY_LOG_PATH}") 2>&1

LOCAL_RUNTIME_JOB_DIR=""
LOCAL_RUNTIME_ROOT_SELECTED=""
REPO_EFFECTIVE="${REPO}"
PYTHON_BIN_EFFECTIVE=".venv/bin/python"

cleanup_local_runtime() {
  set +e
  if [[ -n "\${LOCAL_RUNTIME_JOB_DIR}" && -n "\${LOCAL_RUNTIME_ROOT_SELECTED}" && "${LOCAL_RUNTIME_CLEANUP}" == "1" ]]; then
    case "\${LOCAL_RUNTIME_JOB_DIR}" in
      "\${LOCAL_RUNTIME_ROOT_SELECTED%/}"/*)
        echo "Cleaning local runtime cache: \${LOCAL_RUNTIME_JOB_DIR}"
        rm -rf -- "\${LOCAL_RUNTIME_JOB_DIR}"
        ;;
      *)
        echo "Refusing to clean unexpected local runtime path: \${LOCAL_RUNTIME_JOB_DIR}" >&2
        ;;
    esac
  fi
}
trap cleanup_local_runtime EXIT

select_local_runtime_root() {
  local root
  local candidates
  local -a roots

  if [[ "${LOCAL_RUNTIME_ROOT}" == "auto" ]]; then
    candidates="${LOCAL_RUNTIME_ROOT_CANDIDATES}"
  else
    candidates="${LOCAL_RUNTIME_ROOT}"
    if [[ -n "${LOCAL_RUNTIME_FALLBACK_ROOTS}" ]]; then
      candidates="\${candidates}:${LOCAL_RUNTIME_FALLBACK_ROOTS}"
    fi
  fi

  IFS=: read -r -a roots <<< "\${candidates}"
  for root in "\${roots[@]}"; do
    [[ -z "\${root}" ]] && continue
    if mkdir -p "\${root}" 2>/dev/null && [[ -w "\${root}" ]]; then
      printf '%s\n' "\${root%/}"
      return 0
    fi
    echo "local runtime root is not writable: \${root}" >&2
  done
  return 1
}

copy_runtime_source_to_local() {
  local src="\$1"
  local dst="\$2"
  local copy_tool="${LOCAL_RUNTIME_COPY_TOOL}"
  local item

  if [[ "\${copy_tool}" == "auto" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      copy_tool="rsync"
    else
      copy_tool="cp"
    fi
  fi
  echo "local_runtime_copy_tool_effective=\${copy_tool}"
  if [[ "\${copy_tool}" == "rsync" ]]; then
    rsync -a --delete \
      --exclude .git \
      --exclude .mypy_cache \
      --exclude .pytest_cache \
      --exclude __pycache__ \
      --exclude '*.egg-info' \
      --exclude outputs \
      --exclude graphs \
      --exclude notebook \
      --exclude notebook_outputs \
      "\${src%/}/" "\${dst%/}/"
  else
    mkdir -p "\${dst}"
    for item in .venv src scripts configs docs pyproject.toml uv.lock setup.py README.md; do
      [[ -e "\${src%/}/\${item}" ]] || continue
      cp -a "\${src%/}/\${item}" "\${dst%/}/"
    done
  fi
}

if [[ "${LOCAL_RUNTIME_CACHE}" != "0" ]]; then
  if LOCAL_RUNTIME_ROOT_SELECTED="\$(select_local_runtime_root)"; then
    LOCAL_RUNTIME_JOB_DIR="\${LOCAL_RUNTIME_ROOT_SELECTED}/runtime/\${SLURM_JOB_ID:-manual_${RUN_ID}}_${RUN_NAME}"
    mkdir -p "\${LOCAL_RUNTIME_JOB_DIR}/src"
    echo "======================================================================"
    echo "COPY RUNTIME TO LOCAL STORAGE"
    echo "date=\$(date)"
    echo "runtime_source=${REPO}"
    echo "local_runtime_root_selected=\${LOCAL_RUNTIME_ROOT_SELECTED}"
    echo "local_runtime_job_dir=\${LOCAL_RUNTIME_JOB_DIR}"
    df -h "\${LOCAL_RUNTIME_ROOT_SELECTED}" || true
    if copy_runtime_source_to_local "${REPO}" "\${LOCAL_RUNTIME_JOB_DIR}/src"; then
      REPO_EFFECTIVE="\${LOCAL_RUNTIME_JOB_DIR}/src"
      PYTHON_BIN_EFFECTIVE="\${REPO_EFFECTIVE}/.venv/bin/python"
      if [[ ! -x "\${PYTHON_BIN_EFFECTIVE}" ]]; then
        echo "Local runtime has no executable venv python; falling back to source repo venv." >&2
        PYTHON_BIN_EFFECTIVE="${REPO}/.venv/bin/python"
      fi
    elif [[ "${LOCAL_RUNTIME_CACHE}" == "1" ]]; then
      echo "LOCAL_RUNTIME_CACHE=1 but runtime copy failed." >&2
      exit 2
    else
      echo "LOCAL_RUNTIME_CACHE=auto: runtime copy failed; using REPO directly." >&2
      REPO_EFFECTIVE="${REPO}"
      PYTHON_BIN_EFFECTIVE="${REPO}/.venv/bin/python"
    fi
    echo "repo_effective=\${REPO_EFFECTIVE}"
    echo "python_bin_effective=\${PYTHON_BIN_EFFECTIVE}"
    echo "date=\$(date)"
    echo "======================================================================"
  elif [[ "${LOCAL_RUNTIME_CACHE}" == "1" ]]; then
    echo "LOCAL_RUNTIME_CACHE=1 but no writable local runtime root was found." >&2
    exit 2
  else
    echo "LOCAL_RUNTIME_CACHE=auto: local runtime cache is unavailable; using REPO directly." >&2
  fi
else
  echo "LOCAL_RUNTIME_CACHE=${LOCAL_RUNTIME_CACHE}: using REPO directly."
fi

if [[ "${SKIP_MODULE_LOAD}" == "1" ]]; then
  echo "SKIP_MODULE_LOAD=1: not loading site modules"
else
  module purge
  module load gcc/13.1.0 cmake/3.28 cuda/12.6.0 hdf5/2.0.0 mkl/latest tbb/latest
fi

cd "\${REPO_EFFECTIVE}"

export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export TALESD_GNN_H5_MAX_OPEN_FILES="${TALESD_GNN_H5_MAX_OPEN_FILES}"
export PYTHON_BIN="\${PYTHON_BIN_EFFECTIVE}"
export PYTHONPATH="\${REPO_EFFECTIVE}/src:\${PYTHONPATH:-}"

GRAPH_INPUT_ORIGINAL="${GRAPH_INPUT}"
LOCAL_GRAPH_JOB_DIR=""
LOCAL_GRAPH_ROOT_SELECTED=""
LOCAL_GRAPH_CACHE_DIR=""
LOCAL_GRAPH_READY_FILE=""
LOCAL_GRAPH_TMP_DIR=""
LOCAL_GRAPH_CONTROL_LOCK_DIR=""
LOCAL_GRAPH_CONTROL_LOCK_OWNED=0
LOCAL_GRAPH_REF_FILE=""

cleanup_local_graph_cache() {
  local remaining_refs
  local acquired_cleanup_lock
  set +e

  if [[ -n "\${LOCAL_GRAPH_REF_FILE}" && -n "\${LOCAL_GRAPH_CACHE_DIR}" && "${LOCAL_GRAPH_CLEANUP}" == "1" ]]; then
    acquired_cleanup_lock=0
    if [[ "\${LOCAL_GRAPH_CONTROL_LOCK_OWNED}" == "1" ]]; then
      acquired_cleanup_lock=1
    elif acquire_local_graph_control_lock "\${LOCAL_GRAPH_CONTROL_LOCK_DIR}" "cleanup"; then
      acquired_cleanup_lock=1
    fi

    if [[ "\${acquired_cleanup_lock}" == "1" ]]; then
      rm -f -- "\${LOCAL_GRAPH_REF_FILE}"
      prune_stale_local_graph_refs_locked
      remaining_refs=0
      if [[ -d "\${LOCAL_GRAPH_CACHE_DIR}/.refs" ]]; then
        remaining_refs="\$(find "\${LOCAL_GRAPH_CACHE_DIR}/.refs" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        remaining_refs="\${remaining_refs//[[:space:]]/}"
      fi
      echo "local_graph_ref_removed=\${LOCAL_GRAPH_REF_FILE}"
      echo "local_graph_ref_remaining=\${remaining_refs}"
      if [[ "\${remaining_refs}" == "0" ]]; then
        echo "No remaining local graph references; deleting cache: \${LOCAL_GRAPH_CACHE_DIR}"
        rm -rf -- "\${LOCAL_GRAPH_CACHE_DIR}"
      fi
      if [[ "\${LOCAL_GRAPH_CONTROL_LOCK_OWNED}" == "1" ]]; then
        release_local_graph_control_lock
      fi
    else
      echo "Could not acquire local graph control lock during cleanup: \${LOCAL_GRAPH_CONTROL_LOCK_DIR}" >&2
    fi
    LOCAL_GRAPH_REF_FILE=""
  fi

  if [[ "\${LOCAL_GRAPH_CONTROL_LOCK_OWNED}" == "1" ]]; then
    [[ -n "\${LOCAL_GRAPH_TMP_DIR}" ]] && rm -rf -- "\${LOCAL_GRAPH_TMP_DIR}"
    release_local_graph_control_lock
  fi

  if [[ -n "\${LOCAL_GRAPH_JOB_DIR}" && -n "\${LOCAL_GRAPH_ROOT_SELECTED}" && "${LOCAL_GRAPH_CLEANUP}" == "1" ]]; then
    case "\${LOCAL_GRAPH_JOB_DIR}" in
      "\${LOCAL_GRAPH_ROOT_SELECTED%/}"/*)
        echo "Cleaning local graph cache: \${LOCAL_GRAPH_JOB_DIR}"
        rm -rf -- "\${LOCAL_GRAPH_JOB_DIR}"
        ;;
      *)
        echo "Refusing to clean unexpected local graph cache path: \${LOCAL_GRAPH_JOB_DIR}" >&2
        ;;
    esac
  fi
  cleanup_local_runtime
}
trap cleanup_local_graph_cache EXIT
trap 'echo "Received SIGTERM; cleaning local caches before exit" >&2; exit 143' TERM
trap 'echo "Received SIGINT; cleaning local caches before exit" >&2; exit 130' INT

select_local_graph_root() {
  local root
  local candidates
  local -a roots

  if [[ "${LOCAL_GRAPH_ROOT}" == "auto" ]]; then
    candidates="${LOCAL_GRAPH_ROOT_CANDIDATES}"
  else
    candidates="${LOCAL_GRAPH_ROOT}"
    if [[ -n "${LOCAL_GRAPH_FALLBACK_ROOTS}" ]]; then
      candidates="\${candidates}:${LOCAL_GRAPH_FALLBACK_ROOTS}"
    fi
  fi

  IFS=: read -r -a roots <<< "\${candidates}"
  for root in "\${roots[@]}"; do
    [[ -z "\${root}" ]] && continue
    if mkdir -p "\${root}" 2>/dev/null && [[ -w "\${root}" ]]; then
      printf '%s\n' "\${root%/}"
      return 0
    fi
    echo "local graph root is not writable: \${root}" >&2
  done
  return 1
}

local_graph_lock_is_stale() {
  local lock_dir="\$1"
  local owner_job_id=""
  local now
  local mtime
  local age

  [[ -d "\${lock_dir}" ]] || return 1

  if [[ -f "\${lock_dir}/owner_job_id" ]]; then
    owner_job_id="\$(cat "\${lock_dir}/owner_job_id" 2>/dev/null || true)"
    if [[ -n "\${owner_job_id}" ]] && command -v squeue >/dev/null 2>&1; then
      local squeue_output
      if ! squeue_output="\$(squeue -h -j "\${owner_job_id}" 2>/dev/null)"; then
        return 1
      fi
      if [[ -n "\${squeue_output}" ]]; then
        return 1
      fi
      return 0
    fi
    return 1
  fi

  now="\$(date +%s)"
  mtime="\$(stat -c %Y "\${lock_dir}" 2>/dev/null || printf '%s' "\${now}")"
  age=\$((now - mtime))
  (( age >= ${LOCAL_GRAPH_STALE_LOCK_SEC} ))
}

acquire_local_graph_control_lock() {
  local lock_dir="\$1"
  local purpose="\${2:-local_graph_cache}"
  local waited=0
  local sleep_sec=5

  while ! mkdir "\${lock_dir}" 2>/dev/null; do
    if local_graph_lock_is_stale "\${lock_dir}"; then
      echo "Removing stale local graph lock: \${lock_dir}" >&2
      rm -rf -- "\${lock_dir}"
      continue
    fi
    if (( waited >= ${LOCAL_GRAPH_WAIT_TIMEOUT_SEC} )); then
      echo "Timed out waiting for local graph lock after ${LOCAL_GRAPH_WAIT_TIMEOUT_SEC}s: \${lock_dir}" >&2
      return 124
    fi
    sleep "\${sleep_sec}"
    waited=\$((waited + sleep_sec))
    echo "waiting for local graph lock: purpose=\${purpose} waited=\${waited}s lock=\${lock_dir}"
  done

  LOCAL_GRAPH_CONTROL_LOCK_OWNED=1
  {
    echo "purpose=\${purpose}"
    echo "owner_job_id=\${SLURM_JOB_ID:-}"
    echo "owner_run_name=${RUN_NAME}"
    echo "owner_hostname=\$(hostname 2>/dev/null || true)"
    echo "created_at=\$(date)"
  } > "\${lock_dir}/owner" 2>/dev/null || true
  printf '%s\n' "\${SLURM_JOB_ID:-}" > "\${lock_dir}/owner_job_id" 2>/dev/null || true
}

release_local_graph_control_lock() {
  if [[ "\${LOCAL_GRAPH_CONTROL_LOCK_OWNED}" == "1" && -n "\${LOCAL_GRAPH_CONTROL_LOCK_DIR}" ]]; then
    rm -f -- "\${LOCAL_GRAPH_CONTROL_LOCK_DIR}/owner" "\${LOCAL_GRAPH_CONTROL_LOCK_DIR}/owner_job_id" 2>/dev/null || true
    rmdir "\${LOCAL_GRAPH_CONTROL_LOCK_DIR}" 2>/dev/null || true
    LOCAL_GRAPH_CONTROL_LOCK_OWNED=0
  fi
}

local_graph_ref_is_active() {
  local ref_file="\$1"
  local ref_job_id
  local squeue_output

  ref_job_id="\$(awk -F= '\$1 == "job_id" {print \$2; exit}' "\${ref_file}" 2>/dev/null || true)"
  if [[ -z "\${ref_job_id}" || ! "\${ref_job_id}" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if ! command -v squeue >/dev/null 2>&1; then
    return 0
  fi
  if ! squeue_output="\$(squeue -h -j "\${ref_job_id}" 2>/dev/null)"; then
    return 0
  fi
  [[ -n "\${squeue_output}" ]]
}

prune_stale_local_graph_refs_locked() {
  local ref_file

  [[ -d "\${LOCAL_GRAPH_CACHE_DIR}/.refs" ]] || return 0
  for ref_file in "\${LOCAL_GRAPH_CACHE_DIR}/.refs"/*; do
    [[ -f "\${ref_file}" ]] || continue
    if ! local_graph_ref_is_active "\${ref_file}"; then
      echo "Removing stale local graph ref: \${ref_file}" >&2
      rm -f -- "\${ref_file}"
    fi
  done
}

local_graph_cache_key() {
  local src="\$1"
  local canonical
  local digest
  local safe_name

  canonical="\$(readlink -f "\${src}" 2>/dev/null || printf '%s' "\${src}")"
  safe_name="\$(printf '%s' "\$(basename "\${canonical}")" | tr -c 'A-Za-z0-9_.-' '_')"
  safe_name="\${safe_name:-graph_input}"

  if command -v sha256sum >/dev/null 2>&1; then
    digest="\$(printf '%s' "\${canonical}" | sha256sum | awk '{print substr(\$1, 1, 12)}')"
  else
    digest="\$(printf '%s' "\${canonical}" | cksum | awk '{print \$1}')"
  fi

  printf '%s_%s\n' "\${safe_name}" "\${digest}"
}

copy_graph_input_to_local() {
  local src="\$1"
  local dst="\$2"
  local copy_tool="${LOCAL_GRAPH_COPY_TOOL}"

  if [[ "\${copy_tool}" == "auto" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      copy_tool="rsync"
    else
      copy_tool="cp"
    fi
  fi
  echo "local_graph_copy_tool_effective=\${copy_tool}"

  if [[ "\${copy_tool}" == "rsync" ]]; then
    if ! command -v rsync >/dev/null 2>&1; then
      echo "rsync requested but not found" >&2
      return 127
    fi
    if [[ -d "\${src}" ]]; then
      rsync -a --info=progress2 "\${src%/}/" "\${dst%/}/"
    else
      rsync -a --info=progress2 "\${src}" "\${dst%/}/"
    fi
  else
    if [[ -d "\${src}" ]]; then
      cp -a "\${src%/}/." "\${dst%/}/"
    else
      cp -a "\${src}" "\${dst%/}/"
    fi
  fi
}

prepare_shared_local_graph_cache() {
  local src="\$1"
  local cache_parent="\$2"
  local ref_count
  local stale_tmp

  LOCAL_GRAPH_CACHE_KEY="\$(local_graph_cache_key "\${src}")"
  LOCAL_GRAPH_CACHE_DIR="\${cache_parent}/\${LOCAL_GRAPH_CACHE_KEY}"
  LOCAL_GRAPH_CONTROL_LOCK_DIR="\${LOCAL_GRAPH_CACHE_DIR}.lock"
  LOCAL_GRAPH_READY_FILE="\${LOCAL_GRAPH_CACHE_DIR}/.ready"
  LOCAL_GRAPH_TMP_DIR="\${LOCAL_GRAPH_CACHE_DIR}.tmp.\${SLURM_JOB_ID:-manual_${RUN_ID}}"

  mkdir -p "\${cache_parent}" || return 1

  echo "======================================================================"
  echo "COPY GRAPH INPUT TO SHARED LOCAL CACHE"
  echo "date=\$(date)"
  echo "local_graph_cache=${LOCAL_GRAPH_CACHE}"
  echo "local_graph_cache_scope=${LOCAL_GRAPH_CACHE_SCOPE}"
  echo "local_graph_root_requested=${LOCAL_GRAPH_ROOT}"
  echo "local_graph_root_selected=\${LOCAL_GRAPH_ROOT_SELECTED}"
  echo "graph_input_original=\${src}"
  echo "local_graph_cache_dir=\${LOCAL_GRAPH_CACHE_DIR}"
  du -sh "\${src}" 2>/dev/null || true
  df -h "\${cache_parent}" || true

  acquire_local_graph_control_lock "\${LOCAL_GRAPH_CONTROL_LOCK_DIR}" "prepare_shared_cache" || return \$?

  for stale_tmp in "\${LOCAL_GRAPH_CACHE_DIR}".tmp.*; do
    [[ "\${stale_tmp}" == "\${LOCAL_GRAPH_TMP_DIR}" ]] && continue
    [[ -e "\${stale_tmp}" ]] || continue
    case "\${stale_tmp}" in
      "\${cache_parent}/\${LOCAL_GRAPH_CACHE_KEY}.tmp."*)
        echo "Removing stale local graph temp cache: \${stale_tmp}" >&2
        rm -rf -- "\${stale_tmp}"
        ;;
    esac
  done

  if [[ -f "\${LOCAL_GRAPH_READY_FILE}" ]]; then
    echo "Using existing local graph cache: \${LOCAL_GRAPH_CACHE_DIR}"
  else
    rm -rf -- "\${LOCAL_GRAPH_TMP_DIR}"
    mkdir -p "\${LOCAL_GRAPH_TMP_DIR}"
    if copy_graph_input_to_local "\${src}" "\${LOCAL_GRAPH_TMP_DIR}"; then
      printf '%s\n' "\${src}" > "\${LOCAL_GRAPH_TMP_DIR}/.source"
      touch "\${LOCAL_GRAPH_TMP_DIR}/.ready"
      if [[ -e "\${LOCAL_GRAPH_CACHE_DIR}" && ! -f "\${LOCAL_GRAPH_READY_FILE}" ]]; then
        rm -rf -- "\${LOCAL_GRAPH_CACHE_DIR}"
      fi
      mv "\${LOCAL_GRAPH_TMP_DIR}" "\${LOCAL_GRAPH_CACHE_DIR}"
    else
      COPY_STATUS="\$?"
      echo "Local graph copy failed with status \${COPY_STATUS}" >&2
      rm -rf -- "\${LOCAL_GRAPH_TMP_DIR}"
      release_local_graph_control_lock
      return "\${COPY_STATUS}"
    fi
  fi

  if [[ ! -f "\${LOCAL_GRAPH_READY_FILE}" ]]; then
    echo "Local graph cache is not ready after preparation: \${LOCAL_GRAPH_CACHE_DIR}" >&2
    release_local_graph_control_lock
    return 2
  fi

  mkdir -p "\${LOCAL_GRAPH_CACHE_DIR}/.refs"
  prune_stale_local_graph_refs_locked
  LOCAL_GRAPH_REF_FILE="\${LOCAL_GRAPH_CACHE_DIR}/.refs/\${SLURM_JOB_ID:-manual_${RUN_ID}}_${RUN_NAME}"
  {
    echo "job_id=\${SLURM_JOB_ID:-}"
    echo "run_name=${RUN_NAME}"
    echo "hostname=\$(hostname 2>/dev/null || true)"
    echo "pid=\$\$"
    echo "created_at=\$(date)"
    echo "graph_input_original=\${src}"
  } > "\${LOCAL_GRAPH_REF_FILE}"

  ref_count="\$(find "\${LOCAL_GRAPH_CACHE_DIR}/.refs" -maxdepth 1 -type f 2>/dev/null | wc -l)"
  ref_count="\${ref_count//[[:space:]]/}"
  echo "local_graph_ref_file=\${LOCAL_GRAPH_REF_FILE}"
  echo "local_graph_ref_count=\${ref_count}"

  release_local_graph_control_lock

  if [[ -d "\${src}" ]]; then
    GRAPH_INPUT="\${LOCAL_GRAPH_CACHE_DIR}"
  else
    GRAPH_INPUT="\${LOCAL_GRAPH_CACHE_DIR}/\$(basename "\${src}")"
  fi

  echo "graph_input_effective=\${GRAPH_INPUT}"
  du -sh "\${GRAPH_INPUT}" 2>/dev/null || true
  df -h "\${cache_parent}" || true
  echo "date=\$(date)"
  echo "======================================================================"
}

if [[ "${LOCAL_GRAPH_CACHE}" != "0" ]]; then
  LOCAL_GRAPH_AVAILABLE=1
  if ! LOCAL_GRAPH_ROOT_SELECTED="\$(select_local_graph_root)"; then
    LOCAL_GRAPH_AVAILABLE=0
  fi

  if [[ "\${LOCAL_GRAPH_AVAILABLE}" == "1" ]]; then
    if [[ "${LOCAL_GRAPH_CACHE_SCOPE}" == "shared" ]]; then
      LOCAL_GRAPH_CACHE_PARENT="\${LOCAL_GRAPH_ROOT_SELECTED}/cache"
      if ! prepare_shared_local_graph_cache "\${GRAPH_INPUT_ORIGINAL}" "\${LOCAL_GRAPH_CACHE_PARENT}"; then
        COPY_STATUS="\$?"
        if [[ "${LOCAL_GRAPH_CACHE}" == "auto" ]]; then
          echo "LOCAL_GRAPH_CACHE=auto: falling back to original GRAPH_INPUT." >&2
          LOCAL_GRAPH_AVAILABLE=0
        else
          exit "\${COPY_STATUS}"
        fi
      fi
    else
      LOCAL_GRAPH_JOB_DIR_CANDIDATE="\${LOCAL_GRAPH_ROOT_SELECTED}/\${SLURM_JOB_ID:-manual_${RUN_ID}}_${RUN_NAME}"
      LOCAL_GRAPH_INPUT_DIR="\${LOCAL_GRAPH_JOB_DIR_CANDIDATE}/graphs"

      if mkdir -p "\${LOCAL_GRAPH_INPUT_DIR}"; then
        LOCAL_GRAPH_JOB_DIR="\${LOCAL_GRAPH_JOB_DIR_CANDIDATE}"
        echo "======================================================================"
        echo "COPY GRAPH INPUT TO JOB LOCAL CACHE"
        echo "date=\$(date)"
        echo "local_graph_cache=${LOCAL_GRAPH_CACHE}"
        echo "local_graph_cache_scope=${LOCAL_GRAPH_CACHE_SCOPE}"
        echo "local_graph_root_requested=${LOCAL_GRAPH_ROOT}"
        echo "local_graph_root_selected=\${LOCAL_GRAPH_ROOT_SELECTED}"
        echo "graph_input_original=\${GRAPH_INPUT_ORIGINAL}"
        echo "local_graph_input_dir=\${LOCAL_GRAPH_INPUT_DIR}"
        du -sh "\${GRAPH_INPUT_ORIGINAL}" 2>/dev/null || true
        df -h "\${LOCAL_GRAPH_JOB_DIR}" || true
        if copy_graph_input_to_local "\${GRAPH_INPUT_ORIGINAL}" "\${LOCAL_GRAPH_INPUT_DIR}"; then
          if [[ -d "\${GRAPH_INPUT_ORIGINAL}" ]]; then
            GRAPH_INPUT="\${LOCAL_GRAPH_INPUT_DIR}"
          else
            GRAPH_INPUT="\${LOCAL_GRAPH_INPUT_DIR}/\$(basename "\${GRAPH_INPUT_ORIGINAL}")"
          fi

          echo "graph_input_effective=\${GRAPH_INPUT}"
          du -sh "\${GRAPH_INPUT}" 2>/dev/null || true
          df -h "\${LOCAL_GRAPH_JOB_DIR}" || true
          echo "date=\$(date)"
          echo "======================================================================"
        else
          COPY_STATUS="\$?"
          echo "Local graph copy failed with status \${COPY_STATUS}" >&2
          if [[ "${LOCAL_GRAPH_CACHE}" == "auto" ]]; then
            echo "LOCAL_GRAPH_CACHE=auto: falling back to original GRAPH_INPUT." >&2
            rm -rf -- "\${LOCAL_GRAPH_JOB_DIR}"
            LOCAL_GRAPH_JOB_DIR=""
            LOCAL_GRAPH_ROOT_SELECTED=""
            LOCAL_GRAPH_AVAILABLE=0
          else
            exit "\${COPY_STATUS}"
          fi
        fi
      else
        LOCAL_GRAPH_AVAILABLE=0
      fi
    fi
  fi

  if [[ "\${LOCAL_GRAPH_AVAILABLE}" != "1" ]]; then
    if [[ "${LOCAL_GRAPH_CACHE}" == "auto" ]]; then
      echo "LOCAL_GRAPH_CACHE=auto: local graph cache is unavailable; using original GRAPH_INPUT."
      LOCAL_GRAPH_JOB_DIR=""
      LOCAL_GRAPH_ROOT_SELECTED=""
    else
      echo "LOCAL_GRAPH_CACHE=1 but no writable local graph root was found." >&2
      echo "requested=${LOCAL_GRAPH_ROOT}" >&2
      echo "fallback_roots=${LOCAL_GRAPH_FALLBACK_ROOTS}" >&2
      exit 2
    fi
  fi
fi

for cmd in latex dvipng kpsewhich; do
  if ! command -v "\${cmd}" >/dev/null 2>&1; then
    echo "Missing \${cmd}. Diagnostics use matplotlib text.usetex=True, so install TeX Live before training." >&2
    exit 2
  fi
done
if ! kpsewhich amsmath.sty >/dev/null 2>&1; then
  echo "Missing amsmath.sty. Diagnostics use matplotlib text.usetex=True, so install TeX Live packages before training." >&2
  exit 2
fi

echo "======================================================================"
echo "SERVER WAVEFORM FULL TRAINING"
echo "date=\$(date)"
echo "hostname=\$(hostname)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "partition=${PARTITION}"
echo "nodelist=${NODELIST:-any}"
echo "gpus=${GPUS}"
echo "cpus_per_gpu=${CPUS_PER_GPU}"
echo "cpus_per_task=${CPUS_PER_TASK}"
echo "mem_per_gpu_gb=${MEM_PER_GPU_GB}"
echo "mem=${MEM}"
echo "graph_input_original=\${GRAPH_INPUT_ORIGINAL}"
echo "graph_input_effective=\${GRAPH_INPUT}"
echo "repo_original=${REPO}"
echo "repo_effective=\${REPO_EFFECTIVE}"
echo "python_bin=\${PYTHON_BIN_EFFECTIVE}"
echo "local_runtime_cache=${LOCAL_RUNTIME_CACHE}"
echo "local_runtime_root_requested=${LOCAL_RUNTIME_ROOT}"
echo "local_runtime_root_selected=\${LOCAL_RUNTIME_ROOT_SELECTED:-none}"
echo "local_runtime_root_candidates=${LOCAL_RUNTIME_ROOT_CANDIDATES}"
echo "local_runtime_fallback_roots=${LOCAL_RUNTIME_FALLBACK_ROOTS}"
echo "local_runtime_cleanup=${LOCAL_RUNTIME_CLEANUP}"
echo "local_runtime_copy_tool=${LOCAL_RUNTIME_COPY_TOOL}"
echo "skip_module_load=${SKIP_MODULE_LOAD}"
echo "local_graph_cache=${LOCAL_GRAPH_CACHE}"
echo "local_graph_root_requested=${LOCAL_GRAPH_ROOT}"
echo "local_graph_root_selected=\${LOCAL_GRAPH_ROOT_SELECTED:-none}"
echo "local_graph_root_candidates=${LOCAL_GRAPH_ROOT_CANDIDATES}"
echo "local_graph_fallback_roots=${LOCAL_GRAPH_FALLBACK_ROOTS}"
echo "local_graph_cache_scope=${LOCAL_GRAPH_CACHE_SCOPE}"
echo "local_graph_cleanup=${LOCAL_GRAPH_CLEANUP}"
echo "local_graph_copy_tool=${LOCAL_GRAPH_COPY_TOOL}"
echo "local_graph_wait_timeout_sec=${LOCAL_GRAPH_WAIT_TIMEOUT_SEC}"
echo "local_graph_stale_lock_sec=${LOCAL_GRAPH_STALE_LOCK_SEC}"
echo "run_dir=${RUN_DIR}"
echo "job_log=${LOG_DIR}/${RUN_NAME}.job.log"
echo "epochs=${TRAIN_EPOCHS}"
echo "batch_size=${BATCH_SIZE}"
echo "train_workers=${TRAIN_WORKERS}"
echo "preprocess_workers=${PREPROCESS_WORKERS}"
echo "prefetch_factor=${PREFETCH_FACTOR}"
echo "persistent_workers=${PERSISTENT_WORKERS}"
echo "h5_max_open_files=${H5_MAX_OPEN_FILES}"
echo "pin_memory=${PIN_MEMORY}"
echo "collate_threads=${COLLATE_THREADS}"
echo "epoch_learning_curve=${EPOCH_LEARNING_CURVE}"
echo "best_diagnostics=${BEST_DIAGNOSTICS}"
echo "best_diagnostic_max_graphs=${BEST_DIAGNOSTIC_MAX_GRAPHS}"
echo "training_task=${TRAINING_TASK}"
echo "mass_classification=${MASS_CLASSIFICATION}"
echo "mass_loss_weight=${MASS_LOSS_WEIGHT}"
echo "mass_loss_mode=${MASS_LOSS_MODE}"
echo "mass_ranking_weight=${MASS_RANKING_WEIGHT}"
echo "mass_ranking_margin=${MASS_RANKING_MARGIN}"
echo "classification_arch=${CLASSIFICATION_ARCH}"
echo "loss_mode=${LOSS_MODE}"
echo "quality_prediction=${QUALITY_PREDICTION}"
echo "error_prediction=${ERROR_PREDICTION}"
echo "nll_weight=${NLL_WEIGHT}"
echo "device=${DEVICE}"
echo "This job does not read DST files."
echo "======================================================================"

nvidia-smi

if [[ "${RUN_BUILD}" == "1" ]]; then
  uv sync
  ./build_extensions.sh
else
  echo "RUN_BUILD=0: using existing virtualenv and extension build"
fi

if [[ "${SUMMARIZE_GRAPHS}" == "1" ]]; then
  {
    echo "======================================================================"
    echo "SUMMARIZE GRAPH SHARDS"
    echo "date=\$(date)"
    echo "summary_json=${SUMMARY_DIR}/graph_summary.json"
    echo "summary_log=${SUMMARY_DIR}/graph_summary.log"
    "\${PYTHON_BIN_EFFECTIVE}" scripts/summarize_graph_shards.py "\${GRAPH_INPUT}" \\
      --workers "${PREPROCESS_WORKERS}" \\
      -o "${SUMMARY_DIR}/graph_summary.json"
    echo "date=\$(date)"
    echo "======================================================================"
  } 2>&1 | tee -a "${SUMMARY_DIR}/graph_summary.log"
fi

env \\
  SKIP_BUILD=1 \\
  OUTPUT_ROOT="${OUTPUT_ROOT}" \\
  RUN_ID="${RUN_ID}" \\
  RUN_NAME="${RUN_NAME}" \\
  RUN_DIR="${RUN_DIR}" \\
  GRAPH_INPUT="\${GRAPH_INPUT}" \\
  MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE}" \\
  HIDDEN_DIM="${HIDDEN_DIM}" \\
  LAYERS="${LAYERS}" \\
  DROPOUT="${DROPOUT}" \\
  READOUT_HEADS="${READOUT_HEADS}" \\
  CLASSIFICATION_ARCH="${CLASSIFICATION_ARCH}" \\
  DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM}" \\
  WAVEFORM_ENCODER="${WAVEFORM_ENCODER}" \\
  WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM}" \\
  WAVEFORM_TRANSFORMER_HEADS="${WAVEFORM_TRANSFORMER_HEADS}" \\
  WAVEFORM_TRANSFORMER_LAYERS="${WAVEFORM_TRANSFORMER_LAYERS}" \\
  TRAIN_EPOCHS="${TRAIN_EPOCHS}" \\
  BATCH_SIZE="${BATCH_SIZE}" \\
  LR="${LR}" \\
  WEIGHT_DECAY="${WEIGHT_DECAY}" \\
  LR_SCHEDULER="${LR_SCHEDULER}" \\
  LR_FACTOR="${LR_FACTOR}" \\
  LR_PATIENCE="${LR_PATIENCE}" \\
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \\
  EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS}" \\
  LOSS_MODE="${LOSS_MODE}" \\
  ENERGY_WEIGHT="${ENERGY_WEIGHT}" \\
  CORE_WEIGHT="${CORE_WEIGHT}" \\
  DIRECTION_WEIGHT="${DIRECTION_WEIGHT}" \\
  CORE_SCALE_KM="${CORE_SCALE_KM}" \\
  ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG}" \\
  QUALITY_PREDICTION="${QUALITY_PREDICTION}" \\
  QUALITY_WEIGHT="${QUALITY_WEIGHT}" \\
  QUALITY_ANGULAR_SCALE_DEG="${QUALITY_ANGULAR_SCALE_DEG}" \\
  QUALITY_CORE_SCALE_KM="${QUALITY_CORE_SCALE_KM}" \\
  QUALITY_ENERGY_SCALE="${QUALITY_ENERGY_SCALE}" \\
  ERROR_PREDICTION="${ERROR_PREDICTION}" \\
  ERROR_WEIGHT="${ERROR_WEIGHT}" \\
  ERROR_ANGULAR_SCALE_DEG="${ERROR_ANGULAR_SCALE_DEG}" \\
  ERROR_CORE_SCALE_KM="${ERROR_CORE_SCALE_KM}" \\
  ERROR_ENERGY_SCALE="${ERROR_ENERGY_SCALE}" \\
  NLL_WEIGHT="${NLL_WEIGHT}" \\
  NLL_SIGMA_ENERGY_FLOOR="${NLL_SIGMA_ENERGY_FLOOR}" \\
  NLL_SIGMA_ANGLE_FLOOR_DEG="${NLL_SIGMA_ANGLE_FLOOR_DEG}" \\
  NLL_SIGMA_CORE_FLOOR_KM="${NLL_SIGMA_CORE_FLOOR_KM}" \\
  TRAIN_WORKERS="${TRAIN_WORKERS}" \\
  PREPROCESS_WORKERS="${PREPROCESS_WORKERS}" \\
  COLLATE_THREADS="${COLLATE_THREADS}" \\
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \\
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \\
  H5_MAX_OPEN_FILES="${H5_MAX_OPEN_FILES}" \\
  PIN_MEMORY="${PIN_MEMORY}" \\
  EPOCH_LEARNING_CURVE="${EPOCH_LEARNING_CURVE}" \\
  BEST_DIAGNOSTICS="${BEST_DIAGNOSTICS}" \\
  BEST_DIAGNOSTIC_MAX_GRAPHS="${BEST_DIAGNOSTIC_MAX_GRAPHS}" \\
  TRAINING_TASK="${TRAINING_TASK}" \\
  MASS_CLASSIFICATION="${MASS_CLASSIFICATION}" \\
  MASS_LOSS_WEIGHT="${MASS_LOSS_WEIGHT}" \\
  MASS_LOSS_MODE="${MASS_LOSS_MODE}" \\
  MASS_FOCAL_GAMMA="${MASS_FOCAL_GAMMA}" \\
  MASS_POS_WEIGHT_MODE="${MASS_POS_WEIGHT_MODE}" \\
  MASS_RANKING_WEIGHT="${MASS_RANKING_WEIGHT}" \\
  MASS_RANKING_MARGIN="${MASS_RANKING_MARGIN}" \\
  MASS_COLLAPSE_PATIENCE="${MASS_COLLAPSE_PATIENCE}" \\
  MASS_COLLAPSE_SCORE_STD="${MASS_COLLAPSE_SCORE_STD}" \\
  MASS_COLLAPSE_BALANCED_ACCURACY="${MASS_COLLAPSE_BALANCED_ACCURACY}" \\
  VAL_FRACTION="${VAL_FRACTION}" \\
  TEST_FRACTION="${TEST_FRACTION}" \\
  SPLIT_MODE="${SPLIT_MODE}" \\
  PARTICLE_FILTER="${PARTICLE_FILTER}" \\
  DEVICE="${DEVICE}" \\
  DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT}" \\
  PRECISION_MIN_BIN_COUNT="${PRECISION_MIN_BIN_COUNT}" \\
  MAX_GRAPHS="${MAX_GRAPHS}" \\
  scripts/train_large_existing_graphs.sh
EOF

cat <<EOF
======================================================================
FULL TRAINING SBATCH READY

sbatch_file:
  ${SBATCH_FILE}

run_dir:
  ${RUN_DIR}

job_log:
  ${LOG_DIR}/${RUN_NAME}.job.log

graph_input_original:
  ${GRAPH_INPUT}

local_runtime_cache=${LOCAL_RUNTIME_CACHE}
local_runtime_root_requested=${LOCAL_RUNTIME_ROOT}
local_runtime_root_candidates=${LOCAL_RUNTIME_ROOT_CANDIDATES}
local_runtime_fallback_roots=${LOCAL_RUNTIME_FALLBACK_ROOTS}
local_runtime_cleanup=${LOCAL_RUNTIME_CLEANUP}
local_runtime_copy_tool=${LOCAL_RUNTIME_COPY_TOOL}
local_runtime_copy_timing=inside_allocated_job
skip_module_load=${SKIP_MODULE_LOAD}

local_graph_cache=${LOCAL_GRAPH_CACHE}
local_graph_root_requested=${LOCAL_GRAPH_ROOT}
local_graph_root_candidates=${LOCAL_GRAPH_ROOT_CANDIDATES}
local_graph_fallback_roots=${LOCAL_GRAPH_FALLBACK_ROOTS}
local_graph_cache_scope=${LOCAL_GRAPH_CACHE_SCOPE}
local_graph_cleanup=${LOCAL_GRAPH_CLEANUP}
local_graph_copy_tool=${LOCAL_GRAPH_COPY_TOOL}
local_graph_wait_timeout_sec=${LOCAL_GRAPH_WAIT_TIMEOUT_SEC}
local_graph_stale_lock_sec=${LOCAL_GRAPH_STALE_LOCK_SEC}

partition=${PARTITION}
nodelist=${NODELIST:-any}
time_limit=${TIME_LIMIT}
gpus=${GPUS}
cpus_per_task=${CPUS_PER_TASK}
mem=${MEM}
epochs=${TRAIN_EPOCHS}
batch_size=${BATCH_SIZE}
dropout=${DROPOUT}
weight_decay=${WEIGHT_DECAY}
train_workers=${TRAIN_WORKERS}
preprocess_workers=${PREPROCESS_WORKERS}
prefetch_factor=${PREFETCH_FACTOR}
persistent_workers=${PERSISTENT_WORKERS}
h5_max_open_files=${H5_MAX_OPEN_FILES}
pin_memory=${PIN_MEMORY}
collate_threads=${COLLATE_THREADS}
epoch_learning_curve=${EPOCH_LEARNING_CURVE}
best_diagnostics=${BEST_DIAGNOSTICS}
best_diagnostic_max_graphs=${BEST_DIAGNOSTIC_MAX_GRAPHS}
training_task=${TRAINING_TASK}
mass_classification=${MASS_CLASSIFICATION}
mass_loss_weight=${MASS_LOSS_WEIGHT}
mass_loss_mode=${MASS_LOSS_MODE}
mass_ranking_weight=${MASS_RANKING_WEIGHT}
mass_ranking_margin=${MASS_RANKING_MARGIN}
early_stopping_patience=${EARLY_STOPPING_PATIENCE}
early_stopping_min_epochs=${EARLY_STOPPING_MIN_EPOCHS}
loss_mode=${LOSS_MODE}
quality_prediction=${QUALITY_PREDICTION}
error_prediction=${ERROR_PREDICTION}
nll_weight=${NLL_WEIGHT}
graph_summary_log=${SUMMARY_DIR}/graph_summary.log

This job trains only from local HDF5 graph shards. It does not read DST.
======================================================================
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  exit 0
fi

sbatch "${SBATCH_FILE}"
