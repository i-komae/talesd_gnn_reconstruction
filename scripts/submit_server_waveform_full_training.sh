#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction}"
GRAPH_INPUT="${GRAPH_INPUT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-server_waveform_full_b6000_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"

PARTITION="${PARTITION:-b6000-al9_long}"
GPUS="${GPUS:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-256}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((GPUS * CPUS_PER_GPU))}"
MEM="${MEM:-$((GPUS * MEM_PER_GPU_GB))G}"
TIME_LIMIT="${TIME_LIMIT:-5-00:00:00}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-96}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-6}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-${CPUS_PER_TASK}}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TRAINING_TASK="${TRAINING_TASK:-reconstruction}"
MASS_CLASSIFICATION="${MASS_CLASSIFICATION:-0}"
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
HIDDEN_DIM="${HIDDEN_DIM:-224}"
LAYERS="${LAYERS:-6}"
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
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LR_PATIENCE="${LR_PATIENCE:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_MIN_EPOCHS:-0}"
LOSS_MODE="${LOSS_MODE:-physics-nll}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.5}"
CORE_WEIGHT="${CORE_WEIGHT:-1.2}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.8}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.10}"
ANGULAR_SCALE_DEG="${ANGULAR_SCALE_DEG:-0.75}"
QUALITY_PREDICTION="${QUALITY_PREDICTION:-1}"
QUALITY_WEIGHT="${QUALITY_WEIGHT:-0.2}"
QUALITY_ANGULAR_SCALE_DEG="${QUALITY_ANGULAR_SCALE_DEG:-0.8}"
QUALITY_CORE_SCALE_KM="${QUALITY_CORE_SCALE_KM:-0.04}"
QUALITY_ENERGY_SCALE="${QUALITY_ENERGY_SCALE:-0.10}"
ERROR_PREDICTION="${ERROR_PREDICTION:-1}"
ERROR_WEIGHT="${ERROR_WEIGHT:-0.0}"
ERROR_ANGULAR_SCALE_DEG="${ERROR_ANGULAR_SCALE_DEG:-${QUALITY_ANGULAR_SCALE_DEG}}"
ERROR_CORE_SCALE_KM="${ERROR_CORE_SCALE_KM:-${QUALITY_CORE_SCALE_KM}}"
ERROR_ENERGY_SCALE="${ERROR_ENERGY_SCALE:-${QUALITY_ENERGY_SCALE}}"
NLL_WEIGHT="${NLL_WEIGHT:-0.2}"
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
SUMMARIZE_GRAPHS="${SUMMARIZE_GRAPHS:-1}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
DRY_RUN="${DRY_RUN:-0}"

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
if [[ -z "${GRAPH_INPUT}" ]]; then
  cat >&2 <<EOF
GRAPH_INPUT is required.

The waveform schema changed to rise-aligned raw plus accepted-gapped traces.
Re-export graph HDF5 shards from DST, transfer them to the server, then submit with:

  GRAPH_INPUT=/path/to/new_graph_directory_or_h5 scripts/submit_server_waveform_full_training.sh
EOF
  exit 2
fi
if [[ ! -e "${GRAPH_INPUT}" ]]; then
  echo "graph input not found: ${GRAPH_INPUT}" >&2
  exit 2
fi

SBATCH_DIR="${RUN_DIR}/slurm"
SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${SBATCH_DIR}" "${SLURM_LOG_DIR}" "${SUMMARY_DIR}" "${LOG_DIR}"

SBATCH_FILE="${SBATCH_DIR}/${RUN_NAME}.sbatch"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_LOG_DIR}/%x_%j.out
#SBATCH --error=${SLURM_LOG_DIR}/%x_%j.err

set -euo pipefail

JOB_LOG_PATH="${LOG_DIR}/${RUN_NAME}.job.log"
mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}" "${SLURM_LOG_DIR}"
exec > >(tee -a "\${JOB_LOG_PATH}") 2>&1

module purge
module load gcc/13.1.0 cmake/3.28 cuda/12.6.0 hdf5/2.0.0 mkl/latest tbb/latest

cd "${REPO}"

export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"

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
echo "gpus=${GPUS}"
echo "cpus_per_gpu=${CPUS_PER_GPU}"
echo "cpus_per_task=${CPUS_PER_TASK}"
echo "mem_per_gpu_gb=${MEM_PER_GPU_GB}"
echo "mem=${MEM}"
echo "graph_input=${GRAPH_INPUT}"
echo "run_dir=${RUN_DIR}"
echo "job_log=${LOG_DIR}/${RUN_NAME}.job.log"
echo "epochs=${TRAIN_EPOCHS}"
echo "batch_size=${BATCH_SIZE}"
echo "train_workers=${TRAIN_WORKERS}"
echo "preprocess_workers=${PREPROCESS_WORKERS}"
echo "prefetch_factor=${PREFETCH_FACTOR}"
echo "collate_threads=${COLLATE_THREADS}"
echo "training_task=${TRAINING_TASK}"
echo "mass_classification=${MASS_CLASSIFICATION}"
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
    .venv/bin/python scripts/summarize_graph_shards.py "${GRAPH_INPUT}" \\
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
  GRAPH_INPUT="${GRAPH_INPUT}" \\
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
  TRAINING_TASK="${TRAINING_TASK}" \\
  MASS_CLASSIFICATION="${MASS_CLASSIFICATION}" \\
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

graph_input:
  ${GRAPH_INPUT}

partition=${PARTITION}
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
collate_threads=${COLLATE_THREADS}
training_task=${TRAINING_TASK}
mass_classification=${MASS_CLASSIFICATION}
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
