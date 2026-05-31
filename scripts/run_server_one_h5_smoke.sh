#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_DIR="${GRAPH_DIR:-/dicos_ui_home/ikomae/work/gnn/graphs/waveform_gnn_stream_20260520_123802}"
GRAPH_FILE="${GRAPH_FILE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-server_one_h5_smoke_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
SUMMARY_DIR="${RUN_DIR}/summaries"
CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_DIR}/one_h5_smoke.pt}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/one_h5_smoke.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${SUMMARY_DIR}/one_h5_summary.json}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
MAX_GRAPHS="${MAX_GRAPHS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-192}"
TRAIN_WORKERS="${TRAIN_WORKERS:-6}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DEVICE="${DEVICE:-cuda}"
STABILITY_WAIT_SECONDS="${STABILITY_WAIT_SECONDS:-20}"

HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-5}"
DROPOUT="${DROPOUT:-0.05}"
READOUT_HEADS="${READOUT_HEADS:-4}"
WAVEFORM_ENCODER="${WAVEFORM_ENCODER:-cnn-gru}"
WAVEFORM_EMBEDDING_DIM="${WAVEFORM_EMBEDDING_DIM:-64}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}"

select_graph_file() {
  local candidate
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] || continue
    local size_before size_after
    size_before=$(stat -c %s "${candidate}")
    sleep "${STABILITY_WAIT_SECONDS}"
    size_after=$(stat -c %s "${candidate}")
    if [[ "${size_before}" == "${size_after}" && "${size_before}" != "0" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    echo "skip changing file: ${candidate} (${size_before} -> ${size_after} bytes)" >&2
  done < <(find "${GRAPH_DIR}" -maxdepth 1 -type f -name '*.h5' | sort)
  return 1
}

{
  echo "======================================================================"
  echo "SERVER ONE-H5 SMOKE TEST"
  echo "This script does not read DST files and does not submit a Slurm job."
  echo "Run it inside an interactive Slurm GPU allocation."
  echo "date=$(date)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "graph_dir=${GRAPH_DIR}"
  echo "run_dir=${RUN_DIR}"
  echo "======================================================================"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi is not available in PATH"
  fi

  if [[ -z "${GRAPH_FILE}" ]]; then
    GRAPH_FILE="$(select_graph_file)"
  fi
  if [[ ! -f "${GRAPH_FILE}" ]]; then
    echo "graph file not found: ${GRAPH_FILE}" >&2
    exit 1
  fi

  echo "selected_graph_file=${GRAPH_FILE}"
  ls -lh "${GRAPH_FILE}"

  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "SKIP_BUILD=1: using existing extension build"
  else
    ./build_extensions.sh
  fi

  .venv/bin/python scripts/summarize_graph_shards.py "${GRAPH_FILE}" -o "${SUMMARY_JSON}"

  .venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

  echo "======================================================================"
  echo "START TRAINING SMOKE TEST"
  echo "epochs=${TRAIN_EPOCHS}"
  echo "max_graphs=${MAX_GRAPHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "train_workers=${TRAIN_WORKERS}"
  echo "device=${DEVICE}"
  echo "======================================================================"

  .venv/bin/talesd-gnn train \
    --graphs "${GRAPH_FILE}" \
    -o "${CHECKPOINT}" \
    --epochs "${TRAIN_EPOCHS}" \
    --max-graphs "${MAX_GRAPHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --layers "${LAYERS}" \
    --dropout "${DROPOUT}" \
    --lr-scheduler reduce-on-plateau \
    --lr-factor 0.5 \
    --lr-patience 2 \
    --model-architecture physics \
    --readout-heads "${READOUT_HEADS}" \
    --detector-embedding-dim 0 \
    --waveform-encoder "${WAVEFORM_ENCODER}" \
    --waveform-embedding-dim "${WAVEFORM_EMBEDDING_DIM}" \
    --waveform-transformer-heads 4 \
    --waveform-transformer-layers 1 \
    --loss-mode physics \
    --energy-loss-weight 1.2 \
    --core-loss-weight 1.0 \
    --direction-loss-weight 1.4 \
    --core-loss-scale-km 0.05 \
    --particle-filter all \
    --device "${DEVICE}" \
    --num-workers "${TRAIN_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --collate-backend cpp \
    --collate-threads "${COLLATE_THREADS}" \
    --split-mode source-stratified \
    --test-fraction 0.20 \
    --val-fraction 0.10 \
    --no-diagnostics

  echo "======================================================================"
  echo "SMOKE TEST COMPLETE"
  echo "checkpoint=${CHECKPOINT}"
  echo "metrics=${CHECKPOINT}.metrics.json"
  echo "summary_json=${SUMMARY_JSON}"
  echo "log_path=${LOG_PATH}"
  echo "date=$(date)"
  echo "======================================================================"
} 2>&1 | tee "${LOG_PATH}"
