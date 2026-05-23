#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-waveform_gnn_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${RUN_DIR}/graphs}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}/${RUN_NAME}.h5}"
GRAPH_INPUT="${GRAPH_INPUT:-}"

MAX_EVENTS_PER_FILE="${MAX_EVENTS_PER_FILE:-0}"
ENERGY_SAMPLE_PER_BIN="${ENERGY_SAMPLE_PER_BIN:-50000}"
ENERGY_SAMPLE_STRATIFY_PARTICLE="${ENERGY_SAMPLE_STRATIFY_PARTICLE:-1}"
ENERGY_BIN_WIDTH="${ENERGY_BIN_WIDTH:-0.1}"
ENERGY_OVERSAMPLE_FACTOR="${ENERGY_OVERSAMPLE_FACTOR:-1.0}"
EXPORT_WORKERS="${EXPORT_WORKERS:-6}"
WORKER_MAX_FILES="${WORKER_MAX_FILES:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-24}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-192}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-physics}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LAYERS="${LAYERS:-5}"
DROPOUT="${DROPOUT:-0.05}"
READOUT_HEADS="${READOUT_HEADS:-4}"
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
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
LOSS_MODE="${LOSS_MODE:-physics}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.2}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.4}"
CORE_SCALE_KM="${CORE_SCALE_KM:-0.12}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
DEVICE="${DEVICE:-cpu}"
DIAGNOSTIC_MIN_BIN_COUNT="${DIAGNOSTIC_MIN_BIN_COUNT:-1000}"
MAX_GRAPHS="${MAX_GRAPHS:-}"

CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
SUMMARY_DIR="${RUN_DIR}/summaries"
CONFIG_NAME="${CONFIG_NAME:-${MODEL_ARCHITECTURE}_${WAVEFORM_ENCODER}_h${HIDDEN_DIM}_l${LAYERS}_${PARTICLE_FILTER}_${TRAIN_EPOCHS}epoch}"
CHECKPOINT="${CHECKPOINT_DIR}/${CONFIG_NAME}.pt"
LOG_PATH="${LOG_DIR}/${CONFIG_NAME}.log"
READ_DONE_MARKER="${RUN_DIR}/DST_READ_COMPLETE.txt"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${CONFIG_DIR}" "${SUMMARY_DIR}" "${GRAPH_RUN_DIR}"

cat > "${CONFIG_DIR}/train.env" <<EOF
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GRAPH_RUN_DIR=${GRAPH_RUN_DIR}
GRAPH_OUTPUT=${GRAPH_OUTPUT}
GRAPH_INPUT=${GRAPH_INPUT}
MAX_EVENTS_PER_FILE=${MAX_EVENTS_PER_FILE}
ENERGY_SAMPLE_PER_BIN=${ENERGY_SAMPLE_PER_BIN}
ENERGY_SAMPLE_STRATIFY_PARTICLE=${ENERGY_SAMPLE_STRATIFY_PARTICLE}
ENERGY_BIN_WIDTH=${ENERGY_BIN_WIDTH}
ENERGY_OVERSAMPLE_FACTOR=${ENERGY_OVERSAMPLE_FACTOR}
MODEL_ARCHITECTURE=${MODEL_ARCHITECTURE}
HIDDEN_DIM=${HIDDEN_DIM}
LAYERS=${LAYERS}
WAVEFORM_ENCODER=${WAVEFORM_ENCODER}
WAVEFORM_EMBEDDING_DIM=${WAVEFORM_EMBEDDING_DIM}
DETECTOR_EMBEDDING_DIM=${DETECTOR_EMBEDDING_DIM}
TRAIN_EPOCHS=${TRAIN_EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
VAL_FRACTION=${VAL_FRACTION}
TEST_FRACTION=${TEST_FRACTION}
MAX_GRAPHS=${MAX_GRAPHS}
EOF

{
  echo "run_dir=${RUN_DIR}"
  echo "graph_output=${GRAPH_OUTPUT}"
  echo "graph_input=${GRAPH_INPUT:-}"
  echo "checkpoint=${CHECKPOINT}"
  echo "model_architecture=${MODEL_ARCHITECTURE}"
  echo "waveform_encoder=${WAVEFORM_ENCODER}"
  echo "detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}"
  echo "max_events_per_file=${MAX_EVENTS_PER_FILE}"
  echo "energy_sample_per_bin=${ENERGY_SAMPLE_PER_BIN}"
  echo "energy_sample_stratify_particle=${ENERGY_SAMPLE_STRATIFY_PARTICLE}"
  echo "energy_bin_width=${ENERGY_BIN_WIDTH}"
  echo "train_epochs=${TRAIN_EPOCHS}"
  echo "test_fraction=${TEST_FRACTION}"
  date

  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "SKIP_BUILD=1: using existing extension build"
  else
    ./build_extensions.sh
  fi

  if [[ -z "${GRAPH_INPUT}" ]]; then
    export_energy_args=()
    if [[ -n "${ENERGY_SAMPLE_PER_BIN}" && "${ENERGY_SAMPLE_PER_BIN}" != "0" ]]; then
      export_energy_args=(
        --energy-sample-per-bin "${ENERGY_SAMPLE_PER_BIN}"
        --energy-bin-width "${ENERGY_BIN_WIDTH}"
        --energy-oversample-factor "${ENERGY_OVERSAMPLE_FACTOR}"
      )
      if [[ "${ENERGY_SAMPLE_STRATIFY_PARTICLE}" == "1" ]]; then
        export_energy_args+=(--energy-sample-stratify-particle)
      fi
    fi
    export_cmd=(.venv/bin/talesd-gnn export
      --input-dir /Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313
      --input-dir /Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313
      --input-dir /Volumes/TALE/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313
      --input-dir /Volumes/TALE/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316
      --input-dir /Volumes/TALE/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316
      --input-dir /Volumes/TALE/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316
      -o "${GRAPH_OUTPUT}"
      --kind mc
      --max-events-per-file "${MAX_EVENTS_PER_FILE}"
      --workers "${EXPORT_WORKERS}"
      --worker-max-files "${WORKER_MAX_FILES}"
      --shard-size 50000
      --open-retries 3
      "${export_energy_args[@]}"
      --skip-errors)
    "${export_cmd[@]}"
    GRAPH_INPUT="${GRAPH_OUTPUT}"
  fi

  cat <<EOF | tee "${READ_DONE_MARKER}"
======================================================================
DST FILE READING COMPLETE

Training will now use local waveform HDF5 graph shards:
  ${GRAPH_INPUT}

Network DST input under /Volumes/TALE is no longer needed by this run.
$(date)
======================================================================
EOF

  train_cmd=(.venv/bin/talesd-gnn train
    --graphs "${GRAPH_INPUT}"
    -o "${CHECKPOINT}"
    --epochs "${TRAIN_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --hidden-dim "${HIDDEN_DIM}"
    --layers "${LAYERS}"
    --dropout "${DROPOUT}"
    --lr-scheduler "${LR_SCHEDULER}"
    --lr-factor "${LR_FACTOR}"
    --lr-patience "${LR_PATIENCE}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --model-architecture "${MODEL_ARCHITECTURE}"
    --readout-heads "${READOUT_HEADS}"
    --detector-embedding-dim "${DETECTOR_EMBEDDING_DIM}"
    --waveform-encoder "${WAVEFORM_ENCODER}"
    --waveform-embedding-dim "${WAVEFORM_EMBEDDING_DIM}"
    --waveform-transformer-heads "${WAVEFORM_TRANSFORMER_HEADS}"
    --waveform-transformer-layers "${WAVEFORM_TRANSFORMER_LAYERS}"
    --loss-mode "${LOSS_MODE}"
    --energy-loss-weight "${ENERGY_WEIGHT}"
    --core-loss-weight "${CORE_WEIGHT}"
    --direction-loss-weight "${DIRECTION_WEIGHT}"
    --core-loss-scale-km "${CORE_SCALE_KM}"
    --particle-filter "${PARTICLE_FILTER}"
    --device "${DEVICE}"
    --num-workers "${TRAIN_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --collate-backend cpp
    --collate-threads "${COLLATE_THREADS}"
    --split-mode "${SPLIT_MODE}"
    --test-fraction "${TEST_FRACTION}"
    --val-fraction "${VAL_FRACTION}"
    --diagnostic-energy-bin-width 0.1
    --diagnostic-min-bin-count "${DIAGNOSTIC_MIN_BIN_COUNT}")
  if [[ -n "${MAX_GRAPHS}" ]]; then
    train_cmd+=(--max-graphs "${MAX_GRAPHS}")
  fi
  "${train_cmd[@]}"

  echo "checkpoint=${CHECKPOINT}"
  echo "metrics=${CHECKPOINT}.metrics.json"
  echo "diagnostics_dir=${CHECKPOINT}.diagnostics"
  echo "log_path=${LOG_PATH}"
  date
} 2>&1 | tee "${LOG_PATH}"

echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "log_path=${LOG_PATH}"
echo "dst_read_complete=${READ_DONE_MARKER}"
