#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_DIR="${GRAPH_DIR:-${HOME}/TALE/gnn/outputs/graphs}"
GRAPH_INPUT="${GRAPH_INPUT:-${GRAPH_DIR}/mass_12h_64perfile_6epoch.h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
TAG_PREFIX="${TAG_PREFIX:-species_reco_existing}"
SPECIES_LIST="${SPECIES_LIST:-proton iron}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
COLLATE_THREADS="${COLLATE_THREADS:-0}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.05}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
MODEL_DIR="${OUTPUT_ROOT}/models"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${TAG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"
if [[ "${TEE_LOG:-1}" == "1" ]]; then
  exec > >(tee -a "${LOG_PATH}") 2>&1
fi

echo "tag_prefix=${TAG_PREFIX}"
echo "graph_input=${GRAPH_INPUT}"
echo "output_root=${OUTPUT_ROOT}"
echo "species_list=${SPECIES_LIST}"
echo "train_epochs=${TRAIN_EPOCHS}"
echo "hidden_dim=${HIDDEN_DIM}"
echo "layers=${LAYERS}"
echo "lr=${LR}"
echo "dropout=${DROPOUT}"
echo "batch_size=${BATCH_SIZE}"
echo "train_workers=${TRAIN_WORKERS}"
echo "collate_threads=${COLLATE_THREADS}"
echo "detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}"
echo "split_mode=${SPLIT_MODE}"
echo "loss_mode=scaled-mse"
echo "model_architecture=baseline"
echo "This script does not read DST files."
date

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  echo "SKIP_BUILD=1: using existing extension build"
else
  ./build_extensions.sh
fi

cat <<EOF
======================================================================
LOCAL GRAPH INPUT READY

This run trains species-specific models from local HDF5 graph shards:
  ${GRAPH_INPUT}

No DST files are read by this script.
Network storage is not needed after this line if the graph shards are local.
$(date)
======================================================================
EOF

MAX_GRAPHS_ARG=()
if [[ -n "${MAX_GRAPHS:-}" ]]; then
  MAX_GRAPHS_ARG=(--max-graphs "${MAX_GRAPHS}")
fi

for species in ${SPECIES_LIST}; do
  if [[ "${species}" != "proton" && "${species}" != "iron" ]]; then
    echo "Unsupported species: ${species}" >&2
    exit 2
  fi

  tag="${TAG_PREFIX}_${species}_${TRAIN_EPOCHS}epoch"
  checkpoint="${MODEL_DIR}/${tag}.pt"

  echo "======================================================================"
  echo "Training ${species}-only reconstruction model"
  echo "checkpoint=${checkpoint}"
  echo "======================================================================"

  .venv/bin/talesd-gnn train \
    --graphs "${GRAPH_INPUT}" \
    -o "${checkpoint}" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --layers "${LAYERS}" \
    --dropout "${DROPOUT}" \
    --model-architecture baseline \
    --detector-embedding-dim "${DETECTOR_EMBEDDING_DIM}" \
    --loss-mode scaled-mse \
    --particle-filter "${species}" \
    --device cpu \
    --num-workers "${TRAIN_WORKERS}" \
    --prefetch-factor 2 \
    --collate-backend cpp \
    --collate-threads "${COLLATE_THREADS}" \
    --split-mode "${SPLIT_MODE}" \
    --test-fraction "${TEST_FRACTION}" \
    --val-fraction "${VAL_FRACTION}" \
    --diagnostic-energy-bin-width 0.1 \
    --diagnostic-min-bin-count 20 \
    "${MAX_GRAPHS_ARG[@]}"

  diagnostics_dir="${checkpoint}.diagnostics"
  echo "species=${species}"
  echo "checkpoint=${checkpoint}"
  echo "metrics=${checkpoint}.metrics.json"
  echo "diagnostics_dir=${diagnostics_dir}"
  echo "learning_curve=${diagnostics_dir}/learning_curve.pdf"
  echo "validation_dir=${diagnostics_dir}/validation"
  echo "test_dir=${diagnostics_dir}/test"
done

echo "log_path=${LOG_PATH}"
date
