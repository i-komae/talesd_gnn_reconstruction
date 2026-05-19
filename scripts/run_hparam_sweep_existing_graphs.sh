#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_DIR="${GRAPH_DIR:-${HOME}/TALE/gnn/outputs/graphs}"
GRAPH_INPUT="${GRAPH_INPUT:-${GRAPH_DIR}/mass_12h_64perfile_6epoch.h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/TALE/gnn/outputs/talesd_gnn_reconstruction}"
TAG_PREFIX="${TAG_PREFIX:-hparam_reco_existing}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
TRAIN_WORKERS="${TRAIN_WORKERS:-2}"
COLLATE_THREADS="${COLLATE_THREADS:-1}"
DETECTOR_EMBEDDING_DIM="${DETECTOR_EMBEDDING_DIM:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
SPLIT_MODE="${SPLIT_MODE:-source-stratified}"
PARTICLE_FILTER="${PARTICLE_FILTER:-all}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
LR_PATIENCE="${LR_PATIENCE:-1}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LOSS_MODE="${LOSS_MODE:-scaled-mse}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
CORE_WEIGHT="${CORE_WEIGHT:-1.0}"
DIRECTION_WEIGHT="${DIRECTION_WEIGHT:-1.0}"
MODEL_DIR="${OUTPUT_ROOT}/models"
LOG_DIR="${OUTPUT_ROOT}/logs"
SWEEP_DIR="${OUTPUT_ROOT}/sweeps"
SWEEP_PROGRESS_INTERVAL="${SWEEP_PROGRESS_INTERVAL:-30}"
SWEEP_PROGRESS_POLL_INTERVAL="${SWEEP_PROGRESS_POLL_INTERVAL:-5}"

mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${SWEEP_DIR}"

SWEEP_LOG="${LOG_DIR}/${TAG_PREFIX}_${RUN_ID}_${PARTICLE_FILTER}_sweep.log"
touch "${SWEEP_LOG}"
exec > >(tee -a "${SWEEP_LOG}") 2>&1
source scripts/lib_sweep_progress.sh

CONFIGS=(
  "h128_l4_lr5e4|128|4|5e-4|0.05|1e-4|reduce-on-plateau"
  "h160_l4_lr5e4|160|4|5e-4|0.05|1e-4|reduce-on-plateau"
  "h192_l4_lr5e4|192|4|5e-4|0.05|1e-4|reduce-on-plateau"
  "h160_l5_lr5e4|160|5|5e-4|0.05|1e-4|reduce-on-plateau"
)

ACTIVE_CONFIGS=()
for config in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg_tag _hidden_dim _layers _lr _dropout _weight_decay _lr_scheduler <<< "${config}"
  if [[ -z "${CONFIG_FILTER:-}" || "${cfg_tag}" == *"${CONFIG_FILTER}"* ]]; then
    ACTIVE_CONFIGS+=("${config}")
  fi
done
if (( ${#ACTIVE_CONFIGS[@]} == 0 )); then
  echo "No configs matched CONFIG_FILTER='${CONFIG_FILTER:-}'" >&2
  exit 2
fi

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  echo "SKIP_BUILD=1: using existing extension build"
else
  ./build_extensions.sh
fi

cat <<EOF
======================================================================
HYPERPARAMETER SWEEP READY

This sweep trains from local HDF5 graph shards:
  ${GRAPH_INPUT}

No DST files are read by this script.
Network storage is not needed after this line if the graph shards are local.

run_id=${RUN_ID}
output_root=${OUTPUT_ROOT}
parallel_jobs=${PARALLEL_JOBS}
train_workers_per_job=${TRAIN_WORKERS}
collate_threads_per_job=${COLLATE_THREADS}
train_epochs=${TRAIN_EPOCHS}
split_mode=${SPLIT_MODE}
particle_filter=${PARTICLE_FILTER}
config_filter=${CONFIG_FILTER:-}
loss_mode=${LOSS_MODE}
energy_weight=${ENERGY_WEIGHT}
core_weight=${CORE_WEIGHT}
direction_weight=${DIRECTION_WEIGHT}
detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}
$(date)
======================================================================
EOF

run_one() {
  local config="$1"
  local cfg_tag hidden_dim layers lr dropout weight_decay lr_scheduler
  IFS='|' read -r cfg_tag hidden_dim layers lr dropout weight_decay lr_scheduler <<< "${config}"

  local tag
  tag="$(job_tag_for_config "${config}")"
  local checkpoint="${MODEL_DIR}/${tag}.pt"
  local log_path
  log_path="$(job_log_path_for_config "${config}")"

  {
    echo "tag=${tag}"
    echo "graph_input=${GRAPH_INPUT}"
    echo "checkpoint=${checkpoint}"
    echo "hidden_dim=${hidden_dim}"
    echo "layers=${layers}"
    echo "lr=${lr}"
    echo "dropout=${dropout}"
    echo "weight_decay=${weight_decay}"
    echo "lr_scheduler=${lr_scheduler}"
    echo "loss_mode=${LOSS_MODE}"
    echo "energy_weight=${ENERGY_WEIGHT}"
    echo "core_weight=${CORE_WEIGHT}"
    echo "direction_weight=${DIRECTION_WEIGHT}"
    echo "detector_embedding_dim=${DETECTOR_EMBEDDING_DIM}"
    echo "particle_filter=${PARTICLE_FILTER}"
    echo "This job does not read DST files."
    date

    cmd=(.venv/bin/talesd-gnn train \
      --graphs "${GRAPH_INPUT}" \
      -o "${checkpoint}" \
      --epochs "${TRAIN_EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --lr "${lr}" \
      --weight-decay "${weight_decay}" \
      --hidden-dim "${hidden_dim}" \
      --layers "${layers}" \
      --dropout "${dropout}" \
      --lr-scheduler "${lr_scheduler}" \
      --lr-factor "${LR_FACTOR}" \
      --lr-patience "${LR_PATIENCE}" \
      --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
      --model-architecture baseline \
      --detector-embedding-dim "${DETECTOR_EMBEDDING_DIM}" \
      --loss-mode "${LOSS_MODE}" \
      --energy-loss-weight "${ENERGY_WEIGHT}" \
      --core-loss-weight "${CORE_WEIGHT}" \
      --direction-loss-weight "${DIRECTION_WEIGHT}" \
      --particle-filter "${PARTICLE_FILTER}" \
      --device cpu \
      --num-workers "${TRAIN_WORKERS}" \
      --prefetch-factor 2 \
      --collate-backend cpp \
      --collate-threads "${COLLATE_THREADS}" \
      --split-mode "${SPLIT_MODE}" \
      --test-fraction "${TEST_FRACTION}" \
      --val-fraction "${VAL_FRACTION}" \
      --diagnostic-energy-bin-width 0.1 \
      --diagnostic-min-bin-count 20)
    if [[ -n "${MAX_GRAPHS:-}" ]]; then
      cmd+=(--max-graphs "${MAX_GRAPHS}")
    fi
    if [[ "${NO_DIAGNOSTICS:-0}" == "1" ]]; then
      cmd+=(--no-diagnostics)
    fi
    "${cmd[@]}"

    echo "checkpoint=${checkpoint}"
    echo "metrics=${checkpoint}.metrics.json"
    echo "diagnostics_dir=${checkpoint}.diagnostics"
    date
  } > "${log_path}" 2>&1
}

job_tag_for_config() {
  local config="$1"
  local cfg_tag _hidden_dim _layers _lr _dropout _weight_decay _lr_scheduler
  IFS='|' read -r cfg_tag _hidden_dim _layers _lr _dropout _weight_decay _lr_scheduler <<< "${config}"
  printf "%s" "${TAG_PREFIX}_${RUN_ID}_${PARTICLE_FILTER}_${cfg_tag}_${TRAIN_EPOCHS}epoch"
}

job_display_tag_for_config() {
  local config="$1"
  local cfg_tag _hidden_dim _layers _lr _dropout _weight_decay _lr_scheduler
  IFS='|' read -r cfg_tag _hidden_dim _layers _lr _dropout _weight_decay _lr_scheduler <<< "${config}"
  printf "%s" "${cfg_tag}"
}

job_log_path_for_config() {
  local config="$1"
  printf "%s/%s.log" "${LOG_DIR}" "$(job_tag_for_config "${config}")"
}

running_pids=()
running_tags=()
running_logs=()
failed=0
failed_jobs=0
completed_jobs=0
sweep_total_jobs="${#ACTIVE_CONFIGS[@]}"
sweep_started_at="$(date +%s)"
sweep_last_report_at=0
parallel_jobs_int=$((PARALLEL_JOBS < 1 ? 1 : PARALLEL_JOBS))

sweep_report
for config in "${ACTIVE_CONFIGS[@]}"; do
  sweep_wait_for_slot "${parallel_jobs_int}"
  run_one "${config}" &
  pid="$!"
  idx="${#running_pids[@]}"
  running_pids[$idx]="${pid}"
  running_tags[$idx]="$(job_display_tag_for_config "${config}")"
  running_logs[$idx]="$(job_log_path_for_config "${config}")"
  printf "sweep job started: %-28s pid=%s log=%s\n" "${running_tags[$idx]}" "${pid}" "${running_logs[$idx]}"
  sweep_report
done

sweep_wait_all
sweep_report

metrics_args=()
for config in "${ACTIVE_CONFIGS[@]}"; do
  IFS='|' read -r cfg_tag _hidden_dim _layers _lr _dropout _weight_decay _lr_scheduler <<< "${config}"
  tag="${TAG_PREFIX}_${RUN_ID}_${PARTICLE_FILTER}_${cfg_tag}_${TRAIN_EPOCHS}epoch"
  metrics_path="${MODEL_DIR}/${tag}.pt.metrics.json"
  if [[ -f "${metrics_path}" ]]; then
    metrics_args+=("${metrics_path}")
  fi
done

summary_csv="${SWEEP_DIR}/${TAG_PREFIX}_${RUN_ID}_${PARTICLE_FILTER}_summary.csv"
if (( ${#metrics_args[@]} > 0 )); then
  .venv/bin/python scripts/summarize_metrics.py "${metrics_args[@]}" -o "${summary_csv}"
  echo "summary_csv=${summary_csv}"
else
  echo "No metrics files were produced." >&2
  failed=1
fi

echo "logs=${LOG_DIR}/${TAG_PREFIX}_${RUN_ID}_${PARTICLE_FILTER}_*.log"
echo "sweep_log=${SWEEP_LOG}"
date
exit "${failed}"
