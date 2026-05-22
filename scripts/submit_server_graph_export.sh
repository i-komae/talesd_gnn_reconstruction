#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction}"
GRAPH_ROOT="${GRAPH_ROOT:-/dicos_ui_home/ikomae/work/gnn/graphs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ENERGY_SAMPLE_PER_BIN="${ENERGY_SAMPLE_PER_BIN:-200000}"
RUN_NAME="${RUN_NAME:-server_graph_export_energyflat${ENERGY_SAMPLE_PER_BIN}_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
GRAPH_RUN_DIR="${GRAPH_RUN_DIR:-${GRAPH_ROOT}/${RUN_NAME}}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-${GRAPH_RUN_DIR}/${RUN_NAME}.h5}"

DEFAULT_INPUT_DIRS="/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313:/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313:/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316:/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316"
INPUT_DIRS="${INPUT_DIRS:-${DEFAULT_INPUT_DIRS}}"
INPUT_LISTS="${INPUT_LISTS:-}"
INPUT_FILES="${INPUT_FILES:-}"

KIND="${KIND:-mc}"
CONST_DST="${CONST_DST:-${TALESD_CONST_DST:-}}"
if [[ -z "${CONST_DST}" && -n "${TADIR:-}" ]]; then
  CONST_DST="${TADIR%/}/data/SD/talesdconst_pass2.dst"
fi

PARTITION="${PARTITION:-edr1-al9_large}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEM="${MEM:-256G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"

EXPORT_WORKERS="${EXPORT_WORKERS:-${CPUS_PER_TASK}}"
SUMMARY_WORKERS="${SUMMARY_WORKERS:-${CPUS_PER_TASK}}"
WORKER_MAX_FILES="${WORKER_MAX_FILES:-200}"
MAX_EVENTS="${MAX_EVENTS:-}"
MAX_EVENTS_PER_FILE="${MAX_EVENTS_PER_FILE:-0}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
OPEN_RETRIES="${OPEN_RETRIES:-3}"
OPEN_RETRY_DELAY="${OPEN_RETRY_DELAY:-1.0}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
ENERGY_SAMPLE_STRATIFY_PARTICLE="${ENERGY_SAMPLE_STRATIFY_PARTICLE:-1}"
ENERGY_BIN_WIDTH="${ENERGY_BIN_WIDTH:-0.1}"
ENERGY_OVERSAMPLE_FACTOR="${ENERGY_OVERSAMPLE_FACTOR:-1.0}"
SEED="${SEED:-12345}"
KEEP_NON_MODE0="${KEEP_NON_MODE0:-0}"
SKIP_ERRORS="${SKIP_ERRORS:-1}"

RUN_BUILD="${RUN_BUILD:-1}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/dicos_ui_home/ikomae/work/uv-cache}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${REPO}" ]]; then
  echo "repo not found: ${REPO}" >&2
  exit 2
fi
if [[ "${KIND}" == "mc" && -z "${CONST_DST}" ]]; then
  cat >&2 <<EOF
CONST_DST is required for MC export.

Set one of:
  CONST_DST=/path/to/talesdconst_pass2.dst
  TALESD_CONST_DST=/path/to/talesdconst_pass2.dst
  TADIR=/path/to/TALE
EOF
  exit 2
fi
if [[ "${KIND}" == "mc" && ! -f "${CONST_DST}" ]]; then
  echo "const DST not found: ${CONST_DST}" >&2
  exit 2
fi
if [[ -z "${INPUT_DIRS}${INPUT_LISTS}${INPUT_FILES}" ]]; then
  echo "No DST input specified. Set INPUT_DIRS, INPUT_LISTS, or INPUT_FILES." >&2
  exit 2
fi

q() {
  printf "%q" "$1"
}

SBATCH_DIR="${RUN_DIR}/slurm"
SLURM_LOG_DIR="${RUN_DIR}/slurm_logs"
LOG_DIR="${RUN_DIR}/logs"
CONFIG_DIR="${RUN_DIR}/config"
SUMMARY_DIR="${RUN_DIR}/summaries"
mkdir -p "${SBATCH_DIR}" "${SLURM_LOG_DIR}" "${LOG_DIR}" "${CONFIG_DIR}" "${SUMMARY_DIR}" "${GRAPH_RUN_DIR}"

SBATCH_FILE="${SBATCH_DIR}/${RUN_NAME}.sbatch"

cat > "${CONFIG_DIR}/export_submit.env" <<EOF
REPO=${REPO}
OUTPUT_ROOT=${OUTPUT_ROOT}
GRAPH_ROOT=${GRAPH_ROOT}
RUN_ID=${RUN_ID}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GRAPH_RUN_DIR=${GRAPH_RUN_DIR}
GRAPH_OUTPUT=${GRAPH_OUTPUT}
INPUT_DIRS=${INPUT_DIRS}
INPUT_LISTS=${INPUT_LISTS}
INPUT_FILES=${INPUT_FILES}
KIND=${KIND}
CONST_DST=${CONST_DST}
PARTITION=${PARTITION}
CPUS_PER_TASK=${CPUS_PER_TASK}
MEM=${MEM}
TIME_LIMIT=${TIME_LIMIT}
EXPORT_WORKERS=${EXPORT_WORKERS}
SUMMARY_WORKERS=${SUMMARY_WORKERS}
WORKER_MAX_FILES=${WORKER_MAX_FILES}
MAX_EVENTS=${MAX_EVENTS}
MAX_EVENTS_PER_FILE=${MAX_EVENTS_PER_FILE}
SHARD_SIZE=${SHARD_SIZE}
OPEN_RETRIES=${OPEN_RETRIES}
OPEN_RETRY_DELAY=${OPEN_RETRY_DELAY}
CHUNK_SIZE=${CHUNK_SIZE}
ENERGY_SAMPLE_PER_BIN=${ENERGY_SAMPLE_PER_BIN}
ENERGY_SAMPLE_STRATIFY_PARTICLE=${ENERGY_SAMPLE_STRATIFY_PARTICLE}
ENERGY_BIN_WIDTH=${ENERGY_BIN_WIDTH}
ENERGY_OVERSAMPLE_FACTOR=${ENERGY_OVERSAMPLE_FACTOR}
SEED=${SEED}
KEEP_NON_MODE0=${KEEP_NON_MODE0}
SKIP_ERRORS=${SKIP_ERRORS}
RUN_BUILD=${RUN_BUILD}
UV_CACHE_DIR=${UV_CACHE_DIR}
UV_LINK_MODE=${UV_LINK_MODE}
UV_SYNC_ARGS=${UV_SYNC_ARGS}
OMP_NUM_THREADS=${OMP_NUM_THREADS}
PYTHONUNBUFFERED=${PYTHONUNBUFFERED}
EOF

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SLURM_LOG_DIR}/%x_%j.out
#SBATCH --error=${SLURM_LOG_DIR}/%x_%j.err

set -euo pipefail

JOB_LOG_PATH=$(q "${LOG_DIR}/${RUN_NAME}.job.log")
mkdir -p $(q "${LOG_DIR}") $(q "${SUMMARY_DIR}") $(q "${SLURM_LOG_DIR}") $(q "${GRAPH_RUN_DIR}") $(q "${CONFIG_DIR}")
exec > >(tee -a "\${JOB_LOG_PATH}") 2>&1

module purge
module load gcc/13.1.0 cmake/3.28 hdf5/2.0.0 mkl/latest tbb/latest

cd $(q "${REPO}")

export UV_CACHE_DIR=$(q "${UV_CACHE_DIR}")
export UV_LINK_MODE=$(q "${UV_LINK_MODE}")
export OMP_NUM_THREADS=$(q "${OMP_NUM_THREADS}")
export PYTHONUNBUFFERED=$(q "${PYTHONUNBUFFERED}")
export TALESD_CONST_DST=$(q "${CONST_DST}")

INPUT_DIRS_VALUE=$(q "${INPUT_DIRS}")
INPUT_LISTS_VALUE=$(q "${INPUT_LISTS}")
INPUT_FILES_VALUE=$(q "${INPUT_FILES}")
IFS=':' read -r -a INPUT_DIR_ARRAY <<< "\${INPUT_DIRS_VALUE}"
IFS=':' read -r -a INPUT_LIST_ARRAY <<< "\${INPUT_LISTS_VALUE}"
IFS=':' read -r -a INPUT_FILE_ARRAY <<< "\${INPUT_FILES_VALUE}"

echo "======================================================================"
echo "SERVER CPU DST EXPORT"
echo "date=\$(date)"
echo "hostname=\$(hostname)"
echo "slurm_job_id=\${SLURM_JOB_ID:-}"
echo "partition=${PARTITION}"
echo "cpus_per_task=${CPUS_PER_TASK}"
echo "mem=${MEM}"
echo "time_limit=${TIME_LIMIT}"
echo "export_workers=${EXPORT_WORKERS}"
echo "summary_workers=${SUMMARY_WORKERS}"
echo "worker_max_files=${WORKER_MAX_FILES}"
echo "graph_output=${GRAPH_OUTPUT}"
echo "kind=${KIND}"
echo "const_dst=${CONST_DST}"
echo "max_events=${MAX_EVENTS}"
echo "max_events_per_file=${MAX_EVENTS_PER_FILE}"
echo "energy_sample_per_bin=${ENERGY_SAMPLE_PER_BIN}"
echo "energy_sample_stratify_particle=${ENERGY_SAMPLE_STRATIFY_PARTICLE}"
echo "energy_bin_width=${ENERGY_BIN_WIDTH}"
echo "energy_oversample_factor=${ENERGY_OVERSAMPLE_FACTOR}"
echo "shard_size=${SHARD_SIZE}"
echo "This job reads DST files and writes local HDF5 graph shards."
echo "======================================================================"

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo "RUN_BUILD=1: syncing environment and building extensions"
  uv sync ${UV_SYNC_ARGS}
  ./build_extensions.sh
else
  echo "RUN_BUILD=0: using existing virtualenv and extension build"
fi

if [[ ! -x .venv/bin/talesd-gnn ]]; then
  echo ".venv/bin/talesd-gnn not found. Set RUN_BUILD=1 or run uv sync first." >&2
  exit 2
fi

.venv/bin/python - <<'PY'
import h5py
import numpy
import dstio
print("python imports OK: h5py, numpy, dstio")
PY

export_cmd=(.venv/bin/talesd-gnn export)
input_dst_count=0

for input_dir in "\${INPUT_DIR_ARRAY[@]}"; do
  [[ -n "\${input_dir}" ]] || continue
  if [[ ! -d "\${input_dir}" ]]; then
    echo "input directory not found: \${input_dir}" >&2
    exit 2
  fi
  count=\$(find "\${input_dir}" -type f -name '*.dst.gz' | wc -l | tr -d '[:space:]')
  echo "input_dir=\${input_dir} dst_files=\${count}"
  input_dst_count=\$((input_dst_count + count))
  export_cmd+=(--input-dir "\${input_dir}")
done

for input_list in "\${INPUT_LIST_ARRAY[@]}"; do
  [[ -n "\${input_list}" ]] || continue
  if [[ ! -f "\${input_list}" ]]; then
    echo "input list not found: \${input_list}" >&2
    exit 2
  fi
  count=\$(awk 'NF && \$1 !~ /^#/ {n++} END {print n + 0}' "\${input_list}")
  echo "input_list=\${input_list} dst_files=\${count}"
  input_dst_count=\$((input_dst_count + count))
  export_cmd+=(--input-list "\${input_list}")
done

for input_file in "\${INPUT_FILE_ARRAY[@]}"; do
  [[ -n "\${input_file}" ]] || continue
  if [[ ! -f "\${input_file}" ]]; then
    echo "input DST file not found: \${input_file}" >&2
    exit 2
  fi
  echo "input_file=\${input_file}"
  input_dst_count=\$((input_dst_count + 1))
  export_cmd+=("\${input_file}")
done

if (( input_dst_count <= 0 )); then
  echo "No DST files found in the configured inputs." >&2
  exit 2
fi

export_cmd+=(
  -o $(q "${GRAPH_OUTPUT}")
  --kind $(q "${KIND}")
  --const-dst $(q "${CONST_DST}")
  --workers "${EXPORT_WORKERS}"
  --worker-max-files "${WORKER_MAX_FILES}"
  --chunk-size "${CHUNK_SIZE}"
  --shard-size "${SHARD_SIZE}"
  --open-retries "${OPEN_RETRIES}"
  --open-retry-delay "${OPEN_RETRY_DELAY}"
  --seed "${SEED}"
)

if [[ -n "${MAX_EVENTS}" ]]; then
  export_cmd+=(--max-events "${MAX_EVENTS}")
fi
if [[ "${MAX_EVENTS_PER_FILE}" != "0" ]]; then
  export_cmd+=(--max-events-per-file "${MAX_EVENTS_PER_FILE}")
else
  export_cmd+=(--max-events-per-file 0)
fi
if (( ${ENERGY_SAMPLE_PER_BIN} > 0 )); then
  export_cmd+=(
    --energy-sample-per-bin "${ENERGY_SAMPLE_PER_BIN}"
    --energy-bin-width "${ENERGY_BIN_WIDTH}"
    --energy-oversample-factor "${ENERGY_OVERSAMPLE_FACTOR}"
  )
  if [[ "${ENERGY_SAMPLE_STRATIFY_PARTICLE}" == "1" ]]; then
    export_cmd+=(--energy-sample-stratify-particle)
  fi
fi
if [[ "${KEEP_NON_MODE0}" == "1" ]]; then
  export_cmd+=(--keep-non-mode0)
fi
if [[ "${SKIP_ERRORS}" == "1" ]]; then
  export_cmd+=(--skip-errors)
fi

printf "export command:"
printf " %q" "\${export_cmd[@]}"
printf "\\n"

"\${export_cmd[@]}"

cat <<READ_COMPLETE_EOF | tee $(q "${RUN_DIR}/DST_READ_COMPLETE.txt")
======================================================================
DST FILE READING COMPLETE

All DST inputs configured for this export job have been read.
The remaining training and diagnostic steps should use these HDF5 graph shards:
  ${GRAPH_OUTPUT}

date=\$(date)
======================================================================
READ_COMPLETE_EOF

printf "%s\\n" $(q "${GRAPH_OUTPUT}") > $(q "${CONFIG_DIR}/graph_input.txt")

.venv/bin/python scripts/summarize_graph_shards.py $(q "${GRAPH_OUTPUT}") \\
  --workers "${SUMMARY_WORKERS}" \\
  -o $(q "${SUMMARY_DIR}/graph_summary.json")

cat > $(q "${RUN_DIR}/README.txt") <<README_EOF
Run: ${RUN_NAME}
Created: \$(date)
Purpose: server-side CPU DST export for TALE-SD GNN graph shards.

Important files:
  config/export_submit.env
  config/graph_input.txt
  logs/${RUN_NAME}.job.log
  summaries/graph_summary.json
  DST_READ_COMPLETE.txt

Graph input for training:
  ${GRAPH_OUTPUT}
README_EOF

echo "run_dir=${RUN_DIR}"
echo "graph_input=${GRAPH_OUTPUT}"
echo "graph_summary=${SUMMARY_DIR}/graph_summary.json"
echo "read_done_marker=${RUN_DIR}/DST_READ_COMPLETE.txt"
date
EOF

cat <<EOF
======================================================================
SERVER CPU DST EXPORT SBATCH READY

sbatch_file:
  ${SBATCH_FILE}

run_dir:
  ${RUN_DIR}

graph_output:
  ${GRAPH_OUTPUT}

job_log:
  ${LOG_DIR}/${RUN_NAME}.job.log

partition=${PARTITION}
time_limit=${TIME_LIMIT}
cpus_per_task=${CPUS_PER_TASK}
mem=${MEM}
export_workers=${EXPORT_WORKERS}
summary_workers=${SUMMARY_WORKERS}
energy_sample_per_bin=${ENERGY_SAMPLE_PER_BIN}
energy_sample_stratify_particle=${ENERGY_SAMPLE_STRATIFY_PARTICLE}

Default DST inputs:
  ${INPUT_DIRS}
======================================================================
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  exit 0
fi

sbatch "${SBATCH_FILE}"
