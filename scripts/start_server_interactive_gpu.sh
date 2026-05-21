#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-b6000-al9_short}"
GPUS="${GPUS:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-96}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((GPUS * CPUS_PER_GPU))}"
MEM="${MEM:-$((GPUS * MEM_PER_GPU_GB))G}"
TIME="${TIME:-00:15:00}"

if [[ "${PARTITION}" == a100* && "${ALLOW_A100:-0}" != "1" ]]; then
  cat >&2 <<EOF
Refusing to submit to A100 partition by default: ${PARTITION}

Use one of these instead:
  PARTITION=b6000-al9_short scripts/start_server_interactive_gpu.sh
  PARTITION=v100-al9_short scripts/start_server_interactive_gpu.sh

If A100 is explicitly required, set ALLOW_A100=1.
EOF
  exit 2
fi

cat <<EOF
======================================================================
Starting interactive Slurm GPU session
partition=${PARTITION}
gpus=${GPUS}
cpus_per_gpu=${CPUS_PER_GPU}
cpus_per_task=${CPUS_PER_TASK}
mem_per_gpu_gb=${MEM_PER_GPU_GB}
mem=${MEM}
time=${TIME}
======================================================================
EOF

exec srun --pty \
  --partition="${PARTITION}" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  bash -l
