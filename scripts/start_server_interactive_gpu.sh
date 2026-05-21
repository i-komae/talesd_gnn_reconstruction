#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-b6000-al9_short}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-128G}"
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
cpus_per_task=${CPUS_PER_TASK}
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
