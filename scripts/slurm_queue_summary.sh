#!/usr/bin/env bash
set -euo pipefail

GPU_PARTITIONS_DEFAULT="a100_devel-al9,a100_short-al9,a100-al9,a100_long-al9,v100-al9,v100-al9_short,v100-al9_long,b6000-al9,b6000-al9_short,b6000-al9_long"
GPU_PARTITIONS="${GPU_PARTITIONS:-${GPU_PARTITIONS_DEFAULT}}"
MY_ONLY="${1:-0}"

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

print_sinfo() {
  echo
  echo "##### SLURM INFO #####"
  echo "date     : $(date)"
  echo "login    : $(hostname)"
  echo "user     : ${USER:-unknown}"
  echo "gpu parts: ${GPU_PARTITIONS}"
  echo
  sinfo -p "${GPU_PARTITIONS}" -o "%-24P %5D %10t %-12G %8c %10m %N" || true
}

print_job_table() {
  local title="$1"
  local scope="$2"
  local -a cmd=(squeue -o "%.18i %.22P %.28j %.10u %.2t %.10M %.10l %.6D %R")
  if [[ "${scope}" == "mine" ]]; then
    cmd+=(-u "${USER}")
  fi

  echo
  echo "##### ${title} #####"
  "${cmd[@]}" || true
}

print_summary() {
  local title="$1"
  local scope="$2"
  local -a cmd=(squeue -h -o "%P %t")
  if [[ "${scope}" == "mine" ]]; then
    cmd+=(-u "${USER}")
  fi

  echo
  echo "##### ${title} SUMMARY BY PARTITION #####"
  printf "%-24s %8s %8s %8s %8s\n" "PARTITION" "TOTAL" "PD" "R" "OTHER"

  local data
  data="$("${cmd[@]}" || true)"
  if [[ -z "${data}" ]]; then
    printf "%-24s %8d %8d %8d %8d\n" "(none)" 0 0 0 0
    return
  fi

  printf "%s\n" "${data}" | awk '
    {
      part=$1
      state=$2
      total[part]++
      if (state == "PD") {
        pending[part]++
      } else if (state == "R") {
        running[part]++
      } else {
        other[part]++
      }
    }
    END {
      for (part in total) {
        printf "%-24s %8d %8d %8d %8d\n", part, total[part], pending[part]+0, running[part]+0, other[part]+0
      }
    }' | sort

  echo
  printf "%-24s %8s %8s %8s %8s\n" "TOTAL" "TOTAL" "PD" "R" "OTHER"
  printf "%s\n" "${data}" | awk '
    {
      total++
      if ($2 == "PD") {
        pending++
      } else if ($2 == "R") {
        running++
      } else {
        other++
      }
    }
    END {
      printf "%-24s %8d %8d %8d %8d\n", "all", total+0, pending+0, running+0, other+0
    }'
}

print_pending_reasons() {
  local title="$1"
  local scope="$2"
  local -a cmd=(squeue -t PD -h -o "%P|%R")
  if [[ "${scope}" == "mine" ]]; then
    cmd+=(-u "${USER}")
  fi

  echo
  echo "##### ${title} PENDING REASONS #####"
  local data
  data="$("${cmd[@]}" || true)"
  if [[ -z "${data}" ]]; then
    echo "(none)"
    return
  fi

  printf "%s\n" "${data}" | awk -F'|' '
    {
      key=$1 " | " $2
      count[key]++
    }
    END {
      for (key in count) {
        printf "%6d  %s\n", count[key], key
      }
    }' | sort -nr
}

need_command sinfo
need_command squeue

print_sinfo
print_job_table "MY JOBS" "mine"
print_summary "MY JOBS" "mine"
print_pending_reasons "MY JOBS" "mine"

if [[ "${MY_ONLY}" == "1" ]]; then
  exit 0
fi

print_job_table "ALL JOBS" "all"
print_summary "ALL JOBS" "all"
print_pending_reasons "ALL JOBS" "all"
