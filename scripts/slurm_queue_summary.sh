#!/usr/bin/env bash
set -euo pipefail

GPU_PARTITIONS_DEFAULT="a100_devel-al9,a100_short-al9,a100-al9,a100_long-al9,v100-al9,v100-al9_short,v100-al9_long,b6000-al9,b6000-al9_short,b6000-al9_long"
GPU_PARTITIONS="${GPU_PARTITIONS:-${GPU_PARTITIONS_DEFAULT}}"
MY_ONLY=0
DETAILS=0

usage() {
  cat <<'EOF'
Usage: scripts/slurm_queue_summary.sh [1|--mine-only] [--details]

Default output is compact:
  - GPU node state
  - my jobs
  - my summary
  - all-job summary

Options:
  1, --mine-only  Stop after my jobs and my summary.
  --details       Also print the full all-job table.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    1|--mine-only)
      MY_ONLY=1
      ;;
    --details|--all-details)
      DETAILS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

print_sinfo() {
  echo
  echo "##### SLURM NODE INFO #####"
  echo "date     : $(date)"
  echo "login    : $(hostname)"
  echo "user     : ${USER:-unknown}"
  echo "gpu parts: ${GPU_PARTITIONS}"
  echo
  printf "%-24s %5s %-10s %-12s %8s %10s %s\n" "PARTITION" "NODES" "STATE" "GRES" "CPUS" "MEMORY" "NODELIST"
  sinfo -h -p "${GPU_PARTITIONS}" -o "%P|%D|%t|%G|%c|%m|%N" \
    | awk -F'|' '{printf "%-24s %5s %-10s %-12s %8s %10s %s\n", $1, $2, $3, $4, $5, $6, $7}' \
    || true
}

print_resource_info() {
  echo
  echo "##### RSC INFO #####"
  echo "GPU Used/Total is computed from Slurm CfgTRES/AllocTRES."
  echo

  if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is not available; cannot compute GPU Used/Total."
    return
  fi

  scontrol show node -o | awk -v parts="${GPU_PARTITIONS}" '
    BEGIN {
      nrequested = split(parts, requested, ",")
      nparts = 0
      for (i = 1; i <= nrequested; i++) {
        if (requested[i] != "") {
          nparts++
          order[nparts] = requested[i]
          wanted[requested[i]] = 1
        }
      }
      width = 25
      printf "%-24s %-25s %8s %12s\n", "PARTITION", "GPU USE", "USED%", "USED/TOTAL"
    }

    function has_partition(list, part, values, nvalues, i) {
      nvalues = split(list, values, ",")
      for (i = 1; i <= nvalues; i++) {
        if (values[i] == part) {
          return 1
        }
      }
      return 0
    }

    function gpu_count(tres, values, nvalues, i, item, sum) {
      sum = 0
      nvalues = split(tres, values, ",")
      for (i = 1; i <= nvalues; i++) {
        item = values[i]
        if (item ~ /^gres\/gpu[^=]*=/) {
          sub(/^gres\/gpu[^=]*=/, "", item)
          sum += item + 0
        }
      }
      return sum
    }

    {
      partitions = ""
      cfg_tres = ""
      alloc_tres = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^Partitions=/) {
          partitions = substr($i, 12)
        } else if ($i ~ /^CfgTRES=/) {
          cfg_tres = substr($i, 9)
        } else if ($i ~ /^AllocTRES=/) {
          alloc_tres = substr($i, 11)
        }
      }

      cfg_gpu = gpu_count(cfg_tres)
      alloc_gpu = gpu_count(alloc_tres)
      if (cfg_gpu <= 0) {
        next
      }

      for (i = 1; i <= nparts; i++) {
        part = order[i]
        if (has_partition(partitions, part)) {
          total[part] += cfg_gpu
          used[part] += alloc_gpu
        }
      }
    }

    END {
      for (i = 1; i <= nparts; i++) {
        part = order[i]
        if (total[part] <= 0) {
          continue
        }
        pct = 100.0 * used[part] / total[part]
        filled = int(width * used[part] / total[part] + 0.5)
        if (filled > width) {
          filled = width
        }
        bar = ""
        for (j = 1; j <= width; j++) {
          bar = bar (j <= filled ? "*" : "-")
        }
        printf "%-24s %s %7.1f%% %6d/%-6d\n", part, bar, pct, used[part], total[part]
      }
    }'
}

print_job_table() {
  local title="$1"
  local scope="$2"
  local -a cmd=(squeue -o "%.18i %.22P %.28j %.10u %.10T %.10M %.10l %.6D %R")
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
  printf "%-24s %8s %8s %8s %8s\n" "PARTITION" "TOTAL" "PENDING" "RUNNING" "OTHER"

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
  printf "%-24s %8s %8s %8s %8s\n" "TOTAL" "TOTAL" "PENDING" "RUNNING" "OTHER"
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

print_resource_info
print_sinfo
print_job_table "MY JOBS" "mine"
print_summary "MY JOBS" "mine"
print_pending_reasons "MY JOBS" "mine"

if [[ "${MY_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ "${DETAILS}" == "1" ]]; then
  print_job_table "ALL JOBS" "all"
fi
print_summary "ALL JOBS" "all"
print_pending_reasons "ALL JOBS" "all"
