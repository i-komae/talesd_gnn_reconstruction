#!/usr/bin/env bash
set -euo pipefail

GPU_PARTITIONS_DEFAULT="a100_devel-al9,a100_short-al9,a100-al9,a100_long-al9,v100-al9,v100-al9_short,v100-al9_long,b6000-al9,b6000-al9_short,b6000-al9_long"
GPU_PARTITIONS="${GPU_PARTITIONS:-${GPU_PARTITIONS_DEFAULT}}"
MY_ONLY=0
DETAILS=0
SHOW_NODES=0
SHOW_REASONS=0

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\033[1m'
  CYAN=$'\033[36m'
  RED=$'\033[31m'
  YELLOW=$'\033[33m'
  GREEN=$'\033[32m'
  BLUE=$'\033[34m'
  MAGENTA=$'\033[35m'
  RESET=$'\033[0m'
else
  BOLD=""
  CYAN=""
  RED=""
  YELLOW=""
  GREEN=""
  BLUE=""
  MAGENTA=""
  RESET=""
fi

usage() {
  cat <<'EOF'
Usage: scripts/slurm_queue_summary.sh [1|--mine-only] [--details] [--nodes] [--reasons]

Default output is compact:
  - GPU Used/Total summary
  - my summary
  - GPU queue length by GPU class and partition

Options:
  1, --mine-only  Stop after my summary.
  --details       Also print detailed my-job and all-job tables.
  --nodes         Also print Slurm node state from sinfo.
  --reasons       Also print pending reasons.
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
    --nodes|--node-info)
      SHOW_NODES=1
      ;;
    --reasons|--pending-reasons)
      SHOW_REASONS=1
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
  printf "%s##### SLURM NODE INFO #####%s\n" "${BOLD}${CYAN}" "${RESET}"
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
  printf "%s##### RSC INFO #####%s\n" "${BOLD}${CYAN}" "${RESET}"

  if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is not available; cannot compute GPU Used/Total."
    return
  fi

  scontrol show node -o | awk \
    -v parts="${GPU_PARTITIONS}" \
    -v bold="${BOLD}" \
    -v red="${RED}" \
    -v yellow="${YELLOW}" \
    -v green="${GREEN}" \
    -v blue="${BLUE}" \
    -v magenta="${MAGENTA}" \
    -v reset="${RESET}" '
    BEGIN {
      nrequested = split(parts, requested, ",")
      nclasses = 0
      for (i = 1; i <= nrequested; i++) {
        part = requested[i]
        if (part == "") {
          continue
        }
        if (part ~ /a100/) {
          class = "A100"
        } else if (part ~ /v100/) {
          class = "V100"
        } else if (part ~ /b6000/) {
          class = "B6000"
        } else {
          class = part
        }
        if (!(class in seen_class)) {
          nclasses++
          class_order[nclasses] = class
          seen_class[class] = 1
        }
      }
      width = 25
      printf "%s%-10s %-25s %8s %12s%s\n", bold, "GPU CLASS", "GPU USE", "USED%", "USED/TOTAL", reset
    }

    function node_class(list, values, nvalues, i, part) {
      nvalues = split(list, values, ",")
      for (i = 1; i <= nvalues; i++) {
        part = values[i]
        if (part ~ /a100/) {
          return "A100"
        }
        if (part ~ /v100/) {
          return "V100"
        }
        if (part ~ /b6000/) {
          return "B6000"
        }
      }
      return ""
    }

    function class_color(class) {
      if (class == "A100") {
        return blue
      }
      if (class == "V100") {
        return green
      }
      if (class == "B6000") {
        return magenta
      }
      return ""
    }

    function usage_color(pct) {
      if (pct >= 95.0) {
        return red
      }
      if (pct >= 80.0) {
        return yellow
      }
      return green
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

      class = node_class(partitions)
      if (class == "") {
        next
      }
      total[class] += cfg_gpu
      used[class] += alloc_gpu
    }

    END {
      for (i = 1; i <= nclasses; i++) {
        class = class_order[i]
        if (total[class] <= 0) {
          continue
        }
        pct = 100.0 * used[class] / total[class]
        filled = int(width * used[class] / total[class] + 0.5)
        if (filled > width) {
          filled = width
        }
        bar = ""
        for (j = 1; j <= width; j++) {
          bar = bar (j <= filled ? "*" : "-")
        }
        name_color = class_color(class)
        bar_color = usage_color(pct)
        printf "%s%-10s%s %s%s%s %7.1f%% %6d/%-6d\n", name_color, class, reset, bar_color, bar, reset, pct, used[class], total[class]
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
  printf "%s##### %s #####%s\n" "${BOLD}${CYAN}" "${title}" "${RESET}"
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
  printf "%s##### %s SUMMARY BY PARTITION #####%s\n" "${BOLD}${CYAN}" "${title}" "${RESET}"
  printf "%s%-24s %8s %8s %8s %8s%s\n" "${BOLD}" "PARTITION" "TOTAL" "PENDING" "RUNNING" "OTHER" "${RESET}"

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
    }' | sort | awk -v blue="${BLUE}" -v green="${GREEN}" -v magenta="${MAGENTA}" -v reset="${RESET}" '
    function part_color(part) {
      if (part ~ /a100/) {
        return blue
      }
      if (part ~ /v100/) {
        return green
      }
      if (part ~ /b6000/) {
        return magenta
      }
      return ""
    }
    {
      color = part_color($1)
      printf "%s%-24s%s %8s %8s %8s %8s\n", color, $1, reset, $2, $3, $4, $5
    }'

  echo
  printf "%s%-24s %8s %8s %8s %8s%s\n" "${BOLD}" "TOTAL" "TOTAL" "PENDING" "RUNNING" "OTHER" "${RESET}"
  printf "%s\n" "${data}" | awk -v bold="${BOLD}" -v reset="${RESET}" '
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
      printf "%s%-24s %8d %8d %8d %8d%s\n", bold, "all", total+0, pending+0, running+0, other+0, reset
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
  printf "%s##### %s PENDING REASONS #####%s\n" "${BOLD}${CYAN}" "${title}" "${RESET}"
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

print_gpu_queue_summary() {
  echo
  printf "%s##### GPU QUEUE SUMMARY #####%s\n" "${BOLD}${CYAN}" "${RESET}"

  local data
  data="$(squeue -h -p "${GPU_PARTITIONS}" -o "%P %t" || true)"
  if [[ -z "${data}" ]]; then
    printf "%-24s %8d %8d %8d %8d\n" "(none)" 0 0 0 0
    return
  fi

  printf "%sBY GPU CLASS%s\n" "${BOLD}" "${RESET}"
  printf "%s%-10s %8s %8s %8s %8s%s\n" "${BOLD}" "GPU CLASS" "TOTAL" "PENDING" "RUNNING" "OTHER" "${RESET}"
  printf "%s\n" "${data}" | awk '
    function gpu_class(part) {
      if (part ~ /a100/) {
        return "A100"
      }
      if (part ~ /b6000/) {
        return "B6000"
      }
      if (part ~ /v100/) {
        return "V100"
      }
      return "OTHER"
    }
    {
      class=gpu_class($1)
      state=$2
      total[class]++
      if (state == "PD") {
        pending[class]++
      } else if (state == "R") {
        running[class]++
      } else {
        other[class]++
      }
    }
    END {
      for (class in total) {
        printf "%-10s %8d %8d %8d %8d\n", class, total[class], pending[class]+0, running[class]+0, other[class]+0
      }
    }' | sort | awk -v blue="${BLUE}" -v green="${GREEN}" -v magenta="${MAGENTA}" -v reset="${RESET}" '
    function class_color(class) {
      if (class == "A100") {
        return blue
      }
      if (class == "V100") {
        return green
      }
      if (class == "B6000") {
        return magenta
      }
      return ""
    }
    {
      color = class_color($1)
      printf "%s%-10s%s %8s %8s %8s %8s\n", color, $1, reset, $2, $3, $4, $5
    }'

  echo
  printf "%sBY PARTITION%s\n" "${BOLD}" "${RESET}"
  printf "%s%-24s %8s %8s %8s %8s%s\n" "${BOLD}" "PARTITION" "TOTAL" "PENDING" "RUNNING" "OTHER" "${RESET}"
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
    }' | sort | awk -v blue="${BLUE}" -v green="${GREEN}" -v magenta="${MAGENTA}" -v reset="${RESET}" '
    function part_color(part) {
      if (part ~ /a100/) {
        return blue
      }
      if (part ~ /v100/) {
        return green
      }
      if (part ~ /b6000/) {
        return magenta
      }
      return ""
    }
    {
      color = part_color($1)
      printf "%s%-24s%s %8s %8s %8s %8s\n", color, $1, reset, $2, $3, $4, $5
    }'
}

need_command sinfo
need_command squeue

print_resource_info
if [[ "${SHOW_NODES}" == "1" ]]; then
  print_sinfo
fi
if [[ "${DETAILS}" == "1" ]]; then
  print_job_table "MY JOBS" "mine"
fi
print_summary "MY JOBS" "mine"
if [[ "${SHOW_REASONS}" == "1" ]]; then
  print_pending_reasons "MY JOBS" "mine"
fi

if [[ "${MY_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ "${DETAILS}" == "1" ]]; then
  print_job_table "ALL JOBS" "all"
fi
print_gpu_queue_summary
if [[ "${SHOW_REASONS}" == "1" ]]; then
  print_pending_reasons "ALL JOBS" "all"
fi
