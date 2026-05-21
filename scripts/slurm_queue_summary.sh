#!/usr/bin/env bash
set -euo pipefail

GPU_PARTITIONS_DEFAULT="a100_devel-al9,a100_short-al9,a100-al9,a100_long-al9,b6000-al9,b6000-al9_short,b6000-al9_long,v100-al9,v100-al9_short,v100-al9_long"
GPU_PARTITIONS="${GPU_PARTITIONS:-${GPU_PARTITIONS_DEFAULT}}"
MY_ONLY=0
DETAILS=0
SHOW_NODES=0
SHOW_REASONS=0
PARTITION_FILTER="regular"

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
Usage: scripts/slurm_queue_summary.sh [1] [-g|-c|-a] [-d] [-n] [-r] [-h]

Default output is compact:
  - GPU/CPU/MEM Used/Total summary for GPU partitions
  - regular Slurm partitions grouped by resource class
  - my summary
  - job queue length by resource class and partition

Options:
  1, -m  Stop after my summary.
  -g     GPU partitions only.
  -c     CPU/non-GPU partitions only.
  -a     All visible partitions, including reservations.
  -d     Also print detailed job tables.
  -n     Also print node tables.
  -r     Also print pending reasons.
  -h     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    1|-m|--mine-only)
      MY_ONLY=1
      ;;
    -d|--details|--all-details)
      DETAILS=1
      ;;
    -n|--nodes|--node-info)
      SHOW_NODES=1
      ;;
    -r|--reasons|--pending-reasons)
      SHOW_REASONS=1
      ;;
    -g|--gpu-only|--gpus)
      PARTITION_FILTER="gpu"
      ;;
    -c|--cpu-only|--cpus)
      PARTITION_FILTER="cpu"
      ;;
    -a|--all)
      PARTITION_FILTER="all"
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
  echo "scope    : ${PARTITION_FILTER} partitions"
  echo
  printf "%-24s %5s %-10s %-12s %8s %10s %s\n" "PARTITION" "NODES" "STATE" "GRES" "CPUS" "MEMORY" "NODELIST"
  sinfo -h -o "%P|%D|%t|%G|%c|%m|%N" \
    | awk -F'|' -v filter="${PARTITION_FILTER}" '
      function clean_part(part) {
        sub(/\*$/, "", part)
        return part
      }
      function group_from_part(part) {
        if (part ~ /^reservation_b6000/) return "RESERVATION"
        if (part ~ /a100/) return "A100"
        if (part ~ /b6000/) return "B6000"
        if (part ~ /v100/) return "V100"
        if (part ~ /bigmemory/) return "BIGMEM"
        if (part ~ /^golbal/ || part ~ /^global/) return "GLOBAL"
        if (part ~ /^edr1/) return "EDR1"
        if (part ~ /^edr2/) return "EDR2"
        if (part ~ /^intel/) return "INTEL"
        return "OTHER"
      }
      function is_gpu_group(group) {
        return group == "A100" || group == "B6000" || group == "V100"
      }
      function is_cpu_group(group) {
        return !is_gpu_group(group) && group != "RESERVATION"
      }
      function selected(group) {
        return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
      }
      {
        part = clean_part($1)
        group = group_from_part(part)
        if (!selected(group)) {
          next
        }
        printf "%-24s %5s %-10s %-12s %8s %10s %s\n", $1, $2, $3, $4, $5, $6, $7
      }' \
    || true
}

print_resource_info() {
  echo
  printf "%s##### GPU RSC INFO #####%s\n" "${BOLD}${CYAN}" "${RESET}"

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
        requested_part[part] = 1
        class = class_from_part(part)
        if (!(class in seen_class)) {
          nclasses++
          class_order[nclasses] = class
          seen_class[class] = 1
        }
      }
      width = 25
      printf "%s%-2s %-10s %-4s %-25s %7s %9s %9s%s\n", bold, "", "GPU CLASS", "RSC", "USE", "USED%", "USED", "TOTAL", reset
    }

    function class_from_part(part) {
      if (part ~ /a100/) {
        return "A100"
      }
      if (part ~ /b6000/) {
        return "B6000"
      }
      if (part ~ /v100/) {
        return "V100"
      }
      return part
    }

    function node_class(list, values, nvalues, i, part) {
      nvalues = split(list, values, ",")
      for (i = 1; i <= nvalues; i++) {
        part = values[i]
        if (requested_part[part]) {
          return class_from_part(part)
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

    function cpu_count(tres, values, nvalues, i, item, sum) {
      sum = 0
      nvalues = split(tres, values, ",")
      for (i = 1; i <= nvalues; i++) {
        item = values[i]
        if (item ~ /^cpu=/) {
          sub(/^cpu=/, "", item)
          sum += item + 0
        }
      }
      return sum
    }

    function mem_count(tres, values, nvalues, i, item, sum) {
      sum = 0
      nvalues = split(tres, values, ",")
      for (i = 1; i <= nvalues; i++) {
        item = values[i]
        if (item ~ /^mem=/) {
          sub(/^mem=/, "", item)
          sub(/M$/, "", item)
          sum += item + 0
        }
      }
      return sum
    }

    function usage_bar(used_value, total_value, width, fill, j, filled, bar) {
      if (total_value <= 0) {
        return ""
      }
      filled = int(width * used_value / total_value + 0.5)
      if (filled > width) {
        filled = width
      }
      bar = ""
      for (j = 1; j <= width; j++) {
        bar = bar (j <= filled ? fill : "-")
      }
      return bar
    }

    function gib_string(value_mb) {
      return sprintf("%.1fG", value_mb / 1024.0)
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
      cfg_cpu = cpu_count(cfg_tres)
      alloc_cpu = cpu_count(alloc_tres)
      cfg_mem = mem_count(cfg_tres)
      alloc_mem = mem_count(alloc_tres)
      if (cfg_gpu <= 0) {
        next
      }

      class = node_class(partitions)
      if (class == "") {
        next
      }
      total[class] += cfg_gpu
      used[class] += alloc_gpu
      total_cpu[class] += cfg_cpu
      used_cpu[class] += alloc_cpu
      total_mem[class] += cfg_mem
      used_mem[class] += alloc_mem
    }

    END {
      for (i = 1; i <= nclasses; i++) {
        class = class_order[i]
        if (total[class] <= 0) {
          continue
        }
        pct = 100.0 * used[class] / total[class]
        cpu_pct = total_cpu[class] > 0 ? 100.0 * used_cpu[class] / total_cpu[class] : 0.0
        mem_pct = total_mem[class] > 0 ? 100.0 * used_mem[class] / total_mem[class] : 0.0
        bar = usage_bar(used[class], total[class], width, "*")
        cpu_bar = usage_bar(used_cpu[class], total_cpu[class], width, "*")
        mem_bar = usage_bar(used_mem[class], total_mem[class], width, "*")
        if (cpu_bar == "") {
          cpu_bar = sprintf("%*s", width, "")
        }
        if (mem_bar == "") {
          mem_bar = sprintf("%*s", width, "")
        }
        bar_color = usage_color(pct)
        cpu_bar_color = usage_color(cpu_pct)
        mem_bar_color = usage_color(mem_pct)
        if (printed > 0) {
          print ""
        }
        printed++
        printf "%s*%s  %-10s %-4s %s%s%s %6.1f%% %9d %9d\n", class_color(class), reset, class, "GPU", bar_color, bar, reset, pct, used[class], total[class]
        printf "   %-10s %-4s %s%s%s %6.1f%% %9d %9d\n", "", "CPU", cpu_bar_color, cpu_bar, reset, cpu_pct, used_cpu[class], total_cpu[class]
        printf "   %-10s %-4s %s%s%s %6.1f%% %9s %9s\n", "", "MEM", mem_bar_color, mem_bar, reset, mem_pct, gib_string(used_mem[class]), gib_string(total_mem[class])
      }
    }'
}

print_cpu_resource_info() {
  echo
  printf "%s##### CPU RSC INFO #####%s\n" "${BOLD}${CYAN}" "${RESET}"

  if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is not available; cannot compute CPU/MEM Used/Total."
    return
  fi

  scontrol show node -o | awk \
    -v bold="${BOLD}" \
    -v red="${RED}" \
    -v yellow="${YELLOW}" \
    -v green="${GREEN}" \
    -v blue="${BLUE}" \
    -v magenta="${MAGENTA}" \
    -v cyan="${CYAN}" \
    -v reset="${RESET}" '
    BEGIN {
      nclasses = 0
      width = 25
      printf "%s%-2s %-12s %-4s %-25s %7s %9s %9s%s\n", bold, "", "GROUP", "RSC", "USE", "USED%", "USED", "TOTAL", reset
    }

    function group_rank(group) {
      if (group == "BIGMEM") return 10
      if (group == "GLOBAL") return 20
      if (group == "EDR1") return 30
      if (group == "EDR2") return 40
      if (group == "INTEL") return 50
      return 99
    }

    function group_color(group) {
      if (group == "BIGMEM") return red
      if (group == "GLOBAL") return cyan
      if (group == "EDR1") return blue
      if (group == "EDR2") return green
      if (group == "INTEL") return magenta
      return ""
    }

    function group_from_part(part) {
      if (part ~ /bigmemory/) return "BIGMEM"
      if (part ~ /^edr1/) return "EDR1"
      if (part ~ /^edr2/) return "EDR2"
      if (part ~ /^intel/) return "INTEL"
      if (part ~ /^golbal/ || part ~ /^global/) return "GLOBAL"
      return ""
    }

    function node_group(list, values, nvalues, i, group, fallback) {
      fallback = ""
      nvalues = split(list, values, ",")
      for (i = 1; i <= nvalues; i++) {
        group = group_from_part(values[i])
        if (group == "BIGMEM" || group == "EDR1" || group == "EDR2" || group == "INTEL") {
          return group
        }
        if (group == "GLOBAL") {
          fallback = group
        }
      }
      return fallback
    }

    function tres_count(tres, key, values, nvalues, i, item) {
      nvalues = split(tres, values, ",")
      for (i = 1; i <= nvalues; i++) {
        item = values[i]
        if (item ~ ("^" key "=")) {
          sub("^" key "=", "", item)
          sub(/M$/, "", item)
          return item + 0
        }
      }
      return 0
    }

    function usage_color(pct) {
      if (pct >= 95.0) return red
      if (pct >= 80.0) return yellow
      return green
    }

    function usage_bar(used_value, total_value, width, fill, j, filled, bar) {
      if (total_value <= 0) return ""
      filled = int(width * used_value / total_value + 0.5)
      if (filled > width) filled = width
      bar = ""
      for (j = 1; j <= width; j++) {
        bar = bar (j <= filled ? fill : "-")
      }
      return bar
    }

    function gib_string(value_mb) {
      return sprintf("%.1fG", value_mb / 1024.0)
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

      group = node_group(partitions)
      if (group == "") {
        next
      }
      if (!(group in seen_class)) {
        seen_class[group] = 1
        nclasses++
        class_order[nclasses] = group
      }
      total_cpu[group] += tres_count(cfg_tres, "cpu")
      used_cpu[group] += tres_count(alloc_tres, "cpu")
      total_mem[group] += tres_count(cfg_tres, "mem")
      used_mem[group] += tres_count(alloc_tres, "mem")
    }

    END {
      for (rank = 1; rank <= 99; rank++) {
        for (i = 1; i <= nclasses; i++) {
          group = class_order[i]
          if (group_rank(group) != rank || total_cpu[group] <= 0) {
            continue
          }
          cpu_pct = 100.0 * used_cpu[group] / total_cpu[group]
          mem_pct = total_mem[group] > 0 ? 100.0 * used_mem[group] / total_mem[group] : 0.0
          cpu_bar = usage_bar(used_cpu[group], total_cpu[group], width, "*")
          mem_bar = usage_bar(used_mem[group], total_mem[group], width, "*")
          if (mem_bar == "") {
            mem_bar = sprintf("%*s", width, "")
          }
          cpu_color = usage_color(cpu_pct)
          mem_color = usage_color(mem_pct)
          if (printed > 0) {
            print ""
          }
          printed++
          printf "%s*%s  %-12s %-4s %s%s%s %6.1f%% %9d %9d\n", group_color(group), reset, group, "CPU", cpu_color, cpu_bar, reset, cpu_pct, used_cpu[group], total_cpu[group]
          printf "   %-12s %-4s %s%s%s %6.1f%% %9s %9s\n", "", "MEM", mem_color, mem_bar, reset, mem_pct, gib_string(used_mem[group]), gib_string(total_mem[group])
        }
      }
    }'
}

print_partition_info() {
  echo
  printf "%s##### PARTITION INFO #####%s\n" "${BOLD}${CYAN}" "${RESET}"
  printf "%s%-2s %-12s %-28s %-12s %6s %6s %6s %6s %6s%s\n" \
    "${BOLD}" "" "GROUP" "PARTITION" "TIME_LIMIT" "NODES" "IDLE" "MIX" "ALLOC" "OTHER" "${RESET}"

  sinfo -h -o "%P|%l|%D|%t" | awk -F'|' -v filter="${PARTITION_FILTER}" '
    function clean_part(part) {
      sub(/\*$/, "", part)
      return part
    }
    function group_from_part(part) {
      if (part ~ /^reservation_b6000/) {
        return "RESERVATION"
      }
      if (part ~ /a100/) {
        return "A100"
      }
      if (part ~ /b6000/) {
        return "B6000"
      }
      if (part ~ /v100/) {
        return "V100"
      }
      if (part ~ /bigmemory/) {
        return "BIGMEM"
      }
      if (part ~ /^golbal/ || part ~ /^global/) {
        return "GLOBAL"
      }
      if (part ~ /^edr1/) {
        return "EDR1"
      }
      if (part ~ /^edr2/) {
        return "EDR2"
      }
      if (part ~ /^intel/) {
        return "INTEL"
      }
      return "OTHER"
    }
    function group_rank(group) {
      if (group == "A100") return 10
      if (group == "B6000") return 20
      if (group == "V100") return 30
      if (group == "RESERVATION") return 40
      if (group == "BIGMEM") return 50
      if (group == "GLOBAL") return 60
      if (group == "EDR1") return 70
      if (group == "EDR2") return 80
      if (group == "INTEL") return 90
      return 99
    }
    function is_gpu_group(group) {
      return group == "A100" || group == "B6000" || group == "V100"
    }
    function is_cpu_group(group) {
      return !is_gpu_group(group) && group != "RESERVATION"
    }
    function selected(group) {
      return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
    }
    {
      part = clean_part($1)
      limit[part] = $2
      state = $4
      nodes = $3 + 0
      total[part] += nodes
      if (state == "idle") {
        idle[part] += nodes
      } else if (state == "mix") {
        mix[part] += nodes
      } else if (state == "alloc") {
        alloc[part] += nodes
      } else {
        other[part] += nodes
      }
    }
    END {
      for (part in total) {
        group = group_from_part(part)
        if (!selected(group)) {
          continue
        }
        printf "%02d|%s|%s|%s|%d|%d|%d|%d|%d\n",
          group_rank(group), group, part, limit[part], total[part],
          idle[part]+0, mix[part]+0, alloc[part]+0, other[part]+0
      }
    }' | sort -t'|' -k1,1n -k3,3 | awk -F'|' \
    -v blue="${BLUE}" \
    -v green="${GREEN}" \
    -v magenta="${MAGENTA}" \
    -v yellow="${YELLOW}" \
    -v red="${RED}" \
    -v cyan="${CYAN}" \
    -v reset="${RESET}" '
    function group_color(group) {
      if (group == "A100") return blue
      if (group == "B6000") return magenta
      if (group == "V100") return green
      if (group == "RESERVATION") return yellow
      if (group == "BIGMEM") return red
      if (group == "GLOBAL") return cyan
      if (group == "EDR1") return blue
      if (group == "EDR2") return green
      if (group == "INTEL") return magenta
      return ""
    }
    {
      printf "%s*%s  %-12s %-28s %-12s %6s %6s %6s %6s %6s\n",
        group_color($2), reset, $2, $3, $4, $5, $6, $7, $8, $9
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
  printf "%s%-2s %-12s %-28s %8s %8s %8s %8s%s\n" "${BOLD}" "" "GROUP" "PARTITION" "JOBS" "PENDING" "RUNNING" "OTHER" "${RESET}"

  local data
  data="$("${cmd[@]}" || true)"
  if [[ -z "${data}" ]]; then
    printf "%-2s %-12s %-28s %8d %8d %8d %8d\n" "" "" "(none)" 0 0 0 0
    return
  fi

  printf "%s\n" "${data}" | awk -v filter="${PARTITION_FILTER}" '
    function group_from_part(part) {
      if (part ~ /^reservation_b6000/) return "RESERVATION"
      if (part ~ /a100/) return "A100"
      if (part ~ /b6000/) return "B6000"
      if (part ~ /v100/) return "V100"
      if (part ~ /bigmemory/) return "BIGMEM"
      if (part ~ /^golbal/ || part ~ /^global/) return "GLOBAL"
      if (part ~ /^edr1/) return "EDR1"
      if (part ~ /^edr2/) return "EDR2"
      if (part ~ /^intel/) return "INTEL"
      return "OTHER"
    }
    function group_rank(group) {
      if (group == "A100") return 10
      if (group == "B6000") return 20
      if (group == "V100") return 30
      if (group == "RESERVATION") return 40
      if (group == "BIGMEM") return 50
      if (group == "GLOBAL") return 60
      if (group == "EDR1") return 70
      if (group == "EDR2") return 80
      if (group == "INTEL") return 90
      return 99
    }
    function is_gpu_group(group) {
      return group == "A100" || group == "B6000" || group == "V100"
    }
    function is_cpu_group(group) {
      return !is_gpu_group(group) && group != "RESERVATION"
    }
    function selected(part) {
      group = group_from_part(part)
      return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
    }
    {
      part=$1
      if (!selected(part)) {
        next
      }
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
        group = group_from_part(part)
        printf "%02d|%s|%s|%d|%d|%d|%d\n",
          group_rank(group), group, part, total[part], pending[part]+0, running[part]+0, other[part]+0
      }
    }' | sort -t'|' -k1,1n -k3,3 | awk -F'|' \
    -v blue="${BLUE}" \
    -v green="${GREEN}" \
    -v magenta="${MAGENTA}" \
    -v yellow="${YELLOW}" \
    -v red="${RED}" \
    -v cyan="${CYAN}" \
    -v reset="${RESET}" '
    function group_color(group) {
      if (group == "A100") {
        return blue
      }
      if (group == "V100") {
        return green
      }
      if (group == "B6000") {
        return magenta
      }
      if (group == "RESERVATION") {
        return yellow
      }
      if (group == "BIGMEM") {
        return red
      }
      if (group == "GLOBAL") {
        return cyan
      }
      if (group == "EDR1") {
        return blue
      }
      if (group == "EDR2") {
        return green
      }
      if (group == "INTEL") {
        return magenta
      }
      return ""
    }
    {
      printf "%s*%s  %-12s %-28s %8s %8s %8s %8s\n", group_color($2), reset, $2, $3, $4, $5, $6, $7
    }'

  echo
  printf "%s%-2s %-12s %-28s %8s %8s %8s %8s%s\n" "${BOLD}" "" "GROUP" "TOTAL" "JOBS" "PENDING" "RUNNING" "OTHER" "${RESET}"
  printf "%s\n" "${data}" | awk -v filter="${PARTITION_FILTER}" -v bold="${BOLD}" -v reset="${RESET}" '
    function group_from_part(part) {
      if (part ~ /^reservation_b6000/) return "RESERVATION"
      if (part ~ /a100/) return "A100"
      if (part ~ /b6000/) return "B6000"
      if (part ~ /v100/) return "V100"
      if (part ~ /bigmemory/) return "BIGMEM"
      if (part ~ /^golbal/ || part ~ /^global/) return "GLOBAL"
      if (part ~ /^edr1/) return "EDR1"
      if (part ~ /^edr2/) return "EDR2"
      if (part ~ /^intel/) return "INTEL"
      return "OTHER"
    }
    function is_gpu_group(group) {
      return group == "A100" || group == "B6000" || group == "V100"
    }
    function is_cpu_group(group) {
      return !is_gpu_group(group) && group != "RESERVATION"
    }
    function selected(part) {
      group = group_from_part(part)
      return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
    }
    {
      if (!selected($1)) {
        next
      }
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
      printf "%s%-2s %-12s %-28s %8d %8d %8d %8d%s\n", bold, "", "ALL", "all", total+0, pending+0, running+0, other+0, reset
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

  printf "%s\n" "${data}" | awk -F'|' -v filter="${PARTITION_FILTER}" '
    function group_from_part(part) {
      if (part ~ /^reservation_b6000/) return "RESERVATION"
      if (part ~ /a100/) return "A100"
      if (part ~ /b6000/) return "B6000"
      if (part ~ /v100/) return "V100"
      if (part ~ /bigmemory/) return "BIGMEM"
      if (part ~ /^golbal/ || part ~ /^global/) return "GLOBAL"
      if (part ~ /^edr1/) return "EDR1"
      if (part ~ /^edr2/) return "EDR2"
      if (part ~ /^intel/) return "INTEL"
      return "OTHER"
    }
    function is_gpu_group(group) {
      return group == "A100" || group == "B6000" || group == "V100"
    }
    function is_cpu_group(group) {
      return !is_gpu_group(group) && group != "RESERVATION"
    }
    function selected(part) {
      group = group_from_part(part)
      return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
    }
    {
      if (!selected($1)) {
        next
      }
      key=$1 " | " $2
      count[key]++
    }
    END {
      for (key in count) {
        printf "%6d  %s\n", count[key], key
      }
    }' | sort -nr
}

print_job_queue_summary() {
  local title="JOB QUEUE SUMMARY"
  local group_heading="BY RESOURCE GROUP"
  local group_label="GROUP"
  if [[ "${PARTITION_FILTER}" == "gpu" ]]; then
    title="GPU JOB QUEUE SUMMARY"
    group_heading="BY GPU CLASS"
    group_label="GPU CLASS"
  elif [[ "${PARTITION_FILTER}" == "cpu" ]]; then
    title="CPU JOB QUEUE SUMMARY"
    group_heading="BY CPU GROUP"
  fi

  echo
  printf "%s##### %s #####%s\n" "${BOLD}${CYAN}" "${title}" "${RESET}"

  local data
  data="$(squeue -h -o "%P %t" || true)"
  if [[ -z "${data}" ]]; then
    printf "%-24s %8d %8d %8d %8d\n" "(none)" 0 0 0 0
    return
  fi

  printf "%s%s%s\n" "${BOLD}" "${group_heading}" "${RESET}"
  printf "%s%-2s %-12s %-28s %8s %8s %8s %8s%s\n" "${BOLD}" "" "${group_label}" "" "JOBS" "PENDING" "RUNNING" "OTHER" "${RESET}"
  printf "%s\n" "${data}" | awk -v filter="${PARTITION_FILTER}" '
    function group_from_part(part) {
      if (part ~ /^reservation_b6000/) {
        return "RESERVATION"
      }
      if (part ~ /a100/) {
        return "A100"
      }
      if (part ~ /b6000/) {
        return "B6000"
      }
      if (part ~ /v100/) {
        return "V100"
      }
      if (part ~ /bigmemory/) {
        return "BIGMEM"
      }
      if (part ~ /^golbal/ || part ~ /^global/) {
        return "GLOBAL"
      }
      if (part ~ /^edr1/) {
        return "EDR1"
      }
      if (part ~ /^edr2/) {
        return "EDR2"
      }
      if (part ~ /^intel/) {
        return "INTEL"
      }
      return "OTHER"
    }
    function group_rank(group) {
      if (group == "A100") return 10
      if (group == "B6000") return 20
      if (group == "V100") return 30
      if (group == "RESERVATION") return 40
      if (group == "BIGMEM") return 50
      if (group == "GLOBAL") return 60
      if (group == "EDR1") return 70
      if (group == "EDR2") return 80
      if (group == "INTEL") return 90
      return 99
    }
    function is_gpu_group(group) {
      return group == "A100" || group == "B6000" || group == "V100"
    }
    function is_cpu_group(group) {
      return !is_gpu_group(group) && group != "RESERVATION"
    }
    function selected(group) {
      return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
    }
    {
      group=group_from_part($1)
      if (!selected(group)) {
        next
      }
      state=$2
      total[group]++
      if (state == "PD") {
        pending[group]++
      } else if (state == "R") {
        running[group]++
      } else {
        other[group]++
      }
    }
    END {
      for (group in total) {
        printf "%02d|%s|%d|%d|%d|%d\n",
          group_rank(group), group, total[group], pending[group]+0, running[group]+0, other[group]+0
      }
    }' | sort -t'|' -k1,1n | awk -F'|' \
    -v blue="${BLUE}" \
    -v green="${GREEN}" \
    -v magenta="${MAGENTA}" \
    -v yellow="${YELLOW}" \
    -v red="${RED}" \
    -v cyan="${CYAN}" \
    -v reset="${RESET}" '
    function group_color(group) {
      if (group == "A100") {
        return blue
      }
      if (group == "V100") {
        return green
      }
      if (group == "B6000") {
        return magenta
      }
      if (group == "RESERVATION") {
        return yellow
      }
      if (group == "BIGMEM") {
        return red
      }
      if (group == "GLOBAL") {
        return cyan
      }
      if (group == "EDR1") {
        return blue
      }
      if (group == "EDR2") {
        return green
      }
      if (group == "INTEL") {
        return magenta
      }
      return ""
    }
    {
      printf "%s*%s  %-12s %-28s %8s %8s %8s %8s\n", group_color($2), reset, $2, "", $3, $4, $5, $6
    }'

  echo
  printf "%sBY PARTITION%s\n" "${BOLD}" "${RESET}"
  printf "%s%-2s %-12s %-28s %8s %8s %8s %8s%s\n" "${BOLD}" "" "GROUP" "PARTITION" "JOBS" "PENDING" "RUNNING" "OTHER" "${RESET}"
  printf "%s\n" "${data}" | awk -v filter="${PARTITION_FILTER}" '
    function group_from_part(part) {
      if (part ~ /^reservation_b6000/) {
        return "RESERVATION"
      }
      if (part ~ /a100/) {
        return "A100"
      }
      if (part ~ /b6000/) {
        return "B6000"
      }
      if (part ~ /v100/) {
        return "V100"
      }
      if (part ~ /bigmemory/) {
        return "BIGMEM"
      }
      if (part ~ /^golbal/ || part ~ /^global/) {
        return "GLOBAL"
      }
      if (part ~ /^edr1/) {
        return "EDR1"
      }
      if (part ~ /^edr2/) {
        return "EDR2"
      }
      if (part ~ /^intel/) {
        return "INTEL"
      }
      return "OTHER"
    }
    function group_rank(group) {
      if (group == "A100") return 10
      if (group == "B6000") return 20
      if (group == "V100") return 30
      if (group == "RESERVATION") return 40
      if (group == "BIGMEM") return 50
      if (group == "GLOBAL") return 60
      if (group == "EDR1") return 70
      if (group == "EDR2") return 80
      if (group == "INTEL") return 90
      return 99
    }
    function is_gpu_group(group) {
      return group == "A100" || group == "B6000" || group == "V100"
    }
    function is_cpu_group(group) {
      return !is_gpu_group(group) && group != "RESERVATION"
    }
    function selected(group) {
      return filter == "all" || (filter == "regular" && group != "RESERVATION") || (filter == "gpu" && is_gpu_group(group)) || (filter == "cpu" && is_cpu_group(group))
    }
    {
      part=$1
      group = group_from_part(part)
      if (!selected(group)) {
        next
      }
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
        group = group_from_part(part)
        printf "%02d|%s|%s|%d|%d|%d|%d\n",
          group_rank(group), group, part, total[part], pending[part]+0, running[part]+0, other[part]+0
      }
    }' | sort -t'|' -k1,1n -k3,3 | awk -F'|' \
    -v blue="${BLUE}" \
    -v green="${GREEN}" \
    -v magenta="${MAGENTA}" \
    -v yellow="${YELLOW}" \
    -v red="${RED}" \
    -v cyan="${CYAN}" \
    -v reset="${RESET}" '
    function group_color(group) {
      if (group == "A100") {
        return blue
      }
      if (group == "V100") {
        return green
      }
      if (group == "B6000") {
        return magenta
      }
      if (group == "RESERVATION") {
        return yellow
      }
      if (group == "BIGMEM") {
        return red
      }
      if (group == "GLOBAL") {
        return cyan
      }
      if (group == "EDR1") {
        return blue
      }
      if (group == "EDR2") {
        return green
      }
      if (group == "INTEL") {
        return magenta
      }
      return ""
    }
    {
      printf "%s*%s  %-12s %-28s %8s %8s %8s %8s\n", group_color($2), reset, $2, $3, $4, $5, $6, $7
    }'
}

need_command sinfo
need_command squeue

if [[ "${PARTITION_FILTER}" == "cpu" ]]; then
  print_cpu_resource_info
elif [[ "${PARTITION_FILTER}" != "cpu" ]]; then
  print_resource_info
fi
if [[ "${PARTITION_FILTER}" != "gpu" ]]; then
  print_partition_info
fi
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
print_job_queue_summary
if [[ "${SHOW_REASONS}" == "1" ]]; then
  print_pending_reasons "ALL JOBS" "all"
fi
