#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUMMARY_SCRIPT="${SCRIPT_DIR}/slurm_queue_summary.sh"
INTERVAL="${INTERVAL:-5}"
RESIZE_TERMINAL=1
SUMMARY_ARGS=()

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\033[1m'
  CYAN=$'\033[36m'
  RESET=$'\033[0m'
else
  BOLD=""
  CYAN=""
  RESET=""
fi

usage() {
  cat <<'EOF'
Usage: scripts/slurm_queue_live.sh [-i SEC] [--no-resize] [--] [slurm_queue_summary options]

Periodically redraw scripts/slurm_queue_summary.sh with ANSI colors preserved.

Options:
  -i SEC, --interval SEC  Refresh interval in seconds. Default: 5.
  --no-resize            Do not request terminal resize.
  -h, --help             Show this help.

Examples:
  scripts/slurm_queue_live.sh
  scripts/slurm_queue_live.sh -i 10 -g
  scripts/slurm_queue_live.sh -- -c -p
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interval)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      INTERVAL="$2"
      shift 2
      ;;
    --interval=*)
      INTERVAL="${1#*=}"
      shift
      ;;
    --no-resize)
      RESIZE_TERMINAL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SUMMARY_ARGS+=("$@")
      break
      ;;
    *)
      SUMMARY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! "${INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "invalid interval: ${INTERVAL}" >&2
  exit 2
fi

if [[ ! -x "${SUMMARY_SCRIPT}" ]]; then
  echo "summary script is not executable: ${SUMMARY_SCRIPT}" >&2
  exit 2
fi

terminal_rows() {
  tput lines 2>/dev/null || printf "0"
}

terminal_cols() {
  tput cols 2>/dev/null || printf "0"
}

strip_ansi() {
  sed -E $'s/\x1B\\[[0-9;?]*[ -/]*[@-~]//g'
}

max_line_width() {
  awk '{ if (length($0) > max) max = length($0) } END { print max + 0 }'
}

line_count() {
  awk 'END { print NR + 0 }'
}

request_resize() {
  local rows="$1"
  local cols="$2"
  [[ "${RESIZE_TERMINAL}" == "1" ]] || return 0
  [[ -t 1 ]] || return 0
  [[ "${TERM:-}" != "dumb" ]] || return 0
  [[ "${rows}" -gt 0 && "${cols}" -gt 0 ]] || return 0
  printf '\033[8;%d;%dt' "${rows}" "${cols}"
  RESIZED=1
}

summary_command_label() {
  local label
  label="$(basename "${SUMMARY_SCRIPT}")"
  if [[ "${#SUMMARY_ARGS[@]}" -gt 0 ]]; then
    printf "%s %s" "${label}" "${SUMMARY_ARGS[*]}"
  else
    printf "%s" "${label}"
  fi
}

render_frame() {
  local frame="$1"
  local line
  if [[ -t 1 ]]; then
    printf '\033[H'
    while IFS= read -r line; do
      printf '\033[2K%s\n' "${line}"
    done <<< "${frame}"
    tput ed 2>/dev/null || true
  else
    printf "%s\n" "${frame}"
  fi
}

ORIG_ROWS="$(terminal_rows)"
ORIG_COLS="$(terminal_cols)"
RESIZED=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -t 1 ]]; then
    tput cnorm 2>/dev/null || true
    tput rmcup 2>/dev/null || true
    if [[ "${RESIZED}" == "1" && "${ORIG_ROWS}" -gt 0 && "${ORIG_COLS}" -gt 0 ]]; then
      printf '\033[8;%d;%dt' "${ORIG_ROWS}" "${ORIG_COLS}"
    fi
  fi
  exit "${status}"
}

trap cleanup EXIT INT TERM

if [[ -t 1 ]]; then
  tput smcup 2>/dev/null || true
  tput civis 2>/dev/null || true
fi

while true; do
  output=""
  status=0
  if output="$(FORCE_COLOR=1 "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}" 2>&1)"; then
    status=0
  else
    status=$?
  fi
  updated_at="$(date '+%Y-%m-%d %H:%M:%S')"
  header="${BOLD}${CYAN}Every ${INTERVAL}s: $(summary_command_label)    updated: ${updated_at}    next in: ${INTERVAL}s    Ctrl+C to exit${RESET}"
  frame="${header}"$'\n'"${output}"
  if [[ "${status}" -ne 0 ]]; then
    frame="${frame}"$'\n'$'\n'"slurm_queue_summary exited with status ${status}"
  fi

  plain="$(printf "%s\n" "${frame}" | strip_ansi)"
  needed_rows="$(printf "%s\n" "${plain}" | line_count)"
  needed_cols="$(printf "%s\n" "${plain}" | max_line_width)"
  current_rows="$(terminal_rows)"
  current_cols="$(terminal_cols)"
  target_rows="${current_rows}"
  target_cols="${current_cols}"
  if [[ "${needed_rows}" -gt "${target_rows}" ]]; then
    target_rows="${needed_rows}"
  fi
  if [[ "${needed_cols}" -gt "${target_cols}" ]]; then
    target_cols="${needed_cols}"
  fi

  request_resize "${target_rows}" "${target_cols}"
  render_frame "${frame}"

  sleep "${INTERVAL}"
done
