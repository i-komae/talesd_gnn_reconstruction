#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUMMARY_SCRIPT="${SCRIPT_DIR}/slurm_queue_summary.sh"
INTERVAL="${INTERVAL:-5}"
LIVE_HEADER_MAX_WIDTH="${LIVE_HEADER_MAX_WIDTH:-80}"
SLURM_QUEUE_LIVE_LAYOUT="${SLURM_QUEUE_LIVE_LAYOUT:-auto}"
RESIZE_SETTLE_SECONDS="${RESIZE_SETTLE_SECONDS:-0.05}"
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
Usage: scripts/slurm_queue_live.sh [-i SEC] [--layout auto|wide|vertical] [--no-resize] [--] [slurm_queue_summary options]

Periodically redraw scripts/slurm_queue_summary.sh with ANSI colors preserved.

Options:
  -i SEC, --interval SEC  Refresh interval in seconds. Default: 5.
  --layout MODE           GPU/CPU block layout: auto, wide, or vertical. Default: auto.
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
    --layout)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      SLURM_QUEUE_LIVE_LAYOUT="$2"
      shift 2
      ;;
    --layout=*)
      SLURM_QUEUE_LIVE_LAYOUT="${1#*=}"
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
if [[ "${SLURM_QUEUE_LIVE_LAYOUT}" != "auto" && "${SLURM_QUEUE_LIVE_LAYOUT}" != "wide" && "${SLURM_QUEUE_LIVE_LAYOUT}" != "vertical" ]]; then
  echo "invalid layout: ${SLURM_QUEUE_LIVE_LAYOUT}" >&2
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
  local current_rows
  local current_cols
  [[ "${RESIZE_TERMINAL}" == "1" ]] || return 0
  [[ -t 1 ]] || return 0
  [[ "${TERM:-}" != "dumb" ]] || return 0
  [[ "${rows}" -gt 0 && "${cols}" -gt 0 ]] || return 0
  current_rows="$(terminal_rows)"
  current_cols="$(terminal_cols)"
  if [[ "${current_rows}" == "${rows}" && "${current_cols}" == "${cols}" ]]; then
    return 0
  fi
  printf '\033[8;%d;%dt' "${rows}" "${cols}"
  RESIZED=1
}

capture_summary() {
  local layout="$1"
  if output="$(SLURM_QUEUE_LAYOUT="${layout}" FORCE_COLOR=1 "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}" 2>&1)"; then
    status=0
  else
    status=$?
  fi
}

summary_args_label() {
  if [[ "${#SUMMARY_ARGS[@]}" -gt 0 ]]; then
    printf "%s" "${SUMMARY_ARGS[*]}"
  else
    printf "summary"
  fi
}

truncate_plain() {
  local text="$1"
  local max_width="$2"
  if [[ "${#text}" -le "${max_width}" ]]; then
    printf "%s" "${text}"
    return
  fi
  if [[ "${max_width}" -le 3 ]]; then
    printf "%.*s" "${max_width}" "${text}"
    return
  fi
  printf "%s..." "${text:0:$((max_width - 3))}"
}

make_header() {
  local updated_at="$1"
  local remaining="$2"
  local updated_time
  local text
  updated_time="${updated_at##* }"
  text="Every ${INTERVAL}s: $(summary_args_label) | updated ${updated_time} | next ${remaining}s | Ctrl+C"
  printf "%s%s%s" "${CYAN}" "$(truncate_plain "${text}" "${LIVE_HEADER_MAX_WIDTH}")" "${RESET}"
}

render_frame() {
  local frame="$1"
  local max_rows="${2:-0}"
  local rows
  local visible_frame
  local rendered
  if [[ -t 1 ]]; then
    rows="${max_rows}"
    if [[ "${rows}" -le 0 ]]; then
      rows="$(terminal_rows)"
    fi
    if [[ "${rows}" -gt 0 ]]; then
      visible_frame="$(printf "%s\n" "${frame}" | awk -v max_rows="${rows}" 'NR <= max_rows { print }')"
    else
      visible_frame="${frame}"
    fi
    rendered="$(
      printf "%s\n" "${visible_frame}" \
        | awk -v clear=$'\033[2K' '{ if (NR > 1) printf "\n"; printf "%s%s", clear, $0 }'
    )"
    printf '\033[H%s\033[J' "${rendered}"
  else
    printf "%s\n" "${frame}"
  fi
}

update_header_line() {
  local updated_at="$1"
  local remaining="$2"
  [[ -t 1 ]] || return 0
  printf '\033[H\033[2K%s' "$(make_header "${updated_at}" "${remaining}")"
}

sleep_with_countdown() {
  local updated_at="$1"
  local remaining

  if [[ ! -t 1 || ! "${INTERVAL}" =~ ^[0-9]+$ || "${INTERVAL}" -le 1 ]]; then
    sleep "${INTERVAL}"
    return
  fi

  for ((remaining = INTERVAL; remaining > 0; remaining--)); do
    update_header_line "${updated_at}" "${remaining}"
    sleep 1
  done
}

ORIG_ROWS="$(terminal_rows)"
ORIG_COLS="$(terminal_cols)"
RESIZED=0
ALT_SCREEN_ACTIVE=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -t 1 ]]; then
    # Clear the live frame before leaving the alternate screen.  This also
    # keeps terminals that ignore smcup/rmcup from leaving stale text above
    # the restored shell prompt.
    printf '\033[H\033[J'
    tput cnorm 2>/dev/null || true
    if [[ "${ALT_SCREEN_ACTIVE}" == "1" ]]; then
      tput rmcup 2>/dev/null || true
    fi
    if [[ "${RESIZED}" == "1" && "${ORIG_ROWS}" -gt 0 && "${ORIG_COLS}" -gt 0 ]]; then
      printf '\033[8;%d;%dt' "${ORIG_ROWS}" "${ORIG_COLS}"
    fi
  fi
  exit "${status}"
}

trap cleanup EXIT INT TERM

if [[ -t 1 ]]; then
  if tput smcup 2>/dev/null; then
    ALT_SCREEN_ACTIVE=1
  fi
  tput civis 2>/dev/null || true
fi

AUTO_VERTICAL_LOCK=0
AUTO_VERTICAL_COLS=0

while true; do
  output=""
  status=0
  current_cols="$(terminal_cols)"
  summary_layout="${SLURM_QUEUE_LIVE_LAYOUT}"
  if [[ "${summary_layout}" == "auto" && -t 1 && "${RESIZE_TERMINAL}" == "1" && "${TERM:-}" != "dumb" ]]; then
    if [[ "${AUTO_VERTICAL_LOCK}" == "1" && "${current_cols}" -gt 0 && "${current_cols}" -le "${AUTO_VERTICAL_COLS}" ]]; then
      summary_layout="vertical"
    else
      summary_layout="wide"
    fi
  fi
  capture_summary "${summary_layout}"
  updated_at="$(date '+%Y-%m-%d %H:%M:%S')"
  header="$(make_header "${updated_at}" "${INTERVAL}")"
  frame="${header}"$'\n'"${output}"
  if [[ "${status}" -ne 0 ]]; then
    frame="${frame}"$'\n'$'\n'"slurm_queue_summary exited with status ${status}"
  fi

  plain="$(printf "%s\n" "${frame}" | strip_ansi)"
  needed_rows="$(printf "%s\n" "${plain}" | line_count)"
  needed_cols="$(printf "%s\n" "${plain}" | max_line_width)"
  needed_cols_with_margin=$((needed_cols + 1))

  if [[ "${SLURM_QUEUE_LIVE_LAYOUT}" == "auto" && "${summary_layout}" == "wide" && "${current_cols}" -gt 0 && "${needed_cols_with_margin}" -gt "${current_cols}" ]]; then
    request_resize "$(terminal_rows)" "${needed_cols_with_margin}"
    sleep "${RESIZE_SETTLE_SECONDS}" 2>/dev/null || true
    current_cols="$(terminal_cols)"
    if [[ "${current_cols}" -le 0 || "${needed_cols_with_margin}" -gt "${current_cols}" ]]; then
      capture_summary "vertical"
      header="$(make_header "${updated_at}" "${INTERVAL}")"
      frame="${header}"$'\n'"${output}"
      if [[ "${status}" -ne 0 ]]; then
        frame="${frame}"$'\n'$'\n'"slurm_queue_summary exited with status ${status}"
      fi
      plain="$(printf "%s\n" "${frame}" | strip_ansi)"
      needed_rows="$(printf "%s\n" "${plain}" | line_count)"
      needed_cols="$(printf "%s\n" "${plain}" | max_line_width)"
      needed_cols_with_margin=$((needed_cols + 1))
      AUTO_VERTICAL_LOCK=1
      AUTO_VERTICAL_COLS="${current_cols}"
    else
      AUTO_VERTICAL_LOCK=0
      AUTO_VERTICAL_COLS=0
    fi
  fi
  current_rows="$(terminal_rows)"
  current_cols="$(terminal_cols)"
  target_rows="${current_rows}"
  target_cols="${current_cols}"
  if [[ "${needed_rows}" -gt "${target_rows}" ]]; then
    target_rows="${needed_rows}"
  fi
  # Keep one spare column. Writing printable text into the final terminal
  # column can trigger automatic wrap in some terminals and make the last
  # visible character appear to vanish during redraw.
  needed_cols_with_margin=$((needed_cols + 1))
  if [[ "${needed_cols_with_margin}" -gt "${target_cols}" ]]; then
    target_cols="${needed_cols_with_margin}"
  fi

  request_resize "${target_rows}" "${target_cols}"
  render_frame "${frame}" "${target_rows}"

  sleep_with_countdown "${updated_at}"
done
