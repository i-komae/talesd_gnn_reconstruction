#!/usr/bin/env bash

sweep_format_elapsed() {
  local seconds="$1"
  if (( seconds < 60 )); then
    printf "%ss" "${seconds}"
    return
  fi
  local minutes=$((seconds / 60))
  local sec=$((seconds % 60))
  if (( minutes < 60 )); then
    printf "%dm%02ds" "${minutes}" "${sec}"
    return
  fi
  local hours=$((minutes / 60))
  local minute=$((minutes % 60))
  printf "%dh%02dm" "${hours}" "${minute}"
}

sweep_progress_bar() {
  local done_count="$1"
  local total_count="$2"
  local width="${SWEEP_PROGRESS_WIDTH:-30}"
  local filled=0
  if (( total_count > 0 )); then
    filled=$((done_count * width / total_count))
  fi
  local empty=$((width - filled))
  local bar=""
  local i
  for ((i = 0; i < filled; i++)); do
    bar="${bar}#"
  done
  for ((i = 0; i < empty; i++)); do
    bar="${bar}-"
  done
  printf "[%s]" "${bar}"
}

sweep_latest_status() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    printf "starting"
    return
  fi
  local line
  line="$(awk '
    NF > 0 { fallback=$0 }
    /fit scalers:/ { line=$0 }
    /scan source paths:/ { line=$0 }
    /scan particle labels:/ { line=$0 }
    /scan detector IDs:/ { line=$0 }
    /device=.*data_loader_workers=/ { line=$0 }
    /epoch=[0-9][0-9][0-9][0-9]/ { line=$0 }
    /validation predict:/ { line=$0 }
    /test predict:/ { line=$0 }
    /validation metrics:/ { line=$0 }
    /test metrics:/ { line=$0 }
    /stage_seconds:/ { line=$0 }
    END {
      if (line != "") {
        print line
      } else {
        print fallback
      }
    }
  ' "${log_path}")"
  if [[ -n "${line}" ]]; then
    printf "%s" "${line}"
  else
    printf "log=%s" "${log_path}"
  fi
}

sweep_pid_is_running() {
  local pid="$1"
  local running_pid
  for running_pid in $(jobs -pr); do
    if [[ "${running_pid}" == "${pid}" ]]; then
      return 0
    fi
  done
  return 1
}

sweep_report() {
  local now
  now="$(date +%s)"
  local elapsed=$((now - sweep_started_at))
  local running_count="${#running_pids[@]}"
  local bar
  bar="$(sweep_progress_bar "${completed_jobs}" "${sweep_total_jobs}")"
  printf "sweep progress %s %d/%d done, %d running, %d failed elapsed=%s\n" \
    "${bar}" \
    "${completed_jobs}" \
    "${sweep_total_jobs}" \
    "${running_count}" \
    "${failed_jobs}" \
    "$(sweep_format_elapsed "${elapsed}")"

  local i
  for ((i = 0; i < ${#running_pids[@]}; i++)); do
    printf "  running %-28s pid=%s %s\n" \
      "${running_tags[$i]}" \
      "${running_pids[$i]}" \
      "$(sweep_latest_status "${running_logs[$i]}")"
  done
}

sweep_report_if_due() {
  local now
  now="$(date +%s)"
  if (( now - sweep_last_report_at >= SWEEP_PROGRESS_INTERVAL )); then
    sweep_report
    sweep_last_report_at="${now}"
  fi
}

sweep_poll_jobs() {
  local new_pids=()
  local new_tags=()
  local new_logs=()
  local new_count=0
  local i pid tag log_path status

  for ((i = 0; i < ${#running_pids[@]}; i++)); do
    pid="${running_pids[$i]}"
    tag="${running_tags[$i]}"
    log_path="${running_logs[$i]}"
    if sweep_pid_is_running "${pid}"; then
      new_pids[$new_count]="${pid}"
      new_tags[$new_count]="${tag}"
      new_logs[$new_count]="${log_path}"
      new_count=$((new_count + 1))
      continue
    fi

    if wait "${pid}"; then
      status="OK"
    else
      status="FAILED"
      failed=1
      failed_jobs=$((failed_jobs + 1))
    fi
    completed_jobs=$((completed_jobs + 1))
    printf "sweep job done: %-28s status=%s log=%s\n" "${tag}" "${status}" "${log_path}"
    printf "  final %-28s %s\n" "${tag}" "$(sweep_latest_status "${log_path}")"
  done

  running_pids=()
  running_tags=()
  running_logs=()
  for ((i = 0; i < new_count; i++)); do
    running_pids[$i]="${new_pids[$i]}"
    running_tags[$i]="${new_tags[$i]}"
    running_logs[$i]="${new_logs[$i]}"
  done
}

sweep_wait_for_slot() {
  local parallel_jobs_int="$1"
  while (( ${#running_pids[@]} >= parallel_jobs_int )); do
    sweep_poll_jobs
    sweep_report_if_due
    if (( ${#running_pids[@]} >= parallel_jobs_int )); then
      sleep "${SWEEP_PROGRESS_POLL_INTERVAL}"
    fi
  done
}

sweep_wait_all() {
  while (( ${#running_pids[@]} > 0 )); do
    sweep_poll_jobs
    sweep_report_if_due
    if (( ${#running_pids[@]} > 0 )); then
      sleep "${SWEEP_PROGRESS_POLL_INTERVAL}"
    fi
  done
}
