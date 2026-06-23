#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "submit_server_hetero_light_size_sweep.sh is a legacy wrapper; use submit_server_hetero_source_balanced_size_sweep.sh" >&2
if [[ -n "${LIGHT_TARGETS:-}" && -z "${TARGETS:-}" ]]; then
  export TARGETS="${LIGHT_TARGETS}"
fi
exec "${SCRIPT_DIR}/submit_server_hetero_source_balanced_size_sweep.sh" "$@"
