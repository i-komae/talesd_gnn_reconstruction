#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "submit_server_allsrc_homogeneous_hetero_comparison.sh is a legacy wrapper; use submit_server_source_balanced_homogeneous_hetero_comparison.sh" >&2
exec "${SCRIPT_DIR}/submit_server_source_balanced_homogeneous_hetero_comparison.sh" "$@"
