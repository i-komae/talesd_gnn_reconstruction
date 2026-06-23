#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "submit_server_hetero_light_graph_export.sh is a legacy wrapper; use submit_server_hetero_source_balanced_graph_export.sh" >&2
exec "${SCRIPT_DIR}/submit_server_hetero_source_balanced_graph_export.sh" "$@"
