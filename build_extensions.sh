#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

env UV_CACHE_DIR="$UV_CACHE_DIR" uv sync
env UV_CACHE_DIR="$UV_CACHE_DIR" uv run python setup.py build_ext --inplace
