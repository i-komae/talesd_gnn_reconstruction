#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  cat >&2 <<EOF
Usage: $0 RUN_DIR [GRAPH_OUTPUT_BASE]

Move legacy flat graph shards into RUN_DIR/graphs after a run has finished.
By default, GRAPH_OUTPUT_BASE is read from RUN_DIR/config/train.env.

Set KEEP_COMPAT_SYMLINKS=0 to avoid leaving symlinks at the old flat paths.
EOF
  exit 2
fi

RUN_DIR="$1"
TRAIN_ENV="${RUN_DIR}/config/train.env"
if [[ $# -ge 2 ]]; then
  GRAPH_OUTPUT="$2"
else
  if [[ ! -f "${TRAIN_ENV}" ]]; then
    echo "missing ${TRAIN_ENV}; pass GRAPH_OUTPUT_BASE explicitly" >&2
    exit 2
  fi
  GRAPH_OUTPUT="$(awk -F= '$1 == "GRAPH_OUTPUT" { print substr($0, index($0, "=") + 1); exit }' "${TRAIN_ENV}")"
fi

if [[ -z "${GRAPH_OUTPUT}" ]]; then
  echo "GRAPH_OUTPUT is empty" >&2
  exit 2
fi

OLD_DIR="$(cd "$(dirname "${GRAPH_OUTPUT}")" && pwd)"
BASE_NAME="$(basename "${GRAPH_OUTPUT}")"
STEM="${BASE_NAME%.h5}"
TARGET_DIR="${RUN_DIR}/graphs"
CONFIG_DIR="${RUN_DIR}/config"
KEEP_COMPAT_SYMLINKS="${KEEP_COMPAT_SYMLINKS:-1}"

mkdir -p "${TARGET_DIR}" "${CONFIG_DIR}"

shopt -s nullglob
shards=("${OLD_DIR}/${STEM}"_*.h5)
if (( ${#shards[@]} == 0 && -f "${OLD_DIR}/${BASE_NAME}" )); then
  shards=("${OLD_DIR}/${BASE_NAME}")
fi
if (( ${#shards[@]} == 0 )); then
  echo "no graph shards found for ${GRAPH_OUTPUT}" >&2
  exit 1
fi

echo "run_dir=${RUN_DIR}"
echo "old_graph_base=${GRAPH_OUTPUT}"
echo "target_graph_dir=${TARGET_DIR}"
echo "shards=${#shards[@]}"

for old_path in "${shards[@]}"; do
  file_name="$(basename "${old_path}")"
  new_path="${TARGET_DIR}/${file_name}"
  if [[ "${old_path}" == "${new_path}" ]]; then
    continue
  fi
  if [[ -e "${new_path}" ]]; then
    echo "target already exists: ${new_path}" >&2
    exit 1
  fi
  mv "${old_path}" "${new_path}"
  if [[ "${KEEP_COMPAT_SYMLINKS}" == "1" ]]; then
    ln -s "${new_path}" "${old_path}"
  fi
done

NEW_GRAPH_BASE="${TARGET_DIR}/${BASE_NAME}"
printf "%s\n" "${NEW_GRAPH_BASE}" > "${CONFIG_DIR}/graph_input.txt"
{
  echo "migrated_at=$(date)"
  echo "old_graph_base=${GRAPH_OUTPUT}"
  echo "new_graph_base=${NEW_GRAPH_BASE}"
  echo "keep_compat_symlinks=${KEEP_COMPAT_SYMLINKS}"
  echo "shards=${#shards[@]}"
} > "${CONFIG_DIR}/graph_migration.env"

echo "new_graph_base=${NEW_GRAPH_BASE}"
echo "wrote ${CONFIG_DIR}/graph_input.txt"
