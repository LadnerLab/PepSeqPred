#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

META_PATH="${ROOT_DIR}/data/localsample/meta_500_20_4_10_all.tsv"
EMB_DIR="${ROOT_DIR}/esm2/tests/sample/artifacts/pts"
OUT_PATH="${ROOT_DIR}/data/localsample/sample_data.pt"
ENV_DIR="${ROOT_DIR}/venv"

source "$ENV_DIR/Scripts/activate"
cd "${ROOT_DIR}"

python3 -m linker.linker_cli \
    "${META_PATH}" \
    "${EMB_DIR}" \
    "${OUT_PATH}"
