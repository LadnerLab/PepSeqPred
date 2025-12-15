#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SAMPLE_DIR="${ROOT_DIR}/esm2/tests/sample"
FASTA="${ROOT_DIR}/data/localsample/500_design.fasta"
ENV_DIR="${ROOT_DIR}/venv"

mkdir -p "${SAMPLE_DIR}" "${SAMPLE_DIR}/artifacts" "${SAMPLE_DIR}/logs"

source "${ENV_DIR}/Scripts/activate"

cd "${ROOT_DIR}"

python3 -m esm2.esm_cli \
    --fasta-file "$FASTA" \
    --model-name esm2_t6_8M_UR50D \
    --batch-size 8 \
    --log-dir "logs" \
    --log-json \
    --out-dir "$SAMPLE_DIR"
