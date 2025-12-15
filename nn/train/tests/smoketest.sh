#!/usr/bin/env bash
set -euo pipefail

echo "Starting smoke test ============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

INPUT_DATA="${ROOT_DIR}/data/localsample/sample_data.pt"
SAVE_PATH="${ROOT_DIR}/nn/models/samples"
mkdir -p "${SAVE_PATH}"

EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
SUBSET="${SUBSET:-0}"

echo "Running train_ffnn_cli.py..."

cd "${ROOT_DIR}"

python3 -m nn.train.train_ffnn_cli \
    "${INPUT_DATA}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --wd "${WEIGHT_DECAY}" \
    --val-frac "${VALIDATION_FRACTION}" \
    --save-path "${SAVE_PATH}" \
    --subset "${SUBSET}"

echo "Smoke test complete ============================================================"
