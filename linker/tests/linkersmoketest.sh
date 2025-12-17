#!/usr/bin/env bash
set -euo pipefail

echo "Starting smoke test ============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

META_DIR="${ROOT_DIR}/linker/tests/data"
META_PATH="${META_DIR}/test_meta.tsv"
EMB_DIR="${META_DIR}/embeddings"
OUT_PATH="${META_DIR}/test_dataset.pt"
ENV_DIR="../../venv"

mkdir -p "${EMB_DIR}"

echo "Creating test metadata at ${META_PATH}..."

cat > "${META_PATH}" << 'EOF'
CodeName	AlignStart	AlignStop	FullName	Peptide	Def epitope	Uncertain	Not epitope
TEST_0001	0	30	"ID=PROT1 AC=P00001 OXX=11111"	AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA	1	0	0
TEST_0002	10	40	"ID=PROT2 AC=P00002 OXX=22222"	BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB	0	1	0
TEST_0003	50	80	"ID=PROT3 AC=P00003 OXX=33333"	CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC	0	0	1
EOF

# environment config
echo "Activating virtual environment..."
source "$ENV_DIR/Scripts/activate"

echo "Creating dummy .pt embeddings at ${EMB_DIR}..."

python3 <<PY
import torch
from pathlib import Path

emb_dir = Path("${EMB_DIR}")

seq_len = 400
d_model = 1281

for id in ["PROT1", "PROT2", "PROT3"]:
    index = int(id[-1])
    emb = torch.full((seq_len, d_model), float(index))
    torch.save(emb, emb_dir / f"{id}.pt")
    print(f"Created '{id}.pt' embedding")
PY

echo "Running linker_cli.py..."

cd "${ROOT_DIR}"

python3 -m linker.linker_cli \
    "${META_PATH}" \
    "${OUT_PATH}" \
    --emb-dir "${EMB_DIR}"

echo "Inspecting saved dataset at ${OUT_PATH}..."

python3 <<PY
import sys
sys.path.insert(0, "${ROOT_DIR}")

import torch
from pathlib import Path
from pipelineio.peptidedataset import PeptideDataset

path = Path("${OUT_PATH}")
data = PeptideDataset.load(path)

print(f"Loaded dataset from: {path}")
print(f"Number of samples: {data.embeddings.size(0)}")
print(f"Embedding shape:   {data.embeddings.size()}")
print(f"Targets shape:     {data.targets.size()}")
print(f"First code name:   {data.code_names[0]}")
print(f"First protein ID:  {data.protein_ids[0]}")
print(f"First target vec:  {data.targets[0]}")
PY

echo "Smoke test complete ============================================================"
