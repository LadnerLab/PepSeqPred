#!/bin/bash
#SBATCH --job-name=generate_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/%u/labels_slurm/%A_%a/%x.out
#SBATCH --error=/scratch/%u/labels_slurm/%A_%a/%x.err

usage() {
    echo "Usage:"
    echo "  Array mode: $0 <meta_path> <out_dir> <emb_root>"
    echo "    emb_root should contain shard_000...shard_003"
    echo ""
    echo "  Single mode: $0 <meta_path> <out_dir> <emb_dir> [out_name]"
    echo "    emb_dir should directly contain embedding .pt files"
    echo "    out_name defaults to labels_single.pt"
    echo ""
    echo "Examples:"
    echo "  sbatch --array=0-3 $0 <meta_path> <out_dir> <emb_root>"
    echo "  sbatch $0 <meta_path> <out_dir> <emb_dir> labels_subset_500.pt"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

META_PATH="$1"
OUT_DIR="$2"
EMB_ROOT="$3"
OUT_NAME="${4:-labels_single.pt}"

module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

mkdir -p "$OUT_DIR"

# single or array mode
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SHARD_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
    EMB_DIR="${EMB_ROOT}/shard_${SHARD_ID}"
    OUT_PATH="${OUT_DIR}/labels_shard_${SHARD_ID}.pt"
else
    EMB_DIR="${EMB_ROOT}"
    OUT_PATH="${OUT_DIR}/${OUT_NAME}"

    if [ ! -d "$EMB_DIR" ]; then
        echo "Embedding directory not found: $EMB_DIR"
        exit 1
    fi

    shopt -s nullglob
    PT_FILES=("$EMB_DIR/*.pt")
    shopt -u nullglob
    if [ "${#PT_FILES[@]}" -eq 0 ]; then
        echo "No .pt files found in $EMB_DIR"
        echo "Single mode expects a directory that contains embedding .pt files"
        exit 1
    fi
fi

if [ ! -d "$EMB_DIR" ]; then
    echo "Embedding directory not found: $EMB_DIR"
    exit 1
fi

srun python -u labels.pyz "$META_PATH" "$OUT_PATH" \
  --emb-dir "$EMB_DIR" \
  --restrict-to-embeddings

# command: sbatch generatelabels.sh path/to/meta.tsv path/to/out_dir /scratch/$USER/esm2/artifacts/pts
