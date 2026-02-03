#!/bin/bash
#SBATCH --job-name=generate_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/%u/labels_slurm/%A_%a/%x.out
#SBATCH --error=/scratch/%u/labels_slurm/%A_%a/%x.err

usage() {
    echo "Usage $0 <meta_path> <out_dir> <emb_root>"
    echo "    meta_path: Path to preprocessed metadata."
    echo "    out_dir: Directory to write labels_shard_XXX.pt files."
    echo "    emb_root: Root directory containing shard_000...shard_003."
    echo ""
    echo "Run with SLURM array: sbatch $0 <meta_path> <out_dir> <emb_root>"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "SLURM_ARRAY_TASK_ID is required. Use: sbatch --array=0-3"
    exit 1
fi

META_PATH="$1"
OUT_DIR="$2"
EMB_ROOT="$3"

module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

SHARD_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
SHARD_DIR="${EMB_ROOT}/shard_${SHARD_ID}"
OUT_PATH="${OUT_DIR}/labels_shard_${SHARD_ID}.pt"

mkdir -p "$OUT_DIR"

srun python -u labels.pyz "$META_PATH" "$OUT_PATH" \
  --emb-dir "$SHARD_DIR" \
  --restrict-to-embeddings

# command: sbatch generatelabels.sh path/to/meta.tsv path/to/out_dir /scratch/$USER/esm2/artifacts/pts
