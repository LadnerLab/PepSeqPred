#!/bin/bash
#SBATCH --job-name=link_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/%u/linker_slurm/%x_%j.out
#SBATCH --error=/scratch/%u/linker_slurm/%x_%j.err

# handle incorrect usage
usage() {
    echo "Usage $0 <meta_path> <out_path> <emb_dir_1> [emb_dir2, ..., emb_dir_n]"
    echo "    meta_path: Path to preprocessed metadata."
    echo "    out_path: Desired output .pt file path."
    echo "    emb_dir_*: One or more directories containing .pt embeddings."
}

# need at least metadata, output directory, and one embedding directory
if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

META_PATH="$1"
OUT_PATH="$2"
shift 2
EMB_DIRS=("$@")

# load Python Conda virtual environment (ensure env installed on HPC)
module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

# build script arguments list for linker
PY_ARGS=(
    "$META_PATH"
    "$OUT_PATH"
)
for d in "${EMB_DIRS[@]}"; do
    PY_ARGS+=("--emb-dir" "$d")
done

srun python -u linker_cli.pyz "${PY_ARGS[@]}"
