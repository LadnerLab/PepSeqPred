#!/bin/bash
#SBATCH --job-name=link_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/linker_slurm/%x_%j.out
#SBATCH --error=/scratch/%u/linker_slurm/%x_%j.err

set -euo pipefail

# handle incorrect usage
usage() {
    echo "Usage $0 <meta_path> <emb_dir> <out_path>"
    echo "    meta_path: Path to preprocessed metadata."
    echo "    emb_dir: Path to directory containing ESM-2 embeddings."
    echo "    out_path: Desired output directory."
}

if [ "$#" -ne 3 ]; then
    usage
fi

# set script variables
META_PATH="$1"
EMB_DIR="$2"
OUT_PATH="$3"

# load Python Conda virtual environment (ensure env installed on HPC)
# module purge
# module load anaconda3
# module load cuda
# source $CONDA_PREFIX/etc/profile.d/conda.sh
# conda activate pepseqpred
source "../../venv/Scripts/activate"

python3 -u linker_cli.pyz \
    "${META_PATH}" \
    "${EMB_DIR}" \
    "${OUT_PATH}"
