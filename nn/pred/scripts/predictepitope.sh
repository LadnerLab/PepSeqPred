#!/bin/bash
#SBATCH --job-name=pepseqpred
#SBATCH --output=/scratch/%u/pepseqpred_slurm/%j/%x.out
#SBATCH --error=/scratch/%u/pepseqpred_slurm/%j/%x.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

usage() {
    echo "Usage $0 <checkpoint> <protein_seq> <peptide>"
    echo "    checkpoint: Path to the best model checkpoint .pt file."
    echo "    protein_seq: The entire protein sequence the peptide is derived from as a continuous string."
    echo "    peptide: The peptide sequence as a continuous string."
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

CHECKPOINT=$1
PROTEIN_SEQ=$2
PEPTIDE=$3

module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

srun python -u make_prediction.pyz \
    "${CHECKPOINT}" \
    "${PROTEIN_SEQ}" \
    "${PEPTIDE}" \
    --log-json
