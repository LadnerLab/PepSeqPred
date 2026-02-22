#!/bin/bash
#SBATCH --job-name=predict_epitope
#SBATCH --output=/scratch/%u/predict_epitope_slurm/%j/%x.out
#SBATCH --error=/scratch/%u/predict_epitope_slurm/%j/%x.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:01:00

usage() {
    echo "Usage $0 <checkpoint> <protein_seq> <peptide>"
    echo "    checkpoint: Path to the best model checkpoint .pt file."
    echo "    fasta_input: Path to a FASTA file containing protein sequence(s) and peptides."
    echo "    output_fasta: Path to output FASTA file to write binary epitope masks to."
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

CHECKPOINT=$1
FASTA_INPUT=$2
OUTPUT_FASTA=$3

module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

srun python -u predict.pyz \
    "${CHECKPOINT}" \
    "${FASTA_INPUT}" \
    --output-fasta "${OUTPUT_FASTA}" \
    --log-json
