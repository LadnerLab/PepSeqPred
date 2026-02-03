#!/bin/bash
#SBATCH --job-name=generate_esm_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/slurm/logs/%x_%j.err

# if error, fail loudly
set -euo pipefail

# required input
: "${IN_FASTA:?Set IN_FASTA to the input FASTA path}"

# simple script defaults
MODEL_NAME="${MODEL_NAME:-esm2_t33_650M_UR50D}" # see ESM-2 documentation for other models
BATCH_SIZE="${BATCH_SIZE:-8}" # can probably get away with 16 or 24 on V100, double on A100
HANDLE_LONG="${HANDLE_LONG:-window}" # window or truncate

# HPC helpful variables
NAUID="${NAUID:-$USER}"
HOME_DIR="/home/${NAUID}"
SCRATCH_DIR="/scratch/${NAUID}"

# paths and directories
ESM_CLI="${ESM_CLI:-${HOME_DIR}/esm_cli.py}"
OUT_DIR="${SCRATCH_DIR}/esm2"
LOG_DIR="logs"
EMBEDDING_DIR="artifacts"

mkdir -p "${OUT_DIR}/${LOG_DIR}" "${OUT_DIR}/${EMBEDDING_DIR}"

# load Python conda env (make sure you have all packages installed)
module load anaconda3
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate hura # has numpy, pandas, fair-esm, and torch packages

# run Python script
srun python -u "${ESM_CLI}" \
    --fasta-file "${IN_FASTA}" \
    --model-name "${MODEL_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --handle-long "${HANDLE_LONG}" \
    --log-dir "${LOG_DIR}" \
    --log-json \
    --save-mode pt \
    --out-dir "${OUT_DIR}"

# USAGE: sbatch --export=ALL,IN_FASTA=/scratch/<NAUIDD>/<targets>.fasta generateembeddings.sh
