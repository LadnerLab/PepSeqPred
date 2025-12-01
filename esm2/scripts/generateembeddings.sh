#!/bin/bash
#SBATCH --job-name=gen_esm_embs
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gpus=a100
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/esm2_slurm/%x_%j.out
#SBATCH --error=/scratch/%u/esm2_slurm/%x_%j.err

# if error, fail loudly
set -euo pipefail

# get shard information from SLURM
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

# required input
: "${IN_FASTA:?Set IN_FASTA to the input FASTA path}"

# simple script defaults
MODEL_NAME="${MODEL_NAME:-esm2_t33_650M_UR50D}" # see ESM-2 documentation for other models
MAX_TOKENS="${MAX_TOKENS:-1022}" # max number of tokens in model's context window
BATCH_SIZE="${BATCH_SIZE:-32}" # can probably get away with 16 or 24 on V100, double on A100

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
module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

# set CUDA environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# run Python script
srun python -u "${ESM_CLI}" \
    --fasta-file "${IN_FASTA}" \
    --model-name "${MODEL_NAME}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${SHARD_ID}" \
    --log-dir "${LOG_DIR}" \
    --log-json \
    --out-dir "${OUT_DIR}"

# MONSOON USAGE: sbatch --export=ALL,IN_FASTA=/scratch/$USER/data/<targets>.fasta generateembeddings.sh
