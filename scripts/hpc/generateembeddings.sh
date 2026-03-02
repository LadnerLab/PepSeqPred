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

# for testing
USE_SRUN="${USE_SRUN:-1}"

usage() {
    echo "Usage:"
    echo "  $0 <in_fasta> [out_dir] [model_name] [max_tokens] [batch_size]"
    echo ""
    echo "Positional args override env vars."
    echo "  in_fasta   required (or set IN_FASTA)"
    echo "  out_dir    default: /scratch/\$USER/esm2"
    echo "  model_name default: esm2_t33_650M_UR50D"
    echo "  max_tokens default: 1022"
    echo "  batch_size default: 24"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 /scratch/\$USER/data/targets.fasta"
    echo "  sbatch $0 /scratch/\$USER/data/targets.fasta /scratch/\$USER/esm2 esm2_t33_650M_UR50D 1022 24"
    echo "  sbatch --export=ALL,IN_FASTA=/scratch/\$USER/data/targets.fasta $0"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# get shard information from SLURM
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

# HPC helpful variables
NAUID="${NAUID:-$USER}"
HOME_DIR="/home/${NAUID}"
SCRATCH_DIR="/scratch/${NAUID}"

# positional args with env-var fallback
IN_FASTA="${1:-${IN_FASTA:-}}"
OUT_DIR="${2:-${OUT_DIR:-${SCRATCH_DIR}/esm2}}"
MODEL_NAME="${3:-${MODEL_NAME:-esm2_t33_650M_UR50D}}" # see ESM-2 documentation for other models
MAX_TOKENS="${4:-${MAX_TOKENS:-1022}}" # max number of tokens in model's context window
BATCH_SIZE="${5:-${BATCH_SIZE:-24}}" # can probably get away with 16 or 24 on V100, double on A100

if [ -z "${IN_FASTA}" ]; then
    echo "Missing required input FASTA path."
    usage
    exit 1
fi

# filenaming keys
EMBEDDING_KEY_MODE="${EMBEDDING_KEY_MODE:-id-family}"
KEY_DELIMITER="${KEY_DELIMITER:--}"

# paths and directories
LOG_DIR="logs"
EMBEDDING_DIR="artifacts"

mkdir -p "${OUT_DIR}/${LOG_DIR}" "${OUT_DIR}/${EMBEDDING_DIR}"

# load Python conda env (make sure you have all packages installed)
module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

# turn off srun for testing
if [ "${USE_SRUN}" -eq 1 ]; then
    LAUNCHER="srun"
else
    LAUNCHER=""
fi

# run Python script
${LAUNCHER} python -u esm.pyz \
    --fasta-file "${IN_FASTA}" \
    --model-name "${MODEL_NAME}" \
    --embedding-key-mode "${EMBEDDING_KEY_MODE}" \
    --key-delimiter "${KEY_DELIMITER}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${SHARD_ID}" \
    --log-dir "${LOG_DIR}" \
    --log-json \
    --out-dir "${OUT_DIR}"

# MONSOON USAGE: sbatch --export=ALL,IN_FASTA=/scratch/$USER/data/<targets>.fasta generateembeddings.sh
