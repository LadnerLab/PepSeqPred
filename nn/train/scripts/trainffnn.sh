#!/bin/bash
#SBATCH --job-name=ffnn_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/%u/train_ffnn_slurm/%j/%x.out
#SBATCH --error=/scratch/%u/train_ffnn_slurm/%j/%x.err

# for testing
USE_SRUN="${USE_SRUN:-1}"

# handle incorrect usage
usage() {
    echo "Usage $0 <input_data>"
    echo "    input_data: Path to the .pt input file."
    echo "    See the documentation for more information on other args."
}

# require at least the .pt file
if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

INPUT_DATA=$1
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
WD="${WD:-0.0}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SAVE_PATH="/scratch/$USER/models/ffnn_v1"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${SAVE_PATH}"

# load Python Conda environment
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

${LAUNCHER} python -u train_ffnn.pyz \
    "${INPUT_DATA}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --wd "${WD}" \
    --use-pos-weight \
    --val-frac "${VAL_FRAC}" \
    --save-path "${SAVE_PATH}" \
    --num-workers "${NUM_WORKERS}"
