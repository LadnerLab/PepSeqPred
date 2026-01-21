#!/bin/bash
#SBATCH --job-name=ffnn_optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/optuna_ffnn/%j/%x.out
#SBATCH --error=/scratch/%u/optuna_ffnn/%j/%x.err

# usage: sbatch ./trainffnnoptuna.sh /scratch/$USER/path/to/input_data.pt
usage() {
    echo "Usage: $0 <input_data_pt>"
    exit 1
}
if [ "$#" -lt 1 ]; then
    usage
fi

INPUT_DATA=$1

# tuning controls
STUDY_NAME="${STUDY_NAME:-ffnn_optuna_v1}"
N_TRIALS="${N_TRIALS:-50}"
EPOCHS="${EPOCHS:-25}"
METRIC="${METRIC:-recall}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SUBSET="${SUBSET:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# output paths
SAVE_PATH="${SAVE_PATH:-/scratch/$USER/models/${STUDY_NAME}}"
CSV_PATH="${CSV_PATH:-/scratch/$USER/optuna/${STUDY_NAME}_trials.csv}"

# Optuna storage
STORAGE="${STORAGE:-sqlite:////scratch/$USER/optuna/${STUDY_NAME}.db}"

# architecture search space
ARCH_MODE="${ARCH_MODE:-flat}" # flat, bottleneck, pyramid
DEPTH_MIN="${DEPTH_MIN:-2}"
DEPTH_MAX="${DEPTH_MAX:-6}"
WIDTH_MIN="${WIDTH_MIN:-64}"
WIDTH_MAX="${WIDTH_MAX:-512}"

# optimizer search space
LR_MIN="${LR_MIN:-1e-4}"
LR_MAX="${LR_MAX:-3e-3}"
WD_MIN="${WD_MIN:-1e-8}"
WD_MAX="${WD_MAX:-1e-2}"
BATCH_SIZES="${BATCH_SIZES:-64,128,256,512,1024}"

# pruning and timeout controls
PRUNER_WARMUP="${PRUNER_WARMUP:-2}"
TIMEOUT_S="${TIMEOUT_S:-0}" # timeout in seconds

# build directories
mkdir -p "$SAVE_PATH"
mkdir -p "$(dirname "$CSV_PATH")"
mkdir -p "/scratch/$USER/optuna"
mkdir -p "/scratch/$USER/optuna_ffnn/$SLURM_JOB_ID"

# load Python Conda environment
module purge
module load anaconda3
module load cuda
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pepseqpred

srun python -u train_ffnn_optuna.pyz \
    "$INPUT_DATA" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --n-trials "$N_TRIALS" \
    --epochs "$EPOCHS" \
    --metric "$METRIC" \
    --val-frac "$VAL_FRAC" \
    --subset "$SUBSET" \
    --num-workers "$NUM_WORKERS" \
    --save-path "$SAVE_PATH" \
    --csv-path "$CSV_PATH" \
    --arch-mode "$ARCH_MODE" \
    --depth-min "$DEPTH_MIN" \
    --depth-max "$DEPTH_MAX" \
    --width-min "$WIDTH_MIN" \
    --width-max "$WIDTH_MAX" \
    --batch-sizes "$BATCH_SIZES" \
    --lr-min "$LR_MIN" \
    --lr-max "$LR_MAX" \
    --wd-min "$WD_MIN" \
    --wd-max "$WD_MAX" \
    --use-pos-weight \
    --pruner-warmup "$PRUNER_WARMUP" \
    --timeout-s "$TIMEOUT_S"
