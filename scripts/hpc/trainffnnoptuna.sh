#!/bin/bash
#SBATCH --job-name=ffnn_optuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/optuna_ffnn/%j/%x.out
#SBATCH --error=/scratch/%u/optuna_ffnn/%j/%x.err

# for testing
USE_SRUN="${USE_SRUN:-1}"

# handle incorrect usage
usage() {
    echo "Usage: $0 <embedding_dirs...> -- <label_shards...>"
    echo "  embedding_dirs: one or more directories containing per-protein embeddings (.pt)"
    echo "  label_shards: one or more label shard .pt files"
    echo ""
    echo "Example:"
    echo "  $0 /scratch/$USER/embeddings/shard1 /scratch/$USER/embeddings/shard2 -- /scratch/$USER/labels/labels_00.pt /scratch/$USER/labels/labels_01.pt"
}

# require at least one embedding dir, separator (--), one label shard
if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

# parse embedding dirs until "--" is reached
EMBEDDING_DIRS=()
LABEL_SHARDS=()
PARSE_LABELS=0
for arg in "$@"; do
    if [ "${arg}" = "--" ]; then
        PARSE_LABELS=1
        continue
    fi
    if [ "${PARSE_LABELS}" -eq 0 ]; then
        EMBEDDING_DIRS+=("${arg}")
    else
        LABEL_SHARDS+=("${arg}")
    fi
done

if [ "${#EMBEDDING_DIRS[@]}" -eq 0 ] || [ "${#LABEL_SHARDS[@]}" -eq 0 ]; then
    usage
    exit 1
fi

# tuning controls
STUDY_NAME="${STUDY_NAME:-ffnn_optuna_v1}"
N_TRIALS="${N_TRIALS:-20}"
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-42}"
METRIC="${METRIC:-pr_auc}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SUBSET="${SUBSET:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-1000}"
STRIDE="${STRIDE:-900}"
POS_WEIGHT="${POS_WEIGHT:-13.18999647945325}" # calculated from other script

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

# turn off srun for testing
if [ "${USE_SRUN}" -eq 1 ]; then
    LAUNCHER="srun"
else
    LAUNCHER=""
fi

# DDP timeout setting for testing
DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-5}"
export PEPSEQPRED_DDP_TIMEOUT_MIN="$DDP_TIMEOUT_MIN"

${LAUNCHER} torchrun --nproc_per_node=4 train_ffnn_optuna.pyz \
    --embedding-dirs "${EMBEDDING_DIRS[@]}" \
    --label-shards "${LABEL_SHARDS[@]}" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE" \
    --n-trials "$N_TRIALS" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
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
    --pos-weight "$POS_WEIGHT" \
    --pruner-warmup "$PRUNER_WARMUP" \
    --timeout-s "$TIMEOUT_S" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE"

# USAGE: sbatch trainffnnoptuna.sh /scratch/$USER/esm2/artifacts/pts/shard_000 /scratch/$USER/esm2/artifacts/pts/shard_001 /scratch/$USER/esm2/artifacts/pts/shard_002 /scratch/$USER/esm2/artifacts/pts/shard_003 -- /scratch/$USER/labels/labels_shard_000.pt /scratch/$USER/labels/labels_shard_001.pt /scratch/$USER/labels/labels_shard_002.pt /scratch/$USER/labels/labels_shard_003.pt
