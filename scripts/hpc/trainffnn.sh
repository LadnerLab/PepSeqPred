#!/bin/bash
#SBATCH --job-name=ffnn_v1.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/train_ffnn_slurm/%j_ffnn_v1.0/%x.out
#SBATCH --error=/scratch/%u/train_ffnn_slurm/%j_ffnn_v1.0/%x.err

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

EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}" # ensure batch size is 4 times what you would do for one GPU (for example. 256 = 64 * 4)
LR="${LR:-0.001}"
WD="${WD:-0.0}"
VAL_FRAC="${VAL_FRAC:-0.2}"
POS_WEIGHT="${POS_WEIGHT:-13.18999647945325}" # calculated from previous script
SAVE_PATH="/scratch/$USER/models/ffnn_v1.0"
NUM_WORKERS="${NUM_WORKERS:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-1000}"
STRIDE="${STRIDE:-900}"
SPLIT_TYPE="${SPLIT_TYPE:-id-family}" # id-family or id

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

${LAUNCHER} torchrun --nproc_per_node=4 train_ffnn.pyz \
    --embedding-dirs "${EMBEDDING_DIRS[@]}" \
    --label-shards "${LABEL_SHARDS[@]}" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --wd "$WD" \
    --pos-weight "$POS_WEIGHT" \
    --val-frac "$VAL_FRAC" \
    --split-type "$SPLIT_TYPE" \
    --save-path "$SAVE_PATH" \
    --num-workers "$NUM_WORKERS" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE"

# USAGE: sbatch trainffnn.sh /scratch/$USER/esm2/artifacts/pts/shard_000 /scratch/$USER/esm2/artifacts/pts/shard_001 /scratch/$USER/esm2/artifacts/pts/shard_002 /scratch/$USER/esm2/artifacts/pts/shard_003 -- /scratch/$USER/labels/labels_shard_000.pt /scratch/$USER/labels/labels_shard_001.pt /scratch/$USER/labels/labels_shard_002.pt /scratch/$USER/labels/labels_shard_003.pt
