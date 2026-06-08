#!/bin/bash
#SBATCH --job-name=pepseqpred_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/train_slurm/output/%x_%j.out
#SBATCH --error=/scratch/%u/train_slurm/error/%x_%j.err

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
    echo ""
    echo "Optional environment variables:"
    echo "  N_FOLDS               default: 1 (1=single holdout, >1=K-fold ensemble)"
    echo "  SPLIT_SEEDS           default: 11,22,33,44,55"
    echo "  TRAIN_SEEDS           default: 101,202,303,404,505"
    echo "  SPLIT_STRATEGY        default: size-balanced"
    echo "  SPLIT_REPORT_JSON     default: unset (<save-path>/split_report.json)"
    echo "  THRESHOLD_POLICY      default: max-recall-min-precision"
    echo "  THRESHOLD_MIN_PRECISION default: 0.25"
    echo "  THRESHOLD_MIN_RECALL  default: 0.80"
    echo "  THRESHOLD_FIXED_VALUE default: 0.50"
    echo "  MODEL_HEAD            default: ffnn (ffnn or conv1d)"
    echo "  SEQ_LEN_FEATURE       default: none (none, raw, inverse)"
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

HIDDEN_SIZES="${HIDDEN_SIZES:-150,120,45}"
DROPOUTS="${DROPOUTS:-0.1,0.1,0.1}"
MODEL_HEAD="${MODEL_HEAD:-ffnn}"
SEQ_LEN_FEATURE="${SEQ_LEN_FEATURE:-none}"
CONV_CHANNELS="${CONV_CHANNELS:-64}"
CONV_LAYERS="${CONV_LAYERS:-2}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-9}"
CONV_DROPOUT="${CONV_DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-10}"
BEST_MODEL_METRIC="${BEST_MODEL_METRIC:-pr_auc}"
THRESHOLD_POLICY="${THRESHOLD_POLICY:-max-recall-min-precision}"
THRESHOLD_MIN_PRECISION="${THRESHOLD_MIN_PRECISION:-0.25}"
THRESHOLD_MIN_RECALL="${THRESHOLD_MIN_RECALL:-0.80}"
THRESHOLD_FIXED_VALUE="${THRESHOLD_FIXED_VALUE:-0.50}"
SPLIT_SEEDS="${SPLIT_SEEDS:-11,22,33,44,55}"
TRAIN_SEEDS="${TRAIN_SEEDS:-101,202,303,404,505}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-size-balanced}"
SPLIT_REPORT_JSON="${SPLIT_REPORT_JSON:-}"
N_FOLDS="${N_FOLDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}" # ensure batch size is 4 times what you would do for one GPU (for example. 256 = 64 * 4)
LR="${LR:-0.001}"
WD="${WD:-0.0}"
VAL_FRAC="${VAL_FRAC:-0.2}"
POS_WEIGHT="${POS_WEIGHT:-}" # optional manual override; empty uses train-only auto-compute
SAVE_PATH="/scratch/$USER/models/${SLURM_JOB_NAME:-train_job}"
RESULTS_CSV="${SAVE_PATH}/runs.csv"
ENSEMBLE_MANIFEST="${ENSEMBLE_MANIFEST:-${SAVE_PATH}/ensemble_manifest.json}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-1000}"
STRIDE="${STRIDE:-900}"
SPLIT_TYPE="${SPLIT_TYPE:-id-family}" # id-family or id
LABEL_CACHE_MODE="${LABEL_CACHE_MODE:-current}" # current or all
SAVE_VAL_CURVES="${SAVE_VAL_CURVES:-0}" # 1 to enable validation ROC/PR artifacts
VAL_CURVE_MAX_POINTS="${VAL_CURVE_MAX_POINTS:-2048}"
VAL_PLOT_FORMATS="${VAL_PLOT_FORMATS:-png}"

mkdir -p "${SAVE_PATH}"

# load Python Conda environment
module purge
module load anaconda3
module load cuda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pepseqpred

# turn off srun for testing
if [ "${USE_SRUN}" -eq 1 ]; then
    LAUNCHER="srun"
else
    LAUNCHER=""
fi

TRAIN_ARGS=(
    --n-folds "$N_FOLDS"
    --split-seeds "$SPLIT_SEEDS"
    --train-seeds "$TRAIN_SEEDS"
)
if [ "$N_FOLDS" -gt 1 ]; then
    TRAIN_ARGS+=(--ensemble-manifest "$ENSEMBLE_MANIFEST")
fi

VAL_CURVE_ARGS=()
if [ "${SAVE_VAL_CURVES}" -eq 1 ]; then
    VAL_CURVE_ARGS+=(--save-val-curves)
    VAL_CURVE_ARGS+=(--val-curve-max-points "$VAL_CURVE_MAX_POINTS")
    VAL_CURVE_ARGS+=(--val-plot-formats "$VAL_PLOT_FORMATS")
fi

POS_WEIGHT_ARGS=()
if [ -n "$POS_WEIGHT" ]; then
    POS_WEIGHT_ARGS+=(--pos-weight "$POS_WEIGHT")
fi

SPLIT_REPORT_ARGS=()
if [ -n "$SPLIT_REPORT_JSON" ]; then
    SPLIT_REPORT_ARGS+=(--split-report-json "$SPLIT_REPORT_JSON")
fi

${LAUNCHER} torchrun --nproc_per_node=4 train.pyz \
    --embedding-dirs "${EMBEDDING_DIRS[@]}" \
    --label-shards "${LABEL_SHARDS[@]}" \
    --label-cache-mode "$LABEL_CACHE_MODE" \
    --hidden-sizes "$HIDDEN_SIZES" \
    --dropouts "$DROPOUTS" \
    --model-head "$MODEL_HEAD" \
    --seq-len-feature "$SEQ_LEN_FEATURE" \
    --conv-channels "$CONV_CHANNELS" \
    --conv-layers "$CONV_LAYERS" \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-dropout "$CONV_DROPOUT" \
    --epochs "$EPOCHS" \
    "${TRAIN_ARGS[@]}" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --wd "$WD" \
    "${POS_WEIGHT_ARGS[@]}" \
    --best-model-metric "$BEST_MODEL_METRIC" \
    --threshold-policy "$THRESHOLD_POLICY" \
    --threshold-min-precision "$THRESHOLD_MIN_PRECISION" \
    --threshold-min-recall "$THRESHOLD_MIN_RECALL" \
    --threshold-fixed-value "$THRESHOLD_FIXED_VALUE" \
    --val-frac "$VAL_FRAC" \
    --split-type "$SPLIT_TYPE" \
    --split-strategy "$SPLIT_STRATEGY" \
    "${SPLIT_REPORT_ARGS[@]}" \
    --save-path "$SAVE_PATH" \
    --results-csv "$RESULTS_CSV" \
    --num-workers "$NUM_WORKERS" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    "${VAL_CURVE_ARGS[@]}"

# USAGE: sbatch train.sh /scratch/$USER/esm2/artifacts/pts/shard_000 /scratch/$USER/esm2/artifacts/pts/shard_001 /scratch/$USER/esm2/artifacts/pts/shard_002 /scratch/$USER/esm2/artifacts/pts/shard_003 -- /scratch/$USER/labels/labels_shard_000.pt /scratch/$USER/labels/labels_shard_001.pt /scratch/$USER/labels/labels_shard_002.pt /scratch/$USER/labels/labels_shard_003.pt
