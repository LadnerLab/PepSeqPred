#!/bin/bash
#SBATCH --job-name=evaluate_ffnn
#SBATCH --output=/scratch/%u/evaluate_ffnn_slurm/output/%x_%j.out
#SBATCH --error=/scratch/%u/evaluate_ffnn_slurm/error/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# for testing
USE_SRUN="${USE_SRUN:-1}"

usage() {
    echo "Usage: $0 <model_artifact> [run_root]"
    echo "  model_artifact: Path to trained checkpoint .pt file OR ensemble manifest .json."
    echo "  run_root: Optional directory for all generated artifacts."
    echo ""
    echo "Core behavior:"
    echo "  1) Build a reduced evaluation subset from Cocci reactive/non-reactive peptide lists."
    echo "  2) Generate embeddings for only proteins referenced by selected peptides."
    echo "  3) Build residue label shard from the reduced metadata."
    echo "  4) Run prediction CLI to produce binary FASTA."
    echo "  5) Run evaluate_ffnn_cli for residue-level metrics."
    echo "  6) Run peptide-level 1-count comparison (predicted vs expected labels)."
    echo ""
    echo "Optional environment variables:"
    echo "  USE_SRUN             default: 1 (set 0 to run without srun)"
    echo "  DATA_DIR             default: localdata/FullCocciDataset/CocciData"
    echo "  EVAL_MODE            default: combined (reactive|nonreactive|combined)"
    echo "  RUN_ROOT             default: /scratch/\$USER/evals/\${SLURM_JOB_NAME}/\${EVAL_MODE}"
    echo "  SKIP_IF_EXISTS       default: 1 (skip stage when output sentinel already exists)"
    echo "  RUN_PREP             default: 1"
    echo "  RUN_EMBED            default: 1"
    echo "  RUN_LABELS           default: 1"
    echo "  RUN_PREDICT          default: 1"
    echo "  RUN_EVAL             default: 1"
    echo "  RUN_COMPARE          default: 1"
    echo ""
    echo "Embedding/prediction/eval knobs:"
    echo "  MODEL_NAME           default: esm2_t33_650M_UR50D"
    echo "  MAX_TOKENS           default: 1022"
    echo "  EMBED_BATCH_SIZE     default: 24"
    echo "  THRESHOLD            default: unset"
    echo "  ENSEMBLE_SET_INDEX   default: 1"
    echo "  K_FOLDS              default: unset"
    echo "  SELECT_BEST_SET_RUNS_CSV default: unset"
    echo "  BEST_SET_BY          default: PR_AUC"
    echo "  BEST_SET_AGG         default: mean"
    echo "  BEST_SET_DIRECTION   default: auto"
    echo "  SUBSET               default: 0"
    echo "  EVAL_BATCH_SIZE      default: 64"
    echo "  EVAL_NUM_WORKERS     default: 1"
    echo "  LABEL_CACHE_MODE     default: current"
    echo "  EMIT_FOLD_METRICS    default: 0 (1=true)"
    echo "  INCLUDE_CURVES       default: 0 (1=true)"
    echo "  CURVE_MAX_POINTS     default: 2048"
    echo "  BEST_FOLD_BY         default: pr_auc"
    echo "  BEST_FOLD_DIRECTION  default: auto"
    echo "  PLOT_DIR             default: unset"
    echo "  PLOT_FORMATS         default: png,svg"
    echo "  LOG_LEVEL            default: INFO"
    echo ""
    echo "Optional explicit model architecture flags (forwarded to predict/eval CLIs):"
    echo "  EMB_DIM              default: unset"
    echo "  HIDDEN_SIZES         default: unset"
    echo "  DROPOUTS             default: unset"
    echo "  NUM_CLASSES          default: unset"
    echo "  USE_LAYER_NORM       default: unset (true/false)"
    echo "  USE_RESIDUAL         default: unset (true/false)"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

MODEL_ARTIFACT="$1"

DATA_DIR="${DATA_DIR:-scratch/$USER/data/CWP}"
EVAL_MODE="${EVAL_MODE:-combined}"
RUN_ROOT_DEFAULT="/scratch/$USER/evals/${SLURM_JOB_NAME:-evaluate_ffnn}/${EVAL_MODE}"
RUN_ROOT="${2:-${RUN_ROOT:-$RUN_ROOT_DEFAULT}}"

SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
RUN_PREP="${RUN_PREP:-1}"
RUN_EMBED="${RUN_EMBED:-1}"
RUN_LABELS="${RUN_LABELS:-1}"
RUN_PREDICT="${RUN_PREDICT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

MODEL_NAME="${MODEL_NAME:-esm2_t33_650M_UR50D}"
MAX_TOKENS="${MAX_TOKENS:-1022}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-24}"
THRESHOLD="${THRESHOLD:-}"
ENSEMBLE_SET_INDEX="${ENSEMBLE_SET_INDEX:-1}"
K_FOLDS="${K_FOLDS:-}"
SELECT_BEST_SET_RUNS_CSV="${SELECT_BEST_SET_RUNS_CSV:-}"
BEST_SET_BY="${BEST_SET_BY:-PR_AUC}"
BEST_SET_AGG="${BEST_SET_AGG:-mean}"
BEST_SET_DIRECTION="${BEST_SET_DIRECTION:-max}"
SUBSET="${SUBSET:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-1}"
LABEL_CACHE_MODE="${LABEL_CACHE_MODE:-current}" # current or all
EMIT_FOLD_METRICS="${EMIT_FOLD_METRICS:-1}"
INCLUDE_CURVES="${INCLUDE_CURVES:-1}"
CURVE_MAX_POINTS="${CURVE_MAX_POINTS:-2048}"
BEST_FOLD_BY="${BEST_FOLD_BY:-pr_auc}"
BEST_FOLD_DIRECTION="${BEST_FOLD_DIRECTION:-auto}"
PLOT_DIR="${PLOT_DIR:-}"
PLOT_FORMATS="${PLOT_FORMATS:-png,svg}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# optional model config variables
EMB_DIM="${EMB_DIM:-}"
HIDDEN_SIZES="${HIDDEN_SIZES:-}"
DROPOUTS="${DROPOUTS:-}"
NUM_CLASSES="${NUM_CLASSES:-}"
USE_LAYER_NORM="${USE_LAYER_NORM:-}" # true/false
USE_RESIDUAL="${USE_RESIDUAL:-}" # true/false

case "${EVAL_MODE}" in
    reactive|nonreactive|combined) ;;
    *)
        echo "Invalid EVAL_MODE='${EVAL_MODE}'. Must be reactive|nonreactive|combined."
        exit 1
        ;;
esac

PREP_DIR="${RUN_ROOT}/prepared"
EMBED_DIR="${RUN_ROOT}/embeddings"
EMBED_PT_DIR="${EMBED_DIR}/artifacts/pts"
LABEL_DIR="${RUN_ROOT}/labels"
PREDICT_DIR="${RUN_ROOT}/prediction"
EVAL_DIR="${RUN_ROOT}/evaluation"
COMPARE_DIR="${RUN_ROOT}/peptide_compare"

PREP_META="${PREP_DIR}/eval_metadata.tsv"
PREP_FASTA="${PREP_DIR}/eval_proteins.fasta"
PREP_SUMMARY_JSON="${PREP_DIR}/prepare_summary.json"
EMBED_INDEX_CSV="${EMBED_DIR}/artifacts/eval_embedding_index.csv"
LABEL_SHARD="${LABEL_DIR}/labels_eval.pt"
PRED_FASTA="${PREDICT_DIR}/predictions.fasta"
EVAL_JSON="${EVAL_DIR}/ffnn_eval_summary.json"
COMPARE_CSV="${COMPARE_DIR}/peptide_comparison.csv"
COMPARE_JSON="${COMPARE_DIR}/peptide_comparison_summary.json"

mkdir -p "${PREP_DIR}" "${EMBED_DIR}" "${LABEL_DIR}" "${PREDICT_DIR}" "${EVAL_DIR}" "${COMPARE_DIR}"
mkdir -p "/scratch/$USER/evaluate_ffnn_slurm/output" "/scratch/$USER/evaluate_ffnn_slurm/error"

module purge
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pepseqpred

run_launcher() {
    if [ "${USE_SRUN}" -eq 1 ]; then
        srun "$@"
    else
        "$@"
    fi
}

echo "[evaluateffnn] model_artifact=${MODEL_ARTIFACT}"
echo "[evaluateffnn] data_dir=${DATA_DIR}"
echo "[evaluateffnn] eval_mode=${EVAL_MODE}"
echo "[evaluateffnn] run_root=${RUN_ROOT}"

if [ "${RUN_PREP}" -eq 1 ]; then
    if [ "${SKIP_IF_EXISTS}" -eq 1 ] && [ -s "${PREP_SUMMARY_JSON}" ]; then
        echo "[prepare] skip existing: ${PREP_SUMMARY_JSON}"
    else
        echo "[prepare] building reduced Cocci inputs"
        python -u cocci_eval_pipeline.py prepare \
            --data-dir "${DATA_DIR}" \
            --output-dir "${PREP_DIR}" \
            --mode "${EVAL_MODE}"
    fi
fi

if [ "${RUN_EMBED}" -eq 1 ]; then
    if [ "${SKIP_IF_EXISTS}" -eq 1 ] && [ -s "${EMBED_INDEX_CSV}" ]; then
        echo "[embed] skip existing: ${EMBED_INDEX_CSV}"
    else
        echo "[embed] generating embeddings for reduced protein set"
        run_launcher python -u esm.pyz \
            --fasta-file "${PREP_FASTA}" \
            --out-dir "${EMBED_DIR}" \
            --per-seq-dir "artifacts/pts" \
            --index-csv-path "${EMBED_INDEX_CSV}" \
            --embedding-key-mode "id" \
            --key-delimiter "-" \
            --model-name "${MODEL_NAME}" \
            --max-tokens "${MAX_TOKENS}" \
            --batch-size "${EMBED_BATCH_SIZE}" \
            --num-shards 1 \
            --shard-id 0 \
            --log-dir "logs" \
            --log-level "${LOG_LEVEL}" \
            --log-json
    fi
fi

if [ "${RUN_LABELS}" -eq 1 ]; then
    if [ "${SKIP_IF_EXISTS}" -eq 1 ] && [ -s "${LABEL_SHARD}" ]; then
        echo "[labels] skip existing: ${LABEL_SHARD}"
    else
        echo "[labels] building residue-level labels"
        run_launcher python -u labels.pyz "${PREP_META}" "${LABEL_SHARD}" \
            --emb-dir "${EMBED_PT_DIR}" \
            --restrict-to-embeddings \
            --calc-pos-weight
    fi
fi

PREDICT_ARGS=(
    --output-fasta "${PRED_FASTA}"
    --model-name "${MODEL_NAME}"
    --max-tokens "${MAX_TOKENS}"
    --log-dir "${PREDICT_DIR}/logs"
    --log-level "${LOG_LEVEL}"
    --log-json
)
[ -n "${THRESHOLD}" ] && PREDICT_ARGS+=(--threshold "${THRESHOLD}")
[ -n "${ENSEMBLE_SET_INDEX}" ] && PREDICT_ARGS+=(--ensemble-set-index "${ENSEMBLE_SET_INDEX}")
[ -n "${K_FOLDS}" ] && PREDICT_ARGS+=(--k-folds "${K_FOLDS}")
[ -n "${EMB_DIM}" ] && PREDICT_ARGS+=(--emb-dim "${EMB_DIM}")
[ -n "${HIDDEN_SIZES}" ] && PREDICT_ARGS+=(--hidden-sizes "${HIDDEN_SIZES}")
[ -n "${DROPOUTS}" ] && PREDICT_ARGS+=(--dropouts "${DROPOUTS}")
[ -n "${NUM_CLASSES}" ] && PREDICT_ARGS+=(--num-classes "${NUM_CLASSES}")
case "${USE_LAYER_NORM,,}" in
    true|1|yes|on) PREDICT_ARGS+=(--use-layer-norm) ;;
    false|0|no|off) PREDICT_ARGS+=(--no-use-layer-norm) ;;
esac
case "${USE_RESIDUAL,,}" in
    true|1|yes|on) PREDICT_ARGS+=(--use-residual) ;;
    false|0|no|off) PREDICT_ARGS+=(--no-use-residual) ;;
esac

if [ "${RUN_PREDICT}" -eq 1 ]; then
    if [ "${SKIP_IF_EXISTS}" -eq 1 ] && [ -s "${PRED_FASTA}" ]; then
        echo "[predict] skip existing: ${PRED_FASTA}"
    else
        echo "[predict] generating binary prediction FASTA"
        run_launcher python -u predict.pyz \
            "${MODEL_ARTIFACT}" \
            "${PREP_FASTA}" \
            "${PREDICT_ARGS[@]}"
    fi
fi

EVAL_ARGS=(
    --embedding-dirs "${EMBED_PT_DIR}"
    --label-shards "${LABEL_SHARD}"
    --output-json "${EVAL_JSON}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${EVAL_NUM_WORKERS}"
    --label-cache-mode "${LABEL_CACHE_MODE}"
    --log-dir "${EVAL_DIR}/logs"
    --log-level "${LOG_LEVEL}"
    --log-json
)
[ -n "${THRESHOLD}" ] && EVAL_ARGS+=(--threshold "${THRESHOLD}")
[ -n "${ENSEMBLE_SET_INDEX}" ] && EVAL_ARGS+=(--ensemble-set-index "${ENSEMBLE_SET_INDEX}")
[ -n "${K_FOLDS}" ] && EVAL_ARGS+=(--k-folds "${K_FOLDS}")
[ -n "${SELECT_BEST_SET_RUNS_CSV}" ] && EVAL_ARGS+=(--select-best-set-runs-csv "${SELECT_BEST_SET_RUNS_CSV}")
[ -n "${BEST_SET_BY}" ] && EVAL_ARGS+=(--best-set-by "${BEST_SET_BY}")
[ -n "${BEST_SET_AGG}" ] && EVAL_ARGS+=(--best-set-agg "${BEST_SET_AGG}")
[ -n "${BEST_SET_DIRECTION}" ] && EVAL_ARGS+=(--best-set-direction "${BEST_SET_DIRECTION}")
[ -n "${SUBSET}" ] && EVAL_ARGS+=(--subset "${SUBSET}")
[ -n "${CURVE_MAX_POINTS}" ] && EVAL_ARGS+=(--curve-max-points "${CURVE_MAX_POINTS}")
[ -n "${BEST_FOLD_BY}" ] && EVAL_ARGS+=(--best-fold-by "${BEST_FOLD_BY}")
[ -n "${BEST_FOLD_DIRECTION}" ] && EVAL_ARGS+=(--best-fold-direction "${BEST_FOLD_DIRECTION}")
[ -n "${PLOT_DIR}" ] && EVAL_ARGS+=(--plot-dir "${PLOT_DIR}")
[ -n "${PLOT_FORMATS}" ] && EVAL_ARGS+=(--plot-formats "${PLOT_FORMATS}")
[ -n "${EMB_DIM}" ] && EVAL_ARGS+=(--emb-dim "${EMB_DIM}")
[ -n "${HIDDEN_SIZES}" ] && EVAL_ARGS+=(--hidden-sizes "${HIDDEN_SIZES}")
[ -n "${DROPOUTS}" ] && EVAL_ARGS+=(--dropouts "${DROPOUTS}")
[ -n "${NUM_CLASSES}" ] && EVAL_ARGS+=(--num-classes "${NUM_CLASSES}")
case "${USE_LAYER_NORM,,}" in
    true|1|yes|on) EVAL_ARGS+=(--use-layer-norm) ;;
    false|0|no|off) EVAL_ARGS+=(--no-use-layer-norm) ;;
esac
case "${USE_RESIDUAL,,}" in
    true|1|yes|on) EVAL_ARGS+=(--use-residual) ;;
    false|0|no|off) EVAL_ARGS+=(--no-use-residual) ;;
esac
case "${EMIT_FOLD_METRICS,,}" in
    true|1|yes|on) EVAL_ARGS+=(--emit-fold-metrics) ;;
esac
case "${INCLUDE_CURVES,,}" in
    true|1|yes|on) EVAL_ARGS+=(--include-curves) ;;
esac

if [ "${RUN_EVAL}" -eq 1 ]; then
    if [ "${SKIP_IF_EXISTS}" -eq 1 ] && [ -s "${EVAL_JSON}" ]; then
        echo "[eval] skip existing: ${EVAL_JSON}"
    else
        echo "[eval] running residue-level evaluation CLI"
        if [ -f "evaluate_ffnn.pyz" ]; then
            run_launcher python -u evaluate_ffnn.pyz "${MODEL_ARTIFACT}" "${EVAL_ARGS[@]}"
        elif [ -f "eval_ffnn.pyz" ]; then
            run_launcher python -u eval_ffnn.pyz "${MODEL_ARTIFACT}" "${EVAL_ARGS[@]}"
        else
            run_launcher python -u -m pepseqpred.apps.evaluate_ffnn_cli "${MODEL_ARTIFACT}" "${EVAL_ARGS[@]}"
        fi
    fi
fi

if [ "${RUN_COMPARE}" -eq 1 ]; then
    if [ "${SKIP_IF_EXISTS}" -eq 1 ] && [ -s "${COMPARE_JSON}" ]; then
        echo "[compare] skip existing: ${COMPARE_JSON}"
    else
        echo "[compare] running peptide-level predicted-ones comparison"
        python -u cocci_eval_pipeline.py compare \
            --prediction-fasta "${PRED_FASTA}" \
            --metadata-tsv "${PREP_META}" \
            --label-shard "${LABEL_SHARD}" \
            --output-csv "${COMPARE_CSV}" \
            --output-json "${COMPARE_JSON}"
    fi
fi

echo "[evaluateffnn] done"
echo "  prepare_summary: ${PREP_SUMMARY_JSON}"
echo "  embeddings_index: ${EMBED_INDEX_CSV}"
echo "  labels: ${LABEL_SHARD}"
echo "  predictions_fasta: ${PRED_FASTA}"
echo "  eval_json: ${EVAL_JSON}"
echo "  peptide_compare_csv: ${COMPARE_CSV}"
echo "  peptide_compare_json: ${COMPARE_JSON}"

