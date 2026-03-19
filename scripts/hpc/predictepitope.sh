#!/bin/bash
#SBATCH --job-name=predict_epitope
#SBATCH --output=/scratch/%u/predict_epitope_slurm/output/%x_%j.out
#SBATCH --error=/scratch/%u/predict_epitope_slurm/error/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# for testing
USE_SRUN="${USE_SRUN:-1}"

usage() {
    echo "Usage: $0 <model_artifact> <fasta_input> <output_fasta>"
    echo "  model_artifact: Path to trained checkpoint .pt file OR ensemble manifest .json."
    echo "  fasta_input: Input FASTA file."
    echo "  output_fasta: Output FASTA file for predicted binary masks."
    echo ""
    echo "Optional environment variables:"
    echo "  USE_SRUN           default: 1 (set 0 to run without srun)"
    echo "  MODEL_NAME         default: esm2_t33_650M_UR50D"
    echo "  MAX_TOKENS         default: 1022"
    echo "  THRESHOLD          default: unset (use checkpoint threshold)"
    echo "  ENSEMBLE_SET_INDEX default: 1 (schema v2 manifest only)"
    echo "  K_FOLDS            default: unset (use all valid members)"
    echo "  LOG_DIR            default: logs"
    echo "  LOG_LEVEL          default: INFO"
    echo "  PREDICT_EXTRA_ARGS default: unset (raw extra prediction CLI args)"
    echo ""
    echo "Optional model-config environment variables:"
    echo "  EMB_DIM            default: unset"
    echo "  HIDDEN_SIZES       default: unset (CSV like 150,120,45)"
    echo "  DROPOUTS           default: unset (CSV like 0.1,0.1,0.1)"
    echo "  NUM_CLASSES        default: unset"
    echo "  USE_LAYER_NORM     default: unset (true/false, 1/0, yes/no, on/off)"
    echo "  USE_RESIDUAL       default: unset (true/false, 1/0, yes/no, on/off)"
    echo ""
    echo "Notes:"
    echo "  --log-json is always enabled by this script."
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

MODEL_ARTIFACT="$1"
FASTA_INPUT="$2"
OUTPUT_FASTA="$3"

MODEL_NAME="${MODEL_NAME:-esm2_t33_650M_UR50D}"
MAX_TOKENS="${MAX_TOKENS:-1022}"
THRESHOLD="${THRESHOLD:-}"
ENSEMBLE_SET_INDEX="${ENSEMBLE_SET_INDEX:-1}"
K_FOLDS="${K_FOLDS:-}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# optional model config variables
EMB_DIM="${EMB_DIM:-}"
HIDDEN_SIZES="${HIDDEN_SIZES:-}"
DROPOUTS="${DROPOUTS:-}"
NUM_CLASSES="${NUM_CLASSES:-}"
USE_LAYER_NORM="${USE_LAYER_NORM:-}"    # true/false
USE_RESIDUAL="${USE_RESIDUAL:-}"        # true/false

CLI_ARGS=(
    --output-fasta "${OUTPUT_FASTA}"
    --model-name "${MODEL_NAME}"
    --max-tokens "${MAX_TOKENS}"
    --log-dir "${LOG_DIR}"
    --log-level "${LOG_LEVEL}"
    --log-json
)

# optional decision threshold passes
[ -n "${THRESHOLD}" ] && CLI_ARGS+=(--threshold "${THRESHOLD}")
[ -n "${ENSEMBLE_SET_INDEX}" ] && CLI_ARGS+=(--ensemble-set-index "${ENSEMBLE_SET_INDEX}")
[ -n "${K_FOLDS}" ] && CLI_ARGS+=(--k-folds "${K_FOLDS}")

# optional environment-driven flags
[ -n "${EMB_DIM}" ] && CLI_ARGS+=(--emb-dim "${EMB_DIM}")
[ -n "${HIDDEN_SIZES}" ] && CLI_ARGS+=(--hidden-sizes "${HIDDEN_SIZES}")
[ -n "${DROPOUTS}" ] && CLI_ARGS+=(--dropouts "${DROPOUTS}")
[ -n "${NUM_CLASSES}" ] && CLI_ARGS+=(--num-classes "${NUM_CLASSES}")

case "${USE_LAYER_NORM,,}" in
    true|1|yes|on) CLI_ARGS+=(--use-layer-norm) ;;
    false|0|no|off) CLI_ARGS+=(--no-use-layer-norm) ;;
esac
case "${USE_RESIDUAL,,}" in
    true|1|yes|on) CLI_ARGS+=(--use-residual) ;;
    false|0|no|off) CLI_ARGS+=(--no-use-residual) ;;
esac

module purge
module load anaconda3
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pepseqpred

if [ "${USE_SRUN}" -eq 1 ]; then
    LAUNCHER="srun"
else
    LAUNCHER=""
fi

${LAUNCHER} python -u predict.pyz \
    "${MODEL_ARTIFACT}" \
    "${FASTA_INPUT}" \
    "${CLI_ARGS[@]}"
