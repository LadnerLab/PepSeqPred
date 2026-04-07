#!/bin/bash
usage() {
    echo "Usage: $0 <flagship1_manifest> <flagship2_manifest>"
    echo ""
    echo "Submits seeded Cocci evaluation jobs for two flagship ensemble manifests."
    echo "Each set uses shared prepared/embedding/label artifacts and runs:"
    echo "  predict + eval + peptide compare"
    echo ""
    echo "Required positional args:"
    echo "  flagship1_manifest   Path to flagship1 ensemble_manifest.json"
    echo "  flagship2_manifest   Path to flagship2 ensemble_manifest.json"
    echo ""
    echo "Optional environment variables:"
    echo "  HPC_DIR      default: /home/\$USER/test"
    echo "  SHARED       default: /scratch/\$USER/evals/cocci_eval/combined"
    echo "  OUT_BASE     default: /scratch/\$USER/evals/cocci_eval/seeded_runs"
    echo "  DATA_DIR     default: /scratch/\$USER/data/CWP"
    echo "  SET_START    default: 1"
    echo "  SET_END      default: 10"
    echo "  DRY_RUN      default: 0 (1 = print sbatch commands only)"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

MODEL1="$1"
MODEL2="$2"

HPC_DIR="${HPC_DIR:-/home/$USER/test}"
SHARED="${SHARED:-/scratch/$USER/evals/cocci_eval/combined}"
OUT_BASE="${OUT_BASE:-/scratch/$USER/evals/cocci_eval/seeded_runs}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/data/CWP}"
SET_START="${SET_START:-1}"
SET_END="${SET_END:-10}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${MODEL1}" ]]; then
    echo "Missing flagship1 manifest: ${MODEL1}"
    exit 1
fi
if [[ ! -f "${MODEL2}" ]]; then
    echo "Missing flagship2 manifest: ${MODEL2}"
    exit 1
fi
if [[ ! -d "${SHARED}" ]]; then
    echo "Missing shared run root: ${SHARED}"
    exit 1
fi
if [[ ! -f "${HPC_DIR}/evaluateffnn.sh" ]]; then
    echo "Missing ${HPC_DIR}/evaluateffnn.sh"
    exit 1
fi

submit_one() {
    local model_name="$1"
    local artifact="$2"
    local set_index="$3"

    local set_pad
    set_pad="$(printf '%02d' "${set_index}")"
    local run_root="${OUT_BASE}/${model_name}/set_${set_pad}/combined"

    mkdir -p "${run_root}/prepared" "${run_root}/embeddings/artifacts" "${run_root}/labels"

    # Reuse existing evaluation inputs from the shared combined run.
    ln -sfn "${SHARED}/prepared/eval_metadata.tsv" "${run_root}/prepared/eval_metadata.tsv"
    ln -sfn "${SHARED}/prepared/eval_proteins.fasta" "${run_root}/prepared/eval_proteins.fasta"
    ln -sfn "${SHARED}/embeddings/artifacts/pts" "${run_root}/embeddings/artifacts/pts"
    ln -sfn "${SHARED}/labels/labels_eval.pt" "${run_root}/labels/labels_eval.pt"

    local exports="ALL,DATA_DIR=${DATA_DIR},EVAL_MODE=combined,ENSEMBLE_SET_INDEX=${set_index},EXPECTED_SET_INDEX=${set_index},SKIP_IF_EXISTS=0,RUN_PREP=0,RUN_EMBED=0,RUN_LABELS=0,RUN_PREDICT=1,RUN_EVAL=1,RUN_COMPARE=1,EMIT_FOLD_METRICS=1,INCLUDE_CURVES=1,PLOT_DIR=${run_root}/evaluation/plots"

    local cmd=(
        sbatch
        --chdir="${HPC_DIR}"
        --job-name="ev_${model_name}_s${set_pad}"
        --export="${exports}"
        evaluateffnn.sh
        "${artifact}"
        "${run_root}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run] %q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
}

for s in $(seq "${SET_START}" "${SET_END}"); do
    submit_one "flagship1" "${MODEL1}" "${s}"
    submit_one "flagship2" "${MODEL2}" "${s}"
done

echo "[done] submitted sets ${SET_START}-${SET_END} for flagship1 and flagship2"
