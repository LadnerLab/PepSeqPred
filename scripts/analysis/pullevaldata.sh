#!/bin/bash
usage() {
    echo "Usage:"
    echo "  $0 <remote_user>@<remote_host> [local_out_root]"
    echo ""
    echo "Pull seeded evaluation directories via SCP from HPC."
    echo ""
    echo "Defaults:"
    echo "  remote_user    \$USER"
    echo "  local_out_root localdata/evals/cocci_eval/seeded_runs"
    echo ""
    echo "Remote source layout:"
    echo "  /scratch/<remote_user>/evals/cocci_eval/seeded_runs/<model>/set_XX/combined/evaluation"
    echo ""
    echo "Models copied: flagship1, flagship2"
    echo "Sets copied: set_01 ... set_10"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

# accept user@host [out_root]
if [[ "$1" == *"@"* ]]; then
    REMOTE_USER="${1%@*}"
    REMOTE_HOST="${1#*@}"
    LOCAL_OUT_ROOT="${2:-localdata/evals/cocci_eval/seeded_runs}"
fi

REMOTE_BASE="/scratch/${REMOTE_USER}/evals/cocci_eval/seeded_runs"
MODELS=("flagship1" "flagship2")

mkdir -p "${LOCAL_OUT_ROOT}"

copied=0
failed=0

for model in "${MODELS[@]}"; do
    for set_idx in $(seq -w 1 10); do
        remote_eval_dir="${REMOTE_BASE}/${model}/set_${set_idx}/combined/evaluation"
        local_parent="${LOCAL_OUT_ROOT}/${model}/set_${set_idx}/combined"
        mkdir -p "${local_parent}"

        echo "[scp] ${REMOTE_USER}@${REMOTE_HOST}:${remote_eval_dir} -> ${local_parent}/"
        if scp -r "${REMOTE_USER}@${REMOTE_HOST}:${remote_eval_dir}" "${local_parent}/"; then
            copied=$((copied + 1))
        else
            echo "[error] scp failed: ${REMOTE_USER}@${REMOTE_HOST}:${remote_eval_dir}"
            failed=$((failed + 1))
        fi
    done
done

echo "[done] local_out_root=${LOCAL_OUT_ROOT}"
echo "[done] copied=${copied} failed=${failed}"

if [[ ${failed} -gt 0 ]]; then
    exit 1
fi
