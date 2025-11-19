# if error, fail loudly
set -euo pipefail

# handle required command-line arguments
usage() {
    echo "Usage: $0 <meta_file> <fasta_file> <z_file>"
    echo "  meta_file: Path to metadata file."
    echo "  fasta_file: Path to targets fasta file."
    echo "  z_file: Path to z-score reactivity file."
}

if [ "$#" -ne 3 ]; then 
    usage
fi

META_FILE="$1"
FASTA_FILE="$2"
Z_FILE="$3"

# simple script defaults
SCRIPT="../preprocess_cli.py"
ENV_DIR="../../venv"
IS_EPI_MIN_Z="${IS_EPI_MAX_Z:-20.0}"
IS_EPI_MIN_SUBS="${IS_EPI_MIN_SUBS:-4}"
NOT_EPI_MAX_Z="${NOT_EPI_MAX_Z:-10.0}"
NOT_EPI_MAX_SUBS="${NOT_EPI_MAX_SUBS:-0}" # 0 means use all subjects (0 is converted to None in script)
PREFIX="${PREFIX:-VW_}"

# environment config
echo "[setup] Activating virtual environment..."
source "$ENV_DIR/Scripts/activate"

# run script
python3 -u "$SCRIPT" \
    "$META_FILE" \
    "$FASTA_FILE" \
    "$Z_FILE" \
    --is-epi-z-thresh "$IS_EPI_MIN_Z" \
    --is-epi-min-subs "$IS_EPI_MIN_SUBS" \
    --not-epi-z-thresh "$NOT_EPI_MAX_Z" \
    --not-epi-max-subs "$NOT_EPI_MAX_SUBS" \
    --subject-prefix "$PREFIX" \
    --save-path

# COMMAND: ./preprocessdata.sh ../../data/PV1_meta_2020-11-23.tsv ../../data/fulldesign_2019-02-27_wGBKsw.fasta ../../data/SHERC_combined_wSB_4-24-24_Z-HDI95_avg_round.tsv
