"""prediction_cli.py

Predicts epitope class probabilities for peptides using a trained PepSeqFFNN and ESM-2 protein embeddings. 
The script reads a FASTA file where each record contains a peptide in the header and its parent protein sequence 
in the body, then outputs per peptide prediction probabilities to CSV.

FASTA inputs must follow this pattern (example):

>FKEELDKYFKNHTSPDVDLGDISGINASVV
MKVLIFALLFSLAKAQEGCGIISRKPQPKMEKVSSSRRGVYYNDDIFRSDVLHLTQD...

Where the header line contains the peptide sequence and the body containts the full protein sequence. The protein sequence
should not be broken up into multiple lines.

Usage
-----
>>> # HPC usage with shell script
>>> sbatch ./predictepitope.sh <checkpoint .pt path> <input .fasta path> --output-csv <output .csv path>

>>> # general example usage (illustrative)
>>> python prediction_cli.py <checkpoint .pt path> <input .fasta path> --output-csv <output .csv path> \ 
    --model-name esm2_t33_650M_UR50D --max-tokens 1022 --log-dir logs --log-json
"""
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Iterator, List
import torch
import esm
import pandas as pd
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.embeddings.esm2 import clean_seq
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.predict.inference import infer_emb_dim_from_state, predict

CLASS_NAMES = ["Def epitope", "Uncertain", "Not epitope"]


def read_fasta_pep_prot(fasta_path: Path | str) -> Iterator[Tuple[str, str]]:
    """
    Reads the peptides and protein sequences from a FASTA file.

    Expected pattern (example):
    >FKEELDKYFKNHTSPDVDLGDISGINASVV
    MKVLIFALLFSLAKAQEGCGIISRKPQPKMEKVSSSRRGVYYNDDIFRSDVLHLTQD...

    Parameters
    ----------
        fasta_path : Path or str
            Path to the input FASTA file.

    Yields
    ------
        Tuple[str, str]
            Yields tuples of (peptide, protein_sequence) strings for use in a loop.
    """
    peptide = None
    seq_lines = []
    with open(fasta_path, "r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):
                if peptide:
                    protein_seq = clean_seq("".join(seq_lines))
                    pep = clean_seq(peptide)
                    yield pep, protein_seq

                peptide = line[1:].strip()
                seq_lines = []

            else:
                seq_lines.append(line)

        if peptide:
            protein_seq = clean_seq("".join(seq_lines))
            pep = clean_seq(peptide)
            yield pep, protein_seq


def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Prediction program."""
    parser = argparse.ArgumentParser(
        description="Predict epitope class for a peptide using a trained PepSeqPred model.")
    parser.add_argument("checkpoint",
                        type=Path,
                        help="Path to the trained model checkpoint (.pt).")
    parser.add_argument("fasta_input",
                        type=Path,
                        help="FASTA file where header is a peptide and body is a protein sequence.")
    parser.add_argument("--output-csv",
                        action="store",
                        dest="output_csv",
                        type=Path,
                        default=Path("predictions.csv"),
                        help="Output CSV path for predictions"
                        )
    parser.add_argument("--model-name",
                        action="store",
                        dest="model_name",
                        type=str,
                        default="esm2_t33_650M_UR50D",
                        help="ESM model name to generate embeddings.")
    parser.add_argument("--max-tokens",
                        action="store",
                        dest="max_tokens",
                        type=int,
                        default=1022,
                        help="ESM residue token budget excluding CLS and EOS.")
    parser.add_argument("--log-dir",
                        action="store",
                        dest="log_dir",
                        type=Path,
                        default=Path("logs"),
                        help="Directory for logs.")
    parser.add_argument("--log-level",
                        action="store",
                        dest="log_level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level choice.")
    parser.add_argument("--log-json",
                        action="store_true",
                        dest="log_json",
                        default=False,
                        help="Emit logs as JSON lines for simple parsing.")
    args = parser.parse_args()

    json_indent = 2 if args.log_json else None
    logger = setup_logger(log_dir=args.log_dir,
                          log_level=args.log_level,
                          json_lines=args.log_json,
                          json_indent=json_indent,
                          name="prediction_cli")

    # load ESM embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model, alphabet = getattr(esm.pretrained, args.model_name)()
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    # set model layer to use (33 for default model)
    layer = esm_model.num_layers

    # load model from disk
    checkpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpt, dict) or "model_state_dict" not in checkpt:
        raise ValueError(
            "Model checkpoint format unrecognized, expected dictionary with model_state_dict key")
    state = checkpt["model_state_dict"]

    # build model loaded from disk
    emb_dim = infer_emb_dim_from_state(state)
    psp_model = PepSeqFFNN(emb_dim=emb_dim, hidden_sizes=(
        512, 256, 128), num_classes=len(CLASS_NAMES))
    psp_model.load_state_dict(state)
    psp_model.eval().to(device)

    logger.info("prediction_init",
                extra={"extra": {
                    "checkpoint": str(args.checkpoint),
                    "emb_dim": emb_dim,
                    "device": device
                }})

    # handle fasta file input
    rows: List[Dict[str, Any]] = []
    n = 0
    for pep, prot in read_fasta_pep_prot(args.fasta_input):
        try:
            row = predict(psp_model, esm_model, layer,
                          batch_converter, prot, pep, args.max_tokens, device)
            rows.append(row)
            n += 1

            # log prediction progress every 10 peptides
            if n % 10 == 0:
                logger.info("prediction_progress",
                            extra={"extra": {
                                "processed": n
                            }})

        except Exception as e:
            logger.error("prediction_failed",
                         extra={"extra": {
                             "peptide": pep,
                             "error": str(e)
                         }})

    # store as DataFrame for easy CSV file creation
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    logger.info("prediction_done",
                extra={"extra": {
                    "output_csv": str(args.output_csv),
                    "processed": len(df),
                    "columns": list(df.columns)
                }})


if __name__ == "__main__":
    main()
