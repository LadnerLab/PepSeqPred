"""prediction_cli.py

Predict residue-level binary epitope masks for proteins from FASTA input.

Usage
-----
>>> # HPC usage with shell script
>>> sbatch predictepitope.sh <checkpoint .pt path> <input .fasta path> <output .fasta path>

>>> # general example usage (illustrative)
>>> python prediction_cli.py <checkpoint .pt path> <input .fasta path> --output-fasta <output .fasta path> --model-name esm2_t33_650M_UR50D --max-tokens 1022 --log-dir logs --log-json
"""
import argparse
from pathlib import Path
from typing import Tuple, Iterator
import torch
import esm
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.read import parse_int_csv, parse_float_csv
from pepseqpred.core.embeddings.esm2 import clean_seq
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    infer_decision_threshold,
    predict_protein
)


def read_fasta_records(fasta_path: Path | str) -> Iterator[Tuple[str, str]]:
    """
    Yields (header, protein_sequence) from FASTA, header text is preserved.
    """
    header = None
    seq_lines = []
    with open(fasta_path, "r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, clean_seq("".join(seq_lines))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            yield header, clean_seq("".join(seq_lines))


def _build_cli_model_config(args: argparse.Namespace) -> FFNNModelConfig | None:
    """Builds the model configuration using CLI arguments."""
    any_explicit = any(
        arg is not None
        for arg in (
            args.emb_dim,
            args.hidden_sizes,
            args.dropouts,
            args.use_layer_norm,
            args.use_residual,
            args.num_classes
        )
    )
    if not any_explicit:
        return None

    missing = []
    if args.emb_dim is None:
        missing.append("--emb-dim")
    if args.hidden_sizes is None:
        missing.append("--hidden-sizes")
    if args.dropouts is None:
        missing.append("--dropouts")
    if args.use_layer_norm is None:
        missing.append("--use-layer-norm/--no-use-layer-norm")
    if args.use_residual is None:
        missing.append("--use-residual/--no-use-residual")
    if missing:
        raise ValueError(
            "When using explicit architecture flags, provide all required values: "
            + ", ".join(missing)
        )

    hidden_sizes = parse_int_csv(args.hidden_sizes, "--hidden-sizes")
    dropouts = parse_float_csv(args.dropouts, "--dropouts")
    if len(hidden_sizes) != len(dropouts):
        raise ValueError(
            "--hidden-sizes and --dropouts must be the same length"
        )
    num_classes = int(args.num_classes) if args.num_classes is not None else 1

    return FFNNModelConfig(
        emb_dim=int(args.emb_dim),
        hidden_sizes=tuple(hidden_sizes),
        dropouts=tuple(dropouts),
        num_classes=num_classes,
        use_layer_norm=bool(args.use_layer_norm),
        use_residual=bool(args.use_residual)
    )


def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Prediction program."""
    parser = argparse.ArgumentParser(
        description="Predict residue-level binary epitope masks for proteins in FASTA.")
    parser.add_argument("checkpoint",
                        type=Path,
                        help="Path to the trained model checkpoint (.pt).")
    parser.add_argument("fasta_input",
                        type=Path,
                        help="Input FASTA with taxonomy/protein headers and protein sequences.")
    parser.add_argument("--output-fasta",
                        action="store",
                        dest="output_fasta",
                        type=Path,
                        default=Path("predictions.fasta"),
                        help="Output FASTA path for binary epitope masks."
                        )
    parser.add_argument("--threshold",
                        action="store",
                        dest="threshold",
                        type=float,
                        default=None,
                        help="Optional threshold override within (0.0, 1.0).")
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
    parser.add_argument("--emb-dim",
                        action="store",
                        dest="emb_dim",
                        type=int,
                        default=None,
                        help="Explicit model embedding dimension used in training.")
    parser.add_argument("--hidden-sizes",
                        action="store",
                        dest="hidden_sizes",
                        type=str,
                        default=None,
                        help="Explicit hidden sizes CSV (e.g. 150,120,45).")
    parser.add_argument("--dropouts",
                        action="store",
                        dest="dropouts",
                        type=str,
                        default=None,
                        help="Explicit dropouts CSV (e.g. 0.1,0.1,0.1).")
    parser.add_argument("--num-classes",
                        action="store",
                        dest="num_classes",
                        type=int,
                        default=None,
                        help="Explicit output classes (binary=1).")
    parser.add_argument("--use-layer-norm",
                        action="store_true",
                        dest="use_layer_norm",
                        help="Explicit use layer normalization.")
    parser.add_argument("--use-residual",
                        action="store_true",
                        dest="use_residual",
                        help="Explicit use resdiuals.")
    parser.add_argument("--no-use-layer-norm",
                        action="store_false",
                        dest="use_layer_norm",
                        help="Explicit DO NOT use layer normalization.")
    parser.add_argument("--no-use-residual",
                        action="store_false",
                        dest="use_residual",
                        help="Explicit DO NOT use resdiuals.")
    parser.set_defaults(use_layer_norm=None, use_residual=None)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cli_model_cfg = _build_cli_model_config(args)
    psp_model, model_cfg, model_cfg_src = build_model_from_checkpoint(
        checkpoint,
        device=device,
        model_config=cli_model_cfg
    )
    if model_cfg.num_classes > 1:
        raise ValueError(
            f"Expected binary residue model with num_classes=1, got {model_cfg.num_classes}")

    # handle decision threshold determination
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else infer_decision_threshold(checkpoint)
    )
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("--threshold must be between (0.0, 1.0)")

    logger.info("prediction_init",
                extra={"extra": {
                    "checkpoint": str(args.checkpoint),
                    "device": device,
                    "model_cfg_src": model_cfg_src,
                    "emb_dim": model_cfg.emb_dim,
                    "hidden_sizes": model_cfg.hidden_sizes,
                    "use_layer_norm": model_cfg.use_layer_norm,
                    "use_residual": model_cfg.use_residual,
                    "num_classes": model_cfg.num_classes,
                    "threshold": threshold
                }})

    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    failed = 0
    total_residues = 0
    total_epitopes = 0

    with open(args.output_fasta, "w", encoding="utf-8") as out_f:
        for header, protein_seq in read_fasta_records(args.fasta_input):
            try:
                # try to run inference and write to output file
                pred = predict_protein(
                    psp_model=psp_model,
                    esm_model=esm_model,
                    layer=layer,
                    batch_converter=batch_converter,
                    protein_seq=protein_seq,
                    max_tokens=args.max_tokens,
                    device=device,
                    threshold=threshold
                )
                out_f.write(f">{header}\n{pred['binary_mask']}\n")

                # increment progress
                processed += 1
                total_residues += int(pred["length"])
                total_epitopes += int(pred["n_epitopes"])

                # log every 10 proteins processed
                if processed % 10 == 0:
                    logger.info("prediction_progress",
                                extra={"extra": {
                                    "processed": processed,
                                    "failed": failed
                                }})

            except Exception as e:
                failed += 1
                logger.error("prediction_failed",
                             extra={"extra": {
                                 "header": header,
                                 "error": str(e)
                             }})

    logger.info("prediction_done",
                extra={"extra": {
                    "output_fasta": str(args.output_fasta),
                    "processed": processed,
                    "failed": failed,
                    "total_residues": total_residues,
                    "total_epitopes": total_epitopes
                }})


if __name__ == "__main__":
    main()
