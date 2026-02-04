"""labels_cli.py


Spreads peptide-level labes from preprocessing step across each residue to protein sequence residues. 
Generates an output PT file for downstream model training. Residues are labeled definite epitope, uncertain, 
or not epitope. Because ESM-2 embeddings are generated sharded, the labels PT will correspond to each ESM-2
embedding shard generated.

Usage
-----
>>> # from scripts/hpc/generatelabels.sh
>>> sbatch generatelabels.sh <metadata_tsv> /path/to/out_dir /path/to/root_emb_dir
"""
import time
import argparse
from pathlib import Path
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.labels.builder import ProteinLabelBuilder


def main() -> None:
    """Parses label generation arguments and runs label builder program."""
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Build dense residue labels per protein from metadata.")
    parser.add_argument("meta_path",
                        type=Path,
                        help="Path to metadata TSV file.")
    parser.add_argument("save_path",
                        type=Path,
                        help="Path to output labels .pt file.")
    parser.add_argument("--emb-dir",
                        action="append",
                        dest="emb_dirs",
                        type=Path,
                        required=True,
                        help="Embedding directory. Repeat to add multiple shard roots.")
    parser.add_argument("--restrict-to-embeddings",
                        action="store_true",
                        dest="restrict_to_embeddings",
                        default=False,
                        help="Only build labels for proteins with embeddings in emb dirs.")
    args = parser.parse_args()

    logger = setup_logger(json_lines=True, json_indent=2, name="labels_cli")
    logger.info("run_start", extra={
                "extra": {"save_path": str(args.save_path)}})

    builder = ProteinLabelBuilder(meta_path=args.meta_path,
                                  emb_dirs=args.emb_dirs,
                                  logger=logger,
                                  restrict_to_embeddings=args.restrict_to_embeddings)
    builder.build(save_path=args.save_path)

    logger.info("run_done", extra={"extra": {
        "total_duration_s": round(time.perf_counter() - t0, 3)
    }})


if __name__ == "__main__":
    main()
