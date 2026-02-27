"""esm_cli.py

This module is designed to read a FASTA file with headers in this form (example):
    >ID=A8D0M1_ADE02 AC=A8D0M1 OXX=10515,129951,10509,10508
    MALTCRLRFPVPGFRGRMHRRRGMAGHGLTGGMRRAHHRRRRASHRRMRGGILPLLIPLIA
cleans and batches the target sequence below each header, runs an ESM-2 model to obtain **per-residue** 
embeddings, and writes to individual PT files containing each residue-level embedding per protein sequence, 
plus a CSV metadata file. The script also writes structured logs to stdout to help monitor progress and debug 
any errors.

For sequences that fit within the model token limit (residues + 2 <= 1024), embeddings are computed in 
a single pass with full-sequence context. For longer sequences, overlapping windows are used and stitched 
by averaging overlaps to produce a full-length (L, D+1) matrix.  

This script can be ran on its own, but it is intended for HPC use (e.g., Monsoon). 
See shell script `generateembeddings.sh` or `generateembeddings_cpu.sh` for typical usage.

Usage
-----
>>> # from scripts/hpc/generateembeddings.sh (see shell script for CLI config)
>>> sbatch --export=ALL,IN_FASTA=/scratch/<NAUIDD>/<targets>.fasta generateembeddings.sh
"""
import os
import argparse
from pathlib import Path
from datetime import datetime
import esm
import torch
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.read import read_fasta
from pepseqpred.core.io.keys import (
    build_id_to_family_from_metadata,
    normalize_family_value
)
from pepseqpred.core.embeddings.esm2 import esm_embeddings_from_fasta


def main() -> None:
    """
    Parse CLI arguments, set up logging, run embedding generation script, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Make ESM-2 embeddings from PV1 targets fasta file.")
    parser.add_argument("--log-dir",
                        action="store",
                        dest="log_dir",
                        type=Path,
                        default=Path("logs"),
                        help="Directory for log files.")
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
    parser.add_argument("--per-seq-dir",
                        action="store",
                        dest="per_seq_dir",
                        type=Path,
                        default=Path("artifacts/pts"),
                        help="Directory for individual .pt files when save mode is set to 'pt'.")
    parser.add_argument("--index-csv-path",
                        action="store",
                        dest="idx_csv_path",
                        type=Path,
                        default=Path(
                            f"artifacts/esm_seq_idx_{datetime.now().strftime('%Y-%m-%d_T%H_%M_%S')}.csv"),
                        help="File path to CSV file containing indices for ESM embeddings.")
    parser.add_argument("--out-dir",
                        action="store",
                        dest="out_dir",
                        type=Path,
                        required=True,
                        help="Output directory for all generated files.")
    parser.add_argument("-f", "--fasta-file",
                        action="store",
                        dest="fasta_file",
                        type=Path,
                        required=True,
                        help="Fasta file containing antigens")
    parser.add_argument("--metadata-file",
                        action="store",
                        dest="metadata_file",
                        type=Path,
                        default=None,
                        help="Metadata for naming, if omitted, tries <fasta>.metadata")
    parser.add_argument("--metadata-name-column",
                        action="store",
                        dest="metadata_name_col",
                        type=str,
                        default="Name",
                        help="Metadata column containing full FASTA header strings.")
    parser.add_argument("--metadata-family-column",
                        action="store",
                        dest="metadata_family_col",
                        type=str,
                        default="Family",
                        help="Metadata column containing family values.")
    parser.add_argument("-i", "--id-column",
                        action="store",
                        dest="id_col",
                        type=str,
                        default="ID",
                        help="Name of ID column.")
    parser.add_argument("-s", "--seq-column",
                        action="store",
                        dest="seq_col",
                        type=str,
                        default="Sequence",
                        help="Name of sequence column.")
    parser.add_argument("--embedding-key-mode",
                        action="store",
                        dest="embedding_key_mode",
                        type=str,
                        default="id-family",
                        choices=["id", "id-family"],
                        help="Filename key mode: use ID only, or ID plus viral family.")
    parser.add_argument("--key-delimiter",
                        action="store",
                        dest="key_delimiter",
                        type=str,
                        default="-",
                        help="Delimiter to join ID and viral family when key mode is 'id-family'.")
    parser.add_argument("-m", "--model-name",
                        action="store",
                        dest="model_name",
                        type=str,
                        default="esm2_t33_650M_UR50D",
                        help="ESM model name to use to generate target embeddings.")
    parser.add_argument("-t", "--max-tokens",
                        action="store",
                        dest="max_tokens",
                        type=int,
                        default=1022,
                        help="Size of context window for ESM model.")
    parser.add_argument("-b", "--batch-size",
                        action="store",
                        dest="batch_size",
                        type=int,
                        default=8,
                        help="Batch size for processing targets into ESM embeddings.")
    parser.add_argument("--num-shards",
                        action="store",
                        dest="num_shards",
                        type=int,
                        default=1,
                        help="Number of shards to split input fasta into for parallel processing.")
    parser.add_argument("--shard-id",
                        action="store",
                        dest="shard_id",
                        type=int,
                        default=0,
                        help="0-based index of this shard in [0, num_shards).")

    args = parser.parse_args()
    out_dir = args.out_dir
    json_indent = 2 if args.log_json else None
    logger = setup_logger(log_dir=(out_dir / args.log_dir),
                          log_level=args.log_level,
                          json_lines=args.log_json,
                          json_indent=json_indent,
                          name="esm_cli")

    # initial log
    logger.info("cuda_status",
                extra={"extra": {
                    "SLURM_JOB_ID": os.getenv("SLURM_JOB_ID"),
                    "SLURM_JOB_NAME": os.getenv("SLURM_JOB_NAME"),
                    "cuda_is_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                    "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")
                }})

    metadata_file = None
    if args.embedding_key_mode == "id-family":
        metadata_file = args.metadata_file
        if metadata_file is None:
            inferred_metadata = args.fasta_file.with_suffix(".metadata")
            if inferred_metadata.exists():
                metadata_file = inferred_metadata
        if metadata_file is None:
            raise ValueError(
                "Metadata file is required for --embedding-key-mode='id-family'"
                "Pass --metadata-file or provide '<fasta>.metadata'"
            )

    # read fasta and attach metadata-based family naming
    fasta_df = read_fasta(args.fasta_file)
    if metadata_file is not None and args.embedding_key_mode == "id-family":
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}"
            )
        id_to_family, duplicate_same_family = build_id_to_family_from_metadata(
            metadata_file,
            name_col=args.metadata_name_col,
            family_col=args.metadata_family_col
        )

        fasta_df[args.id_col] = fasta_df[args.id_col].astype(str)
        mapped_family = fasta_df[args.id_col].map(id_to_family)
        missing_id_mask = mapped_family.isna()
        if bool(missing_id_mask.any()):
            missing_examples = (
                fasta_df.loc[missing_id_mask, [args.id_col]][args.id_col]
                .drop_duplicates()
                .head(10)
                .tolist()
            )
            raise ValueError(
                "Metadata naming map did not contain all FASTA IDs, "
                f"missing examples: {missing_examples}"
            )

        fasta_df["viral_family"] = mapped_family.map(normalize_family_value)
        if args.embedding_key_mode == "id-family":
            missing_family_mask = fasta_df["viral_family"].astype(
                str).str.strip() == ""
            if bool(missing_family_mask.any()):
                missing_examples = (
                    fasta_df.loc[missing_family_mask,
                                 [args.id_col]][args.id_col]
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                )
                raise ValueError(
                    "Metadata Family is missing for some FASTA IDs while using id-family keys, "
                    f"examples: {missing_examples}"
                )

        logger.info("metadata_naming_loaded",
                    extra={"extra": {
                        "metadata_file": str(metadata_file),
                        "metadata_name_col": args.metadata_name_col,
                        "metadata_family_col": args.metadata_family_col,
                        "mapped_ids": len(id_to_family),
                        "duplicate_ids_same_family": duplicate_same_family
                    }})

    fasta_df["len"] = fasta_df[args.seq_col].str.len()
    fasta_df = fasta_df.sort_values("len").reset_index(drop=True)

    total_seqs = len(fasta_df)
    num_shards = args.num_shards
    shard_id = args.shard_id
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"shard_id must be in [0, {num_shards}), got {shard_id}.")
    if num_shards > 1:
        # take every num_shards-th row starting at shard_id
        fasta_df = fasta_df.iloc[shard_id::num_shards].reset_index(drop=True)

    # parse out command line arguments
    model_name = args.model_name
    max_tokens = args.max_tokens
    batch_size = args.batch_size
    per_seq_dir = args.per_seq_dir
    idx_csv_path = args.idx_csv_path

    # if running sharded mode, create per-shard output paths
    if num_shards > 1:
        shard_suffix = f"_shard{shard_id:03d}_of_{num_shards:03d}"
        per_seq_dir = per_seq_dir / f"shard_{shard_id:03d}"

        # add suffix to index csv path
        idx_csv_path = idx_csv_path.with_name(
            idx_csv_path.stem + shard_suffix + idx_csv_path.suffix)

    logger.info("run_start",
                extra={"extra": {
                    "model_name": model_name,
                    "max_tokens": max_tokens,
                    "batch_size": batch_size,
                    "metadata_file": str(metadata_file) if metadata_file is not None else None,
                    "embedding_key_mode": args.embedding_key_mode,
                    "key_delimiter": args.key_delimiter,
                    "output_path": str(os.path.abspath(out_dir)),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "torch_version": torch.__version__,
                    "esm_version": esm.__version__,
                    "num_shards": num_shards,
                    "shard_id": shard_id,
                    "total_sequences": total_seqs,
                    "n_sequences_shard": len(fasta_df)
                }})

    index_df, failed_seqs = esm_embeddings_from_fasta(
        fasta_df,
        id_col=args.id_col,
        seq_col=args.seq_col,
        family_col="viral_family",
        model_name=model_name,
        max_tokens=max_tokens,
        batch_size=batch_size,
        per_seq_dir=out_dir/per_seq_dir,
        index_csv_path=out_dir/idx_csv_path,
        key_mode=args.embedding_key_mode,
        key_delimiter=args.key_delimiter,
        logger=logger
    )
    logger.info("run_end", extra={"extra": {
        "failed_count": len(failed_seqs),
        "index_rows": len(index_df)
    }})


if __name__ == "__main__":
    main()
