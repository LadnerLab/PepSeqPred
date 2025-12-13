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
by averaging overlaps to produce a full-length (L, D+1) matrix the token limit are excluded from embedding 
generation.  

This script can be ran on its own, but it is intended for HPC use (e.g., Monsoon). 
See shell script `generateembeddings.sh` or `generateembeddings_cpu.sh` for typical usage.

Usage
-----
>>> # HPC usage with shell script
>>> sbatch --export=ALL,IN_FASTA=/scratch/<NAUIDD>/<targets>.fasta generateembeddings.sh

>>> # general example usage (illustrative)
>>> python esm_cli.py --fasta-file <input.fasta> --model-name esm2_t33_650M_UR50D \ 
                      --batch-size <n> --log-dir <dir> [--log-json] --out-dir <dir>
"""
import os
import argparse
from pathlib import Path
from datetime import datetime
import esm
import torch
from pipelineio.logger import setup_logger
from pipelineio.read import read_fasta
from esm2.embeddings import esm_embeddings_from_fasta

def main() -> None:
    """
    Parse CLI arguments, set up logging, run embedding generation script, and save results.
    """
    parser = argparse.ArgumentParser(description="Make ESM-2 embeddings from PV1 targets fasta file.")
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
                        default=Path(f"artifacts/pts"), 
                        help="Directory for individual .pt files when save mode is set to 'pt'.")
    parser.add_argument("--index-csv-path", 
                        action="store", 
                        dest="idx_csv_path", 
                        type=Path, 
                        default=Path(f"artifacts/esm_seq_idx_{datetime.now().strftime('%Y-%m-%d_T%H_%M_%S')}.csv"), 
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

    # read input fasta file and sort by sequence length
    fasta_df = read_fasta(args.fasta_file)
    fasta_df["len"] = fasta_df[args.seq_col].str.len()
    fasta_df = fasta_df.sort_values("len").reset_index(drop=True)

    total_seqs = len(fasta_df)
    num_shards = args.num_shards
    shard_id = args.shard_id
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}.")
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
        idx_csv_path = idx_csv_path.with_name(idx_csv_path.stem + shard_suffix + idx_csv_path.suffix)

    logger.info("run_start", 
                extra={"extra": {
                    "model_name": model_name, 
                    "max_tokens": max_tokens, 
                    "batch_size": batch_size,  
                    "output_path": str(os.path.abspath(out_dir)), 
                    "device": "cuda" if torch.cuda.is_available() else "cpu", 
                    "torch_version": torch.__version__, 
                    "esm_version": esm.__version__, 
                    "num_shards": num_shards, 
                    "shard_id": shard_id, 
                    "total_sequences": total_seqs, 
                    "n_sequences_shard": len(fasta_df)
                }})


    index_df, failed_seqs = esm_embeddings_from_fasta(fasta_df, 
                                                      id_col=args.id_col, 
                                                      seq_col=args.seq_col, 
                                                      model_name=model_name, 
                                                      max_tokens=max_tokens, 
                                                      batch_size=batch_size, 
                                                      per_seq_dir=out_dir/per_seq_dir,
                                                      index_csv_path=out_dir/idx_csv_path, 
                                                      logger=logger)
    logger.info("run_end", extra={"extra": {
        "failed_count": len(failed_seqs), 
        "index_rows": len(index_df)
    }})
                    
if __name__ == "__main__":
    main()
