"""linker_cli.py

This module is designed to link the model training data and targets together for downstream neural network training and evaluation. For example:

Sample metadata TSV file
------------------------
CodeName	AlignStart	AlignStop	FullName	Peptide	Def epitope	Uncertain	Not epitope \n
TEST_0001	0	30	"ID=PROT1 AC=P00001 OXX=11111"	AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA	1	0	0 \n
TEST_0002	10	40	"ID=PROT2 AC=P00002 OXX=22222"	BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB	0	1	0 \n
TEST_0003	50	80	"ID=PROT3 AC=P00003 OXX=33333"	CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC	0	0	1 \n

Sample embeddings generated (L, D + 1)
--------------------------------------
EMB_1: [[0.123, 1.532, -0.035, ..., 2.445], ..., 519] \n
EMB_2: [[1.623, -0.022, -0.999, ..., 6.013], ..., 36] \n
EMB_3: [[0.321, -6.59, 0.635, ..., -0.001], ..., 1108] \n

This data is combined such that each peptide from the metadata is used to parse out the peptide-level embeddings from the overall protein
sequence's embeddings, then mapped to their targets (labels), along with other information used to make model training as simple as possible.

Usage
-----
>>> # can be done locally, or on HPC
>>> python linker_cli.py \
    <meta_path> \
    <emb_dir> \
    <out_path>
"""
import time
import argparse
from pathlib import Path
from logger import setup_logger
from builder import PeptideDatasetBuilder

def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Linker program."""
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Link generated ESM-2 embeddings with metadata to create model training data.")
    parser.add_argument("meta_path", 
                        type=Path, 
                        help="Path to metadata file.")
    parser.add_argument("save_path", 
                        type=Path, 
                        help="Name of output file to save training data to.")
    parser.add_argument("--emb-dir", 
                        action="append", 
                        dest="emb_dirs", 
                        type=Path, 
                        required=True, 
                        help="Embedding directory. Repeat this flag to add multiple shard roots.")
    
    args = parser.parse_args()
    logger = setup_logger(json_lines=True, 
                          json_indent=2, 
                          name="linker_cli")

    logger.info("run_start", 
                extra={"extra": {
                    "saving_to": str(args.save_path)
                }})
    builder = PeptideDatasetBuilder(meta_path=args.meta_path, 
                                    emb_dirs=args.emb_dirs, 
                                    logger=logger)
    data = builder.build(memmap_dtype="float32")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    data.save(args.save_path)

    logger.info("linking_done", 
                extra={"extra": {
                    "embedding_size": data.embeddings.size(), 
                    "saved_to": str(args.save_path), 
                    "total_duration_s": round(time.perf_counter() - t0, 3)
                }})

if __name__ == "__main__":
    main()
