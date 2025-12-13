import time
import argparse
from pathlib import Path
from pipelineio.logger import setup_logger
from linker.builder import PeptideDatasetBuilder

def main() -> None:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Link generated ESM-2 embeddings with metadata to create model training data.")
    parser.add_argument("meta_path", 
                        type=Path, 
                        help="Path to metadata file.")
    parser.add_argument("emb_dir", 
                        type=Path, 
                        help="Path to the directory storing .pt embedding files.")
    parser.add_argument("save_path", 
                        type=Path, 
                        help="Name of output file to save training data to.")
    
    args = parser.parse_args()
    logger = setup_logger(json_lines=True, 
                          json_indent=2, 
                          name="linker_cli")

    logger.info("run_start", 
                extra={"extra": {
                    "saving_to": str(args.save_path)
                }})
    builder = PeptideDatasetBuilder(meta_path=args.meta_path, 
                                    emb_dir=args.emb_dir, 
                                    logger=logger)
    data = builder.build()
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
