import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from builder import PeptideDatasetBuilder

def setup_logger(log_level: str = "INFO", json_lines: bool = False) -> logging.Logger:
    """
    Creates and sets up a configured logger for this CLI.

    Parameters
    ----------
        log_level : str
            Minimum level for logs. Default is "INFO".
        json_lines : bool
            When True, formats logs as a JSON object. Default is False (`logging` library default format).

    Returns
    -------
        logging.Logger
            Logger named `linker_cli` with a stream handler attached.
    """
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            payload = {"timestamp": datetime.now().isoformat(), 
                       "level": record.levelname, 
                       "message": record.getMessage(), 
                       "logger": record.name, 
                       "where": f"{record.pathname}:{record.lineno}"}
            
            # add all detailed logs using "extra" kwargs
            if hasattr(record, "extra") and isinstance(record.extra, dict):
                payload.update(record.extra)
            
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2)
    
    # create named logger and reset any inherited handlers to avoid duplication
    logger = logging.getLogger("linker_cli")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers[:] = [] # avoid duplicate handlers

    # choose formatter style
    stream_formatter = JSONFormatter() if json_lines else logging.Formatter("%(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger

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
                        type=str, 
                        help="Name of output file to save training data to.")
    
    args = parser.parse_args()
    logger = setup_logger(json_lines=True)

    logger.info("run_start", 
                extra={"extra": {
                    "saving_to": str(args.save_path)
                }})
    builder = PeptideDatasetBuilder(meta_path=args.meta_path, 
                                    emb_dir=args.emb_dir, 
                                    logger=logger)
    data = builder.build()
    data.save(args.save_path)

    logger.info("linking_done", 
                extra={"extra": {
                    "embedding_size": data.embeddings.size(), 
                    "saved_to": str(args.save_path), 
                    "total_duration_s": round(time.perf_counter() - t0, 3)
                }})

if __name__ == "__main__":
    main()
