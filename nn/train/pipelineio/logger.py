import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(log_dir: Optional[Path] = None, log_level: str = "INFO", json_lines: bool = False, json_indent: Optional[int] = None, name: str = Path(__file__).stem) -> logging.Logger:
    """
    Creates and sets up a configured logger for this CLI.

    Parameters
    ----------
        log_dir : Path or None
            Directory where log files will be written. Default is None.
        log_level : str
            Minimum level for logs. Default is "INFO".
        json_lines : bool
            When True, formats logs as a JSON object. Default is False (`logging` library default format).
        json_indent : int or None
            Indent size for JSON logging. Default is None.
        name : str
            The name of the logger to setup. Default is whatever the file name is.

    Returns
    -------
        logging.Logger
            Logger named after the filename with a file handler and a stream handler attached.
    """
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}_{datetime.now().strftime('%Y-%m-%d_T%H_%M_%S')}.log"

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
            
            if json_indent is not None:
                return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=json_indent)
            
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    
    # create named logger and reset any inherited handlers to avoid duplication
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers[:] = [] # avoid duplicate handlers

    # choose formatter style
    formatter = JSONFormatter() if json_lines else logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    stream_formatter = JSONFormatter() if json_lines else logging.Formatter("%(levelname)s %(message)s")

    # add rotating file handlers, ~10 MB of storage (only if save logs)
    if log_dir is not None:
        file_handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger
