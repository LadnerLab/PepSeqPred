import os
import json
import argparse
from pathlib import Path
import pandas as pd
import logging
import time
from datetime import datetime
from read import read_fasta, read_metadata, read_zscores
from process import merge_zscores_metadata, apply_z_threshold
from typing import Optional, Tuple

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
            Logger named `preprocess_cli` with a stream handler attached.
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
    logger = logging.getLogger("preprocess_cli")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers[:] = [] # avoid duplicate handlers

    # choose formatter style
    stream_formatter = JSONFormatter() if json_lines else logging.Formatter("%(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger

def preprocess(meta_path: Path | str, 
               fasta_path: Path | str, 
               z_path: Path | str, 
               fname_col: str = "FullName", 
               code_col: str = "CodeName", 
               is_epitope_z_min: float = 20.0, 
               is_epitope_min_subjects: int = 4, 
               not_epitope_z_max: float = 10.0, 
               not_epitope_max_subjects: Optional[int] = None, 
               prefix: str = "VW_", 
               save_path: Optional[str | Path] = None, 
               logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the entire data preprocessing step for ESM-2 embedding generation and machine learning epitope 
    predictions.

    Parameters
    ----------
        meta_path: Path or str
            Path to metadata file.
        fasta_path : str or Path
            Path to the FASTA file.
        z_path: Path or str
            Path to metadata file.
        fname_col : str
            Column name used to get full names of peptides.
        protein_col : str
            Name of the protein sequence column to be used in the final output DataFrame.
        is_epitope_z_min : float
            The minimum z-score to be used to determine if a peptide could contain an epitope.
        is_epitope_min_subjects : int
            The minimum number of subjects required to be greater than or equal to `is_epitope_z_min`
            in order for the row to be classified as an epitope.
        not_epitope_z_max : float
            The maximum z-score to be used to determine if a peptide does not contain an epitope.
        not_epitope_max_subjects : int or None
            The maximum number of subjects required to be less than `not_epitope_z_max` in order for the 
            row to be classified as not an epitope. Default behavior uses every subject column.
        prefix : str
            The prefix of subject column names. For example, if subject columns are named "VW_010", the prefix
            would be "VW_".
        save_path : Path or str or None
            An optional path to save the merged output DataFrame to a TSV file.
        logger : logging.Logger or None
            Logger to use. If None, uses `preprocess_cli` logger.

    Returns
    -------
        fully_pop_meta_df : pd.DataFrame
            Fully populated PV1 metadata style DataFrame for use in ESM-2 embedding generation and machine learning
            epitope location training/predictions. Optionally, saves DataFrame to a TSV file if `save_path` argument
            is passed to function.
        fasta_df : pd.DataFrame
            The fasta file read into a DataFrame for use in ESM-2 embedding generation and machine learning epitope 
            location training/predictions.
    """
    t0 = time.perf_counter()
    meta_df = read_metadata(meta_path, id_col=fname_col, drop_cols=["Category", "SpeciesID", "Protein", "Encoding"])
    fasta_df = read_fasta(fasta_path, full_name=True)
    z_df = read_zscores(z_path, meta_id_col=code_col)
    logger.info("read_files", 
                extra={"extra": {
                    "meta_size": len(meta_df), 
                    "fasta_size": len(fasta_df), 
                    "z_size": len(z_df), 
                    "read_duration_s": round(time.perf_counter() - t0, 3)
                }})
    
    t1 = time.perf_counter()
    z_df_targets = apply_z_threshold(z_df, 
                                     is_epitope_z_min=is_epitope_z_min, 
                                     is_epitope_min_subjects=is_epitope_min_subjects, 
                                     not_epitope_z_max=not_epitope_z_max, 
                                     not_epitope_max_subjects=not_epitope_max_subjects, 
                                     prefix=prefix)
    logger.info("applied_z_score_thresh", 
                extra={"extra": {
                    "z_one_hot_encoding_duration_s": round(time.perf_counter() - t1, 3)
                }})
    
    t2 = time.perf_counter()
    fully_pop_meta_df = merge_zscores_metadata(z_df_targets, meta_df, id_col=code_col, save_path=save_path)
    logger.info("merged_z_and_meta", 
                extra={"extra": {
                    "merged_size": len(fully_pop_meta_df), 
                    "merged_on": code_col, 
                    "merge_duration_s": round(time.perf_counter() - t2, 3)
                }})

    return fully_pop_meta_df, fasta_df

def main() -> None:
    """
    Parse CLI arguments, set up logging, run data preprocessing script, and save results.
    """
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Preprocess and label model input data from PV1 metadata, targets fasta file, and z-score reactivity datasets.")
    parser.add_argument("meta_file", 
                        type=Path, 
                        help="Path to metadata file.")
    parser.add_argument("fasta_file", 
                        type=Path, 
                        help="Path to targets fasta file.")
    parser.add_argument("z_file", 
                        type=Path, 
                        help="Path to z-score reactivity file.")
    parser.add_argument("--fname-col", 
                        action="store", 
                        dest="fname_col", 
                        type=str, 
                        default="FullName", 
                        help="Name of column containing 'ID=...,AC=...,OXX=...' entries.")
    parser.add_argument("--code-col", 
                        action="store", 
                        dest="code_col", 
                        type=str, 
                        default="CodeName", 
                        help="Name of column containing codenames to map between metadata and z-score files.")
    parser.add_argument("--is-epi-z-thresh", 
                        action="store", 
                        dest="is_epi_z_min", 
                        type=float, 
                        default=20.0, 
                        help="Minimum z-score required for peptide to contain epitopes.")
    parser.add_argument("--is-epi-min-subs", 
                        action="store", 
                        dest="is_epi_min_subs", 
                        type=int, 
                        default=4, 
                        help="Minimum # of subjects required to be at or above minimum z-score for peptide to contain epitopes.")
    parser.add_argument("--not-epi-z-thresh", 
                        action="store", 
                        dest="not_epi_z_max", 
                        type=float, 
                        default=10.0, 
                        help="Maximum z-score required for peptide to NOT contain epitopes.")
    parser.add_argument("--not-epi-max-subs", 
                        action="store", 
                        dest="not_epi_max_subs", 
                        type=int, 
                        default=0, 
                        help="Maximum # of subjects required to be at or below maximum z-score for peptide to NOT contain epitopes. Default is 'None' which means every column is used to disqualify a peptide from containing epitopes.")
    parser.add_argument("--subject-prefix", 
                        action="store", 
                        dest="subject_prefix", 
                        type=str, 
                        default="VW_", 
                        help="Prefix for subject column labels in z-score reactivity data.")
    parser.add_argument("--save-path", 
                        action="store_true", 
                        dest="save_path", 
                        default=False, 
                        help="Store results in a .tsv output file to be used in model training.")
    
    args = parser.parse_args()
    logger = setup_logger(json_lines=True)

    not_epitope_max_subjects = args.not_epi_max_subs if args.not_epi_max_subs != 0 else None

    if args.save_path:
        if not_epitope_max_subjects is None:
            save_path = Path(f"input_data_{int(args.is_epi_z_min)}_{args.is_epi_min_subs}_{int(args.not_epi_z_max)}_all.tsv")
        else:
            save_path = Path(f"input_data_{int(args.is_epi_z_min)}_{args.is_epi_min_subs}_{int(args.not_epi_z_max)}_{not_epitope_max_subjects}.tsv")
    else:
        save_path = None

    is_epitope_z_min = args.is_epi_z_min, 
    is_epitope_min_subjects = args.is_epi_min_subs, 
    not_epitope_z_max = args.not_epi_z_max,  
    logger.info("run_start", 
                extra={"extra": {
                    "is_epi_z_min": is_epitope_z_min, 
                    "is_epi_min_subs": is_epitope_min_subjects, 
                    "not_epi_z_max": not_epitope_z_max, 
                    "not_epi_max_subs": not_epitope_max_subjects, 
                    "save_path": str(save_path)
                }})

    meta_df, fasta_df = preprocess(args.meta_file, 
                         args.fasta_file, 
                         args.z_file, 
                         fname_col=args.fname_col, 
                         code_col=args.code_col,  
                         is_epitope_z_min=is_epitope_z_min, 
                         is_epitope_min_subjects=is_epitope_min_subjects, 
                         not_epitope_z_max=not_epitope_z_max, 
                         not_epitope_max_subjects=not_epitope_max_subjects, 
                         prefix=args.subject_prefix, 
                         save_path=save_path, 
                         logger=logger)
    
    logger.info("preprocessing_done", 
                extra={"extra": {
                    "meta_size": len(meta_df), 
                    "fasta_size": len(fasta_df), 
                    "output_file_path": str(save_path), 
                    "total_duration_s": round(time.perf_counter() - t0, 3)
                }})

if __name__ == "__main__":
    main()
