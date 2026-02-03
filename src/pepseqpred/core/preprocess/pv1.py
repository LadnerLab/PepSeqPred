import logging
import time
from typing import Optional
from pathlib import Path
import pandas as pd
from pepseqpred.core.io.read import read_metadata, read_zscores
from pepseqpred.core.preprocess.zscores import apply_z_threshold, merge_zscores_metadata

# TODO: log/flag all peptides < 30 amino acids in length


def preprocess(meta_path: Path | str,
               z_path: Path | str,
               fname_col: str = "FullName",
               code_col: str = "CodeName",
               is_epitope_z_min: float = 20.0,
               is_epitope_min_subjects: int = 4,
               not_epitope_z_max: float = 10.0,
               not_epitope_max_subjects: Optional[int] = None,
               prefix: str = "VW_",
               save_path: Optional[str | Path] = None,
               logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Runs the entire data preprocessing step for ESM-2 embedding generation and machine learning epitope 
    predictions.

    Parameters
    ----------
        meta_path: Path or str
            Path to metadata file.
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
    """
    t0 = time.perf_counter()
    meta_df = read_metadata(meta_path, id_col=fname_col, drop_cols=[
                            "Category", "SpeciesID", "Protein", "Encoding"])
    z_df = read_zscores(z_path, meta_id_col=code_col)
    logger.info("read_files",
                extra={"extra": {
                    "meta_size": len(meta_df),
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
    fully_pop_meta_df = merge_zscores_metadata(
        z_df_targets, meta_df, id_col=code_col, save_path=save_path)
    logger.info("merged_z_and_meta",
                extra={"extra": {
                    "merged_size": len(fully_pop_meta_df),
                    "merged_on": code_col,
                    "merge_duration_s": round(time.perf_counter() - t2, 3)
                }})

    return fully_pop_meta_df
