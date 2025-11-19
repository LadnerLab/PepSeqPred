from pathlib import Path
from warnings import deprecated
import pandas as pd
from read import read_fasta, read_metadata, read_zscores
from typing import Optional, List

@deprecated("process.py is deprecated and will be removed in the future.")
def merge_fasta_metadata(fasta_df: pd.DataFrame, 
                         meta_df: pd.DataFrame, 
                         id_col: str = "FullName", 
                         seq_col: str = "Sequence", 
                         out_col: str = "Protein") -> pd.DataFrame:
    """
    Performs a left merge using FASTA and metadata DataFrames on the ID column.

    Parameters
    ----------
        fasta_df : pd.DataFrame
            DataFrame containing an ID column and protein sequences.
        meta_df : pd.DataFrame
            DataFrame containing populated metadata columns for PV1 style peptides.
        id_col : str
            The column to merge the DataFrame's on.
        seq_col : str
            Name of protein sequence column in FASTA DataFrame.
        out_col : str
            Name of the protein sequence column to be used in the final output DataFrame.
    
    Returns
    -------
        pd.DataFrame
            The DataFrame left merged the ID column name.
    """
    merged = meta_df.merge(fasta_df[[id_col, seq_col]], on=id_col, how="left", validate="m:1")
    merged[out_col] = merged[seq_col]
    merged.drop(columns=[seq_col], inplace=True)

    return merged

def merge_zscores_metadata(z_df: pd.DataFrame, 
                           meta_df: pd.DataFrame, 
                           id_col: str = "CodeName", 
                           target_cols: List[str] = ["Def epitope", "Uncertain", "Not epitope"], 
                           save_path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Performs a left merge using the z-score reactivity and metadata DataFrames on the ID column.

    Parameters
    ----------
        z_df : pd.DataFrame
            DataFrame containing the z-score reactivity data.
        meta_df : pd.DataFrame
            DataFrame containing populated metadata columns for PV1 style peptides.
        id_col : str
            The column to merge the DataFrame's on.
        target_cols : List[str]
            The names of the target columns to be added to the output DataFrame.
        save_path : Path or str or None
            An optional path to save the merged output DataFrame to a TSV file.

    Returns
    -------
        pd.DataFrame
            The DataFrame left merged on the ID column name.
    """
    merged = meta_df.merge(z_df[[id_col, *target_cols]], on=id_col, how="left")

    if save_path:
        merged.to_csv(save_path, sep="\t", index=False)

    return merged

def apply_z_threshold(z_df: pd.DataFrame, 
                      is_epitope_z_min: float = 20.0, 
                      is_epitope_min_subjects: int = 4, 
                      not_epitope_z_max: float = 10.0, 
                      not_epitope_max_subjects: Optional[int] = None, 
                      prefix: str = "VW_") -> pd.DataFrame:
    """
    Adds three new one-hot encoded columns to the z-score reactivity DataFrame to classify each row as 
    definitely containing an epitope, uncertain about containing an epitope, or does not contain any epitopes.

    Parameters
    ----------
        z_df : pd.DataFrame
            DataFrame containing the z-score reactivity data.
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

    Returns
    -------
        pd.DataFrame
            The z-score reactivity DataFrame with columns "Def epitope", "Uncertain", and "Not epitope" 
            appended and one-hot encoded.
    """
    # assuming df passed in full
    subject_cols = [col for col in z_df.columns if col.startswith(prefix)]

    subjects = z_df.loc[:, subject_cols].apply(pd.to_numeric, errors="coerce")
    num_subjects = subjects.shape[1]
    arr = subjects.to_numpy()
    
    reactive_counts = (arr >= is_epitope_z_min).sum(axis=1)
    nonreactive_counts = (arr < not_epitope_z_max).sum(axis=1)

    def_is = reactive_counts >= is_epitope_min_subjects
    # manually choose non-epitope threshold
    if not_epitope_max_subjects is not None:
        def_not = (nonreactive_counts >= not_epitope_max_subjects) & (reactive_counts == 0)
    else:
        # every subject is nonreactive
        def_not = nonreactive_counts == num_subjects
    uncertain = (~def_is) & (~def_not)

    # drop all zscores since we have targets now
    z_df.drop(columns=subject_cols, inplace=True)

    return z_df.assign(**{"Def epitope": def_is.astype("int8"), 
                        "Uncertain": uncertain.astype("int8"), 
                        "Not epitope": def_not.astype("int8")})

if __name__ == "__main__":
    # testing each function
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)

    fasta_test = read_fasta(Path("../data/fulldesign_2019-02-27_wGBKsw.fasta"), full_name=True)
    metadata_test = read_metadata(Path("../data/PV1_meta_2020-11-23.tsv"))
    zscores_test = read_zscores(Path("../data/SHERC_combined_wSB_4-24-24_Z-HDI95_avg_round.tsv"))

    merged1 = merge_fasta_metadata(fasta_test, metadata_test)
    print(merged1.head())
    z_df = apply_z_threshold(zscores_test)
    merged2 = merge_zscores_metadata(z_df, merged1)
    print(merged2.head())
