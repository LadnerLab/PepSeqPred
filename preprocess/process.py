from pathlib import Path
import pandas as pd
from read import read_fasta, read_metadata, read_zscores
from typing import Optional, List

def merge_fasta_metadata(fasta_df: pd.DataFrame, 
                         meta_df: pd.DataFrame, 
                         id_col: str = "FullName", 
                         seq_col: str = "Sequence", 
                         out_col: str = "Protein") -> pd.DataFrame:
    merged = meta_df.merge(fasta_df[[id_col, seq_col]], on=id_col, how="left", validate="m:1")
    merged[out_col] = merged[seq_col]
    merged.drop(columns=[seq_col], inplace=True)

    return merged

def merge_zscores_metadata(z_df: pd.DataFrame, 
                           meta_df: pd.DataFrame, 
                           id_col: str = "CodeName", 
                           target_cols: List[str] = ["Def epitope", "Uncertain", "Not epitope"], 
                           save_path: Optional[Path | str] = None):
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
