import re
from pathlib import Path
import pandas as pd

def read_fasta(fasta_path: Path | str, full_name: bool = False) -> pd.DataFrame:
    """
    Parses a FASTA file with PV1-style headers into a pandas DataFrame.

    Expected pattern (example):
    >ID=A8D0M1_ADE02 AC=A8D0M1 OXX=10515,129951,10509,10508
    MALTCRLRFPVPGFRGRMHRRRGMAGHGLTGGMRRAHHRRRRASHRRMRGGILPLLIPLIAAAIGAVPGIASVALQAQRH

    Parameters
    ----------
        fasta_path : str or Path
            Path to the FASTA file.
        full_name : bool
            If set to `True`, the header is parsed as one string into one column. Default behavior (`False`)
            parses each part of the header into individual columns ID, AC, and OXX.

    Returns
    -------
        pd.DataFrame
            FASTA data parsed in to a pandas DataFrame object with columns FullName and Sequence if 
            `full_name=False` or ID, AC, OXX if `full_name=True`.

    Raises
    ------
        ValueError
            If a header line does not match the expected pattern of if a sequence line is seen before
            any header.
    """
    header_pattern = re.compile(r"^>ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")
    rows = []
    curr = None
    with open(fasta_path, "r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):
                if curr:
                    curr["Sequence"] = "".join(curr["Sequence"])
                    rows.append(curr)
                
                if not full_name:
                    match_ = header_pattern.match(line)
                    if not match_:
                        raise ValueError(f"Header does not match expected format: '{line}'")

                    curr = {"ID": match_.group(1), 
                            "AC": match_.group(2), 
                            "OXX": match_.group(3), 
                            "Sequence": []}
                else:
                    line = line.split(">")[1]
                    curr = {"FullName": line, 
                            "Sequence": []}
            else:
                if curr is None:
                    raise ValueError("Found sequence before any header")
                curr["Sequence"].append(line)

    if curr:
        curr["Sequence"] = "".join(curr["Sequence"])
        rows.append(curr)

    if not full_name:
        return pd.DataFrame(rows, columns=["ID", "AC", "OXX", "Sequence"])
    else: 
        return pd.DataFrame(rows, columns=["FullName", "Sequence"])
    
def read_metadata(meta_path: Path | str, 
                  id_col: str = "FullName", 
                  category: str = "Category", 
                  peptide_start_idx: str = "AlignStart", 
                  peptide_end_idx: str = "AlignStop") -> pd.DataFrame:
    """
    Parses a metadata TSV file with PV1-style headers into a pandas DataFrame.

    Expected columns:
    CodeName, Category, SpeciesID, Species, Protein, AlignStart, AlignStop, FullName, 
    Peptide, and Encoding.

    Parameters
    ----------
        meta_path: Path or str
            Path to metadata file.
        id_col : str
            Column name used to get full names of peptides.
        category : str
            Column used to determine category. We only care about SetCover (peptides we already have data for).
        peptide_start_idx : 
            Name of column to store the peptide start index within the overall protein sequence.
        peptide_end_idx : str
            Name of column to store the peptide stop index within the overall protein sequence.

    Returns
    -------
        pd.DataFrame
            Metadata file parsed into a DataFrame with populated AlignStart and AlignStop columns, and 
            indices removed from the FullName column.
    """
    meta_df = pd.read_csv(meta_path, sep="\t", dtype=str)
    meta_df = meta_df[meta_df[category] == "SetCover"]

    oxx_idx_pattern = re.compile(r"OXX=[^\s,]+(?:,[^\s,]+)*_(?P<align_start>\d+)_(?P<align_stop>\d+)")
    oxx_match = meta_df[id_col].str.extract(oxx_idx_pattern)

    meta_df[peptide_start_idx] = (pd.to_numeric(oxx_match["align_start"], 
                                                errors="coerce", downcast="integer")) \
                                    .astype("Int64")
    meta_df[peptide_end_idx] = (pd.to_numeric(oxx_match["align_stop"], 
                                              errors="coerce", downcast="integer")) \
                                  .astype("Int64")
    
    oxx_name_pattern = re.compile(r"(\bOXX=)([^\s]+)")
    meta_df[id_col] = meta_df[id_col].str.replace(oxx_name_pattern, 
        lambda m: m.group(1) + ",".join(part.split("_", 1)[0] 
                                        for part in m.group(2).split(",")), 
        regex=True)
    
    return meta_df

def read_zscores(z_path: Path | str, 
                 z_id_col: str = "Sequence name", 
                 meta_id_col: str = "CodeName") -> pd.DataFrame:
    """
    Parses a z-score reactivity TSV file with one ID column and 1 or more subject columns.

    Expected columns:
    Sequence name, VW_001, VW_002, ..., VW_400.

    Parameters
    ----------
        z_path: Path or str
            Path to metadata file.
        z_id_col : str
            Column name used to get ID codename of peptides.
        meta_id_col : str
            Column name we map `z_id_col` to in order to maintain same IDs across DataFrames.

    Returns
    -------
        pd.DataFrame
            Z-score reactivity file parsed into a DataFrame with ID column name changed to match metadata file.
    """
    z_df = pd.read_csv(z_path, sep="\t")
    z_df.rename(columns={z_id_col: meta_id_col}, inplace=True)

    return z_df

if __name__ == "__main__":
    # testing each function
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)

    fasta_test = read_fasta(Path("../data/fulldesign_2019-02-27_wGBKsw.fasta"), full_name=True)
    print(fasta_test.head())
    metadata_test = read_metadata(Path("../data/PV1_meta_2020-11-23.tsv"))
    print(metadata_test.head())
    zscores_test = read_zscores(Path("../data/SHERC_combined_wSB_4-24-24_Z-HDI95_avg_round.tsv"))
    print(zscores_test.head())
