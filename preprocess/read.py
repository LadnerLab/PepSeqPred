import re
from pathlib import Path
import pandas as pd

def read_fasta(fasta_path: Path | str, full_name: bool = False) -> pd.DataFrame:
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
