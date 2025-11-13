import argparse
from pathlib import Path
import pandas as pd
from read import read_fasta, read_metadata, read_zscores
from process import merge_fasta_metadata, merge_zscores_metadata, apply_z_threshold
from typing import Optional

def preprocess(meta_path: Path | str, 
               fasta_path: Path | str, 
               z_path: Path | str, 
               fname_col: str = "FullName", 
               code_col: str = "CodeName", 
               seq_col: str = "Sequence", 
               protein_col: str = "Protein", 
               is_epitope_z_min: float = 20.0, 
               is_epitope_min_subjects: int = 4, 
               not_epitope_z_max: float = 10.0, 
               not_epitope_max_subjects: Optional[int] = None, 
               prefix: str = "VW_", 
               save_path: Optional[str | Path] = None) -> pd.DataFrame:
    meta_df = read_metadata(meta_path, id_col=fname_col)
    fasta_df = read_fasta(fasta_path, full_name=True)
    z_df = read_zscores(z_path, meta_id_col=code_col)

    fasta_meta_df = merge_fasta_metadata(fasta_df, meta_df, 
                                         id_col=fname_col, seq_col=seq_col, out_col=protein_col)
    
    z_df_targets = apply_z_threshold(z_df, 
                                     is_epitope_z_min=is_epitope_z_min, 
                                     is_epitope_min_subjects=is_epitope_min_subjects, 
                                     not_epitope_z_max=not_epitope_z_max, 
                                     not_epitope_max_subjects=not_epitope_max_subjects, 
                                     prefix=prefix)
    
    all_df = merge_zscores_metadata(z_df_targets, fasta_meta_df, id_col=code_col, save_path=save_path)

    return all_df

def main() -> None:
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
    parser.add_argument("--seq-col", 
                        action="store", 
                        dest="seq_col", 
                        type=str, 
                        help="Name of the column containing protein sequences.")
    parser.add_argument("--protein-col", 
                        action="store", 
                        dest="protein_col", 
                        type=str, 
                        default="Protein", 
                        help="Name of output column that will contain protein sequences.")
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
                        dest="is_epi_max_subs", 
                        type=int, 
                        default=None, 
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

    if args.save_path:
        if args.is_epi_max_subs is None:
            save_path = Path(f"input_data_{args.is_epi_z_min}_{args.is_epi_min_subs}_{args.not_epi_z_max}_all")
        else:
            save_path = Path(f"input_data_{args.is_epi_z_min}_{args.is_epi_min_subs}_{args.not_epi_z_max}_{args.is_epi_max_subs}")
    else:
        save_path = None

    full_df = preprocess(args.meta_file, 
                         args.fasta_file, 
                         args.z_file, 
                         fname_col=args.fname_col, 
                         code_col=args.code_col, 
                         seq_col=args.seq_col, 
                         protein_col=args.protein_col, 
                         is_epitope_z_min=args.is_epi_z_min, 
                         is_epitope_min_subjects=args.is_epi_min_subs, 
                         not_epitope_z_max=args.not_epi_z_max, 
                         not_epitope_max_subjects=args.is_epi_max_subs, 
                         prefix=args.subject_prefix, 
                         save_path=save_path)
    
    print("Preprocessing done...\n")
    print(full_df.head())

if __name__ == "__main__":
    main()
