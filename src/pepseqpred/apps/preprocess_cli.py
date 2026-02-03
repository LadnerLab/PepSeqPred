"""preprocess_cli.py

This module is designed to preprocess the metadata and z-score datasets for 
downstream model training by cleaning up the metadata file and appending 
one-hot encoded targets generated from the z-score dataset.

Usage
-----
>>> # from scripts/preprocessdata.sh
>>> ./preprocessdata.sh <metadata_tsv> <zscore_tsv>
"""
import argparse
import time
from pathlib import Path
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.preprocess.pv1 import preprocess


def main() -> None:
    """
    Parse CLI arguments, set up logging, run data preprocessing script, and save results.
    """
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Preprocess and label model input data from PV1 metadata and z-score reactivity datasets.")
    parser.add_argument("meta_file",
                        type=Path,
                        help="Path to metadata file.")
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
    parser.add_argument("--save",
                        action="store_true",
                        dest="save_path",
                        default=False,
                        help="Store results in a .tsv output file to be used in model training.")

    args = parser.parse_args()
    logger = setup_logger(json_lines=True)

    not_epitope_max_subjects = args.not_epi_max_subs if args.not_epi_max_subs != 0 else None

    if args.save:
        if not_epitope_max_subjects is None:
            save_path = Path(
                f"input_data_{int(args.is_epi_z_min)}_{args.is_epi_min_subs}_{int(args.not_epi_z_max)}_all.tsv")
        else:
            save_path = Path(
                f"input_data_{int(args.is_epi_z_min)}_{args.is_epi_min_subs}_{int(args.not_epi_z_max)}_{not_epitope_max_subjects}.tsv")
    else:
        save_path = None

    is_epitope_z_min = args.is_epi_z_min
    is_epitope_min_subjects = args.is_epi_min_subs
    not_epitope_z_max = args.not_epi_z_max
    logger.info("run_start",
                extra={"extra": {
                    "is_epi_z_min": is_epitope_z_min,
                    "is_epi_min_subs": is_epitope_min_subjects,
                    "not_epi_z_max": not_epitope_z_max,
                    "not_epi_max_subs": not_epitope_max_subjects,
                    "save_path": str(save_path)
                }})

    meta_df = preprocess(args.meta_file,
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
                    "output_file_path": str(save_path),
                    "total_duration_s": round(time.perf_counter() - t0, 3)
                }})


if __name__ == "__main__":
    main()
