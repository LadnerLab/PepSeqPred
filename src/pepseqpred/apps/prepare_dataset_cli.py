"""prepare_dataset_cli.py

Normalize PV1/CWP/BKP sources into a shared PV1-compatible training 
contract (i.e., ID=<ID> AC=<AC> OXX=<OXX>).
"""
import argparse
import time
from pathlib import Path
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.preprocess.preparedataset import prepare_dataset


def main() -> None:
    """Parse arguments and run dataset normalization."""
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dataset-specific metadata/labels/targets into a PV1-compatible "
            "contract (i.e., ID=<ID> AC=<AC> OXX=<OXX>) for embedding, label generation, and training."
        )
    )
    parser.add_argument(
        "meta_file",
        type=Path,
        help="Path to metadata TSV source file."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write prepared outputs."
    )
    parser.add_argument(
        "--dataset-kind",
        action="store",
        dest="dataset_kind",
        type=str,
        choices=["pv1", "cwp", "bkp"],
        required=True,
        help="Dataset source kind."
    )
    parser.add_argument(
        "--protein-fasta",
        action="store",
        dest="protein_fasta",
        type=Path,
        required=True,
        help="Protein FASTA used to resolve full protein sequences."
    )
    parser.add_argument(
        "--z-file",
        action="store",
        dest="z_file",
        type=Path,
        default=None,
        help="PV1 z-score TSV (required when --dataset-kind pv1)."
    )
    parser.add_argument(
        "--reactive-codes",
        action="store",
        dest="reactive_codes",
        type=Path,
        default=None,
        help="Reactive code-list TSV (required for cwp/bkp)."
    )
    parser.add_argument(
        "--nonreactive-codes",
        action="store",
        dest="nonreactive_codes",
        type=Path,
        default=None,
        help="Non-reactive code-list TSV (required for cwp/bkp)."
    )
    parser.add_argument(
        "--group-id-offset",
        action="store",
        dest="group_id_offset",
        type=int,
        default=None,
        help=(
            "Optional numeric offset applied to group IDs for cwp/bkp mapping. "
            "Defaults: cwp=100000000, bkp=200000000, pv1=0."
        )
    )

    # PV1 threshold configuration (mirrors preprocess CLI defaults)
    parser.add_argument(
        "--is-epi-z-thresh",
        action="store",
        dest="is_epi_z_min",
        type=float,
        default=20.0,
        help="Minimum z-score required for peptide to contain epitopes (pv1 only)."
    )
    parser.add_argument(
        "--is-epi-min-subs",
        action="store",
        dest="is_epi_min_subs",
        type=int,
        default=4,
        help="Minimum # of subjects at/above z threshold for epitope calls (pv1 only)."
    )
    parser.add_argument(
        "--not-epi-z-thresh",
        action="store",
        dest="not_epi_z_max",
        type=float,
        default=10.0,
        help="Maximum z-score for non-epitope calls (pv1 only)."
    )
    parser.add_argument(
        "--not-epi-max-subs",
        action="store",
        dest="not_epi_max_subs",
        type=int,
        default=0,
        help=(
            "Maximum # subjects below non-epitope z threshold. "
            "Use 0 for all subjects (pv1 only)."
        )
    )
    parser.add_argument(
        "--subject-prefix",
        action="store",
        dest="subject_prefix",
        type=str,
        default="VW_",
        help="Prefix for subject z-score columns (pv1 only)."
    )

    parser.add_argument(
        "--log-level",
        action="store",
        dest="log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level."
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        dest="log_json",
        default=False,
        help="Emit JSON logs."
    )
    args = parser.parse_args()

    logger = setup_logger(
        log_dir=None,
        log_level=args.log_level,
        json_lines=args.log_json,
        json_indent=2 if args.log_json else None,
        name="prepare_dataset_cli"
    )

    if args.group_id_offset is None:
        if args.dataset_kind == "cwp":
            group_id_offset = 100_000_000
        elif args.dataset_kind == "bkp":
            group_id_offset = 200_000_000
        else:
            group_id_offset = 0
    else:
        group_id_offset = int(args.group_id_offset)

    not_epitope_max_subjects = args.not_epi_max_subs if int(
        args.not_epi_max_subs) != 0 else None

    logger.info("run_start", extra={"extra": {
        "dataset_kind": args.dataset_kind,
        "meta_file": str(args.meta_file),
        "output_dir": str(args.output_dir),
        "protein_fasta": str(args.protein_fasta),
        "z_file": str(args.z_file) if args.z_file is not None else None,
        "reactive_codes": str(args.reactive_codes) if args.reactive_codes is not None else None,
        "nonreactive_codes": str(args.nonreactive_codes) if args.nonreactive_codes is not None else None,
        "group_id_offset": group_id_offset
    }})

    summary = prepare_dataset(
        dataset_kind=args.dataset_kind,
        meta_path=args.meta_file,
        output_dir=args.output_dir,
        protein_fasta=args.protein_fasta,
        reactive_codes=args.reactive_codes,
        nonreactive_codes=args.nonreactive_codes,
        z_path=args.z_file,
        is_epitope_z_min=args.is_epi_z_min,
        is_epitope_min_subjects=args.is_epi_min_subs,
        not_epitope_z_max=args.not_epi_z_max,
        not_epitope_max_subjects=not_epitope_max_subjects,
        subject_prefix=args.subject_prefix,
        group_id_offset=group_id_offset,
        logger=logger
    )

    logger.info("run_done", extra={"extra": {
        "dataset_kind": args.dataset_kind,
        "prepared_targets_fasta": summary.get("prepared_targets_fasta"),
        "prepared_labels_metadata_tsv": summary.get("prepared_labels_metadata_tsv"),
        "prepared_embedding_metadata_tsv": summary.get("prepared_embedding_metadata_tsv"),
        "prepare_summary_json": summary.get("prepare_summary_json"),
        "n_targets": summary.get("n_targets"),
        "n_label_rows": summary.get("n_label_rows"),
        "n_label_proteins": summary.get("n_label_proteins"),
        "duration_s": round(time.perf_counter() - t0, 3)
    }})


if __name__ == "__main__":
    main()
