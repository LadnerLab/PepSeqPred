"""prepare_dataset.py

Dataset normalization adapter for multi-source training preparation.

This module converts PV1/CWP/BKP source inputs into a common PV1-compatible
contract used by existing embedding, label, and training CLIs.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Set
import pandas as pd
from pepseqpred.core.io.keys import parse_fullname
from pepseqpred.core.preprocess.pv1 import preprocess as preprocess_pv1


def _build_fullname(protein_id: str, group_numeric: int) -> str:
    """Builds a PV1-style fullname for normalized outputs."""
    return f"ID={protein_id} AC={protein_id} OXX=0,0,0,{int(group_numeric)}"


def _clean_str(value: Any) -> str:
    """Returns stripped string value with null-like tokens collapsed to empty."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    return text


def _record_drop(
    drop_counts: Dict[str, int],
    drop_examples: Dict[str, List[str]],
    reason: str,
    value: str,
    max_examples: int = 20
) -> None:
    """Accumulates drop counts and examples by reason."""
    drop_counts[reason] = int(drop_counts.get(reason, 0)) + 1
    if reason not in drop_examples:
        drop_examples[reason] = []
    examples = drop_examples[reason]
    if len(examples) < max_examples and value not in examples:
        examples.append(value)


def _read_code_set(tsv_path: Path | str) -> Set[str]:
    """Loads code values from first TSV column."""
    tsv_path = Path(tsv_path)
    out: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.reader(in_f, delimiter="\t")
        for row in reader:
            if len(row) == 0:
                continue
            value = str(row[0]).strip()
            if value == "":
                continue
            low = value.lower()
            if low in {"sequence name", "codename", "code_name"}:
                continue
            out.add(value)
    return out


def _header_token_to_accession(token: str) -> str:
    """Returns accession key from FASTA first token."""
    token = str(token).strip()
    if token == "":
        return token
    if token.startswith(("tr|", "sp|")):
        parts = token.split("|")
        if len(parts) >= 2 and parts[1].strip() != "":
            return parts[1].strip()
    return token


def _read_fasta_records(fasta_path: Path | str) -> Iterator[Tuple[str, str]]:
    """Yields (header_without_>, sequence)."""
    header = None
    seq_lines: List[str] = []
    with Path(fasta_path).open("r", encoding="utf-8") as fasta_f:
        for raw in fasta_f:
            line = raw.strip()
            if line == "":
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, "".join(seq_lines)


def _write_fasta_records(out_path: Path | str, records: Iterable[Tuple[str, str]]) -> None:
    """Writes FASTA records."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for header, seq in records:
            out_f.write(f">{header}\n{seq}\n")


def _build_nonpv1_fasta_index(fasta_path: Path | str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Builds accession -> sequence mapping from generic FASTA headers.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, List[str]]]
        - unique accession->sequence map
        - ambiguous accession->list_of_distinct_sequences
    """
    seqs_by_accession: Dict[str, List[str]] = {}
    for header, seq in _read_fasta_records(fasta_path):
        token = header.split()[0] if len(header.split()) > 0 else ""
        accession = _header_token_to_accession(token)
        if accession == "":
            continue
        lst = seqs_by_accession.setdefault(accession, [])
        if seq not in lst:
            lst.append(seq)

    unique: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}
    for accession, seqs in seqs_by_accession.items():
        if len(seqs) == 1:
            unique[accession] = seqs[0]
        elif len(seqs) > 1:
            ambiguous[accession] = seqs
    return unique, ambiguous


def _build_pv1_fasta_index(fasta_path: Path | str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Builds PV1 protein_id -> sequence mapping from PV1-style FASTA headers.
    """
    seqs_by_id: Dict[str, List[str]] = {}
    for header, seq in _read_fasta_records(fasta_path):
        parsed = parse_fullname(header)
        protein_id = str(parsed[0])
        lst = seqs_by_id.setdefault(protein_id, [])
        if seq not in lst:
            lst.append(seq)

    unique: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}
    for protein_id, seqs in seqs_by_id.items():
        if len(seqs) == 1:
            unique[protein_id] = seqs[0]
        elif len(seqs) > 1:
            ambiguous[protein_id] = seqs
    return unique, ambiguous


def _resolve_nonpv1_align_columns(
    df: pd.DataFrame,
    dataset_kind: Literal["cwp", "bkp"]
) -> Tuple[List[str], List[str]]:
    """Picks start/stop fallback columns for non-PV1 datasets."""
    if dataset_kind == "cwp":
        start_candidates = ["StartIndex", "AlignStart", "Start", "alignStart"]
        stop_candidates = ["StopIndex", "AlignStop", "Stop", "alignStop"]
    else:
        start_candidates = ["alignStart", "StartIndex", "AlignStart", "Start"]
        stop_candidates = ["alignStop", "StopIndex", "AlignStop", "Stop"]

    start_cols = [col for col in start_candidates if col in df.columns]
    stop_cols = [col for col in stop_candidates if col in df.columns]
    if len(start_cols) == 0 or len(stop_cols) == 0:
        raise ValueError(
            f"Could not resolve alignment columns for dataset_kind='{dataset_kind}'. "
            f"Found start_cols={start_cols}, stop_cols={stop_cols}, available={list(df.columns)}"
        )
    return start_cols, stop_cols


def _coalesce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Coalesces numeric values across fallback columns left-to-right."""
    series = pd.to_numeric(df[cols[0]], errors="coerce")
    for col in cols[1:]:
        series = series.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return series


def _build_group_numeric_map(tokens: Iterable[str], offset: int) -> Dict[str, int]:
    """Builds deterministic token->numeric map using lexical token order."""
    sorted_tokens = sorted({str(tok).strip()
                           for tok in tokens if str(tok).strip() != ""})
    mapping: Dict[str, int] = {}
    for idx, token in enumerate(sorted_tokens, start=1):
        mapping[token] = int(offset) + idx
    return mapping


def _prepare_nonpv1_rows(
    dataset_kind: Literal["cwp", "bkp"],
    meta_df: pd.DataFrame,
    reactive_codes: set[str],
    nonreactive_codes: set[str],
    protein_seqs_by_accession: Dict[str, str],
    ambiguous_accessions: Dict[str, List[str]],
    group_col: str,
    group_id_offset: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalizes CWP/BKP rows into a shared label metadata contract."""
    if "CodeName" not in meta_df.columns:
        raise ValueError("Metadata must include column 'CodeName'")
    if "SequenceAccession" not in meta_df.columns:
        raise ValueError("Metadata must include column 'SequenceAccession'")
    if group_col not in meta_df.columns:
        raise ValueError(
            f"Metadata must include grouping column '{group_col}'")

    overlap = sorted(list(reactive_codes & nonreactive_codes))
    if len(overlap) > 0:
        preview = overlap[:10]
        raise ValueError(
            f"Reactive/non-reactive code overlap detected ({len(overlap)}), examples={preview}"
        )

    selected_codes = reactive_codes | nonreactive_codes
    selected_df = meta_df[meta_df["CodeName"].astype(
        str).isin(selected_codes)].copy()

    start_cols, stop_cols = _resolve_nonpv1_align_columns(
        selected_df, dataset_kind)
    if "PeptideSequence" in selected_df.columns:
        peptide_col = "PeptideSequence"
    elif "Peptide" in selected_df.columns:
        peptide_col = "Peptide"
    else:
        peptide_col = ""

    selected_df["CodeName"] = selected_df["CodeName"].map(_clean_str)
    selected_df["ProteinID"] = selected_df["SequenceAccession"].map(_clean_str)
    selected_df["GroupToken"] = selected_df[group_col].map(_clean_str)
    selected_df["AlignStart"] = _coalesce_numeric_columns(
        selected_df, start_cols).astype("Int64")
    selected_df["AlignStop"] = _coalesce_numeric_columns(
        selected_df, stop_cols).astype("Int64")
    selected_df["LabelSource"] = selected_df["CodeName"].map(
        lambda code: "reactive" if code in reactive_codes else (
            "nonreactive" if code in nonreactive_codes else "unknown")
    )
    selected_df["Def epitope"] = (
        selected_df["LabelSource"] == "reactive").astype("int8")
    selected_df["Uncertain"] = 0
    selected_df["Not epitope"] = (
        selected_df["LabelSource"] == "nonreactive").astype("int8")

    drop_counts: Dict[str, int] = {}
    drop_examples: Dict[str, List[str]] = {}
    kept_rows: List[Dict[str, Any]] = []

    for row in selected_df.to_dict(orient="records"):
        code_name = _clean_str(row.get("CodeName", ""))
        protein_id = _clean_str(row.get("ProteinID", ""))
        group_token = _clean_str(row.get("GroupToken", ""))

        if protein_id == "":
            _record_drop(drop_counts, drop_examples,
                         "missing_protein_id", code_name)
            continue
        if protein_id in ambiguous_accessions:
            _record_drop(drop_counts, drop_examples,
                         "ambiguous_protein_sequence", protein_id)
            continue
        seq = protein_seqs_by_accession.get(protein_id)
        if seq is None:
            _record_drop(drop_counts, drop_examples,
                         "missing_protein_sequence", protein_id)
            continue

        if group_token == "":
            _record_drop(drop_counts, drop_examples,
                         "missing_group_token", code_name)
            continue

        start_raw = row.get("AlignStart")
        stop_raw = row.get("AlignStop")
        if pd.isna(start_raw) or pd.isna(stop_raw):
            _record_drop(drop_counts, drop_examples,
                         "missing_align_bounds", code_name)
            continue

        try:
            start = int(start_raw)
            stop = int(stop_raw)
        except (TypeError, ValueError):
            _record_drop(drop_counts, drop_examples,
                         "invalid_align_bounds", code_name)
            continue

        if start < 0 or stop <= start or stop > len(seq):
            _record_drop(
                drop_counts,
                drop_examples,
                "out_of_bounds_align",
                f"{code_name}:{start}:{stop}:{len(seq)}",
            )
            continue

        if peptide_col != "":
            peptide = _clean_str(row.get(peptide_col, ""))
        else:
            peptide = ""
        if peptide == "":
            peptide = str(seq[start:stop])

        if len(peptide) == 0:
            _record_drop(drop_counts, drop_examples,
                         "missing_peptide_sequence", code_name)
            continue

        kept_rows.append(
            {
                "CodeName": code_name,
                "ProteinID": protein_id,
                "GroupToken": group_token,
                "AlignStart": start,
                "AlignStop": stop,
                "Peptide": peptide,
                "Def epitope": int(row["Def epitope"]),
                "Uncertain": 0,
                "Not epitope": int(row["Not epitope"])
            }
        )

    normalized_df = pd.DataFrame(kept_rows)
    if normalized_df.empty:
        raise ValueError(
            f"No rows left after normalization for dataset_kind='{dataset_kind}'. "
            f"drop_counts={drop_counts}"
        )

    group_map = _build_group_numeric_map(
        normalized_df["GroupToken"].tolist(), group_id_offset)
    normalized_df["GroupID"] = normalized_df["GroupToken"].map(group_map)
    missing_group_numeric = normalized_df["GroupID"].isna()
    if bool(missing_group_numeric.any()):
        bad_codes = normalized_df.loc[missing_group_numeric, "CodeName"].head(
            10).tolist()
        raise RuntimeError(
            f"Missing GroupID after mapping, examples={bad_codes}")
    normalized_df["GroupID"] = normalized_df["GroupID"].astype(int)

    summary = {
        "selected_codes": int(len(selected_codes)),
        "selected_rows": int(selected_df.shape[0]),
        "normalized_rows": int(normalized_df.shape[0]),
        "drop_counts": drop_counts,
        "drop_examples": drop_examples,
        "start_columns_used": start_cols,
        "stop_columns_used": stop_cols,
        "group_token_column": group_col,
        "group_id_offset": int(group_id_offset),
        "group_mapping": {k: int(v) for k, v in group_map.items()}
    }
    return normalized_df, summary


def _prepare_pv1_rows(
    meta_path: Path | str,
    z_path: Path | str,
    protein_fasta: Path | str,
    is_epitope_z_min: float,
    is_epitope_min_subjects: int,
    not_epitope_z_max: float,
    not_epitope_max_subjects: Optional[int],
    subject_prefix: str,
    logger: Optional[logging.Logger]
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    """Prepares PV1 rows using existing preprocess logic."""
    effective_logger = logger if logger is not None else logging.getLogger(
        "prepare_dataset")
    pre_df = preprocess_pv1(
        meta_path=meta_path,
        z_path=z_path,
        fname_col="FullName",
        code_col="CodeName",
        is_epitope_z_min=is_epitope_z_min,
        is_epitope_min_subjects=is_epitope_min_subjects,
        not_epitope_z_max=not_epitope_z_max,
        not_epitope_max_subjects=not_epitope_max_subjects,
        prefix=subject_prefix,
        save_path=None,
        logger=effective_logger
    )
    required = [
        "CodeName",
        "FullName",
        "Peptide",
        "AlignStart",
        "AlignStop",
        "Def epitope",
        "Uncertain",
        "Not epitope"
    ]
    missing = [col for col in required if col not in pre_df.columns]
    if len(missing) > 0:
        raise ValueError(
            f"PV1 preprocess output missing required columns: {missing}")

    pv1_seqs, ambiguous_pv1 = _build_pv1_fasta_index(protein_fasta)
    if len(pv1_seqs) == 0:
        raise ValueError("PV1 FASTA index is empty")

    drop_counts: Dict[str, int] = {}
    drop_examples: Dict[str, List[str]] = {}
    kept_rows: List[Dict[str, Any]] = []
    groups_by_id: Dict[str, str] = {}

    for row in pre_df.to_dict(orient="records"):
        code_name = _clean_str(row.get("CodeName", ""))
        fullname = _clean_str(row.get("FullName", ""))
        try:
            protein_id, _ac, _oxx, family = parse_fullname(fullname)
        except ValueError:
            _record_drop(drop_counts, drop_examples,
                         "invalid_fullname", code_name)
            continue

        protein_id = _clean_str(protein_id)
        family = _clean_str(family)
        if family == "" or not family.isdigit():
            _record_drop(drop_counts, drop_examples,
                         "invalid_family", f"{protein_id}:{family}")
            continue

        if protein_id in ambiguous_pv1:
            _record_drop(drop_counts, drop_examples,
                         "ambiguous_protein_sequence", protein_id)
            continue
        seq = pv1_seqs.get(protein_id)
        if seq is None:
            _record_drop(drop_counts, drop_examples,
                         "missing_protein_sequence", protein_id)
            continue

        start_raw = row.get("AlignStart")
        stop_raw = row.get("AlignStop")
        if pd.isna(start_raw) or pd.isna(stop_raw):
            _record_drop(drop_counts, drop_examples,
                         "missing_align_bounds", code_name)
            continue
        try:
            start = int(start_raw)
            stop = int(stop_raw)
        except (TypeError, ValueError):
            _record_drop(drop_counts, drop_examples,
                         "invalid_align_bounds", code_name)
            continue
        if start < 0 or stop <= start or stop > len(seq):
            _record_drop(
                drop_counts,
                drop_examples,
                "out_of_bounds_align",
                f"{code_name}:{start}:{stop}:{len(seq)}"
            )
            continue

        prev_family = groups_by_id.get(protein_id)
        if prev_family is None:
            groups_by_id[protein_id] = family
        elif prev_family != family:
            _record_drop(
                drop_counts,
                drop_examples,
                "conflicting_family_for_protein",
                f"{protein_id}:{prev_family}:{family}"
            )
            continue

        kept_rows.append(
            {
                "CodeName": code_name,
                "ProteinID": protein_id,
                "GroupToken": family,
                "AlignStart": start,
                "AlignStop": stop,
                "Peptide": _clean_str(row.get("Peptide", "")),
                "Def epitope": int(row.get("Def epitope", 0)),
                "Uncertain": int(row.get("Uncertain", 0)),
                "Not epitope": int(row.get("Not epitope", 0))
            }
        )

    normalized_df = pd.DataFrame(kept_rows)
    if normalized_df.empty:
        raise ValueError(
            f"No PV1 rows left after normalization. drop_counts={drop_counts}"
        )

    normalized_df["GroupID"] = normalized_df["GroupToken"].astype(int)
    summary = {
        "selected_rows": int(pre_df.shape[0]),
        "normalized_rows": int(normalized_df.shape[0]),
        "drop_counts": drop_counts,
        "drop_examples": drop_examples
    }
    return normalized_df, pv1_seqs, summary


def _finalize_and_write_outputs(
    normalized_df: pd.DataFrame,
    protein_seqs: Dict[str, str],
    out_dir: Path | str
) -> Dict[str, Any]:
    """Writes prepared targets, label metadata, and embedding metadata outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # filter rows to proteins that still have sequence records
    normalized_df = normalized_df[normalized_df["ProteinID"].astype(
        str).isin(set(protein_seqs.keys()))].copy()
    if normalized_df.empty:
        raise ValueError(
            "No rows left after intersecting normalized rows with protein FASTA IDs")

    # ensure one group id per protein
    id_group_counts = (
        normalized_df.groupby("ProteinID")["GroupID"]
        .nunique(dropna=False)
        .reset_index(name="n")
    )
    conflicting = id_group_counts[id_group_counts["n"] > 1]
    if not conflicting.empty:
        examples = conflicting["ProteinID"].head(10).tolist()
        raise ValueError(
            "Found proteins assigned to multiple groups after normalization, "
            f"examples={examples}"
        )

    group_by_id = (
        normalized_df.groupby("ProteinID")["GroupID"]
        .first()
        .to_dict()
    )
    fullname_by_id = {
        str(protein_id): _build_fullname(str(protein_id), int(group_id))
        for protein_id, group_id in group_by_id.items()
    }

    normalized_df["FullName"] = normalized_df["ProteinID"].map(fullname_by_id)
    label_cols = [
        "CodeName",
        "FullName",
        "Peptide",
        "AlignStart",
        "AlignStop",
        "Def epitope",
        "Uncertain",
        "Not epitope",
        "ProteinID",
        "GroupToken",
        "GroupID"
    ]
    label_df = normalized_df[label_cols].copy()
    label_df = label_df.sort_values(
        ["ProteinID", "AlignStart", "CodeName"]).reset_index(drop=True)

    emb_meta_df = (
        label_df[["ProteinID", "FullName", "GroupID"]]
        .drop_duplicates()
        .rename(columns={"FullName": "Name", "GroupID": "Family"})
    )
    emb_meta_df["Family"] = emb_meta_df["Family"].astype(int)
    emb_meta_df = emb_meta_df.sort_values(["ProteinID"]).reset_index(drop=True)

    target_records = []
    for protein_id in sorted(group_by_id.keys()):
        seq = protein_seqs.get(protein_id)
        if seq is None:
            continue
        target_records.append((fullname_by_id[protein_id], seq))
    if len(target_records) == 0:
        raise ValueError(
            "No target FASTA records could be written after normalization")

    targets_path = out_dir / "prepared_targets.fasta"
    labels_meta_path = out_dir / "prepared_labels_metadata.tsv"
    emb_meta_path = out_dir / "prepared_embedding_metadata.tsv"
    _write_fasta_records(targets_path, target_records)
    label_df.to_csv(labels_meta_path, sep="\t", index=False)
    emb_meta_df[["Name", "Family"]].to_csv(
        emb_meta_path, sep="\t", index=False)

    return {
        "prepared_targets_fasta": str(targets_path),
        "prepared_labels_metadata_tsv": str(labels_meta_path),
        "prepared_embedding_metadata_tsv": str(emb_meta_path),
        "n_targets": int(len(target_records)),
        "n_label_rows": int(label_df.shape[0]),
        "n_label_proteins": int(label_df["ProteinID"].nunique())
    }


def prepare_dataset(
    dataset_kind: Literal["pv1", "cwp", "bkp"],
    meta_path: Path | str,
    output_dir: Path | str,
    protein_fasta: Path | str,
    reactive_codes: Optional[Path | str] = None,
    nonreactive_codes: Optional[Path | str] = None,
    z_path: Optional[Path | str] = None,
    is_epitope_z_min: float = 20.0,
    is_epitope_min_subjects: int = 4,
    not_epitope_z_max: float = 10.0,
    not_epitope_max_subjects: Optional[int] = None,
    subject_prefix: str = "VW_",
    group_id_offset: int = 0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Converts dataset-specific sources into a shared PV1-compatible contract.

    Outputs under `output_dir`:
        - prepared_targets.fasta
        - prepared_labels_metadata.tsv
        - prepared_embedding_metadata.tsv
        - prepare_summary.json
    """
    dataset_kind = str(dataset_kind).strip().lower()
    if dataset_kind not in {"pv1", "cwp", "bkp"}:
        raise ValueError(
            f"Unsupported dataset_kind='{dataset_kind}'. Expected one of: pv1,cwp,bkp"
        )

    meta_path = Path(meta_path)
    output_dir = Path(output_dir)
    protein_fasta = Path(protein_fasta)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not protein_fasta.exists():
        raise FileNotFoundError(f"Protein FASTA not found: {protein_fasta}")

    summary: Dict[str, Any] = {
        "dataset_kind": dataset_kind,
        "meta_path": str(meta_path),
        "protein_fasta": str(protein_fasta)
    }

    if dataset_kind == "pv1":
        if z_path is None:
            raise ValueError("--z-file is required when --dataset-kind pv1")
        z_path = Path(z_path)
        if not z_path.exists():
            raise FileNotFoundError(f"Z-score file not found: {z_path}")
        summary["z_path"] = str(z_path)

        normalized_df, pv1_seqs, prep_summary = _prepare_pv1_rows(
            meta_path=meta_path,
            z_path=z_path,
            protein_fasta=protein_fasta,
            is_epitope_z_min=is_epitope_z_min,
            is_epitope_min_subjects=is_epitope_min_subjects,
            not_epitope_z_max=not_epitope_z_max,
            not_epitope_max_subjects=not_epitope_max_subjects,
            subject_prefix=subject_prefix,
            logger=logger
        )
        outputs = _finalize_and_write_outputs(
            normalized_df=normalized_df,
            protein_seqs=pv1_seqs,
            out_dir=output_dir
        )
        summary["normalization"] = prep_summary
    else:
        if reactive_codes is None or nonreactive_codes is None:
            raise ValueError(
                "--reactive-codes and --nonreactive-codes are required "
                "when --dataset-kind is cwp or bkp"
            )
        reactive_codes = Path(reactive_codes)
        nonreactive_codes = Path(nonreactive_codes)
        if not reactive_codes.exists():
            raise FileNotFoundError(
                f"Reactive code file not found: {reactive_codes}")
        if not nonreactive_codes.exists():
            raise FileNotFoundError(
                f"Non-reactive code file not found: {nonreactive_codes}")
        summary["reactive_codes"] = str(reactive_codes)
        summary["nonreactive_codes"] = str(nonreactive_codes)

        meta_df = pd.read_csv(meta_path, sep="\t", dtype=str)
        reactive_set = _read_code_set(reactive_codes)
        nonreactive_set = _read_code_set(nonreactive_codes)

        unique_seqs, ambiguous = _build_nonpv1_fasta_index(protein_fasta)
        if len(unique_seqs) == 0:
            raise ValueError(
                f"No protein sequences resolved from FASTA: {protein_fasta}")

        if dataset_kind == "cwp":
            group_col = "Cluster50ID"
        else:
            group_col = "reClusterID_70"

        normalized_df, prep_summary = _prepare_nonpv1_rows(
            dataset_kind=dataset_kind,
            meta_df=meta_df,
            reactive_codes=reactive_set,
            nonreactive_codes=nonreactive_set,
            protein_seqs_by_accession=unique_seqs,
            ambiguous_accessions=ambiguous,
            group_col=group_col,
            group_id_offset=int(group_id_offset)
        )
        outputs = _finalize_and_write_outputs(
            normalized_df=normalized_df,
            protein_seqs=unique_seqs,
            out_dir=output_dir
        )
        summary["normalization"] = prep_summary
        summary["n_ambiguous_accessions"] = int(len(ambiguous))
        if len(ambiguous) > 0:
            summary["ambiguous_accession_examples"] = sorted(
                list(ambiguous.keys()))[:20]

    summary.update(outputs)
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    summary["prepare_summary_json"] = str(summary_path)

    if logger is not None:
        logger.info(
            "prepare_dataset_done",
            extra={"extra": {
                "dataset_kind": dataset_kind,
                "output_dir": str(output_dir),
                "n_targets": int(summary["n_targets"]),
                "n_label_rows": int(summary["n_label_rows"]),
                "n_label_proteins": int(summary["n_label_proteins"]),
                "summary_path": str(summary_path)
            }}
        )

    return summary
