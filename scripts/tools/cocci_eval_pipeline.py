"""cocci_eval_pipeline.py

Utilities for Cocci FFNN evaluation orchestration:
1) Prepare reduced metadata/FASTA inputs from reactive/non-reactive peptide lists.
2) Compare binary prediction FASTA outputs to peptide-level expected labels.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import pandas as pd


FULLNAME_RE = re.compile(r"^ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")


def read_fasta_records(fasta_path: Path) -> Iterator[Tuple[str, str]]:
    """Yields (header, sequence) records from FASTA."""
    header = None
    seq_lines: List[str] = []
    with fasta_path.open("r", encoding="utf-8") as fasta_f:
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


def write_fasta_records(fasta_path: Path, records: List[Tuple[str, str]]) -> None:
    """Writes FASTA records preserving full headers."""
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as out_f:
        for header, seq in records:
            out_f.write(f">{header}\n{seq}\n")


def parse_protein_id_from_fullname(fullname: str) -> str:
    """Parses protein ID from `ID=... AC=... OXX=...` style fullname."""
    match_ = FULLNAME_RE.match(str(fullname).strip())
    if match_ is None:
        raise ValueError(
            f"Could not parse protein ID from fullname: '{fullname}'")
    return match_.group(1)


def parse_protein_id_from_prediction_header(header: str) -> str:
    """Parses protein ID from prediction FASTA header."""
    header = str(header).strip()
    match_ = FULLNAME_RE.match(header)
    if match_ is not None:
        return match_.group(1)
    # fallback for plain headers
    return header.split()[0]


def build_fullname(protein_id: str, oxx: str) -> str:
    """Builds canonical PV1-style fullname (i.e., ID=<ID> AC=<AC> OXX=<OXX>)."""
    return f"ID={protein_id} AC={protein_id} OXX={oxx}"


def load_code_list(tsv_path: Path) -> List[str]:
    """Loads peptide code names from first TSV column."""
    codes: List[str] = []
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
            codes.append(value)
    return sorted(set(codes))


def _pick_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    """Returns first existing column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


def _available_columns(df: pd.DataFrame, candidates: List[str], label: str) -> List[str]:
    """Returns candidate columns that exist in DataFrame."""
    found = [col for col in candidates if col in df.columns]
    if len(found) == 0:
        raise ValueError(
            f"Could not find {label} columns. Tried: {candidates}")
    return found


def run_prepare(args: argparse.Namespace) -> None:
    """Builds reduced evaluation metadata and FASTA from Cocci source files."""
    data_dir = Path(args.data_dir)
    meta_path = data_dir / args.meta_file
    proteins_fasta_path = data_dir / args.proteins_fasta
    reactive_path = data_dir / args.reactive_file
    nonreactive_path = data_dir / args.nonreactive_file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in [meta_path, proteins_fasta_path, reactive_path, nonreactive_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    meta_df = pd.read_csv(meta_path, sep="\t", dtype=str)
    code_col = _pick_column(
        meta_df, ["CodeName", "Code", "Sequence name"], "code")
    protein_col = _pick_column(
        meta_df,
        ["SequenceAccession", "ProteinID", "ProteinAccession"],
        "protein ID"
    )
    start_cols = _available_columns(
        meta_df,
        ["StartIndex", "AlignStart", "Start"],
        "start index"
    )
    stop_cols = _available_columns(
        meta_df,
        ["StopIndex", "AlignStop", "Stop"],
        "stop index"
    )
    peptide_col = _pick_column(
        meta_df,
        ["PeptideSequence", "Peptide"],
        "peptide sequence"
    )

    reactive_codes = set(load_code_list(reactive_path))
    nonreactive_codes = set(load_code_list(nonreactive_path))
    overlap = sorted(reactive_codes & nonreactive_codes)
    if len(overlap) > 0:
        preview = overlap[:10]
        raise ValueError(
            f"Reactive/non-reactive code overlap detected (first examples): {preview}"
        )

    mode = str(args.mode).strip().lower()
    if mode == "reactive":
        selected_codes = reactive_codes
    elif mode == "nonreactive":
        selected_codes = nonreactive_codes
    elif mode == "combined":
        selected_codes = reactive_codes | nonreactive_codes
    else:
        raise ValueError(
            "--mode must be one of: reactive, nonreactive, combined")

    selected_df = meta_df[meta_df[code_col].isin(selected_codes)].copy()
    if selected_df.empty:
        raise ValueError(
            "Selected peptide subset is empty. Check mode and source files."
        )

    selected_df["CodeName"] = selected_df[code_col].astype(str)
    selected_df["ProteinID"] = selected_df[protein_col].astype(str).str.strip()
    selected_df["Peptide"] = selected_df[peptide_col].astype(str)
    start_numeric = pd.to_numeric(selected_df[start_cols[0]], errors="coerce")
    for col in start_cols[1:]:
        start_numeric = start_numeric.combine_first(
            pd.to_numeric(selected_df[col], errors="coerce")
        )
    stop_numeric = pd.to_numeric(selected_df[stop_cols[0]], errors="coerce")
    for col in stop_cols[1:]:
        stop_numeric = stop_numeric.combine_first(
            pd.to_numeric(selected_df[col], errors="coerce")
        )
    selected_df["AlignStart"] = start_numeric.astype("Int64")
    selected_df["AlignStop"] = stop_numeric.astype("Int64")

    dropped_missing_align_df = selected_df[
        selected_df["AlignStart"].isna() | selected_df["AlignStop"].isna()
    ].copy()
    if not dropped_missing_align_df.empty:
        selected_df = selected_df.drop(dropped_missing_align_df.index).copy()

    if selected_df.empty:
        raise ValueError(
            "No peptides left after dropping rows with missing start/stop alignment."
        )

    selected_df["AlignStart"] = selected_df["AlignStart"].astype(int)
    selected_df["AlignStop"] = selected_df["AlignStop"].astype(int)
    selected_df["PeptideLen"] = selected_df["AlignStop"] - \
        selected_df["AlignStart"]
    if (selected_df["PeptideLen"] <= 0).any():
        bad_rows = selected_df[selected_df["PeptideLen"]
                               <= 0]["CodeName"].head(10)
        raise ValueError(
            f"Found non-positive peptide lengths for codes: {bad_rows.tolist()}"
        )

    label_source: Dict[str, str] = {}
    for code in selected_codes:
        if code in reactive_codes:
            label_source[code] = "reactive"
        elif code in nonreactive_codes:
            label_source[code] = "nonreactive"
        else:
            raise RuntimeError(f"Unexpected label source for code: {code}")

    selected_df["LabelSource"] = selected_df["CodeName"].map(label_source)
    selected_df["Def epitope"] = (
        selected_df["LabelSource"] == "reactive").astype(int)
    selected_df["Uncertain"] = 0
    selected_df["Not epitope"] = (
        selected_df["LabelSource"] == "nonreactive"
    ).astype(int)
    selected_df["FullName"] = selected_df["ProteinID"].map(
        lambda pid: build_fullname(pid, args.oxx)
    )

    proteins_by_id: Dict[str, List[str]] = {}
    for header, seq in read_fasta_records(proteins_fasta_path):
        protein_id = header.split()[0]
        seqs = proteins_by_id.setdefault(protein_id, [])
        if seq not in seqs:
            seqs.append(seq)

    selected_proteins = sorted(set(selected_df["ProteinID"].tolist()))
    missing_proteins = [
        pid for pid in selected_proteins if pid not in proteins_by_id]
    if len(missing_proteins) > 0:
        selected_df = selected_df[
            ~selected_df["ProteinID"].isin(missing_proteins)
        ].copy()

    if selected_df.empty:
        raise ValueError(
            "No peptides left after dropping proteins with missing sequences."
        )

    selected_proteins = sorted(set(selected_df["ProteinID"].tolist()))
    ambiguous_proteins = [
        pid for pid in selected_proteins if len(proteins_by_id.get(pid, [])) > 1
    ]
    if len(ambiguous_proteins) > 0:
        selected_df = selected_df[
            ~selected_df["ProteinID"].isin(ambiguous_proteins)
        ].copy()

    if selected_df.empty:
        raise ValueError(
            "No peptides left after dropping proteins with ambiguous sequences."
        )

    selected_protein_seq: Dict[str, str] = {}
    selected_proteins = sorted(set(selected_df["ProteinID"].tolist()))
    for protein_id in selected_proteins:
        seqs = proteins_by_id[protein_id]
        if len(seqs) > 1:
            raise ValueError(
                f"Ambiguous selected protein ID after filtering: {protein_id}")
        selected_protein_seq[protein_id] = seqs[0]

    invalid_bound_codes: List[str] = []
    for row in selected_df.itertuples(index=False):
        pid = str(row.ProteinID)
        start = int(row.AlignStart)
        stop = int(row.AlignStop)
        seq_len = len(selected_protein_seq[pid])
        if start < 0 or stop > seq_len or stop <= start:
            invalid_bound_codes.append(str(row.CodeName))

    if len(invalid_bound_codes) > 0:
        invalid_set = set(invalid_bound_codes)
        selected_df = selected_df[~selected_df["CodeName"].isin(
            invalid_set)].copy()

    if selected_df.empty:
        raise ValueError(
            "No peptides left after dropping out-of-bounds align windows."
        )

    selected_proteins = sorted(set(selected_df["ProteinID"].tolist()))
    selected_protein_seq = {
        pid: selected_protein_seq[pid] for pid in selected_proteins}

    selected_df = selected_df.sort_values(
        ["ProteinID", "AlignStart", "CodeName"]
    ).reset_index(drop=True)

    out_meta = out_dir / "eval_metadata.tsv"
    out_fasta = out_dir / "eval_proteins.fasta"
    out_summary = out_dir / "prepare_summary.json"

    out_cols = [
        "CodeName",
        "FullName",
        "Peptide",
        "AlignStart",
        "AlignStop",
        "Def epitope",
        "Uncertain",
        "Not epitope",
        "LabelSource",
        "ProteinID",
        "PeptideLen"
    ]
    selected_df.loc[:, out_cols].to_csv(out_meta, sep="\t", index=False)

    fasta_records = [
        (build_fullname(pid, args.oxx), selected_protein_seq[pid])
        for pid in selected_proteins
    ]
    write_fasta_records(out_fasta, fasta_records)

    summary = {
        "mode": mode,
        "data_dir": str(data_dir),
        "meta_path": str(meta_path),
        "proteins_fasta_path": str(proteins_fasta_path),
        "reactive_file": str(reactive_path),
        "nonreactive_file": str(nonreactive_path),
        "output_metadata": str(out_meta),
        "output_fasta": str(out_fasta),
        "n_input_reactive_codes": int(len(reactive_codes)),
        "n_input_nonreactive_codes": int(len(nonreactive_codes)),
        "n_selected_codes": int(len(selected_codes)),
        "n_selected_rows": int(selected_df.shape[0]),
        "n_selected_proteins": int(len(selected_proteins)),
        "n_selected_reactive_rows": int((selected_df["Def epitope"] == 1).sum()),
        "n_selected_nonreactive_rows": int((selected_df["Not epitope"] == 1).sum()),
        "n_dropped_missing_align": int(dropped_missing_align_df.shape[0]),
        "dropped_missing_align_examples": dropped_missing_align_df["CodeName"].head(20).tolist(),
        "n_dropped_missing_proteins": int(len(missing_proteins)),
        "dropped_missing_protein_examples": sorted(missing_proteins)[:20],
        "n_dropped_ambiguous_proteins": int(len(ambiguous_proteins)),
        "dropped_ambiguous_protein_examples": sorted(ambiguous_proteins)[:20],
        "n_dropped_out_of_bounds_align": int(len(set(invalid_bound_codes))),
        "dropped_out_of_bounds_code_examples": sorted(set(invalid_bound_codes))[:20],
        "start_columns_used": start_cols,
        "stop_columns_used": stop_cols,
        "peptide_len_stats": {
            "min": int(selected_df["PeptideLen"].min()),
            "max": int(selected_df["PeptideLen"].max()),
            "mean": float(selected_df["PeptideLen"].mean())
        }
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _safe_mean(values: List[float]) -> float | None:
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


def run_compare(args: argparse.Namespace) -> None:
    """Compares peptide-level predicted 1-counts to expected label 1-counts."""
    pred_fasta = Path(args.prediction_fasta)
    meta_tsv = Path(args.metadata_tsv)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    label_shard_path = Path(args.label_shard) if args.label_shard else None

    if not pred_fasta.exists():
        raise FileNotFoundError(f"Prediction FASTA not found: {pred_fasta}")
    if not meta_tsv.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {meta_tsv}")
    if label_shard_path is not None and not label_shard_path.exists():
        raise FileNotFoundError(f"Label shard not found: {label_shard_path}")

    pred_by_protein: Dict[str, str] = {}
    for header, seq in read_fasta_records(pred_fasta):
        pid = parse_protein_id_from_prediction_header(header)
        seq_clean = str(seq).strip()
        if not set(seq_clean).issubset({"0", "1"}):
            raise ValueError(
                f"Prediction sequence for {pid} contains non-binary characters"
            )
        pred_by_protein[pid] = seq_clean

    meta_df = pd.read_csv(meta_tsv, sep="\t", dtype=str)
    required_cols = [
        "CodeName",
        "FullName",
        "AlignStart",
        "AlignStop",
        "Def epitope",
        "Not epitope"
    ]
    missing_cols = [c for c in required_cols if c not in meta_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"Metadata TSV missing required columns: {missing_cols}")

    label_by_protein = None
    if label_shard_path is not None:
        import torch  # local import to keep prepare mode lightweight

        payload = torch.load(
            label_shard_path, map_location="cpu", weights_only=False)
        labels = payload.get("labels") if isinstance(payload, dict) else None
        if not isinstance(labels, dict):
            raise ValueError(
                f"Label shard must contain dict payload['labels']: {label_shard_path}"
            )
        label_by_protein = labels

    rows_out: List[Dict[str, object]] = []
    missing_prediction_ids: List[str] = []
    missing_label_ids: List[str] = []

    for row in meta_df.to_dict(orient="records"):
        code_name = str(row["CodeName"])
        fullname = str(row["FullName"])
        protein_id = parse_protein_id_from_fullname(fullname)
        start = int(row["AlignStart"])
        stop = int(row["AlignStop"])

        pred_mask = pred_by_protein.get(protein_id)
        if pred_mask is None:
            missing_prediction_ids.append(protein_id)
            continue
        if stop > len(pred_mask) or start < 0 or stop <= start:
            raise ValueError(
                f"Invalid bounds for {code_name}: start={start}, stop={stop}, "
                f"prediction_len={len(pred_mask)}"
            )

        window = pred_mask[start:stop]
        peptide_len = stop - start
        pred_ones = int(window.count("1"))
        pred_zeros = int(peptide_len - pred_ones)

        is_def = int(row["Def epitope"]) == 1
        is_not = int(row["Not epitope"]) == 1
        if is_def and is_not:
            raise ValueError(
                f"Invalid one-hot label for {code_name}: def and not both 1")

        expected_rule_ones = int(peptide_len if is_def else 0)
        label_ones = None
        if label_by_protein is not None:
            labels_tensor = label_by_protein.get(protein_id)
            if labels_tensor is None:
                missing_label_ids.append(protein_id)
            else:
                label_window = labels_tensor[start:stop]
                if label_window.dim() == 2 and label_window.size(1) >= 1:
                    label_ones = int(label_window[:, 0].sum().item())
                elif label_window.dim() == 1:
                    label_ones = int((label_window == 1).sum().item())
                else:
                    raise ValueError(
                        f"Unsupported label tensor shape for protein {protein_id}: "
                        f"{tuple(labels_tensor.shape)}"
                    )

        expected_ones = int(
            label_ones if label_ones is not None else expected_rule_ones)
        delta = int(pred_ones - expected_ones)
        abs_delta = int(abs(delta))
        true_class = "reactive" if is_def else "nonreactive"
        pred_class = "reactive" if pred_ones > 0 else "nonreactive"

        rows_out.append(
            {
                "CodeName": code_name,
                "ProteinID": protein_id,
                "AlignStart": start,
                "AlignStop": stop,
                "PeptideLen": peptide_len,
                "TrueClass": true_class,
                "PredClass": pred_class,
                "PredOnes": pred_ones,
                "PredZeros": pred_zeros,
                "ExpectedOnesRule": expected_rule_ones,
                "LabelOnes": label_ones,
                "ExpectedOnes": expected_ones,
                "DeltaPredMinusExpected": delta,
                "AbsDelta": abs_delta,
                "ExactMatch": int(pred_ones == expected_ones)
            }
        )

    if len(rows_out) == 0:
        raise ValueError("No comparable rows produced in peptide comparison")

    out_df = pd.DataFrame(rows_out).sort_values(
        ["ProteinID", "AlignStart", "CodeName"]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    reactive_mask = out_df["TrueClass"] == "reactive"
    nonreactive_mask = out_df["TrueClass"] == "nonreactive"

    tp = int(((out_df["TrueClass"] == "reactive") & (
        out_df["PredClass"] == "reactive")).sum())
    fp = int(((out_df["TrueClass"] == "nonreactive") &
             (out_df["PredClass"] == "reactive")).sum())
    tn = int(((out_df["TrueClass"] == "nonreactive") & (
        out_df["PredClass"] == "nonreactive")).sum())
    fn = int(((out_df["TrueClass"] == "reactive") & (
        out_df["PredClass"] == "nonreactive")).sum())

    summary = {
        "prediction_fasta": str(pred_fasta),
        "metadata_tsv": str(meta_tsv),
        "label_shard": str(label_shard_path) if label_shard_path is not None else None,
        "output_csv": str(out_csv),
        "n_rows_compared": int(out_df.shape[0]),
        "n_reactive": int(reactive_mask.sum()),
        "n_nonreactive": int(nonreactive_mask.sum()),
        "n_missing_prediction_ids": int(len(set(missing_prediction_ids))),
        "missing_prediction_id_examples": sorted(set(missing_prediction_ids))[:20],
        "n_missing_label_ids": int(len(set(missing_label_ids))),
        "missing_label_id_examples": sorted(set(missing_label_ids))[:20],
        "exact_match_rate": float(out_df["ExactMatch"].mean()),
        "mean_abs_delta": _safe_mean(out_df["AbsDelta"].astype(float).tolist()),
        "mean_pred_ones_overall": _safe_mean(out_df["PredOnes"].astype(float).tolist()),
        "mean_pred_ones_reactive": _safe_mean(
            out_df.loc[reactive_mask, "PredOnes"].astype(float).tolist()
        ),
        "mean_pred_ones_nonreactive": _safe_mean(
            out_df.loc[nonreactive_mask, "PredOnes"].astype(float).tolist()
        ),
        "confusion_peptide_any_positive": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1))
        }
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI parser for prepare/compare subcommands."""
    parser = argparse.ArgumentParser(
        description="Cocci evaluation helper for subset preparation and peptide-level comparison."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare", help="Prepare reduced metadata/FASTA for eval.")
    prepare.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing Cocci dataset source files."
    )
    prepare.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write prepared metadata/FASTA outputs."
    )
    prepare.add_argument(
        "--mode",
        type=str,
        choices=["reactive", "nonreactive", "combined"],
        default="combined",
        help="Subset selection mode."
    )
    prepare.add_argument(
        "--meta-file",
        type=str,
        default="CWP_meta_wClusterInfo_StartStopAdded_50ClustAlignAdded.tsv",
        help="Metadata TSV filename under --data-dir."
    )
    prepare.add_argument(
        "--proteins-fasta",
        type=str,
        default="Coccidioides_combo_FinalDesign_used.faa",
        help="Protein FASTA filename under --data-dir."
    )
    prepare.add_argument(
        "--reactive-file",
        type=str,
        default="CWP_reactive_Z20N4.tsv",
        help="Reactive peptide list filename under --data-dir."
    )
    prepare.add_argument(
        "--nonreactive-file",
        type=str,
        default="CWP_non-reactive_Z10N0.tsv",
        help="Non-reactive peptide list filename under --data-dir."
    )
    prepare.add_argument(
        "--oxx",
        type=str,
        default="0,0,0,0",
        help="OXX token used when writing normalized FullName headers."
    )

    compare = sub.add_parser(
        "compare", help="Compare predicted peptide 1-counts to expected labels."
    )
    compare.add_argument(
        "--prediction-fasta",
        type=Path,
        required=True,
        help="Binary prediction FASTA output from prediction CLI."
    )
    compare.add_argument(
        "--metadata-tsv",
        type=Path,
        required=True,
        help="Prepared metadata TSV generated by prepare stage."
    )
    compare.add_argument(
        "--label-shard",
        type=Path,
        default=None,
        help="Optional label shard PT file to compute expected ones directly from built labels."
    )
    compare.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Per-peptide comparison CSV output path."
    )
    compare.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Summary JSON output path."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
