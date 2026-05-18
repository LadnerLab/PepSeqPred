"""train_ffnn_cli.py

Handles end-to-end training and evaluation of a PepSeqPredFFNN with the goal to predict the locations
of antibody epitopes within a protein sequence downstream. The resulting model will make binary predictions:
definite epitope or not epitope, it handles residues labeled uncertain through a masking process.

The training module utilizes DistributedDataParallel to train the model distributed across multiple GPUs.
It is highly recommended you train this model using an HPC, training locally is often impossible due to time 
constraints and limited compute. For context, our models were often trained using 4 A100 GPUs and take anywhere
from 3 to 5 hours.

Usage
-----
>>> # from scripts/hpc/trainffnn.sh (see shell script for CLI config)
>>> sbatch trainffnn.sh /path/to/emb_shard_dir0 ... /path/to/emb_shard_dirN -- \\ 
                        /path/to/label_shard0.pt ... /path/to/label_shardN.pt
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Mapping, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.write import append_csv_row
from pepseqpred.core.data.proteindataset import ProteinDataset, pad_collate
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.train.trainer import (
    Trainer,
    TrainerConfig,
    ValidationCurveArtifactConfig
)
from pepseqpred.core.train.ddp import init_ddp
from pepseqpred.core.train.split import (
    split_ids,
    split_ids_grouped,
    build_kfold_splits,
    build_grouped_kfold_splits,
    partition_ids_weighted,
    sort_ids_for_locality,
    shuffle_ids_by_group
)
from pepseqpred.core.train.weights import pos_weight_from_label_shards
from pepseqpred.core.train.embedding import infer_emb_dim
from pepseqpred.core.train.seed import set_all_seeds
from pepseqpred.core.io.read import parse_int_csv, parse_float_csv


def summarize_numeric(series: pd.Series) -> Dict[str, Any]:
    """
    Generates a statistical summary given an input series.

    Parameters
    ----------
        series : pd.Series
            The input series which could be one or more metrics from training/eval.

    Returns
    -------
        Dict[str, Any]
            A dictionary containing the count, mean, standard deviation, minimum, and 
            maximum summary statistics for the input series.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna()
    vals = vals[vals.map(lambda x: math.isfinite(float(x)))]
    if vals.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max())
    }


def _sanitize_for_json(value: Any) -> Any:
    """Recursive function to ensure all values are JSON-sanitized."""
    # recurse into dictionary and convert to float or None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    # recurse into list and convert to float or None
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    # convert current value to float or None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _finite_or_none(value: Any) -> float | None:
    """Tries to convert number to float if finite, otherwise returns None."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _parse_plot_formats(raw: str) -> Tuple[str, ...]:
    """Parses comma-separated plot file formats."""
    tokens = [t.strip().lower() for t in str(raw).split(",")]
    formats = tuple(t for t in tokens if len(t) > 0)
    if len(formats) < 1:
        raise ValueError("--val-plot-formats must include at least one format")
    allowed = {"png", "svg", "pdf"}
    bad = [fmt for fmt in formats if fmt not in allowed]
    if len(bad) > 0:
        raise ValueError(
            f"Unsupported --val-plot-formats values={bad}; allowed={sorted(allowed)}"
        )
    return formats


def _as_optional_int(value: Any) -> int | None:
    """Parses optional integer-like value."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _legacy_train_mode_label(n_folds: int) -> str:
    """Resolves legacy-compatible train_mode label from unified n_folds input."""
    return "ensemble-kfold" if int(n_folds) > 1 else "seeded"


_LEGACY_TRAIN_FLAGS: Dict[str, str] = {
    "--train-mode": (
        "Removed. Use --n-folds 1 for holdout runs, or --n-folds K (K>1) for K-fold ensemble runs."
    ),
    "--fold-seed": (
        "Removed. Use --split-seeds (paired with --train-seeds) to control per-set fold assignment seeds."
    ),
    "--ensemble-train-seeds": (
        "Removed. Use --train-seeds as per-set training seeds."
    ),
}


def _match_legacy_train_flags(tokens: Sequence[str]) -> List[str]:
    """Finds removed legacy train-mode flags in parser unknown-token output."""
    found: List[str] = []
    for token in tokens:
        token_s = str(token).strip()
        for flag in _LEGACY_TRAIN_FLAGS:
            if token_s == flag or token_s.startswith(f"{flag}="):
                if flag not in found:
                    found.append(flag)
    return found


def _resolve_pr_zoom_limits(
    fold_evaluations: Sequence[Mapping[str, Any]],
    baseline_y: float | None,
    recall_xmax: float = 0.20,
) -> Tuple[float, float]:
    """Computes stable PR-plot zoom y-limits for highly imbalanced datasets."""
    precision_values: List[float] = []
    xmax = float(recall_xmax)
    if not math.isfinite(xmax) or xmax <= 0.0:
        xmax = 0.20
    for fold_eval in fold_evaluations:
        pr_obj = fold_eval.get("pr_curve")
        if not isinstance(pr_obj, Mapping):
            continue
        if not bool(pr_obj.get("available", False)):
            continue
        recalls = pr_obj.get("recall")
        precisions = pr_obj.get("precision")
        if not isinstance(recalls, list) or not isinstance(precisions, list):
            continue
        if len(recalls) != len(precisions):
            continue
        for recall, precision in zip(recalls, precisions):
            try:
                recall_f = float(recall)
                precision_f = float(precision)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(recall_f) or not math.isfinite(precision_f):
                continue
            if 0.0 < recall_f <= xmax:
                precision_values.append(precision_f)

    if len(precision_values) < 1:
        y_top = 0.05
    else:
        q95 = float(np.quantile(np.asarray(precision_values), 0.95))
        y_top = max(0.02, q95 * 1.15)
    if baseline_y is not None and math.isfinite(float(baseline_y)):
        y_top = max(y_top, float(baseline_y) * 6.0)
    y_top = min(1.0, max(0.02, y_top))
    return (0.0, y_top)


def _resolve_pr_zoom_xmax(
    fold_evaluations: Sequence[Mapping[str, Any]],
    preferred_xmax: float = 0.20,
) -> float:
    """Chooses a PR zoom x-maximum with enough sampled points to draw visible lines."""
    candidate_maxes = [float(preferred_xmax), 0.30, 0.50, 1.00]
    valid_maxes = [x for x in candidate_maxes if math.isfinite(x) and x > 0.0]
    if len(valid_maxes) < 1:
        valid_maxes = [0.20, 0.30, 0.50, 1.00]

    recalls: List[float] = []
    for fold_eval in fold_evaluations:
        pr_obj = fold_eval.get("pr_curve")
        if not isinstance(pr_obj, Mapping):
            continue
        if not bool(pr_obj.get("available", False)):
            continue
        rec = pr_obj.get("recall")
        if not isinstance(rec, list):
            continue
        for raw in rec:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                recalls.append(value)

    if len(recalls) < 1:
        return float(valid_maxes[0])

    for xmax in valid_maxes:
        n_zoom = sum(1 for r in recalls if 0.0 <= r <= xmax)
        if n_zoom >= 2:
            return float(xmax)
    return 1.0


def _plot_fold_curves(
    fold_evaluations: Sequence[Mapping[str, Any]],
    plot_path_base: Path,
    formats: Sequence[str],
    curve_key: str,
    title: str,
    x_label: str,
    y_label: str,
    x_key: str,
    y_key: str,
    metric_key: str,
    chance_line: bool,
    baseline_y: float | None = None,
    secondary_metric_key: str | None = None,
    metric_label: str | None = None,
    secondary_metric_label: str | None = None,
    x_limits: Tuple[float, float] | None = None,
    y_limits: Tuple[float, float] | None = None,
    legend_loc: str = "lower right",
) -> List[str]:
    """Writes a fold-only curve panel for ROC or PR."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install matplotlib to use --save-val-curves."
        ) from e

    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=150)
    plotted = 0
    for fold_eval in fold_evaluations:
        curve_obj = fold_eval.get(curve_key)
        if not isinstance(curve_obj, Mapping):
            continue
        if not bool(curve_obj.get("available", False)):
            continue
        x_vals = curve_obj.get(x_key)
        y_vals = curve_obj.get(y_key)
        if not isinstance(x_vals, list) or not isinstance(y_vals, list):
            continue
        if len(x_vals) < 2 or len(y_vals) < 2:
            continue
        fold_index = _as_optional_int(fold_eval.get("fold_index"))
        fold_label = int(fold_index) if fold_index is not None else (plotted + 1)
        metrics_obj = fold_eval.get("metrics")
        metric_value = float("nan")
        if isinstance(metrics_obj, Mapping):
            raw_metric = metrics_obj.get(metric_key, float("nan"))
            try:
                metric_value = float(raw_metric)
            except (TypeError, ValueError):
                metric_value = float("nan")
        if not math.isfinite(metric_value):
            if curve_key == "pr_curve" and metric_key == "pr_auc":
                raw_ap = curve_obj.get("ap", float("nan"))
                try:
                    metric_value = float(raw_ap)
                except (TypeError, ValueError):
                    metric_value = float("nan")
        primary_label = metric_label if metric_label is not None else metric_key.upper()
        primary_metric_text = (
            f"{primary_label}: {metric_value:.3f}"
            if math.isfinite(metric_value)
            else f"{primary_label}: nan"
        )
        secondary_metric_text = None
        if secondary_metric_key is not None and isinstance(metrics_obj, Mapping):
            raw_secondary = metrics_obj.get(secondary_metric_key, float("nan"))
            try:
                secondary_value = float(raw_secondary)
            except (TypeError, ValueError):
                secondary_value = float("nan")
            if (
                not math.isfinite(secondary_value)
                and curve_key == "pr_curve"
                and secondary_metric_key == "pr_auc_trapz"
            ):
                raw_trapz = curve_obj.get("auprc_trapz", float("nan"))
                try:
                    secondary_value = float(raw_trapz)
                except (TypeError, ValueError):
                    secondary_value = float("nan")
            secondary_label = (
                secondary_metric_label
                if secondary_metric_label is not None
                else secondary_metric_key.upper()
            )
            if math.isfinite(secondary_value):
                secondary_metric_text = f"{secondary_label}: {secondary_value:.3f}"
            else:
                secondary_metric_text = f"{secondary_label}: nan"
        label_parts = [f"Fold {fold_label}", primary_metric_text]
        if secondary_metric_text is not None:
            label_parts.append(secondary_metric_text)
        ax.plot(
            x_vals,
            y_vals,
            linewidth=1.8,
            alpha=0.95,
            label=" | ".join(label_parts),
        )
        plotted += 1

    if chance_line:
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.0,
            color="#C77DBB",
            label="Chance",
        )
    if baseline_y is not None and math.isfinite(float(baseline_y)):
        y_val = float(baseline_y)
        ax.plot(
            [0.0, 1.0],
            [y_val, y_val],
            linestyle="--",
            linewidth=1.0,
            color="#888888",
            label=f"Prevalence baseline: {y_val:.4f}",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_limits is None:
        ax.set_xlim(0.0, 1.0)
    else:
        ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
    if y_limits is None:
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    ax.grid(alpha=0.2)
    if plotted > 0:
        ax.legend(loc=legend_loc, frameon=False, fontsize=9)
    fig.tight_layout()

    saved_paths: List[str] = []
    plot_path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = plot_path_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(str(out_path))
    plt.close(fig)
    return saved_paths


def _build_set_best_fold_curve_payload(
    run_rows: Sequence[Mapping[str, Any]],
    set_index: int,
    val_curve_subdir: str,
) -> Dict[str, Any]:
    """Builds fold-level best-epoch curve payload for one ensemble set."""
    fold_map: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []

    rows_in_set = sorted(
        (
            row
            for row in run_rows
            if int(_as_optional_int(row.get("EnsembleSetIndex")) or 1) == int(set_index)
        ),
        key=lambda row: int(_as_optional_int(row.get("RunIndex")) or int(1e9)),
    )

    for row in rows_in_set:
        fold_idx = _as_optional_int(row.get("FoldIndex"))
        if fold_idx is None:
            continue
        if fold_idx in fold_map:
            warnings.append(
                f"set_index={set_index} fold_index={fold_idx} has multiple runs; keeping first by RunIndex."
            )
            continue

        run_save_dir_raw = row.get("RunSaveDir")
        best_epoch = _as_optional_int(row.get("BestEpoch"))
        status = str(row.get("Status", "")).strip().upper()
        run_index = _as_optional_int(row.get("RunIndex"))

        if status != "OK":
            warnings.append(
                f"Skipping fold {fold_idx} run_index={run_index}: status={status or 'UNKNOWN'}"
            )
            continue
        if run_save_dir_raw is None:
            warnings.append(
                f"Skipping fold {fold_idx} run_index={run_index}: missing RunSaveDir."
            )
            continue
        if best_epoch is None or best_epoch < 0:
            warnings.append(
                f"Skipping fold {fold_idx} run_index={run_index}: invalid BestEpoch={row.get('BestEpoch')}."
            )
            continue

        run_save_dir = Path(str(run_save_dir_raw))
        curve_json = run_save_dir / str(val_curve_subdir) / f"epoch_{best_epoch:04d}_curves.json"
        if not curve_json.exists():
            warnings.append(
                f"Skipping fold {fold_idx} run_index={run_index}: missing curve JSON {curve_json}."
            )
            continue

        try:
            curve_payload = json.loads(curve_json.read_text(encoding="utf-8"))
        except Exception as e:
            warnings.append(
                f"Skipping fold {fold_idx} run_index={run_index}: could not read {curve_json} ({e})."
            )
            continue
        if not isinstance(curve_payload, Mapping):
            warnings.append(
                f"Skipping fold {fold_idx} run_index={run_index}: curve JSON is not an object ({curve_json})."
            )
            continue

        metrics_obj = curve_payload.get("eval_metrics")
        roc_obj = curve_payload.get("roc_curve")
        pr_obj = curve_payload.get("pr_curve")
        if not isinstance(metrics_obj, Mapping):
            metrics_obj = {}
        if not isinstance(roc_obj, Mapping):
            roc_obj = {"available": False, "reason": "missing-roc-curve", "fpr": [], "tpr": []}
        if not isinstance(pr_obj, Mapping):
            pr_obj = {
                "available": False,
                "reason": "missing-pr-curve",
                "recall": [],
                "precision": [],
                "baseline_positive_rate": None,
                "ap": None,
                "auprc_trapz": None,
            }

        fold_map[int(fold_idx)] = {
            "fold_index": int(fold_idx),
            "run_index": int(run_index) if run_index is not None else None,
            "best_epoch": int(best_epoch),
            "run_save_dir": str(run_save_dir),
            "curve_json": str(curve_json),
            "metrics": dict(metrics_obj),
            "roc_curve": dict(roc_obj),
            "pr_curve": dict(pr_obj),
        }

    fold_entries = [fold_map[k] for k in sorted(fold_map.keys())]
    baseline_values: List[float] = []
    for fold_eval in fold_entries:
        pr_obj = fold_eval.get("pr_curve")
        if not isinstance(pr_obj, Mapping):
            continue
        baseline_raw = pr_obj.get("baseline_positive_rate")
        baseline_value = _finite_or_none(baseline_raw)
        if baseline_value is not None:
            baseline_values.append(float(baseline_value))
    pr_baseline = (
        float(sum(baseline_values) / len(baseline_values))
        if len(baseline_values) > 0
        else None
    )
    return {
        "set_index": int(set_index),
        "n_folds_with_curves": int(len(fold_entries)),
        "folds": fold_entries,
        "pr_baseline_mean": pr_baseline,
        "warnings": warnings,
    }


def _write_ensemble_validation_curve_artifacts(
    run_rows: Sequence[Mapping[str, Any]],
    set_index: int,
    output_dir: Path,
    plot_formats: Sequence[str],
    val_curve_subdir: str = "validation_curves",
) -> Dict[str, Any]:
    """Writes fold-level best-epoch ROC/PR plots and sidecar JSON for one set."""
    payload = _build_set_best_fold_curve_payload(
        run_rows=run_rows,
        set_index=set_index,
        val_curve_subdir=val_curve_subdir,
    )
    fold_entries = payload["folds"]
    baseline_y = payload.get("pr_baseline_mean")
    baseline_float = (
        float(baseline_y)
        if baseline_y is not None and math.isfinite(float(baseline_y))
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_status = "ok"
    plot_outputs = {
        "roc_auc_folds": [],
        "pr_auc_folds": [],
        "pr_auc_folds_zoom": [],
    }
    if len(fold_entries) > 0:
        try:
            pr_zoom_xmax = _resolve_pr_zoom_xmax(
                fold_evaluations=fold_entries,
                preferred_xmax=0.20,
            )
            plot_outputs["roc_auc_folds"] = _plot_fold_curves(
                fold_evaluations=fold_entries,
                plot_path_base=output_dir / "roc_auc_folds",
                formats=plot_formats,
                curve_key="roc_curve",
                title="Validation ROC Curves by Fold (Best Epoch)",
                x_label="FPR",
                y_label="TPR",
                x_key="fpr",
                y_key="tpr",
                metric_key="auc",
                chance_line=True,
                metric_label="AUC",
            )
            plot_outputs["pr_auc_folds"] = _plot_fold_curves(
                fold_evaluations=fold_entries,
                plot_path_base=output_dir / "pr_auc_folds",
                formats=plot_formats,
                curve_key="pr_curve",
                title="Validation PR Curves by Fold (Best Epoch)",
                x_label="Recall",
                y_label="Precision",
                x_key="recall",
                y_key="precision",
                metric_key="pr_auc",
                chance_line=False,
                secondary_metric_key="pr_auc_trapz",
                metric_label="AP",
                secondary_metric_label="AUPRC(trapz)",
                baseline_y=baseline_float,
            )
            plot_outputs["pr_auc_folds_zoom"] = _plot_fold_curves(
                fold_evaluations=fold_entries,
                plot_path_base=output_dir / "pr_auc_folds_zoom",
                formats=plot_formats,
                curve_key="pr_curve",
                title="Validation PR Curves by Fold (Best Epoch, Zoomed)",
                x_label="Recall",
                y_label="Precision",
                x_key="recall",
                y_key="precision",
                metric_key="pr_auc",
                chance_line=False,
                secondary_metric_key="pr_auc_trapz",
                metric_label="AP",
                secondary_metric_label="AUPRC(trapz)",
                baseline_y=baseline_float,
                x_limits=(0.0, float(pr_zoom_xmax)),
                y_limits=_resolve_pr_zoom_limits(
                    fold_evaluations=fold_entries,
                    baseline_y=baseline_float,
                    recall_xmax=float(pr_zoom_xmax),
                ),
                legend_loc="upper right",
            )
        except RuntimeError:
            plot_status = "matplotlib_unavailable"
    else:
        plot_status = "no_fold_curves"

    out_payload = {
        "set_index": int(set_index),
        "plot_status": str(plot_status),
        "plot_formats": [str(fmt) for fmt in plot_formats],
        "plot_dir": str(output_dir),
        "n_folds_with_curves": int(payload["n_folds_with_curves"]),
        "pr_baseline_mean": payload.get("pr_baseline_mean"),
        "warnings": list(payload["warnings"]),
        "plot_outputs": plot_outputs,
    }
    out_payload["folds"] = list(fold_entries)
    out_path = output_dir / "ensemble_fold_validation_curves.json"
    out_path.write_text(
        json.dumps(_sanitize_for_json(out_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "set_index": int(set_index),
        "plot_status": str(plot_status),
        "plot_formats": [str(fmt) for fmt in plot_formats],
        "plot_dir": str(output_dir),
        "artifact_json": str(out_path),
        "n_folds_with_curves": int(payload["n_folds_with_curves"]),
        "pr_baseline_mean": payload.get("pr_baseline_mean"),
        "warnings": list(payload["warnings"]),
        "plot_outputs": plot_outputs,
    }


@dataclass
class RunPlan:
    """Execution plan for one train/validation run."""
    run_index: int
    train_mode: str
    split_seed: int
    train_seed: int
    train_ids_all: List[str]
    val_ids_all: List[str]
    save_dir_name: str
    fold_index: int | None = None
    n_folds: int | None = None
    ensemble_set_index: int | None = None
    ensemble_set_split_seed: int | None = None
    ensemble_set_train_seed: int | None = None
    ensemble_set_dir_name: str | None = None


def _check_family_leakage(train_ids: List[str], val_ids: List[str], family_groups: Dict[str, str]) -> None:
    """Raises RuntimeError when any family appears in both train and validation IDs."""
    train_families = {family_groups[pid] for pid in train_ids}
    val_families = {family_groups[pid] for pid in val_ids}
    overlap = train_families & val_families
    if overlap:
        raise RuntimeError(
            f"Family leakage detected: n_overlap={len(overlap)}"
        )


def _build_run_plans(
    args: argparse.Namespace,
    protein_ids: List[str],
    family_groups: Dict[str, str]
) -> Tuple[List[RunPlan], Dict[str, Any]]:
    """Builds run plans from unified seed lists and n-folds configuration."""
    if len(protein_ids) == 0:
        raise ValueError("No proteins found to train on")

    n_folds = int(args.n_folds)
    if n_folds < 1:
        raise ValueError("--n-folds must be >= 1")

    if args.split_seeds is None and args.train_seeds is None:
        split_seeds = [int(args.seed)]
        train_seeds = [int(args.seed)]
    elif args.split_seeds is None or args.train_seeds is None:
        raise ValueError(
            "Provide both --split-seeds and --train-seeds together"
        )
    else:
        split_seeds = parse_int_csv(args.split_seeds, "--split-seeds")
        train_seeds = parse_int_csv(args.train_seeds, "--train-seeds")

    if len(split_seeds) != len(train_seeds):
        raise ValueError(
            "--split-seeds and --train-seeds must be the same length"
        )

    n_sets = len(split_seeds)
    train_mode = _legacy_train_mode_label(n_folds)

    if n_folds == 1:
        run_plans: List[RunPlan] = []
        for run_index, (split_seed, train_seed) in enumerate(zip(split_seeds, train_seeds), start=1):
            if args.split_type == "id-family":
                train_ids_all, val_ids_all = split_ids_grouped(
                    protein_ids, args.val_frac, split_seed, family_groups
                )
                _check_family_leakage(train_ids_all, val_ids_all, family_groups)
            else:
                train_ids_all, val_ids_all = split_ids(
                    protein_ids, args.val_frac, split_seed
                )

            if len(train_ids_all) == 0:
                raise ValueError("Global split produced 0 train IDs")

            run_plans.append(
                RunPlan(
                    run_index=run_index,
                    train_mode=train_mode,
                    split_seed=int(split_seed),
                    train_seed=int(train_seed),
                    train_ids_all=list(train_ids_all),
                    val_ids_all=list(val_ids_all),
                    save_dir_name=f"run_{run_index:03d}_split_{int(split_seed)}_train_{int(train_seed)}",
                    n_folds=1
                )
            )

        return run_plans, {
            "split_seeds": [int(x) for x in split_seeds],
            "train_seeds": [int(x) for x in train_seeds],
            "n_folds": 1,
            "n_sets": int(n_sets),
            "train_mode": train_mode
        }

    run_plans = []
    global_run_index = 1
    for set_index, (set_split_seed, set_train_seed) in enumerate(
        zip(split_seeds, train_seeds),
        start=1
    ):
        if args.split_type == "id-family":
            fold_splits = build_grouped_kfold_splits(
                protein_ids, n_folds=n_folds, seed=int(set_split_seed), groups=family_groups
            )
        else:
            fold_splits = build_kfold_splits(
                protein_ids, n_folds=n_folds, seed=int(set_split_seed)
            )

        if n_sets > 1:
            set_dir_name = f"set_{set_index:03d}_split_{int(set_split_seed)}_train_{int(set_train_seed)}"
        else:
            set_dir_name = None

        for fold_index, (train_ids_all, val_ids_all) in enumerate(fold_splits, start=1):
            if args.split_type == "id-family":
                _check_family_leakage(
                    train_ids_all, val_ids_all, family_groups)

            fold_train_seed = int(set_train_seed)

            if set_dir_name is None:
                save_dir_name = (
                    f"fold_{fold_index:02d}_split_{int(set_split_seed)}_train_{int(fold_train_seed)}")
            else:
                save_dir_name = f"{set_dir_name}/fold_{fold_index:02d}"

            run_plans.append(
                RunPlan(
                    run_index=global_run_index,
                    train_mode=train_mode,
                    split_seed=int(set_split_seed),
                    train_seed=int(fold_train_seed),
                    train_ids_all=list(train_ids_all),
                    val_ids_all=list(val_ids_all),
                    save_dir_name=save_dir_name,
                    fold_index=fold_index,
                    n_folds=n_folds,
                    ensemble_set_index=set_index,
                    ensemble_set_split_seed=int(set_split_seed),
                    ensemble_set_train_seed=int(set_train_seed),
                    ensemble_set_dir_name=set_dir_name
                )
            )
            global_run_index += 1

    return run_plans, {
        "split_seeds": [int(x) for x in split_seeds],
        "train_seeds": [int(x) for x in train_seeds],
        "n_folds": int(n_folds),
        "n_sets": int(n_sets),
        "ensemble_seed_mode": "set-paired",
        "train_mode": train_mode
    }


def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Train FFNN program."""
    parser = argparse.ArgumentParser(
        description="Train PepSeqPred FFNN on protein ESM-2 embeddings for binary residue-level epitope prediction.")
    parser.add_argument("--embedding-dirs",
                        nargs="+",
                        required=True,
                        type=Path,
                        help="One or more directories containing per-protein embeddings")
    parser.add_argument("--label-shards",
                        nargs="+",
                        required=True,
                        type=Path,
                        help="One or more label shard .pt files containing a 'labels' dictionary")
    parser.add_argument("--hidden-sizes",
                        action="store",
                        dest="hidden_sizes",
                        type=str,
                        default="150,120,45",
                        help="The hidden layer sizes")
    parser.add_argument("--dropouts",
                        action="store",
                        dest="dropouts",
                        type=str,
                        default="0.1,0.1,0.1",
                        help="The dropout rates, count must match number of hidden layers")
    parser.add_argument("--use-layer-norm",
                        action="store_true",
                        dest="use_layer_norm",
                        help="If set, layer normalization is applied")
    parser.add_argument("--use-residual",
                        action="store_true",
                        dest="use_residual",
                        help="If set, residuals are used in feed-forward calculation")
    parser.add_argument("--epochs",
                        action="store",
                        dest="epochs",
                        type=int,
                        default=10,
                        help="Number of epochs (training cycles)")
    parser.add_argument("--seed",
                        action="store",
                        dest="seed",
                        type=int,
                        default=42,
                        help="Seed for random split reproducibility")
    parser.add_argument("--batch-size",
                        action="store",
                        dest="batch_size",
                        type=int,
                        default=64,
                        help="Training batch size per epoch")
    parser.add_argument("--lr",
                        action="store",
                        dest="lr",
                        type=float,
                        default=1e-3,
                        help="Model learning rate")
    parser.add_argument("--wd",
                        action="store",
                        dest="weight_decay",
                        type=float,
                        default=0.0,
                        help="Model training weight decay to prevent overfitting by shrinking model weights during training")
    parser.add_argument("--pos-weight",
                        dest="pos_weight",
                        action="store",
                        type=float,
                        default=None,
                        help="Optionally include a pre-calculated postive class weight")
    parser.add_argument("--save-path",
                        action="store",
                        dest="save_path",
                        type=Path,
                        default=Path("checkpoints/ffnn_v1"),
                        help="Path to save the best model to")
    parser.add_argument("--val-frac",
                        action="store",
                        dest="val_frac",
                        type=float,
                        default=0.2,
                        help="Fraction of input data used for validation")
    parser.add_argument("--subset",
                        action="store",
                        dest="subset",
                        type=int,
                        default=0,
                        help="If > 0, use only first N proteins from dataset")
    parser.add_argument("--split-type",
                        type=str,
                        default="id-family",
                        choices=["id", "id-family"],
                        help="Data partition type, use ID only or ID and taxonomic family.")
    parser.add_argument("--num-workers",
                        action="store",
                        dest="num_workers",
                        type=int,
                        default=0,
                        help="Number of worker threads for data loading")
    parser.add_argument("--window-size",
                        action="store",
                        dest="window_size",
                        type=int,
                        default=1000,
                        help="Window size for long protein sequences (<= 0 to disable)")
    parser.add_argument("--stride",
                        action="store",
                        dest="stride",
                        type=int,
                        default=900,
                        help="Stride between windows for long proteins")
    parser.add_argument("--no-collapse-labels",
                        dest="collapse_labels",
                        action="store_false",
                        help="Keep full label vectors when labels are (L, 3)")
    parser.add_argument("--no-pad-last-window",
                        dest="pad_last_window",
                        action="store_false",
                        help="Disable padding of final short window")
    parser.add_argument("--no-cache-label-shard",
                        dest="cache_current_label_shard",
                        action="store_false",
                        help="Reload label shard for each protein instead of caching")
    parser.add_argument("--no-drop-label-after-use",
                        dest="drop_label_after_use",
                        action="store_false",
                        help="Keep labels in memory after each protein is processed")
    parser.add_argument("--label-cache-mode",
                        dest="label_cache_mode",
                        action="store",
                        type=str,
                        default="current",
                        choices=["current", "all"],
                        help="Caching policy for label shard per worker")
    parser.add_argument("--split-seeds",
                        type=str,
                        default=None,
                        help="CSV split seeds (e.g., 11,22,33). Each split seed defines one run set.")
    parser.add_argument("--train-seeds",
                        type=str,
                        default=None,
                        help="CSV train seeds (e.g., 44,55,66). Each train seed pairs with one split seed by index.")
    parser.add_argument("--n-folds",
                        type=int,
                        default=1,
                        help="Number of folds per set. Use 1 for holdout split mode and >1 for K-fold ensemble mode.")
    parser.add_argument("--best-model-metric",
                        type=str,
                        default="loss",
                        choices=["loss", "precision", "recall",
                                 "f1", "mcc", "auc", "auc10", "pr_auc", "res_balanced_acc"],
                        help="Metric used to choose the best model checkpoint per run")
    parser.add_argument("--results-csv",
                        type=Path,
                        default=None,
                        help="Optional CSV output path for per-run results")
    parser.add_argument("--save-val-curves",
                        action="store_true",
                        dest="save_val_curves",
                        default=False,
                        help=(
                            "If set, save per-epoch validation ROC/PR curve data and plots. "
                            "When --n-folds > 1, also writes set-level fold consistency ROC/PR plots "
                            "using each fold's best epoch."
                        ))
    parser.add_argument("--val-curve-max-points",
                        action="store",
                        dest="val_curve_max_points",
                        type=int,
                        default=2048,
                        help="Maximum number of points saved per validation ROC/PR curve.")
    parser.add_argument("--val-plot-formats",
                        action="store",
                        dest="val_plot_formats",
                        type=str,
                        default="png",
                        help="Comma-separated plot formats for validation curves (png,svg,pdf).")
    parser.add_argument("--ensemble-manifest",
                        type=Path,
                        default=None,
                        help="Optional JSON output path for ensemble manifest (written when --n-folds > 1).")

    args, unknown = parser.parse_known_args()
    legacy_flags = _match_legacy_train_flags(unknown)
    if len(legacy_flags) > 0:
        details = " ".join(
            f"{flag}: {_LEGACY_TRAIN_FLAGS[flag]}"
            for flag in legacy_flags
        )
        raise ValueError(
            "Legacy train-mode flags are no longer supported. "
            + details
        )
    if len(unknown) > 0:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    logger = setup_logger(json_lines=True,
                          json_indent=2,
                          name="train_ffnn_cli")

    ddp = init_ddp()
    rank = ddp["rank"] if ddp is not None else 0
    world_size = ddp["world_size"] if ddp is not None else 1

    # log DDP info
    logger.info("ddp_init",
                extra={"extra": {
                    "ddp_enabled": ddp is not None,
                    "world_size": world_size,
                    "rank": rank,
                    "local_rank": ddp["local_rank"] if ddp is not None else 0
                }})

    # disable logging for other ranks when DDP enabled
    if ddp is not None and rank != 0:
        logger.disabled = True

    val_curve_artifacts = None
    if args.save_val_curves:
        if args.val_curve_max_points < 2:
            raise ValueError("--val-curve-max-points must be >= 2")
        val_curve_artifacts = ValidationCurveArtifactConfig(
            max_points=int(args.val_curve_max_points),
            plot_formats=_parse_plot_formats(args.val_plot_formats),
            output_subdir="validation_curves"
        )

    results_csv = args.results_csv or (
        args.save_path / "multi_run_results.csv")
    run_rows: List[Dict[str, Any]] = []

    embedding_dirs = list(args.embedding_dirs)
    label_shards = list(args.label_shards)
    if len(embedding_dirs) == 0 or len(label_shards) == 0:
        raise ValueError("No embedding dirs or label files provided")

    base_dataset = ProteinDataset(
        embedding_dirs=embedding_dirs,
        label_shards=label_shards,
        window_size=args.window_size if args.window_size > 0 else None,
        stride=args.stride,
        collapse_labels=args.collapse_labels,
        pad_last_window=args.pad_last_window,
        return_meta=False,
        cache_current_label_shard=args.cache_current_label_shard,
        drop_label_after_use=args.drop_label_after_use,
        label_cache_mode=args.label_cache_mode
    )
    protein_ids = list(base_dataset.protein_ids)
    if args.subset > 0:
        protein_ids = protein_ids[:args.subset]

    # parition data by ID + family or just ID
    family_groups: Dict[str, str] = {}
    missing_family_ids = 0
    if args.split_type == "id-family":
        for protein_id in protein_ids:
            family = base_dataset.embedding_family_by_id.get(protein_id)
            if family is None or str(family).strip() == "":
                # singleton group when family missing fallback
                family_groups[protein_id] = f"__missing_family__:{protein_id}"
                missing_family_ids += 1
            else:
                family_groups[protein_id] = str(family)

    # estimate relative workload without tensor I/O by using embedding file size.
    id_weights: Dict[str, float] = {}
    for protein_id in protein_ids:
        emb_path = base_dataset.embedding_index.get(protein_id)
        if emb_path is None:
            id_weights[protein_id] = 1.0
            continue
        try:
            id_weights[protein_id] = float(max(1, emb_path.stat().st_size))
        except OSError:
            id_weights[protein_id] = 1.0

    id_groups: Dict[str, str] = {
        protein_id: str(base_dataset.label_index.get(protein_id, ""))
        for protein_id in protein_ids
    }

    run_plans, split_meta = _build_run_plans(args, protein_ids, family_groups)
    if rank == 0:
        logger.info("run_plan_init", extra={"extra": {
            "train_mode": str(split_meta["train_mode"]),
            "n_runs": len(run_plans),
            "split_type": args.split_type,
            "missing_family_ids": missing_family_ids,
            "n_folds": int(split_meta["n_folds"]) if "n_folds" in split_meta else None,
            "n_sets": int(split_meta["n_sets"]) if "n_sets" in split_meta else None,
            "ensemble_seed_mode": split_meta.get("ensemble_seed_mode")
        }})
        if int(split_meta["n_folds"]) > 1:
            logger.info("ensemble_kfold_note", extra={"extra": {
                "message": "--val-frac is ignored in ensemble-kfold mode because folds define validation sets."
            }})

    # parse hidden sizes and dropouts from CSV inputs
    if (not args.hidden_sizes) or (not args.dropouts):
        raise ValueError("--hidden-sizes and --dropouts cannot be empty")
    hidden_sizes = tuple(parse_int_csv(
        args.hidden_sizes, "--hidden-sizes"))
    dropouts = tuple(parse_float_csv(args.dropouts, "--dropouts"))
    if len(hidden_sizes) != len(dropouts):
        raise ValueError(
            "--hidden-sizes and --dropouts must be the same length"
        )

    # per-run loop
    for run_plan in run_plans:
        run_index = run_plan.run_index
        split_seed = run_plan.split_seed
        train_seed = run_plan.train_seed
        set_all_seeds(train_seed)

        train_ids_all = list(run_plan.train_ids_all)
        val_ids_all = list(run_plan.val_ids_all)

        if len(train_ids_all) == 0:
            raise ValueError("Global split produced 0 train IDs")

        if ddp is None:
            train_ids = sort_ids_for_locality(train_ids_all, id_groups)
            val_ids = sort_ids_for_locality(val_ids_all, id_groups)
        else:
            if rank == 0:
                payload = {
                    "train_ids_by_rank": partition_ids_weighted(
                        train_ids_all,
                        world_size,
                        weights=id_weights,
                        groups=id_groups,
                        ensure_non_empty=True
                    ),
                    "val_ids_by_rank": partition_ids_weighted(
                        val_ids_all,
                        world_size,
                        weights=id_weights,
                        groups=id_groups,
                        ensure_non_empty=False
                    )
                }
            else:
                payload = {}

            obj = [payload]
            dist.broadcast_object_list(obj, src=0)
            payload = obj[0]

            train_ids = list(payload["train_ids_by_rank"][rank])
            val_ids = list(payload["val_ids_by_rank"][rank])

            per_rank = [None] * world_size
            dist.all_gather_object(per_rank, {
                "rank": rank,
                "train_ids": len(train_ids),
                "val_ids": len(val_ids)
            })

            if rank == 0:
                logger.info("partition_summary", extra={"extra": {
                    "total_train_ids": len(train_ids_all),
                    "total_val_ids": len(val_ids_all),
                    "per_rank": per_rank
                }})

            if any(int(x["train_ids"]) == 0 for x in per_rank if x is not None):
                raise RuntimeError(
                    "At least one rank received 0 train IDs after weighted partitioning")

        # randomize while preserving label-shard locality
        train_ids = shuffle_ids_by_group(
            train_ids, seed=train_seed, groups=id_groups
        )

        train_data = ProteinDataset(
            embedding_dirs=embedding_dirs,
            label_shards=label_shards,
            protein_ids=train_ids,
            label_index=base_dataset.label_index,
            embedding_index=base_dataset.embedding_index,
            window_size=args.window_size if args.window_size > 0 else None,
            stride=args.stride,
            collapse_labels=args.collapse_labels,
            pad_last_window=args.pad_last_window,
            return_meta=False,
            cache_current_label_shard=args.cache_current_label_shard,
            drop_label_after_use=args.drop_label_after_use,
            label_cache_mode=args.label_cache_mode
        )
        val_data = None
        if len(val_ids) > 0 or ddp is not None:
            val_data = ProteinDataset(
                embedding_dirs=embedding_dirs,
                label_shards=label_shards,
                protein_ids=val_ids,
                label_index=base_dataset.label_index,
                embedding_index=base_dataset.embedding_index,
                window_size=args.window_size if args.window_size > 0 else None,
                stride=args.stride,
                collapse_labels=args.collapse_labels,
                pad_last_window=args.pad_last_window,
                return_meta=False,
                cache_current_label_shard=args.cache_current_label_shard,
                drop_label_after_use=args.drop_label_after_use,
                label_cache_mode=args.label_cache_mode
            )

        # set up data loaders
        pin = torch.cuda.is_available()  # pin memory depending on if CUDA available
        # handle loader worker creation in multi-rank process
        loader_kwargs = {"batch_size": args.batch_size,
                         "shuffle": False,
                         "num_workers": args.num_workers,
                         "pin_memory": pin,
                         "collate_fn": pad_collate}
        if args.num_workers > 0:
            loader_kwargs["multiprocessing_context"] = "spawn"
            # reduce worker respawn overhead
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(train_data, **loader_kwargs)
        val_loader = DataLoader(
            val_data, **loader_kwargs) if val_data is not None else None

        # compute or store positive weight
        pos_weight = None
        if args.pos_weight is not None:
            pos_weight = float(args.pos_weight)
        else:
            pos_weight = pos_weight_from_label_shards(label_shards)

        # build our FFNN model
        emb_dim = infer_emb_dim(base_dataset.embedding_index)
        model = PepSeqFFNN(emb_dim=emb_dim,
                           hidden_sizes=hidden_sizes,
                           dropouts=dropouts,
                           use_layer_norm=args.use_layer_norm,
                           use_residual=args.use_residual,
                           num_classes=1)

        if ddp is not None:
            device = torch.device(f"cuda:{ddp['local_rank']}")
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if ddp is not None:
            model = DDP(model, device_ids=[
                        ddp["local_rank"]], output_device=ddp["local_rank"])

        # setup config and train
        config = TrainerConfig(epochs=args.epochs,
                               batch_size=args.batch_size,
                               learning_rate=args.lr,
                               weight_decay=args.weight_decay,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               pos_weight=pos_weight)
        trainer = Trainer(model=model,
                          train_loader=train_loader,
                          logger=logger,
                          val_loader=val_loader,
                          config=config)

        # run training, only save if rank 0 or single rank run
        if ddp is None or rank == 0:
            run_save_dir = args.save_path / run_plan.save_dir_name
        else:
            run_save_dir = None
        t0 = time.time()
        fit_summary = trainer.fit(
            save_dir=run_save_dir,
            score_key=args.best_model_metric,
            val_curve_artifacts=val_curve_artifacts
        )
        elapsed_s = time.time() - t0

        if rank == 0:
            best_metrics = fit_summary.get("best_metrics") or {}
            best_epoch = int(fit_summary.get("best_epoch", -1))
            best_val_loss = _finite_or_none(
                fit_summary.get("best_val_loss", float("nan")))
            best_score_value = _finite_or_none(
                fit_summary.get("best_score_value", float("nan")))
            display_metric_value = (
                best_val_loss
                if args.best_model_metric == "loss"
                else best_score_value
            )
            threshold = _finite_or_none(
                best_metrics.get("threshold", float("nan")))
            run_valid = (
                best_epoch >= 0
                and (
                    best_val_loss is not None
                    if args.best_model_metric == "loss"
                    else best_score_value is not None
                )
            )
            run_status = "OK" if run_valid else "NO_VALID_SCORE"
            checkpoint_path = (
                run_save_dir / "fully_connected.pt"
                if run_save_dir is not None
                else None
            )
            if not run_valid:
                logger.warning("run_no_valid_score", extra={"extra": {
                    "run_index": run_index,
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "train_mode": run_plan.train_mode,
                    "fold_index": run_plan.fold_index,
                    "ensemble_set_index": run_plan.ensemble_set_index,
                    "best_model_metric": args.best_model_metric,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_score_value": best_score_value
                }})
            row = {
                "RunIndex": run_index,
                "TrainMode": run_plan.train_mode,
                "EnsembleSetIndex": run_plan.ensemble_set_index,
                "EnsembleSetSplitSeed": run_plan.ensemble_set_split_seed,
                "EnsembleSetTrainSeed": run_plan.ensemble_set_train_seed,
                "EnsembleSetDir": (
                    str(args.save_path / run_plan.ensemble_set_dir_name)
                    if run_plan.ensemble_set_dir_name is not None
                    else None
                ),
                "FoldIndex": run_plan.fold_index,
                "NFolds": run_plan.n_folds,
                "SplitSeed": split_seed,
                "TrainSeed": train_seed,
                "RunSaveDir": str(run_save_dir) if run_save_dir is not None else None,
                "CheckpointPath": str(checkpoint_path) if checkpoint_path is not None else None,
                "BestMetricKey": args.best_model_metric,
                "BestMetricValue": display_metric_value,
                "BestEpoch": best_epoch,
                "BestValLoss": best_val_loss,
                "Threshold": threshold,
                "PR_AUC": _finite_or_none(best_metrics.get("pr_auc", float("nan"))),
                "F1": _finite_or_none(best_metrics.get("f1", float("nan"))),
                "MCC": _finite_or_none(best_metrics.get("mcc", float("nan"))),
                "AUC": _finite_or_none(best_metrics.get("auc", float("nan"))),
                "AUC10": _finite_or_none(best_metrics.get("auc10", float("nan"))),
                "BalancedAcc": _finite_or_none(best_metrics.get("res_balanced_acc", float("nan"))),
                "ElapsedSec": _finite_or_none(elapsed_s),
                "Status": run_status
            }
            run_rows.append(row)
            append_csv_row(results_csv, row)

        if ddp is not None:
            dist.barrier()

    # final aggregate and clean-up
    if rank == 0 and run_rows:
        df_runs = pd.DataFrame(run_rows)
        summary_payload = {
            "n_runs": int(len(run_rows)),
            "train_mode": str(split_meta["train_mode"]),
            "split_type": str(args.split_type),
            "best_model_metric": str(args.best_model_metric),
            "split_seeds": [int(x) for x in split_meta["split_seeds"]],
            "train_seeds": [int(x) for x in split_meta["train_seeds"]],
            "metrics": {
                "BestMetricValue": summarize_numeric(df_runs["BestMetricValue"]),
                "PR_AUC": summarize_numeric(df_runs["PR_AUC"]),
                "F1": summarize_numeric(df_runs["F1"]),
                "MCC": summarize_numeric(df_runs["MCC"]),
                "AUC": summarize_numeric(df_runs["AUC"]),
                "AUC10": summarize_numeric(df_runs["AUC10"]),
                "BalancedAcc": summarize_numeric(df_runs["BalancedAcc"]),
                "BestValLoss": summarize_numeric(df_runs["BestValLoss"]),
                "ElapsedSec": summarize_numeric(df_runs["ElapsedSec"])
            }
        }
        if "n_folds" in split_meta:
            summary_payload["n_folds"] = int(split_meta["n_folds"])
        if "n_sets" in split_meta:
            summary_payload["n_sets"] = int(split_meta["n_sets"])
        if "ensemble_seed_mode" in split_meta:
            summary_payload["ensemble_seed_mode"] = str(
                split_meta["ensemble_seed_mode"])
        summary_path = args.save_path / "multi_run_summary.json"
        summary_path.write_text(
            json.dumps(_sanitize_for_json(summary_payload),
                       indent=2, allow_nan=False),
            encoding="utf-8"
        )

        if int(split_meta["n_folds"]) > 1:
            n_sets = int(split_meta.get("n_sets", 1))
            sets_map: Dict[int, Dict[str, Any]] = {}
            for row in run_rows:
                set_idx_raw = row.get("EnsembleSetIndex")
                set_index = int(set_idx_raw) if set_idx_raw is not None else 1

                split_seed_raw = row.get("EnsembleSetSplitSeed")
                train_seed_raw = row.get("EnsembleSetTrainSeed")
                set_split_seed = int(split_seed_raw) if split_seed_raw is not None else int(row["SplitSeed"])
                set_train_seed = int(train_seed_raw) if train_seed_raw is not None else int(row["TrainSeed"])

                if set_index not in sets_map:
                    sets_map[set_index] = {
                        "set_index": set_index,
                        "split_seed": set_split_seed,
                        "train_seed": set_train_seed,
                        "set_dir": row.get("EnsembleSetDir"),
                        "members": []
                    }

                fold_idx_raw = row.get("FoldIndex")
                fold_idx = int(fold_idx_raw) if fold_idx_raw is not None else None
                sets_map[set_index]["members"].append({
                    "member_index": int(row["RunIndex"]),
                    "fold_index": fold_idx,
                    "split_seed": int(row["SplitSeed"]),
                    "train_seed": int(row["TrainSeed"]),
                    "checkpoint": row.get("CheckpointPath"),
                    "threshold": row.get("Threshold"),
                    "status": row.get("Status"),
                    "best_metric_value": row.get("BestMetricValue")
                })

            set_curve_artifacts: Dict[int, Dict[str, Any]] = {}
            if args.save_val_curves:
                for set_index in sorted(sets_map.keys()):
                    entry = sets_map[set_index]
                    set_dir_raw = entry.get("set_dir")
                    if set_dir_raw:
                        curve_out_dir = (
                            Path(str(set_dir_raw))
                            / "validation_curves"
                            / "ensemble_folds"
                        )
                    else:
                        curve_out_dir = (
                            args.save_path
                            / "validation_curves"
                            / "ensemble_folds"
                        )
                    curve_artifacts = _write_ensemble_validation_curve_artifacts(
                        run_rows=run_rows,
                        set_index=int(set_index),
                        output_dir=curve_out_dir,
                        plot_formats=tuple(val_curve_artifacts.plot_formats)
                        if val_curve_artifacts is not None
                        else ("png",),
                    )
                    set_curve_artifacts[int(set_index)] = curve_artifacts
                    for warning in curve_artifacts.get("warnings", []):
                        logger.warning(
                            "ensemble_validation_curve_warning",
                            extra={
                                "extra": {
                                    "set_index": int(set_index),
                                    "message": str(warning),
                                }
                            },
                        )

            set_payloads: List[Dict[str, Any]] = []
            for set_index in sorted(sets_map.keys()):
                entry = sets_map[set_index]
                members = sorted(
                    entry["members"],
                    key=lambda m: (
                        int(m["fold_index"]) if m.get("fold_index") is not None else int(1e9),
                        int(m["member_index"])
                    )
                )
                valid_members = [
                    member for member in members
                    if member.get("status") == "OK" and member.get("checkpoint")
                ]
                set_payload = {
                    "schema_version": 1,
                    "ensemble_type": "kfold_majority_vote",
                    "train_mode": str(split_meta["train_mode"]),
                    "split_type": str(args.split_type),
                    "n_folds": int(split_meta["n_folds"]),
                    "set_index": int(entry["set_index"]),
                    "split_seed": int(entry["split_seed"]),
                    "train_seed": int(entry["train_seed"]),
                    "best_model_metric": str(args.best_model_metric),
                    "n_members": int(len(members)),
                    "n_valid_members": int(len(valid_members)),
                    "voting": {
                        "rule": "majority",
                        "tie_break": "negative"
                    },
                    "members": members
                }
                curve_artifacts = set_curve_artifacts.get(int(set_index))
                if curve_artifacts is not None:
                    set_payload["validation_curve_artifacts"] = curve_artifacts

                if n_sets == 1:
                    set_manifest_path = args.ensemble_manifest or (
                        args.save_path / "ensemble_manifest.json"
                    )
                else:
                    set_dir_raw = entry.get("set_dir")
                    if set_dir_raw:
                        set_manifest_path = Path(set_dir_raw) / "ensemble_manifest.json"
                    else:
                        set_manifest_path = args.save_path / \
                            f"set_{int(entry['set_index']):03d}_ensemble_manifest.json"
                set_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                set_manifest_path.write_text(
                    json.dumps(_sanitize_for_json(set_payload),
                               indent=2, allow_nan=False),
                    encoding="utf-8"
                )
                set_payload["manifest_path"] = str(set_manifest_path)
                set_payloads.append(set_payload)

            if n_sets == 1:
                root_manifest_payload = dict(set_payloads[0])
                root_manifest_payload.pop("manifest_path", None)
                root_manifest_path = args.ensemble_manifest or (
                    args.save_path / "ensemble_manifest.json"
                )
            else:
                root_manifest_payload = {
                    "schema_version": 2,
                    "ensemble_type": "kfold_majority_vote",
                    "train_mode": str(split_meta["train_mode"]),
                    "split_type": str(args.split_type),
                    "n_folds": int(split_meta["n_folds"]),
                    "n_sets": int(n_sets),
                    "ensemble_seed_mode": str(
                        split_meta.get("ensemble_seed_mode", "set-paired")
                    ),
                    "best_model_metric": str(args.best_model_metric),
                    "sets": set_payloads
                }
                root_manifest_path = args.ensemble_manifest or (
                    args.save_path / "ensemble_manifest.json"
                )
            root_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            root_manifest_path.write_text(
                json.dumps(_sanitize_for_json(root_manifest_payload),
                           indent=2, allow_nan=False),
                encoding="utf-8"
            )

    if ddp is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
