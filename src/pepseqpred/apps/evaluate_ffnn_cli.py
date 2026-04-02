"""eval_ffnn_cli.py

Evaluate trained PepSeqPred FFNN checkpoints or ensemble manifests on labeled
embedding shards.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_curve
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.read import parse_float_csv, parse_int_csv
from pepseqpred.core.data.proteindataset import ProteinDataset, pad_collate
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    infer_decision_threshold,
    predict_member_probabilities_from_embedding
)
from pepseqpred.core.train.metrics import compute_eval_metrics


@dataclass(frozen=True)
class EvaluationMember:
    checkpoint: Path
    threshold: float | None
    fold_index: int | None
    member_index: int | None


def _norm_metric_token(value: str) -> str:
    """Normalizes metric names for permissive matching."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _resolve_metric_column(columns: Sequence[str], metric_name: str) -> str:
    """Resolves a requested metric column against runs.csv columns."""
    if metric_name in columns:
        return metric_name

    requested = _norm_metric_token(metric_name)
    lookup: Dict[str, str] = {}
    for col in columns:
        lookup.setdefault(_norm_metric_token(col), col)
    if requested in lookup:
        return lookup[requested]

    raise ValueError(
        f"Metric column '{metric_name}' was not found in runs.csv columns: {list(columns)}"
    )


def _auto_direction_for_metric(metric_name: str) -> str:
    """Returns a default optimization direction for common metric naming patterns."""
    token = _norm_metric_token(metric_name)
    if any(key in token for key in ("loss", "error", "elapsed", "duration", "time")):
        return "min"
    return "max"


def _resolve_best_set_index(
    runs_csv_path: Path,
    metric_name: str,
    agg_name: str,
    direction: str
) -> Tuple[int, Dict[str, Any]]:
    """Selects the best ensemble set index from runs.csv by aggregated metric score."""
    if not runs_csv_path.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv_path}")

    df = pd.read_csv(runs_csv_path)
    if "EnsembleSetIndex" not in df.columns:
        raise ValueError(
            "runs.csv must include 'EnsembleSetIndex' to support best-set selection"
        )
    metric_col = _resolve_metric_column(df.columns, metric_name)

    set_index_series = pd.to_numeric(df["EnsembleSetIndex"], errors="coerce")
    metric_series = pd.to_numeric(df[metric_col], errors="coerce")
    status_mask = pd.Series(True, index=df.index)
    if "Status" in df.columns:
        status_mask = (
            df["Status"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("OK")
        )
    valid_mask = set_index_series.notna() & metric_series.notna() & status_mask
    if int(valid_mask.sum()) < 1:
        raise ValueError(
            "No valid rows found in runs.csv for best-set selection after numeric parsing"
        )

    agg_name_lower = str(agg_name).strip().lower()
    if agg_name_lower not in {"mean", "median", "min", "max"}:
        raise ValueError(
            "--best-set-agg must be one of: mean, median, min, max")

    agg_df = pd.DataFrame(
        {
            "EnsembleSetIndex": set_index_series[valid_mask].astype(int),
            "metric": metric_series[valid_mask].astype(float),
        }
    )
    grouped = (
        agg_df.groupby("EnsembleSetIndex", as_index=False)
        .agg(aggregate_metric=("metric", agg_name_lower))
    )
    if grouped.empty:
        raise ValueError(
            "Could not aggregate any set-level metrics from runs.csv")

    resolved_direction = (
        _auto_direction_for_metric(metric_col)
        if direction == "auto"
        else str(direction).strip().lower()
    )
    if resolved_direction not in {"max", "min"}:
        raise ValueError("--best-set-direction must be one of: auto, max, min")

    if resolved_direction == "max":
        grouped_sorted = grouped.sort_values(
            by=["aggregate_metric", "EnsembleSetIndex"],
            ascending=[False, True]
        )
    else:
        grouped_sorted = grouped.sort_values(
            by=["aggregate_metric", "EnsembleSetIndex"],
            ascending=[True, True]
        )
    best_row = grouped_sorted.iloc[0]
    best_set_index = int(best_row["EnsembleSetIndex"])
    details = {
        "runs_csv_path": str(runs_csv_path),
        "metric_requested": str(metric_name),
        "metric_column": str(metric_col),
        "aggregate": str(agg_name_lower),
        "direction": str(resolved_direction),
        "best_set_index": best_set_index,
        "best_set_score": float(best_row["aggregate_metric"]),
        "set_scores": [
            {
                "ensemble_set_index": int(row["EnsembleSetIndex"]),
                "aggregate_metric": float(row["aggregate_metric"])
            }
            for _, row in grouped_sorted.iterrows()
        ]
    }
    return best_set_index, details


def _build_cli_model_config(args: argparse.Namespace) -> FFNNModelConfig | None:
    """Builds the model configuration from explicit CLI architecture flags."""
    any_explicit = any(
        arg is not None
        for arg in (
            args.emb_dim,
            args.hidden_sizes,
            args.dropouts,
            args.use_layer_norm,
            args.use_residual,
            args.num_classes
        )
    )
    if not any_explicit:
        return None

    missing = []
    if args.emb_dim is None:
        missing.append("--emb-dim")
    if args.hidden_sizes is None:
        missing.append("--hidden-sizes")
    if args.dropouts is None:
        missing.append("--dropouts")
    if args.use_layer_norm is None:
        missing.append("--use-layer-norm/--no-use-layer-norm")
    if args.use_residual is None:
        missing.append("--use-residual/--no-use-residual")
    if missing:
        raise ValueError(
            "When using explicit architecture flags, provide all required values: "
            + ", ".join(missing)
        )

    hidden_sizes = parse_int_csv(args.hidden_sizes, "--hidden-sizes")
    dropouts = parse_float_csv(args.dropouts, "--dropouts")
    if len(hidden_sizes) != len(dropouts):
        raise ValueError(
            "--hidden-sizes and --dropouts must be the same length")

    num_classes = int(args.num_classes) if args.num_classes is not None else 1
    return FFNNModelConfig(
        emb_dim=int(args.emb_dim),
        hidden_sizes=tuple(hidden_sizes),
        dropouts=tuple(dropouts),
        num_classes=num_classes,
        use_layer_norm=bool(args.use_layer_norm),
        use_residual=bool(args.use_residual)
    )


def _as_optional_int(value: Any) -> int | None:
    """Parses optional integer-like value."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_threshold(value: Any) -> float | None:
    """Parses optional valid threshold in (0.0, 1.0)."""
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        return None
    return threshold


def _resolve_member_checkpoint_path(raw_path: Any, manifest_path: Path) -> Path | None:
    """Resolves member checkpoint path relative to manifest if needed."""
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _member_sort_key(member: EvaluationMember) -> Tuple[int, int, str]:
    """Sort key for deterministic member ordering."""
    fold_index = member.fold_index if member.fold_index is not None else int(
        1e9)
    member_index = (
        member.member_index if member.member_index is not None else int(1e9)
    )
    return (fold_index, member_index, str(member.checkpoint))


def _resolve_manifest_members(
    manifest_path: Path,
    ensemble_set_index: int
) -> Tuple[List[EvaluationMember], Dict[str, Any]]:
    """Resolves valid members from schema v1/v2 ensemble manifests."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in ensemble manifest: {manifest_path}"
        ) from e
    if not isinstance(payload, Mapping):
        raise ValueError("Ensemble manifest must be a JSON object")

    schema_version = _as_optional_int(payload.get("schema_version"))
    selected_set: Mapping[str, Any] | None = None
    members_raw: Any = None

    if schema_version == 2:
        sets_raw = payload.get("sets")
        if not isinstance(sets_raw, list):
            raise ValueError(
                "schema_version=2 manifest must contain 'sets' list")
        for entry in sets_raw:
            if not isinstance(entry, Mapping):
                continue
            if _as_optional_int(entry.get("set_index")) == ensemble_set_index:
                selected_set = entry
                break
        if selected_set is None:
            raise ValueError(
                f"Requested --ensemble-set-index={ensemble_set_index} not found in manifest"
            )
        members_raw = selected_set.get("members")
    else:
        selected_set = payload
        members_raw = payload.get("members")

    if not isinstance(members_raw, list):
        raise ValueError("Manifest must contain a 'members' list")

    members: List[EvaluationMember] = []
    for member in members_raw:
        if not isinstance(member, Mapping):
            continue
        status = str(member.get("status", "")).strip().upper()
        checkpoint_path = _resolve_member_checkpoint_path(
            member.get("checkpoint"), manifest_path
        )
        if status != "OK" or checkpoint_path is None:
            continue
        members.append(
            EvaluationMember(
                checkpoint=checkpoint_path,
                threshold=_as_optional_threshold(member.get("threshold")),
                fold_index=_as_optional_int(member.get("fold_index")),
                member_index=_as_optional_int(member.get("member_index"))
            )
        )

    if not members:
        raise ValueError(
            "No valid ensemble members found (requires status=OK and checkpoint path)"
        )

    members_sorted = sorted(members, key=_member_sort_key)
    meta: Dict[str, Any] = {
        "schema_version": schema_version,
        "set_index": (
            _as_optional_int(selected_set.get("set_index"))
            if isinstance(selected_set, Mapping)
            else None
        ),
        "n_members_total": len(members_raw),
        "n_members_valid": len(members_sorted)
    }
    return members_sorted, meta


def _resolve_evaluation_members(
    model_artifact: Path,
    ensemble_set_index: int,
    k_folds: int | None
) -> Tuple[str, List[EvaluationMember], Dict[str, Any]]:
    """Resolves members from either a checkpoint path or ensemble manifest."""
    suffix = model_artifact.suffix.lower()
    if suffix == ".pt":
        members = [
            EvaluationMember(
                checkpoint=model_artifact,
                threshold=None,
                fold_index=1,
                member_index=1
            )
        ]
        mode = "single-checkpoint"
        meta = {
            "schema_version": None,
            "set_index": None,
            "n_members_total": 1,
            "n_members_valid": 1
        }
    elif suffix == ".json":
        members, meta = _resolve_manifest_members(
            manifest_path=model_artifact, ensemble_set_index=ensemble_set_index
        )
        mode = "ensemble-manifest"
    else:
        raise ValueError(
            f"Unsupported model artifact type '{model_artifact.suffix}'. "
            "Expected .pt checkpoint or .json ensemble manifest."
        )

    if k_folds is not None:
        if k_folds < 1:
            raise ValueError("--k-folds must be >= 1")
        if k_folds > len(members):
            raise ValueError(
                f"--k-folds={k_folds} exceeds available valid members={len(members)}"
            )
        members = members[:k_folds]

    return mode, members, meta


def _sanitize_for_json(value: Any) -> Any:
    """Recursively converts non-finite floats to None for strict JSON output."""
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Returns numerator/denominator or NaN when denominator is zero."""
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _augment_metrics_with_confusion_stats(
    metrics: Dict[str, Any], cm: np.ndarray
) -> Dict[str, Any]:
    """Adds confusion-derived metrics and class coverage details."""
    if cm.shape != (2, 2):
        raise ValueError(
            f"Expected confusion matrix shape (2, 2), got {cm.shape}")

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    neg_support = int(tn + fp)
    pos_support = int(fn + tp)

    neg_acc = _safe_ratio(tn, neg_support)
    pos_acc = _safe_ratio(tp, pos_support)
    finite_accs = [x for x in (neg_acc, pos_acc) if math.isfinite(x)]
    balanced_acc = (
        float(sum(finite_accs) / len(finite_accs))
        if len(finite_accs) > 0
        else float("nan")
    )

    metrics["support_neg"] = neg_support
    metrics["support_pos"] = pos_support
    metrics["specificity"] = neg_acc
    metrics["fpr"] = _safe_ratio(fp, neg_support)
    metrics["npv"] = _safe_ratio(tn, int(tn + fn))
    metrics["per_class_acc"] = [neg_acc, pos_acc]
    metrics["res_balanced_acc"] = balanced_acc
    metrics["has_both_classes"] = bool(neg_support > 0 and pos_support > 0)
    return metrics


def _downsample_curve(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int
) -> Tuple[List[float], List[float]]:
    """Downsamples curve points to keep JSON and plot payloads bounded."""
    if x.size != y.size:
        raise ValueError(
            f"Curve x/y size mismatch: x={x.size} y={y.size}")
    if max_points < 2:
        raise ValueError("--curve-max-points must be >= 2")
    if x.size <= max_points:
        return [float(v) for v in x], [float(v) for v in y]

    idx = np.linspace(0, x.size - 1, num=max_points, dtype=np.int64)
    idx = np.unique(idx)
    return [float(v) for v in x[idx]], [float(v) for v in y[idx]]


def _build_roc_curve_payload(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int
) -> Dict[str, Any]:
    """Builds ROC curve points for JSON/plotting."""
    if y_true.size < 1 or np.unique(y_true).size < 2:
        return {
            "available": False,
            "reason": "single-class-labels",
            "fpr": [],
            "tpr": []
        }
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    x_points, y_points = _downsample_curve(
        x=np.asarray(fpr, dtype=np.float64),
        y=np.asarray(tpr, dtype=np.float64),
        max_points=max_points
    )
    return {
        "available": True,
        "reason": None,
        "fpr": x_points,
        "tpr": y_points
    }


def _build_pr_curve_payload(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int
) -> Dict[str, Any]:
    """Builds PR curve points for JSON/plotting."""
    if y_true.size < 1:
        return {
            "available": False,
            "reason": "no-valid-residues",
            "precision": [],
            "recall": [],
            "baseline_positive_rate": None
        }
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    recall = np.asarray(recall, dtype=np.float64)[::-1]
    precision = np.asarray(precision, dtype=np.float64)[::-1]
    recall_points, precision_points = _downsample_curve(
        x=recall,
        y=precision,
        max_points=max_points
    )
    pos_rate = float((y_true == 1).mean()) if y_true.size > 0 else float("nan")
    if not math.isfinite(pos_rate):
        pos_rate = float("nan")
    return {
        "available": True,
        "reason": None,
        "precision": precision_points,
        "recall": recall_points,
        "baseline_positive_rate": (
            float(pos_rate) if math.isfinite(pos_rate) else None
        )
    }


def _empty_eval_output(votes_needed: int | None, include_curves: bool) -> Dict[str, Any]:
    """Returns a canonical empty evaluation payload."""
    out = {
        "processed_proteins": 0,
        "proteins_with_valid_residues": 0,
        "proteins_zero_valid_residues": 0,
        "valid_residues": 0,
        "pos_residues": 0,
        "neg_residues": 0,
        "pred_pos_residues": 0,
        "pred_neg_residues": 0,
        "accuracy": float("nan"),
        "confusion_matrix": [[0, 0], [0, 0]],
        "votes_needed": votes_needed,
        "has_both_classes": False,
        "metrics": {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "auc": float("nan"),
            "auc10": float("nan"),
            "pr_auc": float("nan"),
            "support_neg": 0,
            "support_pos": 0,
            "specificity": float("nan"),
            "fpr": float("nan"),
            "npv": float("nan"),
            "res_balanced_acc": float("nan"),
            "per_class_acc": [float("nan"), float("nan")]
        }
    }
    if include_curves:
        out["roc_curve"] = {
            "available": False,
            "reason": "no-valid-residues",
            "fpr": [],
            "tpr": []
        }
        out["pr_curve"] = {
            "available": False,
            "reason": "no-valid-residues",
            "precision": [],
            "recall": [],
            "baseline_positive_rate": None
        }
    return out


def _build_eval_output_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    processed_proteins: int,
    proteins_zero_valid_residues: int,
    votes_needed: int | None,
    include_curves: bool,
    curve_max_points: int
) -> Dict[str, Any]:
    """Builds a full evaluation summary from flattened residue-level arrays."""
    if y_true.size < 1:
        out = _empty_eval_output(
            votes_needed=votes_needed, include_curves=include_curves)
        out["processed_proteins"] = int(processed_proteins)
        out["proteins_zero_valid_residues"] = int(proteins_zero_valid_residues)
        out["proteins_with_valid_residues"] = int(
            processed_proteins - proteins_zero_valid_residues)
        return out

    metrics = compute_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    cm = np.zeros((2, 2), dtype=np.int64)
    cm[0, 0] = int(((y_true == 0) & (y_pred == 0)).sum())
    cm[0, 1] = int(((y_true == 0) & (y_pred == 1)).sum())
    cm[1, 0] = int(((y_true == 1) & (y_pred == 0)).sum())
    cm[1, 1] = int(((y_true == 1) & (y_pred == 1)).sum())
    metrics = _augment_metrics_with_confusion_stats(metrics=metrics, cm=cm)

    out = {
        "processed_proteins": int(processed_proteins),
        "proteins_with_valid_residues": int(processed_proteins - proteins_zero_valid_residues),
        "proteins_zero_valid_residues": int(proteins_zero_valid_residues),
        "valid_residues": int(y_true.size),
        "pos_residues": int((y_true == 1).sum()),
        "neg_residues": int((y_true == 0).sum()),
        "pred_pos_residues": int((y_pred == 1).sum()),
        "pred_neg_residues": int((y_pred == 0).sum()),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan"),
        "confusion_matrix": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
        "votes_needed": votes_needed,
        "has_both_classes": bool(metrics.get("has_both_classes", False)),
        "metrics": metrics
    }
    if include_curves:
        out["roc_curve"] = _build_roc_curve_payload(
            y_true=y_true, y_prob=y_prob, max_points=curve_max_points
        )
        out["pr_curve"] = _build_pr_curve_payload(
            y_true=y_true, y_prob=y_prob, max_points=curve_max_points
        )
    return out


def _resolve_eval_metric_value(eval_item: Mapping[str, Any], metric_name: str) -> float:
    """Resolves a metric value from eval payload for fold ranking."""
    metrics = eval_item.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Fold evaluation payload missing 'metrics' mapping")
    if metric_name in metrics:
        value = metrics[metric_name]
    else:
        metric_lookup = {
            _norm_metric_token(str(key)): key for key in metrics.keys()
        }
        token = _norm_metric_token(metric_name)
        if token not in metric_lookup:
            raise ValueError(
                f"Metric '{metric_name}' not found in fold metrics keys={list(metrics.keys())}"
            )
        value = metrics[metric_lookup[token]]
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(
            f"Fold ranking metric '{metric_name}' produced non-finite value: {value}"
        )
    return value_float


def _pick_best_fold(
    fold_evaluations: Sequence[Mapping[str, Any]],
    metric_name: str,
    direction: str
) -> Tuple[int, Dict[str, Any]]:
    """Returns best fold array index and fold-level selection metadata."""
    if len(fold_evaluations) < 1:
        raise ValueError(
            "At least one fold evaluation is required to pick a best fold")

    resolved_direction = (
        _auto_direction_for_metric(metric_name)
        if direction == "auto"
        else str(direction).strip().lower()
    )
    if resolved_direction not in {"max", "min"}:
        raise ValueError(
            "--best-fold-direction must be one of: auto, max, min")

    scored: List[Tuple[int, float, int]] = []
    for idx, fold_eval in enumerate(fold_evaluations):
        metric_value = _resolve_eval_metric_value(fold_eval, metric_name)
        fold_index_value = _as_optional_int(fold_eval.get("fold_index"))
        fold_index = int(
            fold_index_value) if fold_index_value is not None else (idx + 1)
        scored.append((idx, metric_value, fold_index))

    if resolved_direction == "max":
        scored_sorted = sorted(
            scored,
            key=lambda row: (-row[1], row[2], row[0])
        )
    else:
        scored_sorted = sorted(
            scored,
            key=lambda row: (row[1], row[2], row[0])
        )
    best_idx, best_score, best_fold_index = scored_sorted[0]
    meta = {
        "metric_requested": str(metric_name),
        "direction": str(resolved_direction),
        "best_fold_index": int(best_fold_index),
        "best_score": float(best_score),
        "fold_scores": [
            {
                "fold_index": int(fold_idx),
                "metric_value": float(score)
            }
            for _, score, fold_idx in scored_sorted
        ]
    }
    return best_idx, meta


def _parse_plot_formats(raw: str) -> List[str]:
    """Parses comma-separated plot file formats."""
    tokens = [t.strip().lower() for t in str(raw).split(",")]
    formats = [t for t in tokens if len(t) > 0]
    if len(formats) < 1:
        raise ValueError("--plot-formats must include at least one format")
    allowed = {"png", "svg", "pdf"}
    bad = [fmt for fmt in formats if fmt not in allowed]
    if len(bad) > 0:
        raise ValueError(
            f"Unsupported --plot-formats values={bad}; allowed={sorted(allowed)}"
        )
    return formats


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
    baseline_y: float | None = None
) -> List[str]:
    """Writes a fold-only curve panel for ROC or PR."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install matplotlib to use --plot-dir."
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
        fold_label = int(
            fold_index) if fold_index is not None else (plotted + 1)
        metrics_obj = fold_eval.get("metrics")
        metric_value = float("nan")
        if isinstance(metrics_obj, Mapping):
            raw_metric = metrics_obj.get(metric_key, float("nan"))
            try:
                metric_value = float(raw_metric)
            except (TypeError, ValueError):
                metric_value = float("nan")
        metric_label = f"{metric_value:.3f}" if math.isfinite(
            metric_value) else "nan"
        ax.plot(
            x_vals,
            y_vals,
            linewidth=1.8,
            alpha=0.95,
            label=f"Fold {fold_label} {metric_key.upper()}: {metric_label}"
        )
        plotted += 1

    if chance_line:
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--",
                linewidth=1.0, color="#C77DBB", label="Chance")
    if baseline_y is not None and math.isfinite(float(baseline_y)):
        y_val = float(baseline_y)
        ax.plot([0.0, 1.0], [y_val, y_val], linestyle="--",
                linewidth=1.0, color="#888888", label=f"Baseline: {y_val:.3f}")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    if plotted > 0:
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()

    saved_paths: List[str] = []
    plot_path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = plot_path_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(str(out_path))
    plt.close(fig)
    return saved_paths


def _plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    plot_path_base: Path,
    formats: Sequence[str],
    title: str
) -> List[str]:
    """Writes confusion matrix heatmap plot with counts and percentages."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install matplotlib to use --plot-dir."
        ) from e

    if confusion_matrix.shape != (2, 2):
        raise ValueError(
            f"Expected confusion matrix shape=(2,2), got {confusion_matrix.shape}"
        )
    cm = confusion_matrix.astype(np.int64, copy=False)
    total = int(cm.sum())
    pct = (
        (cm.astype(np.float64) / float(total)) * 100.0
        if total > 0
        else np.zeros_like(cm, dtype=np.float64)
    )

    fig, ax = plt.subplots(figsize=(6.3, 5.6), dpi=150)
    im = ax.imshow(cm, cmap="Oranges")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Non-epitope residue", "Epitope residue"])
    ax.set_yticks([0, 1], labels=[
                  "True non-epitope residue", "True epitope residue"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    max_val = int(cm.max()) if cm.size > 0 else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > (max_val / 2.0) else "#333333"
            ax.text(
                j,
                i,
                f"{pct[i, j]:.2f}%\n{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=10,
                color=color
            )
    fig.tight_layout()

    saved_paths: List[str] = []
    plot_path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = plot_path_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(str(out_path))
    plt.close(fig)
    return saved_paths


def _evaluate_dataset(
    psp_models: Sequence[torch.nn.Module],
    thresholds: Sequence[float],
    eval_loader: DataLoader,
    device: str,
    logger: Any,
    member_specs: Sequence[EvaluationMember],
    emit_fold_metrics: bool,
    include_curves: bool,
    curve_max_points: int,
    best_fold_by: str,
    best_fold_direction: str,
    plot_dir: Path | None,
    plot_formats: Sequence[str]
) -> Dict[str, Any]:
    """Evaluates one model or an ensemble on the given eval dataset loader."""
    n_members = int(len(psp_models))
    if n_members < 1:
        raise ValueError("At least one model is required for evaluation")
    if len(thresholds) != n_members:
        raise ValueError(
            f"Model/threshold length mismatch: models={n_members} thresholds={len(thresholds)}"
        )
    if len(member_specs) != n_members:
        raise ValueError(
            f"member_specs length mismatch: specs={len(member_specs)} models={n_members}"
        )
    if curve_max_points < 2:
        raise ValueError("--curve-max-points must be >= 2")

    votes_needed = int((n_members // 2) + 1)

    all_true: List[torch.Tensor] = []
    ens_pred: List[torch.Tensor] = []
    ens_prob: List[torch.Tensor] = []

    fold_pred: List[List[torch.Tensor]] = [[] for _ in range(n_members)]
    fold_prob: List[List[torch.Tensor]] = [[] for _ in range(n_members)]

    processed = 0
    zero_valid = 0
    running_valid_residues = 0

    for batch_idx, batch in enumerate(eval_loader, start=1):
        X, y, mask = batch
        batch_size = int(X.size(0))

        for i in range(batch_size):
            processed += 1
            protein_emb = X[i]
            y_row = y[i].to(torch.int64)
            valid_mask = mask[i].bool()
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                zero_valid += 1
                continue

            member_probs_full: List[torch.Tensor] = []
            member_pred_full: List[torch.Tensor] = []
            for model, threshold in zip(psp_models, thresholds):
                probs_member = predict_member_probabilities_from_embedding(
                    psp_model=model,
                    protein_emb=protein_emb,
                    device=device
                )
                member_probs_full.append(probs_member)
                member_pred_full.append(
                    (probs_member >= threshold).to(torch.int64))

            if n_members == 1:
                pred_ensemble = member_pred_full[0]
                prob_ensemble = member_probs_full[0]
            else:
                vote_sum = torch.stack(member_pred_full, dim=0).sum(dim=0)
                pred_ensemble = (vote_sum >= votes_needed).to(torch.int64)
                prob_ensemble = torch.stack(
                    member_probs_full, dim=0).mean(dim=0)

            y_true_valid = y_row[valid_mask].to(torch.int64).cpu()
            ens_pred_valid = pred_ensemble[valid_mask].to(torch.int64).cpu()
            ens_prob_valid = prob_ensemble[valid_mask].to(torch.float32).cpu()

            all_true.append(y_true_valid)
            ens_pred.append(ens_pred_valid)
            ens_prob.append(ens_prob_valid)

            if emit_fold_metrics:
                for member_idx in range(n_members):
                    fold_pred[member_idx].append(
                        member_pred_full[member_idx][valid_mask].to(
                            torch.int64).cpu()
                    )
                    fold_prob[member_idx].append(
                        member_probs_full[member_idx][valid_mask].to(
                            torch.float32).cpu()
                    )

            running_valid_residues += int(y_true_valid.numel())

        if batch_idx % 50 == 0:
            logger.info(
                "eval_progress",
                extra={
                    "extra": {
                        "batches": batch_idx,
                        "processed_proteins": int(processed),
                        "valid_residues": int(running_valid_residues)
                    }
                }
            )

    y_true_np = (
        torch.cat(all_true, dim=0).numpy()
        if len(all_true) > 0
        else np.asarray([], dtype=np.int64)
    )
    ens_pred_np = (
        torch.cat(ens_pred, dim=0).numpy()
        if len(ens_pred) > 0
        else np.asarray([], dtype=np.int64)
    )
    ens_prob_np = (
        torch.cat(ens_prob, dim=0).numpy()
        if len(ens_prob) > 0
        else np.asarray([], dtype=np.float32)
    )

    ensemble_eval = _build_eval_output_from_arrays(
        y_true=y_true_np,
        y_pred=ens_pred_np,
        y_prob=ens_prob_np,
        processed_proteins=int(processed),
        proteins_zero_valid_residues=int(zero_valid),
        votes_needed=(votes_needed if n_members > 1 else None),
        include_curves=include_curves,
        curve_max_points=curve_max_points
    )

    if not emit_fold_metrics:
        return ensemble_eval

    fold_evaluations: List[Dict[str, Any]] = []
    for member_idx, member in enumerate(member_specs):
        fold_pred_np = (
            torch.cat(fold_pred[member_idx], dim=0).numpy()
            if len(fold_pred[member_idx]) > 0
            else np.asarray([], dtype=np.int64)
        )
        fold_prob_np = (
            torch.cat(fold_prob[member_idx], dim=0).numpy()
            if len(fold_prob[member_idx]) > 0
            else np.asarray([], dtype=np.float32)
        )
        fold_eval = _build_eval_output_from_arrays(
            y_true=y_true_np,
            y_pred=fold_pred_np,
            y_prob=fold_prob_np,
            processed_proteins=int(processed),
            proteins_zero_valid_residues=int(zero_valid),
            votes_needed=None,
            include_curves=include_curves,
            curve_max_points=curve_max_points
        )
        fold_eval["fold_index"] = member.fold_index
        fold_eval["member_index"] = member.member_index
        fold_eval["checkpoint"] = str(member.checkpoint)
        fold_eval["threshold"] = float(thresholds[member_idx])
        fold_evaluations.append(fold_eval)

    best_fold_array_idx, best_fold_meta = _pick_best_fold(
        fold_evaluations=fold_evaluations,
        metric_name=best_fold_by,
        direction=best_fold_direction
    )
    best_fold_eval = fold_evaluations[best_fold_array_idx]

    ensemble_eval["folds"] = fold_evaluations
    ensemble_eval["best_fold_selection"] = best_fold_meta
    ensemble_eval["best_fold"] = best_fold_eval

    if plot_dir is not None:
        pr_baseline = None
        pr_obj = best_fold_eval.get("pr_curve")
        if isinstance(pr_obj, Mapping):
            pr_baseline = pr_obj.get("baseline_positive_rate")

        roc_paths = _plot_fold_curves(
            fold_evaluations=fold_evaluations,
            plot_path_base=plot_dir / "roc_auc_folds",
            formats=plot_formats,
            curve_key="roc_curve",
            title="Fold ROC Curves",
            x_label="FPR",
            y_label="TPR",
            x_key="fpr",
            y_key="tpr",
            metric_key="auc",
            chance_line=True
        )
        pr_paths = _plot_fold_curves(
            fold_evaluations=fold_evaluations,
            plot_path_base=plot_dir / "pr_auc_folds",
            formats=plot_formats,
            curve_key="pr_curve",
            title="Fold PR Curves",
            x_label="Recall",
            y_label="Precision",
            x_key="recall",
            y_key="precision",
            metric_key="pr_auc",
            chance_line=False,
            baseline_y=(
                float(pr_baseline)
                if pr_baseline is not None and math.isfinite(float(pr_baseline))
                else None
            )
        )
        cm = np.asarray(best_fold_eval["confusion_matrix"], dtype=np.int64)
        cm_paths = _plot_confusion_matrix(
            confusion_matrix=cm,
            plot_path_base=plot_dir / "confusion_matrix_best_fold",
            formats=plot_formats,
            title="Best Fold Confusion Matrix"
        )
        ensemble_eval["plot_outputs"] = {
            "plot_dir": str(plot_dir),
            "formats": [str(fmt) for fmt in plot_formats],
            "roc_auc_folds": roc_paths,
            "pr_auc_folds": pr_paths,
            "confusion_matrix_best_fold": cm_paths
        }

    return ensemble_eval


def main() -> None:
    """Handles CLI argument parsing and high-level evaluation flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PepSeqPred FFNN checkpoint(s) on labeled embedding shards."
        )
    )
    parser.add_argument(
        "model_artifact",
        type=Path,
        help="Path to model artifact (.pt checkpoint or ensemble manifest .json)."
    )
    parser.add_argument(
        "--embedding-dirs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more directories containing per-protein embeddings."
    )
    parser.add_argument(
        "--label-shards",
        nargs="+",
        required=True,
        type=Path,
        help="One or more label shard .pt files containing a 'labels' dictionary."
    )
    parser.add_argument(
        "--output-json",
        action="store",
        dest="output_json",
        type=Path,
        default=Path("ffnn_eval_summary.json"),
        help="Path to write evaluation summary JSON."
    )
    parser.add_argument(
        "--threshold",
        action="store",
        dest="threshold",
        type=float,
        default=None,
        help="Optional global threshold override in (0.0, 1.0)."
    )
    parser.add_argument(
        "--ensemble-set-index",
        action="store",
        dest="ensemble_set_index",
        type=int,
        default=1,
        help="Set index to use for schema v2 ensemble manifest evaluation."
    )
    parser.add_argument(
        "--k-folds",
        action="store",
        dest="k_folds",
        type=int,
        default=None,
        help="Optional number of ensemble members to evaluate (first K by fold/member order)."
    )
    parser.add_argument(
        "--select-best-set-runs-csv",
        action="store",
        dest="select_best_set_runs_csv",
        type=Path,
        default=None,
        help=(
            "Optional runs.csv path used to auto-select --ensemble-set-index by set-level "
            "aggregate metric."
        )
    )
    parser.add_argument(
        "--best-set-by",
        action="store",
        dest="best_set_by",
        type=str,
        default="PR_AUC",
        help="Metric name (runs.csv column) used for --select-best-set-runs-csv."
    )
    parser.add_argument(
        "--best-set-agg",
        action="store",
        dest="best_set_agg",
        type=str,
        choices=["mean", "median", "min", "max"],
        default="mean",
        help="Aggregation across folds for best-set selection."
    )
    parser.add_argument(
        "--best-set-direction",
        action="store",
        dest="best_set_direction",
        type=str,
        choices=["auto", "max", "min"],
        default="auto",
        help="Optimization direction for best-set selection metric."
    )
    parser.add_argument(
        "--subset",
        action="store",
        dest="subset",
        type=int,
        default=0,
        help="If > 0, evaluate only first N proteins by deterministic protein_id sort order."
    )
    parser.add_argument(
        "--batch-size",
        action="store",
        dest="batch_size",
        type=int,
        default=64,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--num-workers",
        action="store",
        dest="num_workers",
        type=int,
        default=0,
        help="Number of data loader workers."
    )
    parser.add_argument(
        "--emit-fold-metrics",
        action="store_true",
        dest="emit_fold_metrics",
        default=False,
        help="Emit per-fold evaluation payloads in addition to ensemble evaluation."
    )
    parser.add_argument(
        "--include-curves",
        action="store_true",
        dest="include_curves",
        default=False,
        help="Include ROC and PR curve points in evaluation JSON payloads."
    )
    parser.add_argument(
        "--curve-max-points",
        action="store",
        dest="curve_max_points",
        type=int,
        default=2048,
        help="Maximum number of curve points saved per ROC/PR curve."
    )
    parser.add_argument(
        "--best-fold-by",
        action="store",
        dest="best_fold_by",
        type=str,
        default="pr_auc",
        help="Metric used to pick the best fold when --emit-fold-metrics is enabled."
    )
    parser.add_argument(
        "--best-fold-direction",
        action="store",
        dest="best_fold_direction",
        type=str,
        choices=["auto", "max", "min"],
        default="auto",
        help="Optimization direction used for --best-fold-by."
    )
    parser.add_argument(
        "--plot-dir",
        action="store",
        dest="plot_dir",
        type=Path,
        default=None,
        help="Optional directory to write fold ROC/PR and best-fold confusion matrix plots."
    )
    parser.add_argument(
        "--plot-formats",
        action="store",
        dest="plot_formats",
        type=str,
        default="png,svg",
        help="Comma-separated plot formats (png,svg,pdf)."
    )
    parser.add_argument(
        "--label-cache-mode",
        action="store",
        dest="label_cache_mode",
        type=str,
        choices=["current", "all"],
        default="current",
        help="Label shard cache strategy used by ProteinDataset."
    )
    parser.add_argument(
        "--log-dir",
        action="store",
        dest="log_dir",
        type=Path,
        default=Path("logs"),
        help="Directory for logs."
    )
    parser.add_argument(
        "--log-level",
        action="store",
        dest="log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level choice."
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        dest="log_json",
        default=False,
        help="Emit logs as JSON lines for simple parsing."
    )
    parser.add_argument(
        "--emb-dim",
        action="store",
        dest="emb_dim",
        type=int,
        default=None,
        help="Explicit model embedding dimension used in training."
    )
    parser.add_argument(
        "--hidden-sizes",
        action="store",
        dest="hidden_sizes",
        type=str,
        default=None,
        help="Explicit hidden sizes CSV (e.g. 150,120,45)."
    )
    parser.add_argument(
        "--dropouts",
        action="store",
        dest="dropouts",
        type=str,
        default=None,
        help="Explicit dropouts CSV (e.g. 0.1,0.1,0.1)."
    )
    parser.add_argument(
        "--num-classes",
        action="store",
        dest="num_classes",
        type=int,
        default=None,
        help="Explicit output classes (binary=1)."
    )
    parser.add_argument(
        "--use-layer-norm",
        action="store_true",
        dest="use_layer_norm",
        help="Explicitly use layer normalization."
    )
    parser.add_argument(
        "--use-residual",
        action="store_true",
        dest="use_residual",
        help="Explicitly use residual connections."
    )
    parser.add_argument(
        "--no-use-layer-norm",
        action="store_false",
        dest="use_layer_norm",
        help="Explicitly disable layer normalization."
    )
    parser.add_argument(
        "--no-use-residual",
        action="store_false",
        dest="use_residual",
        help="Explicitly disable residual connections."
    )
    parser.set_defaults(use_layer_norm=None, use_residual=None)
    args = parser.parse_args()

    if args.ensemble_set_index < 1:
        raise ValueError("--ensemble-set-index must be >= 1")
    if args.k_folds is not None and args.k_folds < 1:
        raise ValueError("--k-folds must be >= 1")
    if args.threshold is not None and (args.threshold <= 0.0 or args.threshold >= 1.0):
        raise ValueError("--threshold must be between (0.0, 1.0)")
    if args.subset < 0:
        raise ValueError("--subset must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.curve_max_points < 2:
        raise ValueError("--curve-max-points must be >= 2")

    plot_formats: List[str] = []
    if args.plot_dir is not None:
        plot_formats = _parse_plot_formats(args.plot_formats)
        args.emit_fold_metrics = True
        args.include_curves = True

    selected_set_meta: Dict[str, Any] | None = None
    if args.select_best_set_runs_csv is not None:
        if args.model_artifact.suffix.lower() != ".json":
            raise ValueError(
                "--select-best-set-runs-csv requires a .json ensemble manifest model artifact"
            )
        selected_set_index, selected_set_meta = _resolve_best_set_index(
            runs_csv_path=args.select_best_set_runs_csv,
            metric_name=args.best_set_by,
            agg_name=args.best_set_agg,
            direction=args.best_set_direction
        )
        args.ensemble_set_index = int(selected_set_index)

    json_indent = 2 if args.log_json else None
    logger = setup_logger(
        log_dir=args.log_dir,
        log_level=args.log_level,
        json_lines=args.log_json,
        json_indent=json_indent,
        name="eval_ffnn_cli"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_mode, member_specs, artifact_meta = _resolve_evaluation_members(
        model_artifact=args.model_artifact,
        ensemble_set_index=args.ensemble_set_index,
        k_folds=args.k_folds
    )

    cli_model_cfg = _build_cli_model_config(args)
    psp_models: List[torch.nn.Module] = []
    member_thresholds: List[float] = []
    member_model_cfgs: List[FFNNModelConfig] = []
    member_model_cfg_srcs: List[str] = []

    for spec in member_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")

        checkpoint = torch.load(
            spec.checkpoint, map_location="cpu", weights_only=True)
        psp_model, model_cfg, model_cfg_src = build_model_from_checkpoint(
            checkpoint=checkpoint, device=device, model_config=cli_model_cfg
        )
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else (
                spec.threshold
                if spec.threshold is not None
                else infer_decision_threshold(checkpoint)
            )
        )
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(
                f"Invalid threshold for member checkpoint: {spec.checkpoint}")

        psp_models.append(psp_model)
        member_thresholds.append(float(threshold))
        member_model_cfgs.append(model_cfg)
        member_model_cfg_srcs.append(model_cfg_src)

    if not psp_models:
        raise ValueError("No evaluation models were loaded")

    emb_dims = {int(cfg.emb_dim) for cfg in member_model_cfgs}
    if len(emb_dims) != 1:
        raise ValueError(
            "All ensemble members must share emb_dim for shared-embedding evaluation, "
            f"got {sorted(emb_dims)}"
        )

    base_dataset = ProteinDataset(
        embedding_dirs=args.embedding_dirs,
        label_shards=args.label_shards,
        protein_ids=None,
        window_size=None,
        stride=1,
        collapse_labels=True,
        pad_last_window=False,
        return_meta=False,
        cache_current_label_shard=True,
        drop_label_after_use=False,
        label_cache_mode=args.label_cache_mode
    )
    candidate_ids = list(base_dataset.protein_ids)
    if args.subset > 0:
        selected_ids = list(candidate_ids[: int(args.subset)])
    else:
        selected_ids = list(candidate_ids)
    if len(selected_ids) == 0:
        raise ValueError("No proteins available for evaluation")

    eval_dataset = ProteinDataset(
        embedding_dirs=args.embedding_dirs,
        label_shards=args.label_shards,
        protein_ids=selected_ids,
        label_index=base_dataset.label_index,
        embedding_index=base_dataset.embedding_index,
        window_size=None,
        stride=1,
        collapse_labels=True,
        pad_last_window=False,
        return_meta=False,
        cache_current_label_shard=True,
        drop_label_after_use=False,
        label_cache_mode=args.label_cache_mode
    )

    pin = torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin,
        "collate_fn": pad_collate
    }
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    eval_loader = DataLoader(eval_dataset, **loader_kwargs)

    first_cfg = member_model_cfgs[0]
    logger.info(
        "eval_init",
        extra={
            "extra": {
                "model_artifact": str(args.model_artifact),
                "artifact_mode": artifact_mode,
                "manifest_schema_version": artifact_meta.get("schema_version"),
                "ensemble_set_index": artifact_meta.get("set_index"),
                "requested_ensemble_set_index": int(args.ensemble_set_index),
                "selected_set_via_runs_csv": bool(
                    args.select_best_set_runs_csv is not None
                ),
                "selected_set_runs_csv": (
                    str(args.select_best_set_runs_csv)
                    if args.select_best_set_runs_csv is not None
                    else None
                ),
                "k_folds": args.k_folds,
                "n_members": int(len(psp_models)),
                "threshold_mode": (
                    "global-override"
                    if args.threshold is not None
                    else "per-member-default"
                ),
                "threshold": float(member_thresholds[0]) if len(member_thresholds) == 1 else None,
                "member_thresholds": [float(x) for x in member_thresholds],
                "device": device,
                "model_cfg_src": (
                    str(member_model_cfg_srcs[0])
                    if len(set(member_model_cfg_srcs)) == 1
                    else "mixed"
                ),
                "emb_dim": int(first_cfg.emb_dim),
                "hidden_sizes": [int(x) for x in first_cfg.hidden_sizes],
                "use_layer_norm": bool(first_cfg.use_layer_norm),
                "use_residual": bool(first_cfg.use_residual),
                "num_classes": int(first_cfg.num_classes),
                "emit_fold_metrics": bool(args.emit_fold_metrics),
                "include_curves": bool(args.include_curves),
                "curve_max_points": int(args.curve_max_points),
                "best_fold_by": str(args.best_fold_by),
                "best_fold_direction": str(args.best_fold_direction),
                "plot_dir": str(args.plot_dir) if args.plot_dir is not None else None,
                "plot_formats": plot_formats if len(plot_formats) > 0 else None,
                "n_candidate_proteins": int(len(candidate_ids)),
                "n_selected_proteins": int(len(selected_ids))
            }
        }
    )

    eval_out = _evaluate_dataset(
        psp_models=psp_models,
        thresholds=member_thresholds,
        eval_loader=eval_loader,
        device=device,
        logger=logger,
        member_specs=member_specs,
        emit_fold_metrics=bool(args.emit_fold_metrics),
        include_curves=bool(args.include_curves),
        curve_max_points=int(args.curve_max_points),
        best_fold_by=str(args.best_fold_by),
        best_fold_direction=str(args.best_fold_direction),
        plot_dir=args.plot_dir,
        plot_formats=plot_formats
    )
    if not bool(eval_out.get("has_both_classes", False)):
        logger.warning(
            "eval_single_class_labels",
            extra={
                "extra": {
                    "pos_residues": int(eval_out["pos_residues"]),
                    "neg_residues": int(eval_out["neg_residues"]),
                    "note": (
                        "Evaluation labels contain only one class; ROC-style metrics may be undefined "
                        "and positive-class metrics may be degenerate."
                    ),
                }
            },
        )

    summary = {
        "model_artifact": str(args.model_artifact),
        "artifact_mode": artifact_mode,
        "manifest_schema_version": artifact_meta.get("schema_version"),
        "ensemble_set_index": artifact_meta.get("set_index"),
        "requested_ensemble_set_index": int(args.ensemble_set_index),
        "selected_set_runs_csv": (
            str(args.select_best_set_runs_csv)
            if args.select_best_set_runs_csv is not None
            else None
        ),
        "selected_set_meta": selected_set_meta,
        "best_set_by": (
            str(args.best_set_by)
            if args.select_best_set_runs_csv is not None
            else None
        ),
        "best_set_agg": (
            str(args.best_set_agg)
            if args.select_best_set_runs_csv is not None
            else None
        ),
        "best_set_direction": (
            str(args.best_set_direction)
            if args.select_best_set_runs_csv is not None
            else None
        ),
        "k_folds": args.k_folds,
        "n_members": int(len(psp_models)),
        "threshold_mode": (
            "global-override" if args.threshold is not None else "per-member-default"
        ),
        "threshold": float(member_thresholds[0]) if len(member_thresholds) == 1 else None,
        "member_thresholds": [float(x) for x in member_thresholds],
        "emit_fold_metrics": bool(args.emit_fold_metrics),
        "include_curves": bool(args.include_curves),
        "curve_max_points": int(args.curve_max_points),
        "best_fold_by": str(args.best_fold_by),
        "best_fold_direction": str(args.best_fold_direction),
        "plot_dir": str(args.plot_dir) if args.plot_dir is not None else None,
        "plot_formats": plot_formats if len(plot_formats) > 0 else None,
        "subset": int(args.subset),
        "n_candidate_proteins": int(len(candidate_ids)),
        "n_selected_proteins": int(len(selected_ids)),
        "embedding_dirs": [str(p) for p in args.embedding_dirs],
        "label_shards": [str(p) for p in args.label_shards],
        "device": device,
        "model_cfg_src": (
            str(member_model_cfg_srcs[0])
            if len(set(member_model_cfg_srcs)) == 1
            else "mixed"
        ),
        "emb_dim": int(first_cfg.emb_dim),
        "hidden_sizes": [int(x) for x in first_cfg.hidden_sizes],
        "use_layer_norm": bool(first_cfg.use_layer_norm),
        "use_residual": bool(first_cfg.use_residual),
        "num_classes": int(first_cfg.num_classes),
        "evaluation": eval_out
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_sanitize_for_json(summary), indent=2, allow_nan=False),
        encoding="utf-8"
    )

    logger.info(
        "eval_done",
        extra={
            "extra": {
                "output_json": str(args.output_json),
                "processed_proteins": int(eval_out["processed_proteins"]),
                "valid_residues": int(eval_out["valid_residues"]),
                "accuracy": eval_out["accuracy"],
                "metrics": eval_out["metrics"]
            }
        }
    )


if __name__ == "__main__":
    main()
