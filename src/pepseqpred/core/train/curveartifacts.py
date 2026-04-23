"""curve_artifacts.py

Validation ROC/PR curve artifact helpers for training.

This module builds deterministic, downsampled ROC/PR payloads from validation
arrays, writes JSON sidecars, and optionally renders plots when matplotlib is
available.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve
)


def _sanitize_for_json(value: Any) -> Any:
    """Recursively converts non-finite floats to None for strict JSON output."""
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _downsample_curve(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int
) -> Tuple[List[float], List[float]]:
    """Downsamples curve points deterministically to keep payloads bounded."""
    if x.size != y.size:
        raise ValueError(f"Curve x/y size mismatch: x={x.size} y={y.size}")
    if max_points < 2:
        raise ValueError("--val-curve-max-points must be >= 2")
    if x.size <= max_points:
        return [float(v) for v in x], [float(v) for v in y]

    idx = np.linspace(0, x.size - 1, num=max_points, dtype=np.int64)
    idx = np.unique(idx)
    return [float(v) for v in x[idx]], [float(v) for v in y[idx]]


def build_roc_curve_payload(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int
) -> Dict[str, Any]:
    """Builds ROC curve payload from validation labels/probabilities."""
    if max_points < 2:
        raise ValueError("--val-curve-max-points must be >= 2")
    if y_true.size < 1 or np.unique(y_true).size < 2:
        return {
            "available": False,
            "reason": "single-class-labels",
            "fpr": [],
            "tpr": []
        }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr_points, tpr_points = _downsample_curve(
        x=np.asarray(fpr, dtype=np.float64),
        y=np.asarray(tpr, dtype=np.float64),
        max_points=max_points
    )
    return {
        "available": True,
        "reason": None,
        "fpr": fpr_points,
        "tpr": tpr_points
    }


def _compute_pr_auc_trapezoid(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes PR AUC by trapezoidal integration."""
    if y_true.size < 1:
        return float("nan")

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        return 1.0 if int(unique_labels[0]) == 1 else 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    recall = np.asarray(recall, dtype=np.float64)[::-1]
    precision = np.asarray(precision, dtype=np.float64)[::-1]
    return float(np.trapezoid(precision, recall))


def build_pr_curve_payload(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int
) -> Dict[str, Any]:
    """Builds PR curve payload from validation labels/probabilities."""
    if max_points < 2:
        raise ValueError("--val-curve-max-points must be >= 2")
    if y_true.size < 1:
        return {
            "available": False,
            "reason": "no-valid-residues",
            "precision": [],
            "recall": [],
            "baseline_positive_rate": None,
            "ap": None,
            "auprc_trapz": None
        }

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    recall = np.asarray(recall, dtype=np.float64)[::-1]
    precision = np.asarray(precision, dtype=np.float64)[::-1]

    try:
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap = float("nan")
    try:
        auprc_trapz = float(_compute_pr_auc_trapezoid(y_true, y_prob))
    except Exception:
        auprc_trapz = float("nan")

    recall_points, precision_points = _downsample_curve(
        x=recall,
        y=precision,
        max_points=max_points
    )
    pos_rate = float((y_true == 1).mean()) if y_true.size > 0 else float("nan")
    return {
        "available": True,
        "reason": None,
        "precision": precision_points,
        "recall": recall_points,
        "baseline_positive_rate": (
            float(pos_rate) if math.isfinite(pos_rate) else None
        ),
        "ap": float(ap) if math.isfinite(ap) else None,
        "auprc_trapz": float(auprc_trapz) if math.isfinite(auprc_trapz) else None
    }


def _plot_curve(
    plt: Any,
    curve_payload: Mapping[str, Any],
    path_base: Path,
    formats: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    x_key: str,
    y_key: str,
    chance_line: bool = False,
    baseline_y: float | None = None
) -> List[str]:
    """Plots a single curve panel and writes one file per requested format."""
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=150)

    available = bool(curve_payload.get("available", False))
    if available:
        x_vals = curve_payload.get(x_key, [])
        y_vals = curve_payload.get(y_key, [])
        if not isinstance(x_vals, list) or not isinstance(y_vals, list):
            x_vals = []
            y_vals = []
        if len(x_vals) >= 2 and len(y_vals) >= 2:
            ax.plot(x_vals, y_vals, linewidth=2.0, color="#1f77b4")
        else:
            available = False

    if not available:
        reason = str(curve_payload.get("reason", "unavailable"))
        ax.text(
            0.5,
            0.5,
            f"Curve unavailable:\n{reason}",
            ha="center",
            va="center",
            fontsize=11
        )

    if chance_line:
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--",
                linewidth=1.0, color="#888888")
    if baseline_y is not None and math.isfinite(float(baseline_y)):
        y_val = float(baseline_y)
        ax.plot([0.0, 1.0], [y_val, y_val], linestyle="--",
                linewidth=1.0, color="#B05A2B")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    saved_paths: List[str] = []
    path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = path_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved_paths.append(str(out_path))
    plt.close(fig)
    return saved_paths


def write_validation_curve_artifacts(
    epoch: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Mapping[str, Any] | None,
    output_dir: Path | str,
    max_points: int = 2048,
    plot_formats: Sequence[str] = ("png",)
) -> Dict[str, Any]:
    """Writes validation curve JSON payload and optional ROC/PR plots."""
    if max_points < 2:
        raise ValueError("--val-curve-max-points must be >= 2")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_np = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
    y_prob_np = np.asarray(y_prob).reshape(-1).astype(np.float64, copy=False)
    if y_true_np.size != y_prob_np.size:
        raise ValueError(
            f"Expected y_true/y_prob length match, got y_true={y_true_np.size} y_prob={y_prob_np.size}"
        )

    fmts = [str(fmt).strip().lower() for fmt in plot_formats if str(fmt).strip()]
    if len(fmts) < 1:
        raise ValueError("--val-plot-formats must include at least one format")
    allowed = {"png", "svg", "pdf"}
    bad = [fmt for fmt in fmts if fmt not in allowed]
    if len(bad) > 0:
        raise ValueError(
            f"Unsupported --val-plot-formats values={bad}; allowed={sorted(allowed)}"
        )

    roc_payload = build_roc_curve_payload(
        y_true=y_true_np,
        y_prob=y_prob_np,
        max_points=max_points
    )
    pr_payload = build_pr_curve_payload(
        y_true=y_true_np,
        y_prob=y_prob_np,
        max_points=max_points
    )

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "n_residues": int(y_true_np.size),
        "max_points": int(max_points),
        "plot_formats": [str(fmt) for fmt in fmts],
        "eval_metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
        "roc_curve": roc_payload,
        "pr_curve": pr_payload
    }

    epoch_tag = f"epoch_{int(epoch):04d}"
    json_path = output_dir / f"{epoch_tag}_curves.json"
    json_path.write_text(
        json.dumps(_sanitize_for_json(payload), indent=2, allow_nan=False),
        encoding="utf-8"
    )

    artifact_paths: Dict[str, Any] = {
        "curve_json": str(json_path),
        "roc_auc_plots": [],
        "pr_auc_plots": [],
        "plot_status": "ok"
    }

    try:
        import matplotlib.pyplot as plt
    except Exception:
        artifact_paths["plot_status"] = "matplotlib_unavailable"
        return artifact_paths

    baseline_y = pr_payload.get("baseline_positive_rate")
    baseline_float = (
        float(baseline_y)
        if baseline_y is not None and math.isfinite(float(baseline_y))
        else None
    )

    artifact_paths["roc_auc_plots"] = _plot_curve(
        plt=plt,
        curve_payload=roc_payload,
        path_base=output_dir / f"{epoch_tag}_roc_auc",
        formats=fmts,
        title=f"Validation ROC Curve (Epoch {int(epoch)})",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        x_key="fpr",
        y_key="tpr",
        chance_line=True
    )
    artifact_paths["pr_auc_plots"] = _plot_curve(
        plt=plt,
        curve_payload=pr_payload,
        path_base=output_dir / f"{epoch_tag}_pr_auc",
        formats=fmts,
        title=f"Validation PR Curve (Epoch {int(epoch)})",
        x_label="Recall",
        y_label="Precision",
        x_key="recall",
        y_key="precision",
        chance_line=False,
        baseline_y=baseline_float
    )
    return artifact_paths
