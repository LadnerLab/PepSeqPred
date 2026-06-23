"""Threshold selection utilities for PepSeqPred classification outputs."""

from typing import Any, Dict, Sequence, Tuple

import numpy as np


THRESHOLD_POLICIES: Tuple[str, ...] = (
    "max-recall-min-precision",
    "best-f1",
    "best-mcc",
    "min-recall-max-precision",
    "fixed",
)

DEFAULT_THRESHOLD_GRID: Tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
)


def _safe_divide(n: float, d: float) -> float:
    """Checks for divide by zero before division operation."""
    return float(n / d) if d != 0.0 else 0.0


def _validate_probability(value: float, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0 or out >= 1.0:
        raise ValueError(f"{name} must be in (0.0, 1.0)")
    return out


def _validate_constraint(value: float, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")
    return out


def _as_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true_np = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
    y_prob_np = np.asarray(y_prob).reshape(-1).astype(np.float64, copy=False)
    if y_true_np.shape[0] != y_prob_np.shape[0]:
        raise ValueError(
            f"y_true/y_prob length mismatch: {y_true_np.shape[0]} != {y_prob_np.shape[0]}"
        )
    if y_prob_np.size > 0 and not bool(np.isfinite(y_prob_np).all()):
        raise ValueError("y_prob must contain only finite values")
    return y_true_np, y_prob_np


def _mcc_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0.0:
        return 0.0
    return float(((tp * tn) - (fp * fn)) / np.sqrt(denom))


def _row_from_counts(
    threshold: float,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Dict[str, Any]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    mcc = _mcc_from_counts(tp, fp, tn, fn)
    total = int(tp + fp + tn + fn)
    pred_pos = int(tp + fp)
    pred_neg = int(tn + fn)
    support_pos = int(tp + fn)
    support_neg = int(tn + fp)
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "support_pos": support_pos,
        "support_neg": support_neg,
        "valid_residues": total,
        "pred_pos": pred_pos,
        "pred_neg": pred_neg,
        "pred_pos_frac": _safe_divide(pred_pos, total),
    }


def _confusion_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[int, int, int, int]:
    """Builds a confusion matrix given model probabilities for threshold calculation."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    # return order: true pos, false pos, true neg, false neg
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    return tp, fp, tn, fn


def _row_from_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    tp, fp, tn, fn = _confusion_from_probs(y_true, y_prob, threshold)
    return _row_from_counts(threshold, tp, fp, tn, fn)


def _candidate_rows(y_true: np.ndarray, y_prob: np.ndarray) -> Sequence[Dict[str, Any]]:
    if y_true.size == 0:
        return []

    order = np.argsort(y_prob, kind="mergesort")[::-1]
    y_sorted = y_true.astype(np.int64, copy=False)[order]
    p_sorted = y_prob.astype(np.float64, copy=False)[order]

    is_pos = (y_sorted == 1).astype(np.int64)
    is_neg = 1 - is_pos
    tp_cum = np.cumsum(is_pos)
    fp_cum = np.cumsum(is_neg)
    total_pos = int(is_pos.sum())
    total_neg = int(is_neg.sum())
    last_index = np.flatnonzero(np.r_[p_sorted[1:] != p_sorted[:-1], True])

    rows = []
    for index in last_index:
        tp = int(tp_cum[index])
        fp = int(fp_cum[index])
        fn = int(total_pos - tp)
        tn = int(total_neg - fp)
        rows.append(_row_from_counts(float(p_sorted[index]), tp, fp, tn, fn))

    # If possible, include the valid threshold just below 1.0 that predicts
    # no positive residues. This keeps the legacy selector behavior.
    near_one = float(np.nextafter(1.0, 0.0))
    if float(p_sorted[0]) < near_one:
        rows.append(_row_from_counts(near_one, 0, 0, total_neg, total_pos))

    return rows


def _empty_selection(
    policy: str,
    min_precision: float,
    min_recall: float,
    fixed_threshold: float,
) -> Dict[str, Any]:
    return {
        "threshold": float("nan"),
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "mcc": float("nan"),
        "support_pos": 0,
        "support_neg": 0,
        "valid_residues": 0,
        "pred_pos": 0,
        "pred_neg": 0,
        "pred_pos_frac": float("nan"),
        "status": "no_valid_residues",
        "policy": policy,
        "min_precision": float(min_precision),
        "min_recall": float(min_recall),
        "fixed_threshold": float(fixed_threshold),
    }


def _with_policy(
    row: Dict[str, Any],
    *,
    policy: str,
    status: str,
    min_precision: float,
    min_recall: float,
    fixed_threshold: float,
) -> Dict[str, Any]:
    out = dict(row)
    out["status"] = status
    out["policy"] = policy
    out["min_precision"] = float(min_precision)
    out["min_recall"] = float(min_recall)
    out["fixed_threshold"] = float(fixed_threshold)
    return out


def _best_row(rows: Sequence[Dict[str, Any]], keys: Sequence[Tuple[str, bool]]) -> Dict[str, Any]:
    def key_fn(row: Dict[str, Any]) -> Tuple[float, ...]:
        values = []
        for key, higher_is_better in keys:
            value = float(row[key])
            values.append(value if higher_is_better else -value)
        return tuple(values)

    return max(rows, key=key_fn)


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    policy: str = "max-recall-min-precision",
    min_precision: float = 0.25,
    min_recall: float = 0.80,
    fixed_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Selects a decision threshold from validation labels and probabilities."""
    policy = str(policy).strip().lower()
    if policy not in THRESHOLD_POLICIES:
        raise ValueError(
            f"Unsupported threshold policy '{policy}'. Expected one of: {', '.join(THRESHOLD_POLICIES)}"
        )
    min_precision = _validate_constraint(min_precision, "min_precision")
    min_recall = _validate_constraint(min_recall, "min_recall")
    fixed_threshold = _validate_probability(fixed_threshold, "fixed_threshold")
    y_true_np, y_prob_np = _as_arrays(y_true, y_prob)

    if y_true_np.size == 0:
        return _empty_selection(policy, min_precision, min_recall, fixed_threshold)

    if policy == "fixed":
        row = _row_from_threshold(y_true_np, y_prob_np, fixed_threshold)
        return _with_policy(
            row,
            policy=policy,
            status="ok",
            min_precision=min_precision,
            min_recall=min_recall,
            fixed_threshold=fixed_threshold,
        )

    rows = _candidate_rows(y_true_np, y_prob_np)
    if len(rows) == 0:
        return _empty_selection(policy, min_precision, min_recall, fixed_threshold)

    if policy == "max-recall-min-precision":
        eligible = [row for row in rows if float(row["precision"]) >= min_precision]
        if eligible:
            row = _best_row(
                eligible,
                (
                    ("recall", True),
                    ("precision", True),
                    ("threshold", True),
                ),
            )
            return _with_policy(
                row,
                policy=policy,
                status="ok",
                min_precision=min_precision,
                min_recall=min_recall,
                fixed_threshold=fixed_threshold,
            )
        row = _best_row(rows, (("precision", True), ("threshold", True)))
        return _with_policy(
            row,
            policy=policy,
            status="min_precision_unreachable",
            min_precision=min_precision,
            min_recall=min_recall,
            fixed_threshold=fixed_threshold,
        )

    if policy == "best-f1":
        row = _best_row(
            rows,
            (
                ("f1", True),
                ("precision", True),
                ("recall", True),
                ("threshold", True),
            ),
        )
        return _with_policy(
            row,
            policy=policy,
            status="ok",
            min_precision=min_precision,
            min_recall=min_recall,
            fixed_threshold=fixed_threshold,
        )

    if policy == "best-mcc":
        row = _best_row(
            rows,
            (
                ("mcc", True),
                ("f1", True),
                ("precision", True),
                ("recall", True),
                ("threshold", True),
            ),
        )
        return _with_policy(
            row,
            policy=policy,
            status="ok",
            min_precision=min_precision,
            min_recall=min_recall,
            fixed_threshold=fixed_threshold,
        )

    eligible = [row for row in rows if float(row["recall"]) >= min_recall]
    if eligible:
        row = _best_row(
            eligible,
            (
                ("precision", True),
                ("recall", True),
                ("threshold", True),
            ),
        )
        return _with_policy(
            row,
            policy=policy,
            status="ok",
            min_precision=min_precision,
            min_recall=min_recall,
            fixed_threshold=fixed_threshold,
        )

    row = _best_row(
        rows,
        (
            ("recall", True),
            ("precision", True),
            ("threshold", True),
        ),
    )
    return _with_policy(
        row,
        policy=policy,
        status="min_recall_unreachable",
        min_precision=min_precision,
        min_recall=min_recall,
        fixed_threshold=fixed_threshold,
    )


def threshold_diagnostic_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> Sequence[Dict[str, Any]]:
    """Computes fixed-threshold diagnostics for validation/evaluation payloads."""
    y_true_np, y_prob_np = _as_arrays(y_true, y_prob)
    if y_true_np.size == 0:
        return []
    rows = []
    for threshold in thresholds:
        threshold_f = _validate_probability(threshold, "threshold")
        row = _row_from_threshold(y_true_np, y_prob_np, threshold_f)
        rows.append(row)
    return rows


def find_threshold_max_recall_min_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.50,
) -> Dict[str, Any]:
    """
    Finds the threshold that maximizes recall subject to precision >= min_precision.

    This compatibility wrapper preserves the public helper used by older code.
    """
    return select_threshold(
        y_true,
        y_prob,
        policy="max-recall-min-precision",
        min_precision=min_precision,
    )
