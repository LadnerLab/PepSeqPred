"""threshold.py

Threshold selection utilities for PepSeqPred classification outputs.

Provides helpers to compute confusion statistics from probabilities and to
select a threshold that maximizes recall subject to a minimum precision.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np


def _safe_divide(n: float, d: float) -> float:
    """Checks for divide by zero before division operation."""
    return float(n / d) if d != 0.0 else 0.0


def _confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    """Builds a confusion matrix given model probabilities for threshold calculation."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true = y_true.astype(np.int64)

    # return order: true pos, false pos, true neg, false neg
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    return tp, fp, tn, fn


def find_threshold_max_recall_min_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.50
) -> Dict[str, Any]:
    """
    Finds the threshold that maximizes recall subject such that `precision >= min_precision`. 
    If no precision meets this constraint, the best possible threshold is returned.

    Parameters
    ----------
        y_true : ndarray
            An array of the true labels for a batch of residues.
        y_prob : ndarray
            An array of the model's estimated probabilities that a residue is an epitope for a given batch.
        min_precision : float
            Minimum accepted precision while recall is optimized. Default is `0.50`.

    Returns
    -------
        Dict[str, Any]
            A dictionary containing the most optimal threshold, confusion matrix 
            used to calculate the most optimal threshold, precision and recall 
            at that threshold, the minimum precision accepted, and the status 
            which is either `"ok"` if precision >= `min_precision` otherwise 
            `"min_precision_unreachable"`.
    """
    if y_true.size == 0:
        return {
            "threshold": float("nan"),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "status": "no_valid_residues",
            "min_precision": min_precision
        }

    # sort highest to lowest predicted probs
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

    best: Optional[Dict[str, Any]] = None
    best_fallback: Optional[Dict[str, Any]] = None

    def _consider(row: Dict[str, Any]) -> None:
        """Best threshold and best fallback computation."""
        nonlocal best, best_fallback
        if best_fallback is None or (
            row["precision"] > best_fallback["precision"] or (
                row["precision"] == best_fallback["precision"] and
                row["threshold"] > best_fallback["threshold"]
            )
        ):
            best_fallback = row
        if row["precision"] >= min_precision:
            if best is None or (
                row["recall"] > best["recall"] or (
                    row["recall"] == best["recall"] and
                    row["precision"] > best["precision"]
                ) or (
                    row["recall"] == best["recall"] and
                    row["precision"] == best["precision"] and
                    row["threshold"] > best["threshold"]
                )
            ):
                best = row

    for index in last_index:
        thresh = float(p_sorted[index])
        tp = int(tp_cum[index])
        fp = int(fp_cum[index])
        fn = int(total_pos - tp)
        tn = int(total_neg - fp)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)

        row = {"threshold": float(thresh),
               "tp": tp, "fp": fp, "tn": tn, "fn": fn,
               "precision": precision,
               "recall": recall}

        # get best fallback and best thresholds
        _consider(row)

    # if max prob is below valid threshold (<1), represents "predict no positives"
    near_one = float(np.nextafter(1.0, 0.0))
    if float(p_sorted[0]) < near_one:
        _consider({
            "threshold": near_one,
            "tp": 0, "fp": 0, "tn": total_neg, "fn": total_pos,
            "precision": 0.0,
            "recall": 0.0
        })

    if best is not None:
        best["status"] = "ok"
        best["min_precision"] = min_precision
        return best

    # minimum precision contraint failed
    if best_fallback is not None:
        best_fallback["status"] = "min_precision_unreachable"
        best_fallback["min_precision"] = min_precision

    return best_fallback if best_fallback is not None else {
        "threshold": float("nan"),
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "precision": float("nan"),
        "recall": float("nan"),
        "status": "no_valid_residues",
        "min_precision": min_precision
    }
