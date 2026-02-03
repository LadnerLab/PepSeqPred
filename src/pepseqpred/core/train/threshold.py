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


def find_threshold_max_recall_min_precision(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.50, num_thresholds: int = 999) -> Dict[str, Any]:
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
        num_thresholds : int
            The number of different thresholds to try between 0.001 and 0.999. Default is `999`.

    Returns
    -------
        Dict[str, Any]
            A dictionary containing the most optimal threshold, confusion matrix 
            used to calculate the most optimal threshold, precision and recall 
            at that threshold, the minimum precision accepted, and the status 
            which is either `"ok"` if precision >= `min_precision` otherwise 
            `"min_precision_unreachable"`.
    """
    thresholds = np.linspace(0.001, 0.999, num_thresholds, dtype=np.float64)

    best: Optional[Dict[str, Any]] = None
    best_fallback: Optional[Dict[str, Any]] = None

    for thresh in thresholds:
        tp, fp, tn, fn = _confusion_from_probs(y_true, y_prob, thresh)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)

        row = {"threshold": float(thresh),
               "tp": tp, "fp": fp, "tn": tn, "fn": fn,
               "precision": precision,
               "recall": recall}

        # always hold fallback which maximizes threshold in case precision is always < min_precision
        if best_fallback is None:
            best_fallback = row
        elif (row["precision"] > best_fallback["precision"]) or (
                row["precision"] == best_fallback["precision"] and
                row["threshold"] > best_fallback["threshold"]):
            best_fallback = row

        # best case when precision > min_precision and/or recall is increasing
        if precision >= min_precision:
            if best is None:
                best = row
            elif (row["recall"] > best["recall"] or (
                  row["recall"] == best["recall"] and
                  row["precision"] > best["precision"])):
                best = row

    if best is not None:
        best["status"] = "ok"
        best["min_precision"] = min_precision
        return best

    # minimum precision contraint failed
    best_fallback["status"] = "min_precision_unreachable"
    best_fallback["min_precision"] = min_precision
    return best_fallback
