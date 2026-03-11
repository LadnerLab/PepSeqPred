"""metrics.py

Evaluation metric helpers for PepSeqPred training.

Provides a convenience function to compute common binary classification metrics
from labels, predictions, and probabilities.
"""

from typing import Dict, Any, Union, Sequence
import numpy as np
import torch
from sklearn.metrics import (precision_recall_fscore_support,
                             average_precision_score,
                             matthews_corrcoef,
                             roc_auc_score,
                             roc_curve,
                             auc)


ArrayLike1D = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int]]


def _to_numpy_1d(x: ArrayLike1D) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def compute_eval_metrics(y_true: ArrayLike1D, y_pred: ArrayLike1D, y_prob: ArrayLike1D) -> Dict[str, Any]:
    """
    Computes evaluation metrics given true lables, predicted labels, and predicted probabilities.

    Parameters
    ----------
        y_true : Tensor
            True class labels.
        y_pred : Tensor
            Predicted class labels.
        y_prob : Tensor
            Predicted class probabilities.

    Returns
    -------
        metrics : Dict[str, Any]
            Dictionary of evaluation metrics.
    """
    metrics: Dict[str, Any] = {}

    y_true_np = _to_numpy_1d(y_true).astype(np.int64, copy=False)
    y_pred_np = _to_numpy_1d(y_pred).astype(np.int64, copy=False)
    y_prob_np = _to_numpy_1d(y_prob).astype(np.float64, copy=False)

    # calculate precision, recall, and f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="binary", zero_division=0)
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    # Avoid sklearn warning when both tensors contain only one shared label.
    if np.unique(np.concatenate((y_true_np, y_pred_np))).size < 2:
        metrics["mcc"] = 0.0
    else:
        metrics["mcc"] = float(matthews_corrcoef(y_true_np, y_pred_np))

    has_both_classes = np.unique(y_true_np).size >= 2
    if not has_both_classes:
        only_class = int(y_true_np[0]) if y_true_np.size > 0 else 0
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = 1.0 if only_class == 1 else 0.0
        metrics["auc10"] = float("nan")
        return metrics

    # ROC AUC
    try:
        metrics["auc"] = float(roc_auc_score(y_true_np, y_prob_np))

    except Exception:
        metrics["auc"] = float("nan")

    # PR AUC
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_np, y_prob_np))

    except Exception:
        metrics["pr_auc"] = float("nan")

    # AUC10 calculation]
    try:
        fpr, tpr, _ = roc_curve(y_true_np, y_prob_np)
        mask = fpr <= 0.10
        if mask.sum() >= 2:
            metrics["auc10"] = float(auc(fpr[mask], tpr[mask]) / 0.10)

        else:
            metrics["auc10"] = float("nan")

    except Exception:
        metrics["auc10"] = float("nan")

    return metrics
