import torch
from typing import Dict, Any
from sklearn.metrics import (precision_recall_fscore_support,
                             average_precision_score,
                             matthews_corrcoef,
                             roc_auc_score,
                             roc_curve,
                             auc)


def compute_eval_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor) -> Dict[str, Any]:
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

    # calculate precesion, recall, f1, and mcc
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # ROC AUC
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    except Exception:
        metrics["auc"] = float("nan")

    # PR AUC
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    except Exception:
        metrics["pr_auc"] = float("nan")

    # AUC10 calculation]
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        mask = fpr <= 0.10
        if mask.sum() >= 2:
            metrics["auc10"] = float(auc(fpr[mask], tpr[mask]) / 0.10)

        else:
            metrics["auc10"] = float("nan")

    except Exception:
        metrics["auc10"] = float("nan")

    return metrics
