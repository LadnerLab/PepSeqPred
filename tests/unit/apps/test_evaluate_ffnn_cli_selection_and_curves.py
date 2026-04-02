from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pepseqpred.apps.evaluate_ffnn_cli import (
    _build_pr_curve_payload,
    _build_roc_curve_payload,
    _pick_best_fold,
    _resolve_best_set_index,
)

pytestmark = pytest.mark.unit


def test_resolve_best_set_index_by_mean_pr_auc(tmp_path: Path):
    runs_csv = tmp_path / "runs.csv"
    pd.DataFrame(
        {
            "EnsembleSetIndex": [1, 1, 2, 2, 3, 3],
            "FoldIndex": [1, 2, 1, 2, 1, 2],
            "PR_AUC": [0.20, 0.30, 0.41, 0.39, 0.31, 0.30],
        }
    ).to_csv(runs_csv, index=False)

    best_set, meta = _resolve_best_set_index(
        runs_csv_path=runs_csv,
        metric_name="PR_AUC",
        agg_name="mean",
        direction="max",
    )

    assert best_set == 2
    assert meta["metric_column"] == "PR_AUC"
    assert meta["best_set_index"] == 2
    assert meta["best_set_score"] == pytest.approx(0.40)


def test_resolve_best_set_index_auto_direction_uses_min_for_loss(tmp_path: Path):
    runs_csv = tmp_path / "runs.csv"
    pd.DataFrame(
        {
            "EnsembleSetIndex": [1, 1, 2, 2],
            "BestValLoss": [1.2, 1.1, 0.9, 1.0],
        }
    ).to_csv(runs_csv, index=False)

    best_set, meta = _resolve_best_set_index(
        runs_csv_path=runs_csv,
        metric_name="bestvalloss",
        agg_name="mean",
        direction="auto",
    )

    assert best_set == 2
    assert meta["direction"] == "min"
    assert meta["best_set_score"] == pytest.approx(0.95)


def test_build_roc_curve_payload_handles_single_class():
    y_true = np.asarray([0, 0, 0], dtype=np.int64)
    y_prob = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    out = _build_roc_curve_payload(y_true=y_true, y_prob=y_prob, max_points=64)
    assert out["available"] is False
    assert out["reason"] == "single-class-labels"
    assert out["fpr"] == []
    assert out["tpr"] == []


def test_build_curves_and_best_fold_selection():
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)

    roc = _build_roc_curve_payload(y_true=y_true, y_prob=y_prob, max_points=5)
    pr = _build_pr_curve_payload(y_true=y_true, y_prob=y_prob, max_points=5)
    assert roc["available"] is True
    assert len(roc["fpr"]) >= 2
    assert len(roc["fpr"]) <= 5
    assert pr["available"] is True
    assert pr["baseline_positive_rate"] == pytest.approx(0.5)

    fold_evaluations = [
        {"fold_index": 1, "metrics": {"pr_auc": 0.71}},
        {"fold_index": 2, "metrics": {"pr_auc": 0.79}},
        {"fold_index": 3, "metrics": {"pr_auc": 0.75}},
    ]
    best_idx, meta = _pick_best_fold(
        fold_evaluations=fold_evaluations,
        metric_name="PR_AUC",
        direction="auto",
    )
    assert best_idx == 1
    assert meta["best_fold_index"] == 2
    assert meta["best_score"] == pytest.approx(0.79)
