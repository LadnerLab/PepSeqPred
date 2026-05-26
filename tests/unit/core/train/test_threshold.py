import numpy as np
import pytest
from pepseqpred.core.train.threshold import (
    DEFAULT_THRESHOLD_GRID,
    find_threshold_max_recall_min_precision,
    select_threshold,
    threshold_diagnostic_grid,
)

pytestmark = pytest.mark.unit


def test_threshold_no_valid_residues():
    out = find_threshold_max_recall_min_precision(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        min_precision=0.25,
    )
    assert out["status"] == "no_valid_residues"


def test_threshold_respects_min_precision():
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)
    y_prob = np.array([0.9, 0.8, 0.4, 0.1], dtype=np.float64)
    out = find_threshold_max_recall_min_precision(
        y_true, y_prob, min_precision=0.5)
    assert out["status"] == "ok"
    assert out["precision"] >= 0.5


def test_threshold_min_precision_unreachable_falls_back_to_best_precision():
    y_true = np.array([1, 0, 0], dtype=np.int64)
    y_prob = np.array([0.1, 0.9, 0.8], dtype=np.float64)
    out = select_threshold(
        y_true,
        y_prob,
        policy="max-recall-min-precision",
        min_precision=0.8,
    )
    assert out["status"] == "min_precision_unreachable"
    assert out["threshold"] == pytest.approx(0.1)
    assert out["precision"] == pytest.approx(1.0 / 3.0)


def test_threshold_best_f1_and_mcc_pick_less_conservative_operating_point():
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)
    y_prob = np.array([0.9, 0.4, 0.8, 0.1], dtype=np.float64)
    f1_out = select_threshold(y_true, y_prob, policy="best-f1")
    mcc_out = select_threshold(y_true, y_prob, policy="best-mcc")
    assert f1_out["status"] == "ok"
    assert f1_out["threshold"] == pytest.approx(0.4)
    assert f1_out["f1"] == pytest.approx(0.8)
    assert mcc_out["threshold"] == pytest.approx(0.4)


def test_threshold_min_recall_max_precision():
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)
    y_prob = np.array([0.9, 0.3, 0.8, 0.1], dtype=np.float64)
    out = select_threshold(
        y_true,
        y_prob,
        policy="min-recall-max-precision",
        min_recall=1.0,
    )
    assert out["status"] == "ok"
    assert out["threshold"] == pytest.approx(0.3)
    assert out["recall"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(2.0 / 3.0)


def test_threshold_fixed_policy_and_grid():
    y_true = np.array([1, 0, 1, 0], dtype=np.int64)
    y_prob = np.array([0.7, 0.6, 0.2, 0.1], dtype=np.float64)
    out = select_threshold(
        y_true,
        y_prob,
        policy="fixed",
        fixed_threshold=0.5,
    )
    assert out["status"] == "ok"
    assert out["policy"] == "fixed"
    assert out["threshold"] == pytest.approx(0.5)
    assert out["tp"] == 1
    assert out["fp"] == 1
    grid = threshold_diagnostic_grid(y_true, y_prob)
    assert len(grid) == len(DEFAULT_THRESHOLD_GRID)
    assert {row["threshold"] for row in grid} == set(DEFAULT_THRESHOLD_GRID)


def test_threshold_invalid_inputs_raise():
    y_true = np.array([1, 0], dtype=np.int64)
    y_prob = np.array([0.9, 0.1], dtype=np.float64)
    with pytest.raises(ValueError, match="Unsupported threshold policy"):
        select_threshold(y_true, y_prob, policy="bogus")
    with pytest.raises(ValueError, match="fixed_threshold"):
        select_threshold(y_true, y_prob, policy="fixed", fixed_threshold=1.0)
    with pytest.raises(ValueError, match="length mismatch"):
        select_threshold(y_true, np.array([0.5], dtype=np.float64))
