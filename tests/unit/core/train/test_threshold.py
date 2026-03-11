import numpy as np
import pytest
from pepseqpred.core.train.threshold import find_threshold_max_recall_min_precision

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
