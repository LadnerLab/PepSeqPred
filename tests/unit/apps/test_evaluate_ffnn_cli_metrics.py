import math
import numpy as np
import pytest
from pepseqpred.apps.evaluate_ffnn_cli import (
    _augment_metrics_with_confusion_stats,
)

pytestmark = pytest.mark.unit


def test_augment_metrics_with_confusion_stats_single_class_negative():
    cm = np.array([[73, 27], [0, 0]], dtype=np.int64)
    metrics = _augment_metrics_with_confusion_stats(metrics={}, cm=cm)

    assert metrics["support_neg"] == 100
    assert metrics["support_pos"] == 0
    assert metrics["has_both_classes"] is False
    assert metrics["specificity"] == pytest.approx(0.73)
    assert metrics["fpr"] == pytest.approx(0.27)
    assert metrics["npv"] == pytest.approx(1.0)
    assert metrics["per_class_acc"][0] == pytest.approx(0.73)
    assert math.isnan(metrics["per_class_acc"][1])
    assert metrics["res_balanced_acc"] == pytest.approx(0.73)


def test_augment_metrics_with_confusion_stats_two_class():
    cm = np.array([[40, 10], [5, 45]], dtype=np.int64)
    metrics = _augment_metrics_with_confusion_stats(metrics={}, cm=cm)

    assert metrics["support_neg"] == 50
    assert metrics["support_pos"] == 50
    assert metrics["has_both_classes"] is True
    assert metrics["specificity"] == pytest.approx(0.8)
    assert metrics["fpr"] == pytest.approx(0.2)
    assert metrics["npv"] == pytest.approx(40 / 45)
    assert metrics["per_class_acc"] == pytest.approx([0.8, 0.9])
    assert metrics["res_balanced_acc"] == pytest.approx(0.85)
