import json
import numpy as np
import pytest
from pepseqpred.core.train.curveartifacts import (
    _downsample_curve,
    build_roc_curve_payload,
    build_pr_curve_payload,
    write_validation_curve_artifacts
)

pytestmark = pytest.mark.unit


def test_downsample_curve_is_deterministic():
    x = np.linspace(0.0, 1.0, num=11, dtype=np.float64)
    y = x ** 2
    x_out, y_out = _downsample_curve(x=x, y=y, max_points=5)

    assert len(x_out) == 5
    assert len(y_out) == 5
    assert x_out[0] == pytest.approx(0.0)
    assert x_out[-1] == pytest.approx(1.0)
    assert y_out[0] == pytest.approx(0.0)
    assert y_out[-1] == pytest.approx(1.0)


def test_build_roc_curve_payload_handles_single_class():
    y_true = np.zeros(8, dtype=np.int64)
    y_prob = np.linspace(0.0, 1.0, num=8, dtype=np.float64)
    out = build_roc_curve_payload(y_true=y_true, y_prob=y_prob, max_points=32)

    assert out["available"] is False
    assert out["reason"] == "single-class-labels"
    assert out["fpr"] == []
    assert out["tpr"] == []


def test_build_pr_curve_payload_handles_no_valid_residues():
    y_true = np.asarray([], dtype=np.int64)
    y_prob = np.asarray([], dtype=np.float64)
    out = build_pr_curve_payload(y_true=y_true, y_prob=y_prob, max_points=32)

    assert out["available"] is False
    assert out["reason"] == "no-valid-residues"
    assert out["precision"] == []
    assert out["recall"] == []
    assert out["ap"] is None
    assert out["auprc_trapz"] is None


def test_write_validation_curve_artifacts_rejects_invalid_max_points(tmp_path):
    y_true = np.asarray([0, 1], dtype=np.int64)
    y_prob = np.asarray([0.2, 0.8], dtype=np.float64)
    with pytest.raises(ValueError, match="val-curve-max-points"):
        write_validation_curve_artifacts(
            epoch=0,
            y_true=y_true,
            y_prob=y_prob,
            metrics={},
            output_dir=tmp_path,
            max_points=1,
            plot_formats=("png",)
        )


def test_write_validation_curve_artifacts_writes_json_and_plots_when_available(tmp_path):
    y_true = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    y_prob = np.asarray([0.05, 0.9, 0.65, 0.3, 0.8, 0.25], dtype=np.float64)
    out = write_validation_curve_artifacts(
        epoch=0,
        y_true=y_true,
        y_prob=y_prob,
        metrics={"auc": float("nan"), "pr_auc": 0.77},
        output_dir=tmp_path,
        max_points=16,
        plot_formats=("png",)
    )

    json_path = tmp_path / "epoch_0000_curves.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["epoch"] == 0
    assert payload["eval_metrics"]["auc"] is None
    assert payload["eval_metrics"]["pr_auc"] == pytest.approx(0.77)
    assert isinstance(payload["roc_curve"]["fpr"], list)
    assert isinstance(payload["pr_curve"]["precision"], list)

    if out["plot_status"] == "ok":
        assert (tmp_path / "epoch_0000_roc_auc.png").exists()
        assert (tmp_path / "epoch_0000_pr_auc.png").exists()
    else:
        assert out["plot_status"] == "matplotlib_unavailable"
