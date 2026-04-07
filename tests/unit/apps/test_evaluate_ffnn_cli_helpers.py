import argparse
import json
import math
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import torch

import pepseqpred.apps.evaluate_ffnn_cli as eval_mod
from pepseqpred.apps.evaluate_ffnn_cli import (
    EvaluationMember,
    _as_optional_int,
    _as_optional_threshold,
    _build_cli_model_config,
    _build_eval_output_from_arrays,
    _build_pr_curve_payload,
    _compute_pr_auc_trapezoid,
    _downsample_curve,
    _empty_eval_output,
    _evaluate_dataset,
    _parse_plot_formats,
    _pick_best_fold,
    _resolve_best_set_index,
    _resolve_evaluation_members,
    _resolve_eval_metric_value,
    _resolve_manifest_members,
    _resolve_metric_column,
    _resolve_pr_zoom_limits,
    _safe_ratio,
    _sanitize_for_json,
)

pytestmark = pytest.mark.unit


@contextmanager
def _scratch_dir():
    root = Path("localdata") / "tmp_pytest_eval_helpers"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _cli_args(**overrides):
    base = {
        "emb_dim": None,
        "hidden_sizes": None,
        "dropouts": None,
        "use_layer_norm": None,
        "use_residual": None,
        "num_classes": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_metric_column_and_best_set_index_validations():
    assert _resolve_metric_column(["PR_AUC", "Loss"], "prauc") == "PR_AUC"
    with pytest.raises(ValueError, match="Metric column"):
        _resolve_metric_column(["PR_AUC"], "mcc")

    with _scratch_dir() as tmp_path:
        missing_path = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match="runs.csv not found"):
            _resolve_best_set_index(
                runs_csv_path=missing_path,
                metric_name="PR_AUC",
                agg_name="mean",
                direction="max"
            )

        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"PR_AUC": [0.2, 0.4]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="EnsembleSetIndex"):
            _resolve_best_set_index(
                runs_csv_path=bad_csv,
                metric_name="PR_AUC",
                agg_name="mean",
                direction="max"
            )

        no_valid = tmp_path / "no_valid.csv"
        pd.DataFrame(
            {
                "EnsembleSetIndex": [1, 2],
                "PR_AUC": [0.2, 0.3],
                "Status": ["FAIL", "FAIL"]
            }
        ).to_csv(no_valid, index=False)
        with pytest.raises(ValueError, match="No valid rows"):
            _resolve_best_set_index(
                runs_csv_path=no_valid,
                metric_name="PR_AUC",
                agg_name="mean",
                direction="max"
            )

        good_csv = tmp_path / "good.csv"
        pd.DataFrame(
            {
                "EnsembleSetIndex": [1, 1, 2, 2],
                "PR_AUC": [0.6, 0.5, 0.7, 0.65]
            }
        ).to_csv(good_csv, index=False)
        with pytest.raises(ValueError, match="best-set-agg"):
            _resolve_best_set_index(
                runs_csv_path=good_csv,
                metric_name="PR_AUC",
                agg_name="bad",
                direction="max"
            )
        with pytest.raises(ValueError, match="best-set-direction"):
            _resolve_best_set_index(
                runs_csv_path=good_csv,
                metric_name="PR_AUC",
                agg_name="mean",
                direction="weird"
            )


def test_build_cli_model_config_and_optional_parsers():
    assert _build_cli_model_config(_cli_args()) is None

    with pytest.raises(ValueError, match="provide all required values"):
        _build_cli_model_config(_cli_args(emb_dim=4))

    with pytest.raises(ValueError, match="same length"):
        _build_cli_model_config(
            _cli_args(
                emb_dim=4,
                hidden_sizes="8,4",
                dropouts="0.1",
                use_layer_norm=True,
                use_residual=False,
                num_classes=1
            )
        )

    cfg = _build_cli_model_config(
        _cli_args(
            emb_dim=4,
            hidden_sizes="8,4",
            dropouts="0.1,0.2",
            use_layer_norm=True,
            use_residual=False,
            num_classes=1
        )
    )
    assert cfg is not None
    assert cfg.emb_dim == 4
    assert cfg.hidden_sizes == (8, 4)
    assert cfg.dropouts == (0.1, 0.2)

    assert _as_optional_int(None) is None
    assert _as_optional_int("3") == 3
    assert _as_optional_int("abc") is None
    assert _as_optional_threshold(None) is None
    assert _as_optional_threshold("0.4") == pytest.approx(0.4)
    assert _as_optional_threshold("bad") is None
    assert _as_optional_threshold(float("inf")) is None
    assert _as_optional_threshold(0.0) is None


def test_manifest_and_evaluation_member_resolution():
    with _scratch_dir() as tmp_path:
        ckpt_1 = tmp_path / "m1.pt"
        ckpt_2 = tmp_path / "m2.pt"
        ckpt_1.touch()
        ckpt_2.touch()

        manifest_v2 = {
            "schema_version": 2,
            "sets": [
                {"set_index": 1, "members": [{"status": "OK", "checkpoint": str(ckpt_1.resolve())}]},
                {
                    "set_index": 2,
                    "members": [
                        {"status": "OK", "checkpoint": str(ckpt_2.resolve()), "fold_index": 2, "member_index": 1, "threshold": 0.55},
                        {"status": "FAIL", "checkpoint": str(ckpt_1.resolve()), "fold_index": 1, "member_index": 1}
                    ]
                }
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_v2), encoding="utf-8")

        members, meta = _resolve_manifest_members(manifest_path, ensemble_set_index=2)
        assert len(members) == 1
        assert members[0].checkpoint == ckpt_2.resolve()
        assert members[0].threshold == pytest.approx(0.55)
        assert meta["set_index"] == 2

        mode, resolved, _ = _resolve_evaluation_members(
            model_artifact=manifest_path,
            ensemble_set_index=2,
            k_folds=1
        )
        assert mode == "ensemble-manifest"
        assert len(resolved) == 1

        mode, single_members, _ = _resolve_evaluation_members(
            model_artifact=ckpt_1,
            ensemble_set_index=1,
            k_folds=None
        )
        assert mode == "single-checkpoint"
        assert len(single_members) == 1

        with pytest.raises(ValueError, match="k-folds must be >= 1"):
            _resolve_evaluation_members(
                model_artifact=manifest_path,
                ensemble_set_index=2,
                k_folds=0
            )
        with pytest.raises(ValueError, match="exceeds available valid members"):
            _resolve_evaluation_members(
                model_artifact=ckpt_1,
                ensemble_set_index=1,
                k_folds=2
            )
        bad_suffix = tmp_path / "model.bin"
        bad_suffix.touch()
        with pytest.raises(ValueError, match="Unsupported model artifact type"):
            _resolve_evaluation_members(
                model_artifact=bad_suffix,
                ensemble_set_index=1,
                k_folds=None
            )


def test_json_curve_and_metric_helpers(monkeypatch):
    cleaned = _sanitize_for_json(
        {"a": float("nan"), "b": [1.0, float("inf"), {"c": -1.0}]}
    )
    assert cleaned["a"] is None
    assert cleaned["b"][1] is None
    assert _safe_ratio(1, 0) != _safe_ratio(1, 0)
    assert _safe_ratio(3, 2) == pytest.approx(1.5)

    with pytest.raises(ValueError, match="size mismatch"):
        _downsample_curve(np.asarray([0.0, 1.0]), np.asarray([0.0]), max_points=4)
    with pytest.raises(ValueError, match=">= 2"):
        _downsample_curve(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), max_points=1)
    x_out, y_out = _downsample_curve(
        np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        max_points=3
    )
    assert len(x_out) <= 3
    assert len(y_out) <= 3

    empty_pr = _build_pr_curve_payload(
        y_true=np.asarray([], dtype=np.int64),
        y_prob=np.asarray([], dtype=np.float64),
        max_points=8
    )
    assert empty_pr["available"] is False
    assert empty_pr["reason"] == "no-valid-residues"

    monkeypatch.setattr(eval_mod, "average_precision_score", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(eval_mod.np, "trapezoid", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    pr_with_exceptions = _build_pr_curve_payload(
        y_true=np.asarray([0, 1], dtype=np.int64),
        y_prob=np.asarray([0.1, 0.9], dtype=np.float64),
        max_points=8
    )
    assert pr_with_exceptions["available"] is True
    assert pr_with_exceptions["ap"] is None
    assert pr_with_exceptions["auprc_trapz"] is None

    assert math.isnan(_compute_pr_auc_trapezoid(
        y_true=np.asarray([], dtype=np.int64),
        y_prob=np.asarray([], dtype=np.float64),
    ))
    assert _compute_pr_auc_trapezoid(
        y_true=np.asarray([0, 0], dtype=np.int64),
        y_prob=np.asarray([0.2, 0.3], dtype=np.float64),
    ) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="at least one format"):
        _parse_plot_formats(" , ")
    with pytest.raises(ValueError, match="Unsupported"):
        _parse_plot_formats("png,jpg")
    assert _parse_plot_formats("png, svg") == ["png", "svg"]

    fold_evals = [
        {"pr_curve": {"available": True, "recall": [0.0, 0.1, 0.3], "precision": [0.2, 0.4, 0.8]}},
        {"pr_curve": {"available": False, "recall": [], "precision": []}},
        {"pr_curve": {"available": True, "recall": [0.1, "bad"], "precision": [0.5, 0.9]}},
    ]
    y_low, y_high = _resolve_pr_zoom_limits(fold_evals, baseline_y=0.03)
    assert y_low == pytest.approx(0.0)
    assert 0.02 <= y_high <= 1.0

    with pytest.raises(ValueError, match="missing 'metrics' mapping"):
        _resolve_eval_metric_value({}, "pr_auc")
    with pytest.raises(ValueError, match="not found"):
        _resolve_eval_metric_value({"metrics": {"auc": 0.8}}, "mcc")
    with pytest.raises(ValueError, match="non-finite value"):
        _resolve_eval_metric_value({"metrics": {"pr_auc": float("nan")}}, "pr_auc")
    assert _resolve_eval_metric_value({"metrics": {"PR_AUC": 0.71}}, "pr_auc") == pytest.approx(0.71)


def test_empty_eval_and_fold_selection_paths():
    empty = _empty_eval_output(votes_needed=2, include_curves=True)
    assert empty["votes_needed"] == 2
    assert empty["roc_curve"]["available"] is False
    assert empty["pr_curve"]["available"] is False

    out = _build_eval_output_from_arrays(
        y_true=np.asarray([], dtype=np.int64),
        y_pred=np.asarray([], dtype=np.int64),
        y_prob=np.asarray([], dtype=np.float64),
        processed_proteins=3,
        proteins_zero_valid_residues=2,
        votes_needed=None,
        include_curves=False,
        curve_max_points=16
    )
    assert out["processed_proteins"] == 3
    assert out["proteins_with_valid_residues"] == 1
    assert out["valid_residues"] == 0

    with pytest.raises(ValueError, match="At least one fold"):
        _pick_best_fold([], metric_name="PR_AUC", direction="auto")
    with pytest.raises(ValueError, match="best-fold-direction"):
        _pick_best_fold(
            [{"fold_index": 1, "metrics": {"pr_auc": 0.7}}],
            metric_name="PR_AUC",
            direction="weird"
        )

    idx, meta = _pick_best_fold(
        fold_evaluations=[
            {"fold_index": 2, "metrics": {"loss": 0.4}},
            {"fold_index": 1, "metrics": {"loss": 0.5}},
        ],
        metric_name="loss",
        direction="auto"
    )
    assert idx == 0
    assert meta["direction"] == "min"
    assert meta["best_fold_index"] == 2


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, event, extra):
        self.messages.append((event, extra))


def test_evaluate_dataset_validation_errors():
    logger = _FakeLogger()
    with pytest.raises(ValueError, match="At least one model"):
        _evaluate_dataset(
            psp_models=[],
            thresholds=[],
            eval_loader=[],
            device="cpu",
            logger=logger,
            member_specs=[],
            emit_fold_metrics=False,
            include_curves=False,
            curve_max_points=8,
            best_fold_by="pr_auc",
            best_fold_direction="auto",
            plot_dir=None,
            plot_formats=["png"]
        )

    dummy_model = object()
    dummy_member = EvaluationMember(
        checkpoint=Path("m.pt"), threshold=0.5, fold_index=1, member_index=1
    )
    with pytest.raises(ValueError, match="Model/threshold length mismatch"):
        _evaluate_dataset(
            psp_models=[dummy_model],
            thresholds=[],
            eval_loader=[],
            device="cpu",
            logger=logger,
            member_specs=[dummy_member],
            emit_fold_metrics=False,
            include_curves=False,
            curve_max_points=8,
            best_fold_by="pr_auc",
            best_fold_direction="auto",
            plot_dir=None,
            plot_formats=["png"]
        )
    with pytest.raises(ValueError, match="member_specs length mismatch"):
        _evaluate_dataset(
            psp_models=[dummy_model],
            thresholds=[0.5],
            eval_loader=[],
            device="cpu",
            logger=logger,
            member_specs=[],
            emit_fold_metrics=False,
            include_curves=False,
            curve_max_points=8,
            best_fold_by="pr_auc",
            best_fold_direction="auto",
            plot_dir=None,
            plot_formats=["png"]
        )
    with pytest.raises(ValueError, match="curve-max-points must be >= 2"):
        _evaluate_dataset(
            psp_models=[dummy_model],
            thresholds=[0.5],
            eval_loader=[],
            device="cpu",
            logger=logger,
            member_specs=[dummy_member],
            emit_fold_metrics=False,
            include_curves=False,
            curve_max_points=1,
            best_fold_by="pr_auc",
            best_fold_direction="auto",
            plot_dir=None,
            plot_formats=["png"]
        )


def test_evaluate_dataset_ensemble_and_single_member(monkeypatch):
    logger = _FakeLogger()
    model_a = object()
    model_b = object()

    def _fake_probs(psp_model, protein_emb, device):
        _ = device
        if float(protein_emb[0, 0].item()) < 0.5:
            if psp_model is model_a:
                return torch.tensor([0.2, 0.8, 0.4], dtype=torch.float32)
            return torch.tensor([0.9, 0.3, 0.6], dtype=torch.float32)
        if psp_model is model_a:
            return torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32)
        return torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)

    monkeypatch.setattr(eval_mod, "predict_member_probabilities_from_embedding", _fake_probs)

    X = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32
    )
    y = torch.tensor(
        [
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=torch.int64
    )
    mask = torch.tensor(
        [
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.int64
    )
    eval_loader = [(X, y, mask)]
    member_specs = [
        EvaluationMember(checkpoint=Path("m1.pt"), threshold=0.5, fold_index=1, member_index=1),
        EvaluationMember(checkpoint=Path("m2.pt"), threshold=0.5, fold_index=2, member_index=1),
    ]

    ensemble_out = _evaluate_dataset(
        psp_models=[model_a, model_b],
        thresholds=[0.5, 0.5],
        eval_loader=eval_loader,
        device="cpu",
        logger=logger,
        member_specs=member_specs,
        emit_fold_metrics=True,
        include_curves=True,
        curve_max_points=16,
        best_fold_by="pr_auc",
        best_fold_direction="auto",
        plot_dir=None,
        plot_formats=["png"]
    )
    assert ensemble_out["processed_proteins"] == 2
    assert ensemble_out["proteins_zero_valid_residues"] == 1
    assert ensemble_out["valid_residues"] == 3
    assert ensemble_out["votes_needed"] == 2
    assert len(ensemble_out["folds"]) == 2
    assert "best_fold_selection" in ensemble_out
    assert "best_fold" in ensemble_out

    single_out = _evaluate_dataset(
        psp_models=[model_a],
        thresholds=[0.5],
        eval_loader=eval_loader,
        device="cpu",
        logger=logger,
        member_specs=[member_specs[0]],
        emit_fold_metrics=False,
        include_curves=False,
        curve_max_points=16,
        best_fold_by="pr_auc",
        best_fold_direction="auto",
        plot_dir=None,
        plot_formats=["png"]
    )
    assert single_out["processed_proteins"] == 2
    assert single_out["votes_needed"] is None
    assert "folds" not in single_out
