import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import optuna
import pandas as pd
import pytest
import torch

import pepseqpred.apps.train_ffnn_cli as train_cli
import pepseqpred.apps.train_ffnn_optuna_cli as optuna_cli

pytestmark = pytest.mark.unit


def _mk_case_dir(tag: str) -> Path:
    case_dir = Path("localdata") / f"unit_realcov_{tag}_{uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _write_training_artifacts(case_dir: Path, *, all_uncertain: bool) -> tuple[Path, Path]:
    emb_dir = case_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    label_shard = case_dir / "labels_000.pt"

    labels = {}
    pos_count = 0
    neg_count = 0
    ids = [("P001", "111"), ("P002", "111"), ("P003", "222"), ("P004", "222")]
    for i, (protein_id, family) in enumerate(ids):
        torch.manual_seed(10 + i)
        emb = torch.randn(6, 4, dtype=torch.float32)
        torch.save(emb, emb_dir / f"{protein_id}-{family}.pt")

        if all_uncertain:
            y = torch.tensor(
                [[0.0, 1.0, 0.0]] * 6,
                dtype=torch.float32
            )
        else:
            y = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
            pos_count += int((y == 1).sum().item())
            neg_count += int((y == 0).sum().item())
        labels[protein_id] = y

    if all_uncertain:
        # Keep positive class weight sane for the test run.
        pos_count = 1
        neg_count = 1

    torch.save(
        {"labels": labels, "class_stats": {"pos_count": pos_count, "neg_count": neg_count}},
        label_shard
    )
    return emb_dir, label_shard


def _run_main(entrypoint, argv: list[str]) -> None:
    old_argv = list(sys.argv)
    sys.argv = argv
    try:
        entrypoint()
    finally:
        sys.argv = old_argv


def test_train_cli_helper_parsers_and_numeric_summary():
    summary_empty = train_cli.summarize_numeric(
        pd.Series([float("nan"), float("inf"), -float("inf")])
    )
    assert summary_empty["count"] == 0
    assert summary_empty["mean"] is None

    summary_ok = train_cli.summarize_numeric(pd.Series([1.0, 2.0, 3.0]))
    assert summary_ok["count"] == 3
    assert summary_ok["mean"] == pytest.approx(2.0)

    assert train_cli._finite_or_none("nan") is None
    assert train_cli._finite_or_none("abc") is None
    assert train_cli._finite_or_none("1.5") == pytest.approx(1.5)

    assert train_cli._parse_plot_formats("png, svg") == ("png", "svg")
    with pytest.raises(ValueError, match="val-plot-formats"):
        train_cli._parse_plot_formats(",,")
    with pytest.raises(ValueError, match="Unsupported"):
        train_cli._parse_plot_formats("png,jpg")


def test_train_ffnn_cli_real_no_valid_score_with_val_curve_artifacts():
    case_dir = _mk_case_dir("ffnn_no_valid")
    emb_dir, label_shard = _write_training_artifacts(
        case_dir, all_uncertain=True
    )
    save_dir = case_dir / "train_out"

    try:
        _run_main(
            train_cli.main,
            [
                "train_ffnn_cli.py",
                "--embedding-dirs",
                str(emb_dir),
                "--label-shards",
                str(label_shard),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--hidden-sizes",
                "8",
                "--dropouts",
                "0.1",
                "--val-frac",
                "0.5",
                "--split-seeds",
                "11",
                "--train-seeds",
                "101",
                "--best-model-metric",
                "f1",
                "--save-val-curves",
                "--val-curve-max-points",
                "64",
                "--val-plot-formats",
                "png",
                "--save-path",
                str(save_dir),
                "--results-csv",
                str(save_dir / "runs.csv"),
            ],
        )

        run_dirs = sorted(save_dir.glob("run_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "fully_connected.pt").exists()
        assert (run_dirs[0] / "validation_curves" /
                "epoch_0000_curves.json").exists()

        runs_df = pd.read_csv(save_dir / "runs.csv")
        assert int(runs_df.shape[0]) == 1
        assert str(runs_df.iloc[0]["BestMetricKey"]) == "f1"
        assert str(runs_df.iloc[0]["Status"]) == "NO_VALID_SCORE"

        summary = json.loads(
            (save_dir / "multi_run_summary.json").read_text(encoding="utf-8")
        )
        assert int(summary["n_runs"]) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_train_ffnn_cli_real_ensemble_manifest_generation():
    case_dir = _mk_case_dir("ffnn_ensemble")
    emb_dir, label_shard = _write_training_artifacts(
        case_dir, all_uncertain=False
    )
    save_dir = case_dir / "ensemble_out"

    try:
        _run_main(
            train_cli.main,
            [
                "train_ffnn_cli.py",
                "--embedding-dirs",
                str(emb_dir),
                "--label-shards",
                str(label_shard),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--hidden-sizes",
                "8",
                "--dropouts",
                "0.1",
                "--train-mode",
                "ensemble-kfold",
                "--split-type",
                "id-family",
                "--n-folds",
                "2",
                "--split-seeds",
                "17,19",
                "--train-seeds",
                "101,202",
                "--save-path",
                str(save_dir),
                "--results-csv",
                str(save_dir / "runs.csv"),
                "--ensemble-manifest",
                str(save_dir / "ensemble_manifest.json"),
            ],
        )

        payload = json.loads(
            (save_dir / "ensemble_manifest.json").read_text(encoding="utf-8")
        )
        assert payload["train_mode"] == "ensemble-kfold"
        assert payload["n_sets"] == 2
        assert len(payload["sets"]) == 2
        assert all(int(x["n_members"]) == 2 for x in payload["sets"])
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_train_ffnn_optuna_cli_real_with_storage_and_helpers():
    assert optuna_cli._broadcast_params({"a": 1}, None) == {"a": 1}

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
    trial_flat = study.ask()
    sizes_flat, drop_flat, depth_flat, _ = optuna_cli.build_hidden_sizes(
        trial=trial_flat,
        depth_min=2,
        depth_max=2,
        width_min=64,
        width_max=64,
        mode="flat"
    )
    assert depth_flat == 2
    assert sizes_flat == (64, 64)
    assert len(drop_flat) == 2

    trial_bottle = study.ask()
    sizes_bottle, _, _, _ = optuna_cli.build_hidden_sizes(
        trial=trial_bottle,
        depth_min=3,
        depth_max=3,
        width_min=32,
        width_max=96,
        mode="bottleneck"
    )
    assert all(
        sizes_bottle[i] >= sizes_bottle[i + 1]
        for i in range(len(sizes_bottle) - 1)
    )

    trial_pyramid = study.ask()
    sizes_pyramid, _, _, _ = optuna_cli.build_hidden_sizes(
        trial=trial_pyramid,
        depth_min=3,
        depth_max=3,
        width_min=32,
        width_max=96,
        mode="pyramid"
    )
    assert all(
        sizes_pyramid[i] <= sizes_pyramid[i + 1]
        for i in range(len(sizes_pyramid) - 1)
    )

    case_dir = _mk_case_dir("optuna_storage")
    emb_dir, label_shard = _write_training_artifacts(
        case_dir, all_uncertain=False
    )
    save_dir = case_dir / "optuna_out"
    csv_path = save_dir / "trials.csv"
    storage_uri = f"sqlite:///{(case_dir / 'study.db').as_posix()}"

    try:
        _run_main(
            optuna_cli.main,
            [
                "train_ffnn_optuna_cli.py",
                "--embedding-dirs",
                str(emb_dir),
                "--label-shards",
                str(label_shard),
                "--storage",
                storage_uri,
                "--n-trials",
                "1",
                "--epochs",
                "1",
                "--val-frac",
                "0.5",
                "--subset",
                "4",
                "--batch-sizes",
                "2",
                "--num-workers",
                "0",
                "--metric",
                "auc",
                "--arch-mode",
                "flat",
                "--depth-min",
                "1",
                "--depth-max",
                "1",
                "--width-min",
                "64",
                "--width-max",
                "64",
                "--save-path",
                str(save_dir),
                "--csv-path",
                str(csv_path),
                "--study-name",
                "unit_realcov_study",
            ],
        )

        best_payload = json.loads(
            (save_dir / "best_trial.json").read_text(encoding="utf-8")
        )
        assert best_payload["study_name"] == "unit_realcov_study"
        assert best_payload["metric"] == "auc"
        assert csv_path.exists()
        assert (case_dir / "study.db").exists()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
