import sys
import json
from pathlib import Path
import pytest
import pepseqpred.apps.train_ffnn_cli as train_cli
import pepseqpred.apps.train_ffnn_optuna_cli as optuna_cli

pytestmark = pytest.mark.integration


def test_train_ffnn_cli_main_inprocess(training_artifacts, tmp_path: Path, monkeypatch):
    save_dir = tmp_path / "train_out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ffnn_cli.py",
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
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
            "--save-path",
            str(save_dir),
            "--results-csv",
            str(save_dir / "runs.csv")
        ]
    )

    train_cli.main()

    run_dirs = list(save_dir.glob("run_*"))
    assert run_dirs
    assert (run_dirs[0] / "fully_connected.pt").exists()
    assert (save_dir / "runs.csv").exists()
    assert (save_dir / "multi_run_summary.json").exists()


def test_train_ffnn_cli_main_inprocess_with_val_curves(
    training_artifacts, tmp_path: Path, monkeypatch
):
    save_dir = tmp_path / "train_out_curves"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ffnn_cli.py",
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
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
            "--save-path",
            str(save_dir),
            "--results-csv",
            str(save_dir / "runs.csv"),
            "--save-val-curves",
            "--val-curve-max-points",
            "128",
            "--val-plot-formats",
            "png"
        ]
    )

    train_cli.main()

    run_dirs = list(save_dir.glob("run_*"))
    assert run_dirs
    curves_dir = run_dirs[0] / "validation_curves"
    assert (curves_dir / "epoch_0000_curves.json").exists()

    roc_plot = curves_dir / "epoch_0000_roc_auc.png"
    pr_plot = curves_dir / "epoch_0000_pr_auc.png"
    if roc_plot.exists() or pr_plot.exists():
        assert roc_plot.exists()
        assert pr_plot.exists()


def test_train_ffnn_cli_ensemble_kfold_inprocess(training_artifacts, tmp_path: Path, monkeypatch):
    save_dir = tmp_path / "ensemble_out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ffnn_cli.py",
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
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
        ]
    )

    train_cli.main()

    set_dirs = sorted(save_dir.glob("set_*"))
    assert len(set_dirs) == 2
    for set_dir in set_dirs:
        fold_dirs = sorted(set_dir.glob("fold_*"))
        assert len(fold_dirs) == 2
        assert all((fold_dir / "fully_connected.pt").exists() for fold_dir in fold_dirs)
        assert (set_dir / "ensemble_manifest.json").exists()

    assert (save_dir / "runs.csv").exists()
    assert (save_dir / "multi_run_summary.json").exists()
    manifest_path = save_dir / "ensemble_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["train_mode"] == "ensemble-kfold"
    assert payload["n_sets"] == 2
    assert len(payload["sets"]) == 2
    assert [(x["split_seed"], x["train_seed"]) for x in payload["sets"]] == [
        (17, 101),
        (19, 202),
    ]
    assert all(int(x["n_members"]) == 2 for x in payload["sets"])
    assert all(Path(x["manifest_path"]).exists() for x in payload["sets"])


def test_train_ffnn_cli_ensemble_kfold_with_aggregate_val_curves(
    training_artifacts, tmp_path: Path, monkeypatch
):
    save_dir = tmp_path / "ensemble_out_curves"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ffnn_cli.py",
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
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
            "--split-type",
            "id-family",
            "--n-folds",
            "2",
            "--split-seeds",
            "17",
            "--train-seeds",
            "101",
            "--save-val-curves",
            "--val-curve-max-points",
            "128",
            "--val-plot-formats",
            "png",
            "--save-path",
            str(save_dir),
            "--results-csv",
            str(save_dir / "runs.csv"),
        ],
    )

    train_cli.main()

    fold_dirs = sorted(save_dir.glob("fold_*"))
    assert len(fold_dirs) == 2
    assert all((fold_dir / "fully_connected.pt").exists() for fold_dir in fold_dirs)
    assert all(
        (fold_dir / "validation_curves" / "epoch_0000_curves.json").exists()
        for fold_dir in fold_dirs
    )

    aggregate_json = (
        save_dir
        / "validation_curves"
        / "ensemble_folds"
        / "ensemble_fold_validation_curves.json"
    )
    assert aggregate_json.exists()
    aggregate_payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
    assert int(aggregate_payload["set_index"]) == 1
    assert int(aggregate_payload["n_folds_with_curves"]) == 2
    assert len(aggregate_payload["folds"]) == 2

    manifest_payload = json.loads(
        (save_dir / "ensemble_manifest.json").read_text(encoding="utf-8")
    )
    curve_meta = manifest_payload.get("validation_curve_artifacts")
    assert isinstance(curve_meta, dict)
    assert Path(curve_meta["artifact_json"]).exists()

    if str(curve_meta.get("plot_status")) == "ok":
        roc_paths = [Path(p) for p in curve_meta["plot_outputs"]["roc_auc_folds"]]
        pr_paths = [Path(p) for p in curve_meta["plot_outputs"]["pr_auc_folds"]]
        pr_zoom_paths = [Path(p)
                         for p in curve_meta["plot_outputs"]["pr_auc_folds_zoom"]]
        assert len(roc_paths) == 1
        assert len(pr_paths) == 1
        assert len(pr_zoom_paths) == 1
        assert roc_paths[0].exists()
        assert pr_paths[0].exists()
        assert pr_zoom_paths[0].exists()


@pytest.mark.slow
def test_train_ffnn_optuna_cli_main_inprocess(
    training_artifacts, tmp_path: Path, monkeypatch
):
    save_dir = tmp_path / "optuna_out"
    csv_path = save_dir / "trials.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ffnn_optuna_cli.py",
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
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
            "test_smoke"
        ]
    )

    optuna_cli.main()

    assert (save_dir / "best_trial.json").exists()
    assert csv_path.exists()
