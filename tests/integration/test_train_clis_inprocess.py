import sys
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
