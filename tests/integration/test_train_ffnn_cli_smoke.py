import os
from pathlib import Path
import subprocess
import sys
import pytest

pytestmark = pytest.mark.integration


def test_train_ffnn_cli_smoke(training_artifacts, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not current_pythonpath
        else f"{src_path}{os.pathsep}{current_pythonpath}"
    )

    save_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "pepseqpred.apps.train_ffnn_cli",
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
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, env=env
    )
    assert proc.returncode == 0, proc.stderr

    run_dirs = list(save_dir.glob("run_*"))
    assert run_dirs
    assert (run_dirs[0] / "fully_connected.pt").exists()
    assert (save_dir / "runs.csv").exists()
    assert (save_dir / "multi_run_summary.json").exists()
