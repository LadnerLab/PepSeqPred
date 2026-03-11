import os
from pathlib import Path
import subprocess
import sys
import types
import pytest
import torch
import pepseqpred.apps.prediction_cli as prediction_cli

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


class FakeAlphabet:
    def get_batch_converter(self):
        def _batch_converter(pairs):
            labels = [name for name, _seq in pairs]
            seqs = [seq for _name, seq in pairs]
            max_len = max((len(seq) for seq in seqs), default=0)
            tokens = torch.zeros((len(seqs), max_len + 2), dtype=torch.long)
            for i, seq in enumerate(seqs):
                seq_len = len(seq)
                tokens[i, 1:1 + seq_len] = 1
                tokens[i, 1 + seq_len] = 2
            return labels, seqs, tokens

        return _batch_converter


class FakeESMModel:
    num_layers = 1

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, batch_tokens, repr_layers, return_contacts=False):
        _ = return_contacts
        batch_size, token_len = batch_tokens.shape
        # append_seq_len -> final emb dim is 4 (matches training fixture)
        rep_dim = 3
        reps = torch.ones((batch_size, token_len, rep_dim),
                          dtype=torch.float32)
        return {"representations": {repr_layers[0]: reps}}


def test_train_then_predict_e2e(training_artifacts, tmp_path: Path, monkeypatch):
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

    train_cmd = [
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
        train_cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr

    run_dirs = sorted(save_dir.glob("run_*"))
    assert run_dirs
    checkpoint = run_dirs[0] / "fully_connected.pt"
    assert checkpoint.exists()

    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeESMModel(), FakeAlphabet())
    )
    monkeypatch.setattr(prediction_cli.esm, "pretrained", fake_pretrained)

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">protein_e2e\nACDEFG\n", encoding="utf-8")
    output_fasta = tmp_path / "predictions.fasta"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prediction_cli.py",
            str(checkpoint),
            str(fasta),
            "--output-fasta",
            str(output_fasta),
            "--model-name",
            "fake_model",
            "--threshold",
            "0.5",
        ],
    )

    prediction_cli.main()

    lines = [line.strip() for line in output_fasta.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert lines[0] == ">protein_e2e"
    assert len(lines[1]) == 6
    assert set(lines[1]).issubset({"0", "1"})
