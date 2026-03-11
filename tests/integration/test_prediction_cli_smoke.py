import sys
import types
from pathlib import Path
import pytest
import torch
import pepseqpred.apps.prediction_cli as prediction_cli
from pepseqpred.core.models.ffnn import PepSeqFFNN

pytestmark = pytest.mark.integration


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
        rep_dim = 3  # append_seq_len -> final emb dim is 4
        reps = torch.ones((batch_size, token_len, rep_dim),
                          dtype=torch.float32)
        return {"representations": {repr_layers[0]: reps}}


def _write_checkpoint(path: Path) -> None:
    model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(3,),
        dropouts=(0.0,),
        use_layer_norm=False,
        use_residual=False,
        num_classes=1,
    )
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": {"threshold": 0.5},
        },
        path,
    )


def test_prediction_cli_smoke(monkeypatch, tmp_path: Path):
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeESMModel(), FakeAlphabet())
    )
    monkeypatch.setattr(prediction_cli.esm, "pretrained", fake_pretrained)

    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)

    fasta = tmp_path / "input.fasta"
    fasta.write_text(
        ">protein_1\nACDEFG\n>protein_2\nLMNPQ\n", encoding="utf-8")

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
    assert lines[0] == ">protein_1"
    assert len(lines[1]) == 6
    assert set(lines[1]).issubset({"0", "1"})
    assert lines[2] == ">protein_2"
    assert len(lines[3]) == 5
    assert set(lines[3]).issubset({"0", "1"})
