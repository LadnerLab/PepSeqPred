import types
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from pepseqpred.api import PepSeqPredictor, predict_fasta, predict_sequence
from pepseqpred.core.models.ffnn import PepSeqFFNN

pytestmark = pytest.mark.unit


@contextmanager
def _scratch_dir():
    root = Path("localdata") / "tmp_pytest_predictor_api"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


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
        rep_dim = 3  # append_seq_len -> final emb dim becomes 4
        reps = torch.ones((batch_size, token_len, rep_dim),
                          dtype=torch.float32)
        return {"representations": {repr_layers[0]: reps}}


def _write_checkpoint(path: Path, threshold: float = 0.37) -> None:
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
            "metrics": {"threshold": float(threshold)},
        },
        path,
    )


def test_predictor_from_artifact_predict_sequence(monkeypatch):
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeESMModel(), FakeAlphabet())
    )
    monkeypatch.setattr(
        "pepseqpred.api.predictor.esm.pretrained", fake_pretrained)

    with _scratch_dir() as tmp_path:
        checkpoint = tmp_path / "model.pt"
        _write_checkpoint(checkpoint)

        predictor = PepSeqPredictor.from_artifact(
            model_artifact=checkpoint,
            model_name="fake_model",
            device="cpu",
        )
        out = predictor.predict_sequence("ACDEFG", header="protein_1")

        assert out.header == "protein_1"
        assert out.length == 6
        assert len(out.binary_mask) == 6
        assert set(out.binary_mask).issubset({"0", "1"})
        assert out.n_members == 1
        assert len(out.member_thresholds) == 1
        assert out.member_thresholds[0] == pytest.approx(0.37)


def test_predict_fasta_helper_writes_output(monkeypatch):
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeESMModel(), FakeAlphabet())
    )
    monkeypatch.setattr(
        "pepseqpred.api.predictor.esm.pretrained", fake_pretrained)

    with _scratch_dir() as tmp_path:
        checkpoint = tmp_path / "model.pt"
        _write_checkpoint(checkpoint)

        fasta = tmp_path / "input.fasta"
        fasta.write_text(">p1\nACDEFG\n>p2\nLMNPQ\n", encoding="utf-8")
        output_fasta = tmp_path / "predictions.fasta"

        outputs = predict_fasta(
            model_artifact=checkpoint,
            fasta_input=fasta,
            output_fasta=output_fasta,
            model_name="fake_model",
            device="cpu",
        )

        assert len(outputs) == 2
        assert output_fasta.exists()

        lines = [
            line.strip()
            for line in output_fasta.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert lines[0] == ">p1"
        assert len(lines[1]) == 6
        assert lines[2] == ">p2"
        assert len(lines[3]) == 5


def test_predict_sequence_helper(monkeypatch):
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeESMModel(), FakeAlphabet())
    )
    monkeypatch.setattr(
        "pepseqpred.api.predictor.esm.pretrained", fake_pretrained)

    with _scratch_dir() as tmp_path:
        checkpoint = tmp_path / "model.pt"
        _write_checkpoint(checkpoint)

        out = predict_sequence(
            model_artifact=checkpoint,
            protein_seq="ACDEFG",
            header="direct_call",
            model_name="fake_model",
            device="cpu",
        )

        assert out.header == "direct_call"
        assert out.length == 6
        assert len(out.binary_mask) == 6
