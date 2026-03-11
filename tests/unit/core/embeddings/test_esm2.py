import logging
import types
from pathlib import Path
import pandas as pd
import pytest
import torch
import pepseqpred.core.embeddings.esm2 as esm2

pytestmark = pytest.mark.unit


class FakeAlphabet:
    def get_batch_converter(self):
        def _convert(pairs):
            labels = [name for name, _ in pairs]
            seqs = [seq for _, seq in pairs]
            max_len = max((len(s) for s in seqs), default=0)
            tokens = torch.zeros((len(seqs), max_len + 2), dtype=torch.long)
            for i, seq in enumerate(seqs):
                seq_len = len(seq)
                tokens[i, 1:1 + seq_len] = 1
                tokens[i, 1 + seq_len] = 2
            return labels, seqs, tokens

        return _convert


class FakeModel(torch.nn.Module):
    def __init__(self, embed_dim: int = 3, expose_embed_dim: bool = True):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self._embed_dim = embed_dim
        if expose_embed_dim:
            self.embed_dim = embed_dim

    def forward(self, batch_tokens, repr_layers, return_contacts=False):
        _ = return_contacts
        batch_size, token_len = batch_tokens.shape
        rep = torch.ones(
            (batch_size, token_len, self._embed_dim),
            dtype=torch.float32,
            device=batch_tokens.device
        )
        return {"representations": {repr_layers[0]: rep}}


def test_clean_seq_token_batches_and_append_len():
    assert esm2.clean_seq("acdx-*\n") == "ACDX"

    batches = list(
        esm2.token_packed_batches(
            [("a", "A" * 3), ("b", "B" * 3), ("c", "C" * 9)],
            max_tokens=10,
            max_tokens_per_seq=1022
        )
    )
    assert len(batches) == 2
    assert [x[0] for x in batches[0]] == ["a", "b"]

    arr = torch.ones((5, 3), dtype=torch.float32).numpy()
    out = esm2.append_seq_len(arr, 5)
    assert out.shape == (5, 4)
    assert (out[:, -1] == 5).all()


def test_compute_window_embedding_cpu_paths():
    token = torch.tensor([[0, 1, 1, 1, 1, 1, 2]], dtype=torch.long)

    model_1 = FakeModel(embed_dim=3, expose_embed_dim=True)
    out_1 = esm2.compute_window_embedding(
        token, model_1, layer=1, device="cpu", window_size=3, stride=2
    )
    assert out_1.shape == (5, 3)
    assert torch.allclose(out_1, torch.ones_like(out_1))

    model_2 = FakeModel(embed_dim=2, expose_embed_dim=False)
    out_2 = esm2.compute_window_embedding(
        token, model_2, layer=1, device="cpu", window_size=3, stride=2
    )
    assert out_2.shape == (5, 2)
    assert torch.allclose(out_2, torch.ones_like(out_2))


def test_esm_embeddings_from_fasta_short_and_long(monkeypatch, tmp_path: Path):
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeModel(embed_dim=3), FakeAlphabet())
    )
    monkeypatch.setattr(esm2.esm, "pretrained", fake_pretrained)

    df = pd.DataFrame(
        [
            {"ID": "P1", "Sequence": "ACD", "viral_family": "111"},
            {"ID": "P2", "Sequence": "ACDEFGH", "viral_family": "222"}
        ]
    )

    per_seq = tmp_path / "pts"
    idx_csv = tmp_path / "idx.csv"

    index_df, failed = esm2.esm_embeddings_from_fasta(
        df,
        id_col="ID",
        seq_col="Sequence",
        family_col="viral_family",
        model_name="fake_model",
        max_tokens=6,
        batch_size=2,
        per_seq_dir=per_seq,
        index_csv_path=idx_csv,
        key_mode="id-family",
        key_delimiter="-",
        logger=logging.getLogger("esm2_test")
    )

    assert failed == []
    assert len(index_df) == 2
    assert set(index_df["handle"]) == {"short", "long"}
    for key in index_df["id"].tolist():
        assert (per_seq / f"{key}.pt").exists()


def test_esm_embeddings_from_fasta_key_validation(monkeypatch, tmp_path: Path):
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeModel(embed_dim=3), FakeAlphabet())
    )
    monkeypatch.setattr(esm2.esm, "pretrained", fake_pretrained)

    df = pd.DataFrame([{"ID": "P1", "Sequence": "ACD"}])

    with pytest.raises(ValueError, match="Unsupported key_mode"):
        esm2.esm_embeddings_from_fasta(
            df,
            model_name="fake_model",
            key_mode="bad",
            per_seq_dir=tmp_path / "pts1",
            index_csv_path=tmp_path / "idx1.csv",
            logger=logging.getLogger("esm2_test")
        )

    with pytest.raises(ValueError, match="Missing required family column"):
        esm2.esm_embeddings_from_fasta(
            df,
            model_name="fake_model",
            key_mode="id-family",
            per_seq_dir=tmp_path / "pts2",
            index_csv_path=tmp_path / "idx2.csv",
            logger=logging.getLogger("esm2_test")
        )

    df_conflict = pd.DataFrame(
        [
            {"ID": "P1", "Sequence": "AAA"},
            {"ID": "P1", "Sequence": "CCC"}
        ]
    )
    with pytest.raises(
        ValueError, match="Conflicting sequences map to the same embedding key"
    ):
        esm2.esm_embeddings_from_fasta(
            df_conflict,
            model_name="fake_model",
            key_mode="id",
            per_seq_dir=tmp_path / "pts3",
            index_csv_path=tmp_path / "idx3.csv",
            logger=logging.getLogger("esm2_test")
        )
