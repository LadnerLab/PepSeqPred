import types

import pytest
import torch

import pepseqpred.core.predict.inference as inf
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.predict.inference import FFNNModelConfig

pytestmark = pytest.mark.unit


def _make_checkpoint_state() -> dict:
    model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(3,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False
    )
    return {"model_state_dict": model.state_dict()}


def test_infer_model_config_error_and_residual_paths():
    with pytest.raises(ValueError, match="Expected 2D tensor"):
        inf.infer_model_config_from_state(
            {"ff_model.0.linear.weight": "not-a-tensor"}
        )

    with pytest.raises(ValueError, match="Could not infer output layer"):
        inf.infer_model_config_from_state(
            {"ff_model.0.linear.weight": torch.randn(4, 3)}
        )

    with pytest.raises(ValueError, match="Hidden layer mismatch"):
        inf.infer_model_config_from_state(
            {
                "ff_model.0.linear.weight": torch.randn(4, 3),
                "ff_model.1.linear.weight": torch.randn(5, 6),
                "ff_model.2.weight": torch.randn(1, 5),
            }
        )

    with pytest.raises(ValueError, match="Output in_features"):
        inf.infer_model_config_from_state(
            {
                "ff_model.0.linear.weight": torch.randn(4, 3),
                "ff_model.1.weight": torch.randn(1, 5),
            }
        )

    cfg = inf.infer_model_config_from_state(
        {
            "ff_model.0.linear.weight": torch.randn(4, 3),
            "ff_model.0.skip.weight": torch.randn(4, 3),
            "ff_model.0.layer_norm.weight": torch.randn(4),
            "ff_model.1.weight": torch.randn(1, 4),
        }
    )
    assert cfg.emb_dim == 3
    assert cfg.hidden_sizes == (4,)
    assert cfg.use_layer_norm is True
    assert cfg.use_residual is True


def test_build_model_from_checkpoint_validation_errors():
    with pytest.raises(ValueError, match="Expected checkpoint dict"):
        inf.build_model_from_checkpoint({})

    with pytest.raises(ValueError, match="must be a mapping"):
        inf.build_model_from_checkpoint({"model_state_dict": 123})

    ckpt = _make_checkpoint_state()
    with pytest.raises(ValueError, match="emb_dim must be > 0"):
        inf.build_model_from_checkpoint(
            ckpt,
            model_config=FFNNModelConfig(
                emb_dim=0,
                hidden_sizes=(3,),
                dropouts=(0.0,),
                num_classes=1,
                use_layer_norm=False,
                use_residual=False
            )
        )
    with pytest.raises(ValueError, match="same length"):
        inf.build_model_from_checkpoint(
            ckpt,
            model_config=FFNNModelConfig(
                emb_dim=4,
                hidden_sizes=(3, 2),
                dropouts=(0.0,),
                num_classes=1,
                use_layer_norm=False,
                use_residual=False
            )
        )
    with pytest.raises(ValueError, match="incompatible"):
        inf.build_model_from_checkpoint(
            ckpt,
            model_config=FFNNModelConfig(
                emb_dim=99,
                hidden_sizes=(3,),
                dropouts=(0.0,),
                num_classes=1,
                use_layer_norm=False,
                use_residual=False
            )
        )


def test_infer_decision_threshold_edge_cases():
    assert inf.infer_decision_threshold(123, default=0.5) == pytest.approx(0.5)
    assert inf.infer_decision_threshold({"metrics": None}, default=0.5) == pytest.approx(0.5)
    assert inf.infer_decision_threshold({"metrics": {"threshold": None}}, default=0.5) == pytest.approx(0.5)
    assert inf.infer_decision_threshold({"metrics": {"threshold": "bad"}}, default=0.5) == pytest.approx(0.5)
    assert inf.infer_decision_threshold({"metrics": {"threshold": 0.0}}, default=0.5) == pytest.approx(0.5)
    assert inf.infer_decision_threshold({"metrics": {"threshold": 1.1}}, default=0.5) == pytest.approx(0.5)


def test_prediction_payload_and_member_probability_errors():
    with pytest.raises(ValueError, match="Expected probs shape"):
        inf._build_prediction_payload(
            probs=torch.zeros((2, 2)),
            mask=torch.zeros(4, dtype=torch.int64),
            threshold=0.5
        )
    with pytest.raises(ValueError, match="Expected mask shape"):
        inf._build_prediction_payload(
            probs=torch.zeros(4),
            mask=torch.zeros((2, 2), dtype=torch.int64),
            threshold=0.5
        )
    with pytest.raises(ValueError, match="matching probs/mask lengths"):
        inf._build_prediction_payload(
            probs=torch.zeros(4),
            mask=torch.zeros(3, dtype=torch.int64),
            threshold=0.5
        )

    payload = inf._build_prediction_payload(
        probs=torch.tensor([0.2, 0.8, 0.6], dtype=torch.float32),
        mask=torch.tensor([0, 1, 1], dtype=torch.int64),
        threshold=0.5
    )
    assert payload["binary_mask"] == "011"


class _FakeBatchConverter:
    def __call__(self, pairs):
        _name, seq = pairs[0]
        tokens = torch.zeros((1, len(seq) + 2), dtype=torch.long)
        return [pairs[0][0]], [seq], tokens


class _FakeEsmModel:
    def __call__(self, batch_tokens, repr_layers, return_contacts=False):
        _ = return_contacts
        seq_len = int(batch_tokens.size(1) - 2)
        rep_dim = 3
        reps = torch.ones((1, seq_len + 2, rep_dim), dtype=torch.float32)
        return {"representations": {repr_layers[0]: reps}}


def test_embed_protein_seq_paths(monkeypatch):
    with pytest.raises(ValueError, match="empty after cleaning"):
        monkeypatch.setattr(inf, "clean_seq", lambda _x: "")
        inf.embed_protein_seq(
            protein_seq="AA",
            esm_model=_FakeEsmModel(),
            layer=1,
            batch_converter=_FakeBatchConverter(),
            device="cpu",
            max_tokens=10
        )

    monkeypatch.setattr(inf, "clean_seq", lambda s: s)
    out_short = inf.embed_protein_seq(
        protein_seq="ABCDE",
        esm_model=_FakeEsmModel(),
        layer=1,
        batch_converter=_FakeBatchConverter(),
        device="cpu",
        max_tokens=10
    )
    assert out_short.shape == (5, 4)

    monkeypatch.setattr(
        inf,
        "compute_window_embedding",
        lambda *_args, **_kwargs: torch.ones((5, 3), dtype=torch.float32)
    )
    out_long = inf.embed_protein_seq(
        protein_seq="ABCDE",
        esm_model=_FakeEsmModel(),
        layer=1,
        batch_converter=_FakeBatchConverter(),
        device="cpu",
        max_tokens=1
    )
    assert out_long.shape == (5, 4)


class _BadShapeModel(torch.nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _ = X
        return torch.randn(2, 3)


class _GoodShapeModel(torch.nn.Module):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        length = int(X.size(1))
        return torch.zeros((1, length), dtype=torch.float32)


def test_prediction_functions_error_branches(monkeypatch):
    with pytest.raises(ValueError, match="shape \\(L, E\\)"):
        inf.predict_member_probabilities_from_embedding(
            psp_model=_GoodShapeModel(),
            protein_emb=torch.zeros((4,), dtype=torch.float32),
            device="cpu"
        )
    with pytest.raises(ValueError, match="Expected logits shape"):
        inf.predict_member_probabilities_from_embedding(
            psp_model=_BadShapeModel(),
            protein_emb=torch.zeros((4, 3), dtype=torch.float32),
            device="cpu"
        )

    probs = inf.predict_member_probabilities_from_embedding(
        psp_model=_GoodShapeModel(),
        protein_emb=torch.zeros((4, 3), dtype=torch.float32),
        device="cpu"
    )
    assert probs.shape == (4,)

    with pytest.raises(ValueError, match="threshold must be in"):
        inf.predict_from_embedding(
            psp_model=_GoodShapeModel(),
            protein_emb=torch.zeros((3, 3), dtype=torch.float32),
            device="cpu",
            threshold=0.0
        )

    monkeypatch.setattr(
        inf,
        "predict_member_probabilities_from_embedding",
        lambda **_kwargs: torch.tensor([0.2, 0.9, 0.7], dtype=torch.float32)
    )
    payload = inf.predict_from_embedding(
        psp_model=_GoodShapeModel(),
        protein_emb=torch.zeros((3, 3), dtype=torch.float32),
        device="cpu",
        threshold=0.5
    )
    assert payload["binary_mask"] == "011"


def test_predict_ensemble_and_predict_protein_edge_cases(monkeypatch):
    with pytest.raises(ValueError, match="At least one model"):
        inf.predict_ensemble_from_embedding(
            psp_models=[],
            protein_emb=torch.zeros((3, 3), dtype=torch.float32),
            device="cpu",
            thresholds=[]
        )
    with pytest.raises(ValueError, match="length mismatch"):
        inf.predict_ensemble_from_embedding(
            psp_models=[_GoodShapeModel()],
            protein_emb=torch.zeros((3, 3), dtype=torch.float32),
            device="cpu",
            thresholds=[]
        )
    with pytest.raises(ValueError, match="each threshold"):
        inf.predict_ensemble_from_embedding(
            psp_models=[_GoodShapeModel()],
            protein_emb=torch.zeros((3, 3), dtype=torch.float32),
            device="cpu",
            thresholds=[1.0]
        )

    model_a = _GoodShapeModel()
    model_b = _GoodShapeModel()

    def _bad_probs(**_kwargs):
        return torch.zeros((1, 3), dtype=torch.float32)

    monkeypatch.setattr(inf, "predict_member_probabilities_from_embedding", _bad_probs)
    with pytest.raises(ValueError, match="Expected member probs shape"):
        inf.predict_ensemble_from_embedding(
            psp_models=[model_a, model_b],
            protein_emb=torch.zeros((3, 3), dtype=torch.float32),
            device="cpu",
            thresholds=[0.5, 0.5]
        )

    def _length_mismatch(psp_model, **_kwargs):
        if psp_model is model_a:
            return torch.tensor([0.1, 0.8, 0.9], dtype=torch.float32)
        return torch.tensor([0.1, 0.8], dtype=torch.float32)

    monkeypatch.setattr(inf, "predict_member_probabilities_from_embedding", _length_mismatch)
    with pytest.raises(ValueError, match="same sequence length"):
        inf.predict_ensemble_from_embedding(
            psp_models=[model_a, model_b],
            protein_emb=torch.zeros((3, 3), dtype=torch.float32),
            device="cpu",
            thresholds=[0.5, 0.5]
        )

    with pytest.raises(ValueError, match="threshold must be in"):
        inf.predict_protein(
            psp_model=_GoodShapeModel(),
            esm_model=types.SimpleNamespace(),
            layer=1,
            batch_converter=types.SimpleNamespace(),
            protein_seq="ACDE",
            max_tokens=10,
            device="cpu",
            threshold=1.0
        )
