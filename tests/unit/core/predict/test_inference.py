import pytest
import torch
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    infer_decision_threshold,
    infer_model_config_from_state,
    normalize_state_dict_keys,
    predict_ensemble_from_embedding,
    predict_from_embedding
)

pytestmark = pytest.mark.unit


def _make_checkpoint(
    emb_dim: int = 4,
    hidden_sizes: tuple[int, ...] = (8,),
    dropouts: tuple[float, ...] = (0.0,),
    use_layer_norm: bool = False,
    use_residual: bool = False
):
    model = PepSeqFFNN(
        emb_dim=emb_dim,
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        num_classes=1,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual,
    )
    return {"model_state_dict": model.state_dict(), "metrics": {"threshold": 0.37}}


def test_normalize_state_dict_keys_strips_module_prefix():
    ckpt = _make_checkpoint()
    state = ckpt["model_state_dict"]
    prefixed = {f"module.{k}": v for k, v in state.items()}
    out = normalize_state_dict_keys(prefixed)
    assert set(out.keys()) == set(state.keys())


def test_infer_model_config_from_state_happy_path():
    ckpt = _make_checkpoint(emb_dim=4, hidden_sizes=(8,), dropouts=(0.0,))
    cfg = infer_model_config_from_state(ckpt["model_state_dict"])
    assert cfg.emb_dim == 4
    assert cfg.hidden_sizes == (8,)
    assert cfg.num_classes == 1
    assert cfg.use_layer_norm is False
    assert cfg.use_residual is False


def test_infer_model_config_from_state_ambiguous_residual_raises():
    ckpt = _make_checkpoint(emb_dim=4, hidden_sizes=(
        4,), dropouts=(0.0,), use_residual=False)
    with pytest.raises(ValueError, match="Cannot infer use_residual"):
        infer_model_config_from_state(ckpt["model_state_dict"])


def test_build_model_from_checkpoint_handles_ddp_prefixed_state():
    ckpt = _make_checkpoint()
    prefixed_state = {f"module.{k}": v for k,
                      v in ckpt["model_state_dict"].items()}
    model, cfg, cfg_src = build_model_from_checkpoint(
        {"model_state_dict": prefixed_state},
        device="cpu",
    )
    assert cfg_src == "state_dict"
    assert cfg.emb_dim == 4

    x = torch.randn(2, 5, 4)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 5)


def test_build_model_from_checkpoint_rejects_invalid_num_classes():
    ckpt = _make_checkpoint()
    bad_cfg = FFNNModelConfig(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=2,
        use_layer_norm=False,
        use_residual=False
    )
    with pytest.raises(ValueError, match="num_classes=1"):
        build_model_from_checkpoint(ckpt, device="cpu", model_config=bad_cfg)


def test_infer_decision_threshold_uses_default_on_invalid():
    ckpt = {"metrics": {"threshold": 1.2}}
    assert infer_decision_threshold(ckpt, default=0.5) == 0.5


def test_infer_decision_threshold_reads_checkpoint_metric():
    ckpt = {"metrics": {"threshold": 0.42}}
    assert infer_decision_threshold(ckpt, default=0.5) == pytest.approx(0.42)


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self, logits_1d: list[float]):
        super().__init__()
        logits = torch.tensor(logits_1d, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("logits", logits)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _ = X
        return self.logits


def test_predict_ensemble_from_embedding_uses_majority_vote():
    emb = torch.zeros((4, 3), dtype=torch.float32)
    models = [
        _ConstantLogitModel([2.0, 2.0, -2.0, -2.0]),
        _ConstantLogitModel([2.0, -2.0, 2.0, -2.0]),
        _ConstantLogitModel([-2.0, 2.0, 2.0, -2.0]),
    ]
    out = predict_ensemble_from_embedding(
        psp_models=models,
        protein_emb=emb,
        device="cpu",
        thresholds=[0.5, 0.5, 0.5]
    )
    assert out["binary_mask"] == "1110"
    assert out["n_members"] == 3
    assert out["votes_needed"] == 2


def test_predict_ensemble_from_embedding_even_tie_is_negative():
    emb = torch.zeros((2, 3), dtype=torch.float32)
    models = [
        _ConstantLogitModel([2.0, -2.0]),
        _ConstantLogitModel([-2.0, 2.0]),
    ]
    out = predict_ensemble_from_embedding(
        psp_models=models,
        protein_emb=emb,
        device="cpu",
        thresholds=[0.5, 0.5]
    )
    assert out["binary_mask"] == "00"
    assert out["votes_needed"] == 2


def test_predict_ensemble_from_embedding_k1_matches_single():
    emb = torch.zeros((3, 3), dtype=torch.float32)
    model = _ConstantLogitModel([2.0, -2.0, 2.0])
    single = predict_from_embedding(
        psp_model=model,
        protein_emb=emb,
        device="cpu",
        threshold=0.5
    )
    ensemble = predict_ensemble_from_embedding(
        psp_models=[model],
        protein_emb=emb,
        device="cpu",
        thresholds=[0.5]
    )
    assert ensemble["binary_mask"] == single["binary_mask"]
    assert ensemble["n_epitopes"] == single["n_epitopes"]
