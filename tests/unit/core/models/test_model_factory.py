import pytest
import torch

from pepseqpred.core.models.factory import (
    PepSeqModelConfig,
    build_pepseq_model,
    model_config_from_mapping,
    model_config_to_dict,
)
from pepseqpred.core.models.ffnn import PepSeqConvFFNN, PepSeqFFNN

pytestmark = pytest.mark.unit


def test_conv_head_preserves_residue_logit_shape():
    model = PepSeqConvFFNN(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        conv_channels=3,
        conv_layers=2,
        conv_kernel_size=5,
        conv_dropout=0.0,
    )

    out = model(torch.randn(2, 7, 4))

    assert out.shape == (2, 7)


def test_conv_head_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="positive odd"):
        PepSeqConvFFNN(
            emb_dim=4,
            hidden_sizes=(8,),
            dropouts=(0.0,),
            conv_kernel_size=4,
        )


def test_model_factory_builds_ffnn_and_conv_heads():
    base = PepSeqModelConfig(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False,
    )
    assert isinstance(build_pepseq_model(base), PepSeqFFNN)

    conv = PepSeqModelConfig(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False,
        model_head="conv1d",
        conv_channels=3,
        conv_layers=1,
        conv_kernel_size=3,
        conv_dropout=0.0,
    )
    assert isinstance(build_pepseq_model(conv), PepSeqConvFFNN)

    payload = model_config_to_dict(conv)
    assert payload["hidden_sizes"] == [8]
    assert payload["dropouts"] == [0.0]
    assert "seq_len_feature" not in payload
    assert model_config_from_mapping(payload) == conv


def test_model_config_seq_len_feature_serialization():
    cfg = PepSeqModelConfig(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False,
        seq_len_feature="raw",
    )
    payload = model_config_to_dict(cfg)
    assert payload["seq_len_feature"] == "raw"
    assert model_config_from_mapping(payload) == cfg

    inverse = PepSeqModelConfig(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False,
        seq_len_feature="inverse",
    )
    assert model_config_to_dict(inverse)["seq_len_feature"] == "inverse"

    omitted = dict(payload)
    omitted.pop("seq_len_feature")
    assert model_config_from_mapping(omitted).seq_len_feature is None

    for bad in ("none", None, "bad"):
        bad_payload = dict(payload)
        bad_payload["seq_len_feature"] = bad
        with pytest.raises(ValueError, match="seq_len_feature"):
            model_config_from_mapping(bad_payload)
