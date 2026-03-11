import argparse
import pytest
from pepseqpred.apps.prediction_cli import _build_cli_model_config

pytestmark = pytest.mark.unit


def _args(**overrides):
    base = {
        "emb_dim": None,
        "hidden_sizes": None,
        "dropouts": None,
        "use_layer_norm": None,
        "use_residual": None,
        "num_classes": None
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_cli_model_config_returns_none_if_no_explicit_flags():
    assert _build_cli_model_config(_args()) is None


def test_build_cli_model_config_requires_all_fields_when_any_explicit():
    with pytest.raises(ValueError, match="provide all required values"):
        _build_cli_model_config(_args(emb_dim=4))


def test_build_cli_model_config_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        _build_cli_model_config(
            _args(
                emb_dim=4,
                hidden_sizes="8,4",
                dropouts="0.1",
                use_layer_norm=True,
                use_residual=False,
                num_classes=1
            )
        )


def test_build_cli_model_config_happy_path():
    cfg = _build_cli_model_config(
        _args(
            emb_dim=4,
            hidden_sizes="8,4",
            dropouts="0.1,0.2",
            use_layer_norm=True,
            use_residual=False,
            num_classes=1
        )
    )
    assert cfg.emb_dim == 4
    assert cfg.hidden_sizes == (8, 4)
    assert cfg.dropouts == (0.1, 0.2)
    assert cfg.use_layer_norm is True
    assert cfg.use_residual is False
    assert cfg.num_classes == 1
