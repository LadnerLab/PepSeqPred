"""Model configuration and construction helpers for PepSeqPred classifiers."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Tuple
import torch.nn as nn
from pepseqpred.core.models.ffnn import PepSeqConvFFNN, PepSeqFFNN

MODEL_HEAD_FFNN = "ffnn"
MODEL_HEAD_CONV1D = "conv1d"
MODEL_HEADS = (MODEL_HEAD_FFNN, MODEL_HEAD_CONV1D)


@dataclass(frozen=True)
class PepSeqModelConfig:
    """Architecture configuration for a PepSeqPred residue classifier."""

    emb_dim: int
    hidden_sizes: Tuple[int, ...]
    dropouts: Tuple[float, ...]
    num_classes: int
    use_layer_norm: bool
    use_residual: bool
    model_head: str = MODEL_HEAD_FFNN
    conv_channels: int = 64
    conv_layers: int = 2
    conv_kernel_size: int = 9
    conv_dropout: float = 0.1


def normalize_model_head(model_head: str | None) -> str:
    """Normalize and validate a model-head token."""
    head = str(model_head or MODEL_HEAD_FFNN).strip().lower()
    if head not in MODEL_HEADS:
        raise ValueError(
            f"model_head must be one of: {', '.join(MODEL_HEADS)}")
    return head


def validate_model_config(config: PepSeqModelConfig) -> PepSeqModelConfig:
    """Validate architecture config values and return a normalized config."""
    model_head = normalize_model_head(config.model_head)
    hidden_sizes = tuple(int(x) for x in config.hidden_sizes)
    dropouts = tuple(float(x) for x in config.dropouts)
    if len(hidden_sizes) != len(dropouts):
        raise ValueError("hidden_sizes and dropouts must have the same length")
    if int(config.emb_dim) <= 0:
        raise ValueError("emb_dim must be > 0")
    if int(config.num_classes) != 1:
        raise ValueError(
            f"Inference expects binary residue model (num_classes=1), got {config.num_classes}"
        )
    if int(config.conv_channels) <= 0:
        raise ValueError("conv_channels must be > 0")
    if int(config.conv_layers) < 1:
        raise ValueError("conv_layers must be >= 1")
    if int(config.conv_kernel_size) < 1 or int(config.conv_kernel_size) % 2 != 1:
        raise ValueError("conv_kernel_size must be a positive odd integer")
    if float(config.conv_dropout) < 0.0 or float(config.conv_dropout) > 1.0:
        raise ValueError("conv_dropout must be in [0.0, 1.0]")
    return PepSeqModelConfig(
        emb_dim=int(config.emb_dim),
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        num_classes=int(config.num_classes),
        use_layer_norm=bool(config.use_layer_norm),
        use_residual=bool(config.use_residual),
        model_head=model_head,
        conv_channels=int(config.conv_channels),
        conv_layers=int(config.conv_layers),
        conv_kernel_size=int(config.conv_kernel_size),
        conv_dropout=float(config.conv_dropout),
    )


def model_config_from_mapping(raw: Mapping[str, Any]) -> PepSeqModelConfig:
    """Build a validated model config from checkpoint or CLI metadata."""
    if not isinstance(raw, Mapping):
        raise ValueError("model_config must be a mapping")

    missing = [
        key for key in (
            "emb_dim",
            "hidden_sizes",
            "dropouts",
            "num_classes",
            "use_layer_norm",
            "use_residual",
        )
        if key not in raw
    ]
    if missing:
        raise ValueError(
            "model_config missing required fields: " + ", ".join(missing)
        )

    return validate_model_config(
        PepSeqModelConfig(
            emb_dim=int(raw["emb_dim"]),
            hidden_sizes=tuple(int(x) for x in raw["hidden_sizes"]),
            dropouts=tuple(float(x) for x in raw["dropouts"]),
            num_classes=int(raw["num_classes"]),
            use_layer_norm=bool(raw["use_layer_norm"]),
            use_residual=bool(raw["use_residual"]),
            model_head=normalize_model_head(raw.get("model_head", MODEL_HEAD_FFNN)),
            conv_channels=int(raw.get("conv_channels", 64)),
            conv_layers=int(raw.get("conv_layers", 2)),
            conv_kernel_size=int(raw.get("conv_kernel_size", 9)),
            conv_dropout=float(raw.get("conv_dropout", 0.1)),
        )
    )


def model_config_to_dict(config: PepSeqModelConfig) -> dict[str, Any]:
    """Return JSON-serializable model config metadata."""
    normalized = validate_model_config(config)
    out = asdict(normalized)
    out["hidden_sizes"] = [int(x) for x in normalized.hidden_sizes]
    out["dropouts"] = [float(x) for x in normalized.dropouts]
    return out


def build_pepseq_model(config: PepSeqModelConfig) -> nn.Module:
    """Construct a PepSeqPred model from a validated config."""
    config = validate_model_config(config)
    if config.model_head == MODEL_HEAD_FFNN:
        return PepSeqFFNN(
            emb_dim=config.emb_dim,
            hidden_sizes=config.hidden_sizes,
            dropouts=config.dropouts,
            num_classes=config.num_classes,
            use_layer_norm=config.use_layer_norm,
            use_residual=config.use_residual,
        )
    if config.model_head == MODEL_HEAD_CONV1D:
        return PepSeqConvFFNN(
            emb_dim=config.emb_dim,
            hidden_sizes=config.hidden_sizes,
            dropouts=config.dropouts,
            num_classes=config.num_classes,
            use_layer_norm=config.use_layer_norm,
            use_residual=config.use_residual,
            conv_channels=config.conv_channels,
            conv_layers=config.conv_layers,
            conv_kernel_size=config.conv_kernel_size,
            conv_dropout=config.conv_dropout,
        )
    raise ValueError(f"Unsupported model_head={config.model_head}")
