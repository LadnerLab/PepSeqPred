"""Sequence-length feature mode helpers."""

from typing import Any

SEQ_LEN_FEATURE_NONE = "none"
SEQ_LEN_FEATURE_RAW = "raw"
SEQ_LEN_FEATURE_INVERSE = "inverse"
SEQ_LEN_FEATURE_AUTO = "auto"

EMBEDDING_SEQ_LEN_FEATURES = (
    SEQ_LEN_FEATURE_NONE,
    SEQ_LEN_FEATURE_RAW,
    SEQ_LEN_FEATURE_INVERSE,
)
MODEL_SEQ_LEN_FEATURES = (
    SEQ_LEN_FEATURE_RAW,
    SEQ_LEN_FEATURE_INVERSE,
)
PREDICTION_SEQ_LEN_FEATURES = (
    SEQ_LEN_FEATURE_AUTO,
    SEQ_LEN_FEATURE_NONE,
    SEQ_LEN_FEATURE_RAW,
    SEQ_LEN_FEATURE_INVERSE,
)


def normalize_embedding_seq_len_feature(value: Any) -> str:
    """Normalize an embedding-generation sequence-length feature mode."""
    if value is None:
        return SEQ_LEN_FEATURE_NONE
    mode = str(value).strip().lower()
    if mode not in EMBEDDING_SEQ_LEN_FEATURES:
        raise ValueError(
            "seq_len_feature must be one of: "
            + ", ".join(EMBEDDING_SEQ_LEN_FEATURES)
        )
    return mode


def normalize_model_seq_len_feature(value: Any) -> str | None:
    """Normalize model-config sequence-length feature metadata."""
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in MODEL_SEQ_LEN_FEATURES:
        raise ValueError(
            "model_config.seq_len_feature must be either 'raw' or 'inverse' "
            "when present"
        )
    return mode


def cli_seq_len_feature_to_model(value: Any) -> str | None:
    """Convert a CLI mode into model-config metadata."""
    mode = normalize_embedding_seq_len_feature(value)
    if mode == SEQ_LEN_FEATURE_NONE:
        return None
    return mode


def model_seq_len_feature_to_embedding(value: Any) -> str:
    """Convert model-config metadata into an embedding feature mode."""
    mode = normalize_model_seq_len_feature(value)
    if mode is None:
        return SEQ_LEN_FEATURE_NONE
    return mode


def normalize_prediction_seq_len_feature(value: Any) -> str:
    """Normalize a prediction-time sequence-length feature mode."""
    if value is None:
        return SEQ_LEN_FEATURE_AUTO
    mode = str(value).strip().lower()
    if mode not in PREDICTION_SEQ_LEN_FEATURES:
        raise ValueError(
            "seq_len_feature must be one of: "
            + ", ".join(PREDICTION_SEQ_LEN_FEATURES)
        )
    return mode
