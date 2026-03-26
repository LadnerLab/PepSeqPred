"""Prediction utilities and artifact resolution for PepSeqPred."""

from pepseqpred.core.predict.artifacts import PredictionMember, resolve_prediction_members
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    embed_protein_seq,
    infer_decision_threshold,
    infer_model_config_from_state,
    normalize_state_dict_keys,
    predict_ensemble_from_embedding,
    predict_from_embedding,
    predict_member_probabilities_from_embedding,
    predict_protein
)

__all__ = [
    "PredictionMember",
    "resolve_prediction_members",
    "FFNNModelConfig",
    "build_model_from_checkpoint",
    "embed_protein_seq",
    "infer_decision_threshold",
    "infer_model_config_from_state",
    "normalize_state_dict_keys",
    "predict_member_probabilities_from_embedding",
    "predict_from_embedding",
    "predict_ensemble_from_embedding",
    "predict_protein"
]
