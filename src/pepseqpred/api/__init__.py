"""Public-facing API for PepSeqPred inference."""

from pepseqpred.api.predictor import (
    PepSeqPredictor,
    load_predictor,
    predict_fasta,
    predict_sequence
)
from pepseqpred.api.types import PredictionResult

__all__ = [
    "PepSeqPredictor",
    "PredictionResult",
    "load_predictor",
    "predict_fasta",
    "predict_sequence"
]
