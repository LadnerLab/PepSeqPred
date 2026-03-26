"""Top-level package exports for PepSeqPred."""

from importlib.metadata import PackageNotFoundError, version

from pepseqpred.api import (
    PepSeqPredictor,
    PredictionResult,
    load_predictor,
    predict_fasta,
    predict_sequence
)

try:
    __version__ = version("pepseqpred")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "PepSeqPredictor",
    "PredictionResult",
    "load_predictor",
    "predict_sequence",
    "predict_fasta"
]
