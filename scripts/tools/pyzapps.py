from typing import Dict, Any


APPS: Dict[str, Any] = {
    "esm": "pepseqpred.apps.esm_cli:main",
    "labels": "pepseqpred.apps.labels_cli:main",
    "preprocess": "pepseqpred.apps.preprocess_cli:main",
    "train_ffnn": "pepseqpred.apps.train_ffnn_cli:main",
    "train_ffnn_optuna": "pepseqpred.apps.train_ffnn_optuna_cli:main",
    "predict": "pepseqpred.apps.prediction_cli:main"
}
