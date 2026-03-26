"""predictor.py

User-facing inference API for PepSeqPred.
"""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional
import esm
import torch
from pepseqpred.api.types import PredictionResult
from pepseqpred.core.embeddings.esm2 import clean_seq
from pepseqpred.core.predict.artifacts import resolve_prediction_members
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    embed_protein_seq,
    infer_decision_threshold,
    predict_ensemble_from_embedding,
    predict_from_embedding
)

_DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"


def _coerce_threshold(threshold: float | None) -> float | None:
    if threshold is None:
        return None
    threshold = float(threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be between (0.0, 1.0)")
    return threshold


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but torch.cuda.is_available() is False")
    return device


def _load_esm(model_name: str, device: str) -> Tuple[torch.nn.Module, Any, int]:
    try:
        loader = getattr(esm.pretrained, model_name)
    except AttributeError as e:
        raise ValueError(f"Unsupported ESM model name: '{model_name}'") from e

    esm_model, alphabet = loader()
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    layer = esm_model.num_layers
    return esm_model, batch_converter, layer


def _read_fasta_records(fasta_path: Path | str) -> Iterator[Tuple[str, str]]:
    header = None
    seq_lines: list[str] = []
    with open(fasta_path, "r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, clean_seq("".join(seq_lines))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            yield header, clean_seq("".join(seq_lines))


def _normalize_sequence_input(
    sequences: Mapping[str, str] | Iterable[str] | Iterable[Tuple[str, str]]
) -> List[Tuple[str | None, str]]:
    if isinstance(sequences, (str, bytes)):
        raise TypeError(
            "sequences must be a mapping or iterable of sequences, not a single string"
        )

    if isinstance(sequences, Mapping):
        return [(str(k), str(v)) for k, v in sequences.items()]

    normalized: list[tuple[str | None, str]] = []
    for item in sequences:
        if isinstance(item, tuple) and len(item) == 2:
            header, seq = item
            normalized.append((str(header), str(seq)))
        else:
            normalized.append((None, str(item)))
    return normalized


class PepSeqPredictor:
    """Predictor API for residue-level epitope inference."""

    def __init__(
        self,
        *,
        psp_models: Sequence[torch.nn.Module],
        member_thresholds: Sequence[float],
        esm_model: torch.nn.Module,
        batch_converter: Any,
        layer: int,
        device: str,
        model_name: str,
        max_tokens: int,
        artifact_mode: str,
        artifact_meta: Mapping[str, Any]
    ) -> None:
        self._psp_models = tuple(psp_models)
        self._member_thresholds = tuple(float(x) for x in member_thresholds)
        self._esm_model = esm_model
        self._batch_converter = batch_converter
        self._layer = int(layer)
        self._device = device
        self._model_name = model_name
        self._max_tokens = int(max_tokens)
        self._artifact_mode = artifact_mode
        self._artifact_meta = dict(artifact_meta)

    @classmethod
    def from_artifact(
        cls,
        model_artifact,
        *,
        ensemble_set_index: int = 1,
        k_folds: Optional[int] = None,
        threshold: Optional[float] = None,
        model_name: str = _DEFAULT_ESM_MODEL,
        max_tokens: int = 1022,
        device: str = "auto",
        model_config: Optional[FFNNModelConfig] = None
    ) -> "PepSeqPredictor":
        threshold = _coerce_threshold(threshold)
        if ensemble_set_index < 1:
            raise ValueError("ensemble_set_index must be >= 1")
        if k_folds is not None and k_folds < 1:
            raise ValueError("k_folds must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        model_artifact = Path(model_artifact)
        if not model_artifact.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_artifact}")

        resolved_device = _resolve_device(device)
        esm_model, batch_converter, layer = _load_esm(
            model_name, resolved_device)

        artifact_mode, member_specs, artifact_meta = resolve_prediction_members(
            model_artifact=model_artifact,
            ensemble_set_index=ensemble_set_index,
            k_folds=k_folds,
        )

        psp_models: List[torch.nn.Module] = []
        member_thresholds: List[float] = []
        member_model_cfgs = []

        for spec in member_specs:
            if not spec.checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {spec.checkpoint}")

            checkpoint = torch.load(
                spec.checkpoint, map_location="cpu", weights_only=True)
            psp_model, model_cfg, _ = build_model_from_checkpoint(
                checkpoint=checkpoint,
                device=resolved_device,
                model_config=model_config,
            )

            member_threshold = (
                threshold
                if threshold is not None
                else (
                    spec.threshold
                    if spec.threshold is not None
                    else infer_decision_threshold(checkpoint)
                )
            )
            if member_threshold <= 0.0 or member_threshold >= 1.0:
                raise ValueError(
                    f"Invalid threshold for member checkpoint: {spec.checkpoint}")

            psp_models.append(psp_model)
            member_thresholds.append(float(member_threshold))
            member_model_cfgs.append(model_cfg)

        if not psp_models:
            raise ValueError("No prediction models were loaded")

        emb_dims = {int(cfg.emb_dim) for cfg in member_model_cfgs}
        if len(emb_dims) != 1:
            raise ValueError(
                f"All ensemble members must share emb_dim for shared-embedding inference, "
                f"got {sorted(emb_dims)}"
            )

        return cls(
            psp_models=psp_models,
            member_thresholds=member_thresholds,
            esm_model=esm_model,
            batch_converter=batch_converter,
            layer=layer,
            device=resolved_device,
            model_name=model_name,
            max_tokens=max_tokens,
            artifact_mode=artifact_mode,
            artifact_meta=artifact_meta
        )

    @property
    def artifact_mode(self) -> str:
        return self._artifact_mode

    @property
    def artifact_meta(self) -> Mapping[str, Any]:
        return dict(self._artifact_meta)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def n_members(self) -> int:
        return len(self._psp_models)

    def _predict_from_embedding(
        self,
        protein_emb: torch.Tensor,
        *,
        threshold: Optional[float] = None
    ) -> Tuple[Dict[str, Any], Tuple[float, ...]]:
        threshold = _coerce_threshold(threshold)
        if threshold is None:
            thresholds = self._member_thresholds
        else:
            thresholds = tuple(float(threshold) for _ in self._psp_models)

        # handle 1 to many models
        if len(self._psp_models) == 1:
            payload = predict_from_embedding(
                psp_model=self._psp_models[0],
                protein_emb=protein_emb,
                device=self._device,
                threshold=float(thresholds[0])
            )
        else:
            payload = predict_ensemble_from_embedding(
                psp_models=self._psp_models,
                protein_emb=protein_emb,
                device=self._device,
                thresholds=thresholds
            )

        return payload, thresholds

    def _payload_to_result(
        self,
        *,
        header: Optional[str],
        sequence: str,
        payload: Mapping[str, Any],
        used_thresholds: Tuple[float, ...]
    ) -> PredictionResult:
        raw_threshold = payload.get("threshold")
        threshold_value: Optional[float] = None
        if raw_threshold is not None:
            try:
                parsed = float(raw_threshold)
            except (TypeError, ValueError):
                parsed = float("nan")
            if math.isfinite(parsed):
                threshold_value = parsed

        payload_member_thresholds = payload.get("member_thresholds")
        if isinstance(payload_member_thresholds, list):
            member_thresholds = tuple(float(x)
                                      for x in payload_member_thresholds)
        else:
            member_thresholds = tuple(float(x) for x in used_thresholds)

        raw_votes_needed = payload.get("votes_needed")
        votes_needed = (int(raw_votes_needed)
                        if raw_votes_needed is not None else None)
        n_members = int(payload.get("n_members", len(self._psp_models)))

        meta = {
            "artifact_mode": self._artifact_mode,
            "artifact_meta": dict(self._artifact_meta),
            "model_name": self._model_name,
            "device": self._device,
            "max_tokens": self._max_tokens
        }

        return PredictionResult(
            header=header,
            sequence=sequence,
            binary_mask=str(payload["binary_mask"]),
            length=int(payload["length"]),
            n_epitopes=int(payload["n_epitopes"]),
            frac_epitope=float(payload["frac_epitope"]),
            p_epitope_mean=float(payload["p_epitope_mean"]),
            p_epitope_max=float(payload["p_epitope_max"]),
            threshold=threshold_value,
            artifact_mode=self._artifact_mode,
            n_members=n_members,
            votes_needed=votes_needed,
            member_thresholds=member_thresholds,
            meta=meta
        )

    def predict_sequence(
        self,
        protein_seq: str,
        *,
        header: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> PredictionResult:
        sequence = clean_seq(protein_seq)
        if not sequence:
            raise ValueError("Protein sequence is empty after cleaning")

        protein_emb = embed_protein_seq(
            protein_seq=sequence,
            esm_model=self._esm_model,
            layer=self._layer,
            batch_converter=self._batch_converter,
            device=self._device,
            max_tokens=self._max_tokens
        )
        payload, used_thresholds = self._predict_from_embedding(
            protein_emb=protein_emb,
            threshold=threshold
        )
        return self._payload_to_result(
            header=header,
            sequence=sequence,
            payload=payload,
            used_thresholds=used_thresholds
        )

    def predict_fasta(
        self,
        fasta_input: Path | str,
        *,
        threshold: Optional[float] = None
    ) -> List[PredictionResult]:
        return [
            self.predict_sequence(seq, header=header, threshold=threshold)
            for header, seq in _read_fasta_records(fasta_input)
        ]

    def write_fasta_predictions(
        self,
        fasta_input: Path | str,
        output_fasta: Path | str,
        *,
        threshold: Optional[float] = None
    ) -> List[PredictionResult]:
        results = self.predict_fasta(
            fasta_input=fasta_input,
            threshold=threshold
        )
        output_fasta = Path(output_fasta)
        output_fasta.parent.mkdir(parents=True, exist_ok=True)

        # write directly to output fasta before returning
        with output_fasta.open("w", encoding="utf-8") as out_f:
            for index, result in enumerate(results, start=1):
                fasta_header = (result.header
                                if result.header
                                else f"sequence_{index:05d}")
                out_f.write(f">{fasta_header}\n{result.binary_mask}\n")
        return results


def load_predictor(
    model_artifact: Path | str,
    *,
    ensemble_set_index: int = 1,
    k_folds: Optional[int] = None,
    threshold: Optional[float] = None,
    model_name: str = _DEFAULT_ESM_MODEL,
    max_tokens: int = 1022,
    device: str = "auto",
    model_config: Optional[FFNNModelConfig] = None
) -> PepSeqPredictor:
    return PepSeqPredictor.from_artifact(
        model_artifact=model_artifact,
        ensemble_set_index=ensemble_set_index,
        k_folds=k_folds,
        threshold=threshold,
        model_name=model_name,
        max_tokens=max_tokens,
        device=device,
        model_config=model_config,
    )


def predict_sequence(
    model_artifact: Path | str,
    protein_seq: str,
    *,
    header: Optional[str] = None,
    threshold: Optional[float] = None,
    ensemble_set_index: int = 1,
    k_folds: Optional[int] = None,
    model_name: str = _DEFAULT_ESM_MODEL,
    max_tokens: int = 1022,
    device: str = "auto",
    model_config: Optional[FFNNModelConfig] = None
) -> PredictionResult:
    predictor = load_predictor(
        model_artifact=model_artifact,
        ensemble_set_index=ensemble_set_index,
        k_folds=k_folds,
        threshold=threshold,
        model_name=model_name,
        max_tokens=max_tokens,
        device=device,
        model_config=model_config
    )
    return predictor.predict_sequence(
        protein_seq=protein_seq,
        header=header,
        threshold=threshold
    )


def predict_fasta(
    model_artifact: Path | str,
    fasta_input: Path | str,
    *,
    output_fasta: Optional[Path | str] = None,
    threshold: Optional[float] = None,
    ensemble_set_index: int = 1,
    k_folds: Optional[int] = None,
    model_name: str = _DEFAULT_ESM_MODEL,
    max_tokens: int = 1022,
    device: str = "auto",
    model_config: Optional[FFNNModelConfig] = None
) -> List[PredictionResult]:
    predictor = load_predictor(
        model_artifact=model_artifact,
        ensemble_set_index=ensemble_set_index,
        k_folds=k_folds,
        threshold=threshold,
        model_name=model_name,
        max_tokens=max_tokens,
        device=device,
        model_config=model_config
    )
    if output_fasta is None:
        return predictor.predict_fasta(
            fasta_input=fasta_input,
            threshold=threshold
        )
    return predictor.write_fasta_predictions(
        fasta_input=fasta_input,
        output_fasta=output_fasta,
        threshold=threshold
    )
