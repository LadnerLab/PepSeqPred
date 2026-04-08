"""predictor.py

User-facing inference API for PepSeqPred.
"""
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional
import esm
import torch
from pepseqpred.api.pretrainedregistry import (
    list_pretrained_models as _list_pretrained_models,
    open_pretrained_model_artifact
)
from pepseqpred.api.types import PredictionResult, PretrainedModelInfo
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
    """Validate an optional threshold and normalize it to float."""
    if threshold is None:
        return None
    threshold = float(threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be between (0.0, 1.0)")
    return threshold


def _resolve_device(device: str) -> str:
    """Resolve the compute device, honoring the `auto` alias."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but torch.cuda.is_available() is False")
    return device


def _load_esm(model_name: str, device: str) -> Tuple[torch.nn.Module, Any, int]:
    """Load an ESM model/alphabet pair and move the model to `device`."""
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
    """Yield cleaned `(header, sequence)` records from a FASTA file."""
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
    """Normalize heterogeneous sequence inputs into `(header, sequence)` pairs."""
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
    """Residue-level epitope predictor with shared embedding inference.

    Parameters
    ----------
        psp_models : Sequence[torch.nn.Module]
            Loaded single-member or ensemble FFNN models used for inference.
        member_thresholds : Sequence[float]
            Decision thresholds applied to model probabilities.
        esm_model : torch.nn.Module
            Loaded ESM model used to generate residue embeddings.
        batch_converter : Any
            Batch converter returned by the selected ESM alphabet.
        layer : int
            ESM representation layer index used for embedding extraction.
        device : str
            Device string (`"cpu"`/`"cuda"`/`"cuda:N"`) used for inference.
        model_name : str
            Name of the ESM backbone used by this predictor instance.
        max_tokens : int
            Maximum residue token budget per ESM pass (excluding CLS/EOS).
        artifact_mode : str
            Artifact resolution mode (`single-checkpoint` or `ensemble-manifest`).
        artifact_meta : Mapping[str, Any]
            Additional metadata returned during artifact member resolution.
        pretrained_meta : Mapping[str, Any] | None
            Optional metadata describing a bundled pretrained source.

    Returns
    -------
        PepSeqPredictor
            Configured predictor instance ready for sequence/FASTA inference.

    Raises
    ------
        ValueError
            If threshold coercion or numeric normalization fails.
    """

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
        artifact_meta: Mapping[str, Any],
        pretrained_meta: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Initialize predictor state from already loaded models and metadata."""
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
        self._pretrained_meta = (
            dict(pretrained_meta) if pretrained_meta is not None else {}
        )

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
        """Build a predictor from a checkpoint `.pt` or manifest `.json` artifact.

        Parameters
        ----------
            model_artifact : Path | str
                Path to a checkpoint file or ensemble manifest file.
            ensemble_set_index : int
                Ensemble set index to resolve when `model_artifact` is schema-v2 JSON.
            k_folds : int | None
                Optional number of sorted member checkpoints to keep.
            threshold : float | None
                Optional global threshold override applied to all members.
            model_name : str
                ESM model name used for embedding generation.
            max_tokens : int
                Maximum residue token budget per ESM pass.
            device : str
                Device selector string or `"auto"`.
            model_config : FFNNModelConfig | None
                Optional explicit FFNN architecture config for checkpoint loading.

        Returns
        -------
            PepSeqPredictor
                Fully initialized predictor instance.

        Raises
        ------
            FileNotFoundError
                If the artifact or any resolved member checkpoint does not exist.
            ValueError
                If argument values are invalid or resolved members are inconsistent.
        """
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

    @classmethod
    def from_pretrained(
        cls,
        model_id: Optional[str] = None,
        *,
        threshold: Optional[float] = None,
        k_folds: Optional[int] = None,
        max_tokens: int = 1022,
        device: str = "auto",
        model_config: Optional[FFNNModelConfig] = None
    ) -> "PepSeqPredictor":
        """Build a predictor from a bundled pretrained model id or alias.

        Parameters
        ----------
            model_id : str | None
                Canonical model id or alias. `None` resolves to the default model.
            threshold : float | None
                Optional global threshold override applied to all ensemble members.
            k_folds : int | None
                Optional number of sorted member checkpoints to keep.
            max_tokens : int
                Maximum residue token budget per ESM pass.
            device : str
                Device selector string or `"auto"`.
            model_config : FFNNModelConfig | None
                Optional explicit FFNN architecture config for checkpoint loading.

        Returns
        -------
            PepSeqPredictor
                Predictor initialized from packaged pretrained resources.

        Raises
        ------
            FileNotFoundError
                If packaged pretrained artifacts are missing.
            ValueError
                If `model_id` is unknown or inference arguments are invalid.
        """
        with open_pretrained_model_artifact(model_id) as (artifact_path, info):
            predictor = cls.from_artifact(
                model_artifact=artifact_path,
                ensemble_set_index=1,
                k_folds=k_folds,
                threshold=threshold,
                model_name=info.expected_esm_model,
                max_tokens=max_tokens,
                device=device,
                model_config=model_config
            )

        predictor._pretrained_meta = {
            "model_id": info.model_id,
            "aliases": list(info.aliases),
            "is_default": bool(info.is_default),
            "expected_esm_model": info.expected_esm_model,
            "provenance": dict(info.provenance)
        }
        return predictor

    @property
    def artifact_mode(self) -> str:
        """Get the artifact resolution mode used by this predictor."""
        return self._artifact_mode

    @property
    def artifact_meta(self) -> Mapping[str, Any]:
        """Get a copy of resolved artifact metadata."""
        return dict(self._artifact_meta)

    @property
    def pretrained_meta(self) -> Mapping[str, Any]:
        """Get a copy of pretrained metadata, if this predictor is pretrained."""
        return dict(self._pretrained_meta)

    @property
    def model_name(self) -> str:
        """Get the ESM backbone name used by this predictor."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get the active inference device string."""
        return self._device

    @property
    def n_members(self) -> int:
        """Get the number of ensemble members loaded in this predictor."""
        return len(self._psp_models)

    def _predict_from_embedding(
        self,
        protein_emb: torch.Tensor,
        *,
        threshold: Optional[float] = None
    ) -> Tuple[Dict[str, Any], Tuple[float, ...]]:
        """Run single-model or ensemble inference from a precomputed embedding."""
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
        """Convert low-level prediction payload values into `PredictionResult`."""
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
        if self._pretrained_meta:
            meta["pretrained"] = dict(self._pretrained_meta)

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
        """Predict a residue-level epitope mask for a single protein sequence.

        Parameters
        ----------
            protein_seq : str
                Raw protein sequence to clean, embed, and score.
            header : str | None
                Optional identifier to carry through into the output result.
            threshold : float | None
                Optional override threshold applied to all member predictions.

        Returns
        -------
            PredictionResult
                Structured output containing binary mask summary fields and metadata.

        Raises
        ------
            ValueError
                If the cleaned sequence is empty or threshold is out of bounds.
        """
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
        """Predict residue-level masks for all sequences in a FASTA input.

        Parameters
        ----------
            fasta_input : Path | str
                Input FASTA file path.
            threshold : float | None
                Optional global threshold override applied during prediction.

        Returns
        -------
            List[PredictionResult]
                Prediction outputs in FASTA record order.

        Raises
        ------
            FileNotFoundError
                If the FASTA file cannot be found.
            ValueError
                If any record fails sequence cleaning or threshold validation.
        """
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
        """Predict FASTA records and write binary masks to an output FASTA file.

        Parameters
        ----------
            fasta_input : Path | str
                Input FASTA file path.
            output_fasta : Path | str
                Output FASTA path for writing binary mask sequences.
            threshold : float | None
                Optional global threshold override applied during prediction.

        Returns
        -------
            List[PredictionResult]
                Prediction outputs matching records written to `output_fasta`.

        Raises
        ------
            FileNotFoundError
                If the input FASTA path cannot be found.
            ValueError
                If sequence cleaning or threshold validation fails.
        """
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
    """Create a predictor from an artifact path.

    Parameters
    ----------
        model_artifact : Path | str
            Path to a checkpoint `.pt` or ensemble manifest `.json`.
        ensemble_set_index : int
            Ensemble set index for schema-v2 root manifests.
        k_folds : int | None
            Optional number of sorted members to keep.
        threshold : float | None
            Optional global threshold override.
        model_name : str
            ESM model name used for embedding generation.
        max_tokens : int
            Maximum residue token budget per ESM pass.
        device : str
            Device selector string or `"auto"`.
        model_config : FFNNModelConfig | None
            Optional explicit FFNN architecture config.

    Returns
    -------
        PepSeqPredictor
            Initialized predictor instance.

    Raises
    ------
        FileNotFoundError
            If artifact or member checkpoints are missing.
        ValueError
            If arguments or resolved members are invalid.
    """
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


def list_pretrained_models() -> List[PretrainedModelInfo]:
    """List bundled pretrained model metadata entries.

    Returns
    -------
        List[PretrainedModelInfo]
            Metadata for canonical pretrained ids and aliases.

    Raises
    ------
        FileNotFoundError
            If packaged pretrained resources are unavailable at runtime.
    """
    return _list_pretrained_models()


def load_pretrained_predictor(
    model_id: Optional[str] = None,
    *,
    threshold: Optional[float] = None,
    k_folds: Optional[int] = None,
    max_tokens: int = 1022,
    device: str = "auto",
    model_config: Optional[FFNNModelConfig] = None
) -> PepSeqPredictor:
    """Create a predictor from a bundled pretrained model id or alias.

    Parameters
    ----------
        model_id : str | None
            Canonical id or alias. `None` selects the default pretrained model.
        threshold : float | None
            Optional global threshold override.
        k_folds : int | None
            Optional number of sorted members to keep.
        max_tokens : int
            Maximum residue token budget per ESM pass.
        device : str
            Device selector string or `"auto"`.
        model_config : FFNNModelConfig | None
            Optional explicit FFNN architecture config.

    Returns
    -------
        PepSeqPredictor
            Initialized predictor using bundled pretrained artifacts.

    Raises
    ------
        FileNotFoundError
            If packaged pretrained resources are missing.
        ValueError
            If `model_id` or prediction arguments are invalid.
    """
    return PepSeqPredictor.from_pretrained(
        model_id=model_id,
        threshold=threshold,
        k_folds=k_folds,
        max_tokens=max_tokens,
        device=device,
        model_config=model_config
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
    """Convenience wrapper that loads a predictor and predicts one sequence.

    Parameters
    ----------
        model_artifact : Path | str
            Path to checkpoint `.pt` or manifest `.json`.
        protein_seq : str
            Protein sequence to score.
        header : str | None
            Optional identifier to include in the result.
        threshold : float | None
            Optional global threshold override.
        ensemble_set_index : int
            Ensemble set index for schema-v2 root manifests.
        k_folds : int | None
            Optional number of sorted ensemble members to keep.
        model_name : str
            ESM model name used for embedding generation.
        max_tokens : int
            Maximum residue token budget per ESM pass.
        device : str
            Device selector string or `"auto"`.
        model_config : FFNNModelConfig | None
            Optional explicit FFNN architecture config.

    Returns
    -------
        PredictionResult
            Structured prediction for the provided sequence.

    Raises
    ------
        FileNotFoundError
            If artifact or member checkpoints are missing.
        ValueError
            If sequence cleaning or inference arguments are invalid.
    """
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
    """Convenience wrapper to predict FASTA input with one artifact load.

    Parameters
    ----------
        model_artifact : Path | str
            Path to checkpoint `.pt` or manifest `.json`.
        fasta_input : Path | str
            Input FASTA file path.
        output_fasta : Path | str | None
            Optional output path to write binary mask FASTA.
        threshold : float | None
            Optional global threshold override.
        ensemble_set_index : int
            Ensemble set index for schema-v2 root manifests.
        k_folds : int | None
            Optional number of sorted ensemble members to keep.
        model_name : str
            ESM model name used for embedding generation.
        max_tokens : int
            Maximum residue token budget per ESM pass.
        device : str
            Device selector string or `"auto"`.
        model_config : FFNNModelConfig | None
            Optional explicit FFNN architecture config.

    Returns
    -------
        List[PredictionResult]
            Prediction outputs in FASTA record order.

    Raises
    ------
        FileNotFoundError
            If artifact, input FASTA, or resolved checkpoints are missing.
        ValueError
            If thresholds or sequence inputs are invalid.
    """
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
