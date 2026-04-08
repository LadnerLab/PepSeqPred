"""types.py

Public prediction result types for the user-facing API.
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionResult:
    """Structured residue-level prediction output for one protein sequence.

    Parameters
    ----------
        header : str | None
            Optional sequence identifier associated with the prediction.
        sequence : str
            Cleaned amino-acid sequence used for inference.
        binary_mask : str
            Residue-level binary mask where `"1"` denotes epitope-positive residues.
        length : int
            Sequence length after cleaning.
        n_epitopes : int
            Number of positive residues in `binary_mask`.
        frac_epitope : float
            Fraction of positive residues (`n_epitopes / length`).
        p_epitope_mean : float
            Mean predicted epitope probability across residues.
        p_epitope_max : float
            Maximum predicted epitope probability across residues.
        threshold : float | None
            Applied threshold for single-model outputs; optional for ensemble summaries.
        artifact_mode : str
            Artifact loading mode used by the predictor.
        n_members : int
            Number of model members used for inference.
        votes_needed : int | None
            Majority vote requirement for ensemble predictions.
        member_thresholds : tuple[float, ...]
            Threshold values applied to each model member.
        meta : Mapping[str, Any]
            Additional contextual metadata (artifact/pretrained/device details).

    Returns
    -------
        PredictionResult
            Immutable dataclass instance describing one sequence prediction.

    Raises
    ------
        TypeError
            If required fields are missing during dataclass construction.
    """
    header: str | None
    sequence: str
    binary_mask: str
    length: int
    n_epitopes: int
    frac_epitope: float
    p_epitope_mean: float
    p_epitope_max: float
    threshold: float | None
    artifact_mode: str
    n_members: int
    votes_needed: int | None
    member_thresholds: tuple[float, ...]
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class PretrainedModelInfo:
    """Metadata describing one bundled pretrained PepSeqPred model entry.

    Parameters
    ----------
        model_id : str
            Canonical immutable pretrained identifier.
        aliases : tuple[str, ...]
            User-facing aliases that resolve to `model_id`.
        description : str
            Human-readable summary of the pretrained entry.
        expected_esm_model : str
            ESM backbone name expected by this pretrained model.
        n_members : int
            Number of ensemble members bundled for this entry.
        is_default : bool
            Whether this entry is the default when `model_id` is omitted.
        provenance : Mapping[str, Any]
            Provenance metadata for traceability of bundled artifacts.

    Returns
    -------
        PretrainedModelInfo
            Immutable dataclass instance describing one bundled model entry.

    Raises
    ------
        TypeError
            If required fields are missing during dataclass construction.
    """
    model_id: str
    aliases: tuple[str, ...]
    description: str
    expected_esm_model: str
    n_members: int
    is_default: bool
    provenance: Mapping[str, Any]
