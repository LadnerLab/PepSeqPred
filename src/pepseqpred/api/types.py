"""types.py

Public prediction result types for the user-facing API.
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionResult:
    """Structured residue-level prediction output for one protein sequence."""
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
