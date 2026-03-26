"""artifacts.py

Shared model artifact resolution for prediction/evaluation APIs.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Optional


@dataclass(frozen=True)
class PredictionMember:
    checkpoint: Path
    threshold: float | None
    fold_index: int | None
    member_index: int | None


def _as_optional_int(value: Any) -> int | None:
    """Tries to return integer from unknown value, else None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_threshold(value: Any) -> float | None:
    """Tries to return float threshold from value, else None."""
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        return None
    return threshold


def _resolve_member_checkpoint_path(raw_path: Any, manifest_path: Path) -> Path | None:
    """Tries to resolve member checkpoint raw path, else None."""
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _member_sort_key(member: PredictionMember) -> Tuple[int, int, str]:
    """Gets the member sort key as tuple given PredictionMember."""
    fold_index = member.fold_index if member.fold_index is not None else int(
        1e9)
    member_index = member.member_index if member.member_index is not None else int(
        1e9)
    return (fold_index, member_index, str(member.checkpoint))


def _resolve_manifest_members(
    manifest_path: Path,
    ensemble_set_index: int
) -> Tuple[List[PredictionMember], Dict[str, Any]]:
    """Resolves manifest members into a sorted list of PredictionMembers and metadata."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in ensemble manifest: {manifest_path}") from e

    if not isinstance(payload, Mapping):
        raise ValueError("Ensemble manifest must be a JSON object")

    schema_version = _as_optional_int(payload.get("schema_version"))
    selected_set: Mapping[str, Any] | None = None
    members_raw: Any = None

    if schema_version == 2:
        sets_raw = payload.get("sets")
        if not isinstance(sets_raw, list):
            raise ValueError(
                "schema_version=2 manifest must contain 'sets' list")

        for entry in sets_raw:
            if not isinstance(entry, Mapping):
                continue
            if _as_optional_int(entry.get("set_index")) == ensemble_set_index:
                selected_set = entry
                break

        if selected_set is None:
            raise ValueError(
                f"Requested ensemble_set_index={ensemble_set_index} not found in manifest"
            )
        members_raw = selected_set.get("members")
    else:
        selected_set = payload
        members_raw = payload.get("members")

    if not isinstance(members_raw, list):
        raise ValueError("Manifest must contain a 'members' list")

    members: List[PredictionMember] = []
    for member in members_raw:
        if not isinstance(member, Mapping):
            continue

        status = str(member.get("status", "")).strip().upper()
        checkpoint_path = _resolve_member_checkpoint_path(
            member.get("checkpoint"), manifest_path
        )
        if status != "OK" or checkpoint_path is None:
            continue

        members.append(
            PredictionMember(
                checkpoint=checkpoint_path,
                threshold=_as_optional_threshold(member.get("threshold")),
                fold_index=_as_optional_int(member.get("fold_index")),
                member_index=_as_optional_int(member.get("member_index"))
            )
        )

    if not members:
        raise ValueError(
            "No valid ensemble members found (requires status=OK and checkpoint path)"
        )

    members_sorted = sorted(members, key=_member_sort_key)
    meta: Dict[str, Any] = {
        "schema_version": schema_version,
        "set_index": _as_optional_int(selected_set.get("set_index"))
        if isinstance(selected_set, Mapping)
        else None,
        "n_members_total": len(members_raw),
        "n_members_valid": len(members_sorted)
    }
    return members_sorted, meta


def resolve_prediction_members(
    model_artifact: Path | str,
    ensemble_set_index: int = 1,
    k_folds: Optional[int] = None
) -> Tuple[str, List[PredictionMember], Dict[str, Any]]:
    """Resolve members from a .pt checkpoint or ensemble-manifest .json artifact."""
    model_artifact = Path(model_artifact)
    suffix = model_artifact.suffix.lower()

    # handle 1 or more models (single checkpoint or ensemble of k models)
    if suffix == ".pt":
        members = [
            PredictionMember(
                checkpoint=model_artifact,
                threshold=None,
                fold_index=1,
                member_index=1
            )
        ]
        mode = "single-checkpoint"
        meta = {
            "schema_version": None,
            "set_index": None,
            "n_members_total": 1,
            "n_members_valid": 1
        }
    elif suffix == ".json":
        members, meta = _resolve_manifest_members(
            manifest_path=model_artifact,
            ensemble_set_index=ensemble_set_index
        )
        mode = "ensemble-manifest"
    else:
        raise ValueError(
            f"Unsupported model artifact type '{model_artifact.suffix}' "
            "Expected .pt checkpoint or .json ensemble manifest."
        )

    # only return k folds if provided
    if k_folds is not None:
        if k_folds < 1:
            raise ValueError("k_folds must be >= 1")
        if k_folds > len(members):
            raise ValueError(
                f"k_folds={k_folds} exceeds valid member count={len(members)}"
            )
        members = members[:k_folds]

    return mode, members, meta
