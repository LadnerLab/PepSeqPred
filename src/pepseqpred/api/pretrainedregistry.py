"""pretrainedregistry.py

Registry and resource resolution for bundled pretrained predictor artifacts.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Iterator, Mapping, Tuple, List

from pepseqpred.api.types import PretrainedModelInfo

_RESOURCE_PACKAGE = "pepseqpred.api"
_RESOURCE_ROOT = "pretrained_artifacts"
_DEFAULT_MODEL_ID = "flagship2-v1"


@dataclass(frozen=True)
class _RegistryEntry:
    """Internal immutable row describing one bundled pretrained model."""
    model_id: str
    aliases: Tuple[str, ...]
    description: str
    expected_esm_model: str
    artifact_subdir: str
    artifact_name: str
    n_members: int
    provenance: Mapping[str, Any]


_REGISTRY: Tuple[_RegistryEntry, ...] = (
    _RegistryEntry(
        model_id="flagship1-v1",
        aliases=("flagship1",),
        description=(
            "Bundled 5-fold flagship ensemble (set 1), tuned for residue-level "
            "epitope inference."
        ),
        expected_esm_model="esm2_t33_650M_UR50D",
        artifact_subdir="flagship1-v1",
        artifact_name="ensemble_manifest.json",
        n_members=5,
        provenance={
            "source_run": "ffnn_ens_1.0_27423100",
            "source_set_index": 1,
            "source_split_type": "id-family",
            "source_train_mode": "ensemble-kfold",
            "source_note": (
                "phaseA artifact bundled in-repo; swap registry artifact payload when "
                "phaseB flagship checkpoint set is available."
            )
        }
    ),
    _RegistryEntry(
        model_id="flagship2-v1",
        aliases=("flagship2", "default"),
        description=(
            "Bundled 5-fold flagship ensemble (set 1), tuned for residue-level "
            "epitope inference."
        ),
        expected_esm_model="esm2_t33_650M_UR50D",
        artifact_subdir="flagship2-v1",
        artifact_name="ensemble_manifest.json",
        n_members=5,
        provenance={
            "source_run": "ffnn_ens_1.1_27450180",
            "source_set_index": 1,
            "source_split_type": "id-family",
            "source_train_mode": "ensemble-kfold",
            "source_note": (
                "phaseA artifact bundled in-repo; swap registry artifact payload when "
                "phaseB flagship checkpoint set is available."
            )
        }
    )
)

_REGISTRY_BY_ID = {entry.model_id: entry for entry in _REGISTRY}


def _entry_to_info(entry: _RegistryEntry) -> PretrainedModelInfo:
    """Convert an internal registry row into a public metadata dataclass."""
    return PretrainedModelInfo(
        model_id=entry.model_id,
        aliases=entry.aliases,
        description=entry.description,
        expected_esm_model=entry.expected_esm_model,
        n_members=entry.n_members,
        is_default=(entry.model_id == _DEFAULT_MODEL_ID),
        provenance=dict(entry.provenance),
    )


def list_pretrained_models() -> List[PretrainedModelInfo]:
    """Return canonical bundled pretrained model definitions."""
    return [_entry_to_info(entry) for entry in _REGISTRY]


def canonicalize_pretrained_model_id(model_id: str | None) -> str:
    """Resolve a canonical pretrained model id from alias/default input.

    Parameters
    ----------
        model_id : str | None
            Canonical id, alias, empty string, or `None`.

    Returns
    -------
        str
            Canonical model id present in the registry.

    Raises
    ------
        ValueError
            If the input does not map to any known canonical id or alias.
    """
    if model_id is None:
        return _DEFAULT_MODEL_ID
    token = str(model_id).strip()
    if not token:
        return _DEFAULT_MODEL_ID

    token_lc = token.lower()

    for entry in _REGISTRY:
        if token_lc == entry.model_id.lower():
            return entry.model_id
        if any(token_lc == alias.lower() for alias in entry.aliases):
            return entry.model_id

    canonical_ids = ", ".join(entry.model_id for entry in _REGISTRY)
    alias_names = ", ".join(
        alias for entry in _REGISTRY for alias in entry.aliases)
    raise ValueError(
        f"Unknown pretrained model_id '{model_id}'. "
        f"Known ids: {canonical_ids}. Known aliases: {alias_names}."
    )


def get_pretrained_model_info(model_id: str | None) -> PretrainedModelInfo:
    """Return metadata for a canonical/alias/default pretrained id input.

    Parameters
    ----------
        model_id : str | None
            Canonical id, alias, empty string, or `None`.

    Returns
    -------
        PretrainedModelInfo
            Metadata for the resolved canonical pretrained model.

    Raises
    ------
        ValueError
            If `model_id` does not resolve to a known model entry.
    """
    canonical_id = canonicalize_pretrained_model_id(model_id)
    return _entry_to_info(_REGISTRY_BY_ID[canonical_id])


@contextmanager
def open_pretrained_model_artifact(
    model_id: str | None,
) -> Iterator[Tuple[Path, PretrainedModelInfo]]:
    """Yield a filesystem path to a bundled manifest plus model metadata.

    Parameters
    ----------
        model_id : str | None
            Canonical pretrained id or alias. `None` selects the default model.

    Yields
    ------
        Iterator[Tuple[Path, PretrainedModelInfo]]
            Context-managed iterator yielding manifest path and resolved metadata.

    Raises
    ------
        FileNotFoundError
            If bundled pretrained resources are missing at runtime.
        ValueError
            If `model_id` does not resolve to a known model entry.
    """

    canonical_id = canonicalize_pretrained_model_id(model_id)
    entry = _REGISTRY_BY_ID[canonical_id]

    artifact_dir = files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_ROOT).joinpath(
        entry.artifact_subdir
    )
    artifact_manifest = artifact_dir.joinpath(entry.artifact_name)
    if not artifact_manifest.is_file():
        raise FileNotFoundError(
            f"Bundled pretrained manifest is missing for '{entry.model_id}': "
            f"{_RESOURCE_ROOT}/{entry.artifact_subdir}/{entry.artifact_name}"
        )

    # Directory-level extraction keeps relative checkpoint paths valid when needed.
    with as_file(artifact_dir) as artifact_dir_path:
        manifest_path = Path(artifact_dir_path) / entry.artifact_name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Bundled pretrained manifest does not exist at resolved path: "
                f"{manifest_path}"
            )
        yield manifest_path, _entry_to_info(entry)
