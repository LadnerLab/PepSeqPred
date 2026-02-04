"""embedding.py

Embedding inspection utilities for training.

Provides helpers to infer embedding dimensionality from stored per-protein
embedding tensors.
"""

from pathlib import Path
from typing import Dict
import torch


def infer_emb_dim(emb_index: Dict[str, Path]) -> int:
    """
    Infer embedding dimension by loading the first embedding in the index.

    Parameters
    ----------
        emb_index : Dict[str, Path]
            Mapping of protein IDs to embedding file paths.

    Returns
    -------
        int
            The embedding dimension (D) inferred from the first tensor of shape (L, D).

    Raises
    ------
        ValueError
            If `emb_index` is empty.
        ValueError
            If the loaded embedding is not a 2D tensor of shape (L, D).
    """
    if not emb_index:
        raise ValueError("No embedding found in provided directories")
    first_path = next(iter(emb_index.values()))
    emb = torch.load(first_path, map_location="cpu", weights_only=True)
    if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
        raise ValueError(
            f"Expected embedding tensor of shape (L, D) in {first_path}, got {type(emb)} with shape {getattr(emb, 'shape', None)}")
    return int(emb.size(1))
