from pathlib import Path
from typing import Dict
import torch


def infer_emb_dim(emb_index: Dict[str, Path]) -> int:
    """Infers the embedding dimension by loading first embedding from index. We expect embedding vectors to be of length 1281."""
    if not emb_index:
        raise ValueError("No embedding found in provided directories")
    first_path = next(iter(emb_index.values()))
    emb = torch.load(first_path, map_location="cpu", weights_only=True)
    if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
        raise ValueError(
            f"Expected embedding tensor of shape (L, D) in {first_path}, got {type(emb)} with shape {getattr(emb, 'shape', None)}")
    return int(emb.size(1))
