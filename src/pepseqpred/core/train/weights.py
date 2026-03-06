"""weights.py

Class weighting helpers for PepSeqPred training.

Provides utilities to count positive/negative residues and compute a global
positive class weight across DDP ranks.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


def compute_pos_neg_counts(loader: DataLoader) -> Tuple[int, int]:
    """
    Calculates the positive and negative counts of each class.

    Parameters
    ----------
        loader : DataLoader
            Model training data with likely imbalanced classes.

    Returns
    -------
        Tuple[int, int]
            (pos_count, neg_count) over valid (masked) residues.
    """
    neg, pos = 0, 0
    for batch in loader:
        if len(batch) == 2:
            _, y = batch
            mask = None
        else:
            _, y, mask = batch
        y = y.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            y = y[mask]
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    return pos, neg


def global_pos_weight(local_pos: int, local_neg: int, ddp: Dict[str, Any] | None) -> float:
    """
    Compute global negative/positive class weight across ranks if DDP is running.

    Parameters
    ----------
        local_pos : int
            Local count of positive residues.
        local_neg : int
            Local count of negative residues.
        ddp : Dict[str, Any] | None
            DDP metadata dict, or `None` if DDP is disabled.

    Returns
    -------
        float
            Ratio of negatives to positives, aggregated across ranks when DDP is enabled.
    """
    if ddp is None:
        return float(local_neg / max(local_pos, 1))
    t = torch.tensor([local_pos, local_neg], device=torch.device("cuda"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    pos = int(t[0].item())
    neg = int(t[1].item())
    return float(neg / max(pos, 1))


def pos_weight_from_label_shards(label_shards: List[Path]) -> float:
    """
    Calculates the positive (definite epitope) class weight across label shard files.

    Parameters
    ----------
        label_shards : List[Path]
            The file paths to each label shard.

    Returns
    -------
        float
            The weight of the positive class.

    Raises
    ------
        ValueError
            When the label payload is missing the `class_stats` dictionary.
    """
    total_pos, total_neg = 0, 0
    for shard in label_shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        stats = payload.get("class_stats")
        if stats is None:
            raise ValueError(
                f"{shard} missing class_stats (rebuild labels with --calc-pos-weight)"
            )
        total_pos += int(stats["pos_count"])
        total_neg += int(stats["neg_counts"])
    return float(total_neg / max(1, total_pos))
