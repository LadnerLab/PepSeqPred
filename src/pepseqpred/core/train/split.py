"""split.py

Dataset splitting helpers for PepSeqPred training.

Provides utilities to split protein IDs into train/validation sets and to shard
ID lists across DDP ranks for parallel trials.
"""

import random
from typing import List, Dict, Tuple, Any


def split_ids(ids: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Split protein IDs into training and validation subsets.

    Parameters
    ----------
        ids : List[str]
            List of protein IDs to split.
        val_frac : float
            Fraction of IDs to allocate to validation.
        seed : int
            Seed used to shuffle IDs before splitting.

    Returns
    -------
        Tuple[List[str], List[str]]
        ---------------------------
            `(train_ids, val_ids)` after shuffling and splitting.
    """
    if val_frac < 0.0 or val_frac > 1.0:
        return ids, []
    ids = list(ids)
    range_ = random.Random(seed)
    range_.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return ids[n_val:], ids[:n_val]


def shard_ids_by_rank(ids: List[str], ddp: Dict[str, Any] | None) -> List[str]:
    """
    Shard an ID list across ranks for DDP hyperparameter optimization.

    Parameters
    ----------
        ids : List[str]
            List of protein IDs to shard.
        ddp : Dict[str, Any] | None
            DDP metadata dict with `rank` and `world_size`, or `None` if DDP is disabled.

    Returns
    -------
        List[str]
            Subset of IDs assigned to the current rank.
    """
    if ddp is None:
        return list(ids)
    rank = ddp["rank"]
    world_size = ddp["world_size"]
    return list(ids)[rank::world_size]
