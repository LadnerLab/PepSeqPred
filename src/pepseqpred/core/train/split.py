import random
from typing import List, Dict, Tuple, Any


def split_ids(ids: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """Splits protein IDs into training and validation subsets."""
    if val_frac < 0.0 or val_frac > 1.0:
        return ids, []
    ids = list(ids)
    range_ = random.Random(seed)
    range_.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return ids[n_val:], ids[:n_val]


def shard_ids_by_rank(ids: List[str], ddp: Dict[str, Any] | None) -> List[str]:
    """Shards an ID list across ranks for DDP hyperparameter optimization."""
    if ddp is None:
        return list(ids)
    rank = ddp["rank"]
    world_size = ddp["world_size"]
    return list(ids)[rank::world_size]
