"""split.py

Dataset splitting helpers for PepSeqPred training.

Provides utilities to split protein IDs into train/validation sets and to shard
ID lists across DDP ranks for parallel trials.
"""

import math
import random
from typing import List, Dict, Tuple, Any, Optional, Set


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


def split_ids_grouped(
    ids: List[str],
    val_frac: float,
    seed: int,
    groups: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """
    Split protein IDs into training and validation subsets while keeping (taxonomic) groups intact.

    Any IDs missing from `groups` are treated as singleton groups.
    """
    ids = list(ids)
    if val_frac < 0.0 or val_frac > 1.0:
        return ids, []
    if val_frac == 0.0:
        return ids, []
    if val_frac == 1.0:
        return [], ids
    if len(ids) == 0:
        return [], []

    n_val = int(len(ids) * val_frac)

    # build group --> member IDs
    group_to_ids: Dict[str, List[str]] = {}
    for protein_id in ids:
        # use protein_id as fallback
        group_id = str(groups.get(protein_id, protein_id))
        group_to_ids.setdefault(group_id, []).append(protein_id)

    range_ = random.Random(seed)
    group_items = list(group_to_ids.items())
    # randomize order of equal-sized groupings
    range_.shuffle(group_items)
    # descending order families by size (not based on original insertions since shuffled)
    group_items.sort(key=lambda x: -len(x[1]))

    val_groups: Set[str] = set()
    val_count = 0
    remaining = sum(len(members) for _, members in group_items)

    for group_id, members in group_items:
        group_size = len(members)
        remaining -= group_size

        if val_count == n_val:
            continue

        diff_take = abs((val_count + group_size) - n_val)
        diff_skip = abs(val_count - n_val)
        take = diff_take < diff_skip

        # if skipping makes target unreachable, force take
        if (not take) and (val_count < n_val) and ((val_count + remaining) < n_val):
            take = True

        if take:
            val_groups.add(group_id)
            val_count += group_size

    val_ids = [
        pid for pid in ids if str(groups.get(pid, pid)) in val_groups
    ]
    train_ids = [
        pid for pid in ids if str(groups.get(pid, pid)) not in val_groups
    ]
    return train_ids, val_ids


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


def sort_ids_for_locality(ids: List[str],
                          groups: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Sort IDs so items from the same group (for example label shard path) are contiguous.

    Parameters
    ----------
        ids : List[str]
            Protein IDs to sort.
        groups : Optional[Dict[str, str]]
            Optional group dictionary mapping how to group protein IDs to keep them contiguous.
    Returns
    -------
        List[str]
            Sorted list of protein IDs optionally grouped together in contiguous manner.
    """
    groups = groups or {}
    return sorted(list(ids), key=lambda protein_id: (groups.get(protein_id, ""), protein_id))


def partition_ids_weighted(ids: List[str],
                           world_size: int,
                           weights: Optional[Dict[str, float]] = None,
                           ensure_non_empty: bool = False,
                           groups: Optional[Dict[str, str]] = None,
                           sort_within_rank: bool = True) -> List[List[str]]:
    """
    Partition IDs across ranks using greedy load balancing.

    IDs are sorted from largest estimated workload to smallest. Each ID is then
    assigned to the rank with the current lowest total workload, which helps
    reduce long-tail stragglers in DDP.

    Parameters
    ----------
        ids : List[str]
            Protein IDs to partition across ranks.
        world_size : int
            Number of ranks (partitions) to create.
        weights : Optional[Dict[str, float]]
            Optional per-ID workload estimates. Missing or invalid entries default
            to 1.0. A common proxy is embedding file size in bytes.
        ensure_non_empty : bool
            If True, enforce that each rank receives at least one ID. Raises if
            `len(ids) < world_size` or if an empty partition is produced.
        groups : Optional[Dict[str, str]]
            Optional group dictionary mapping how to group protein IDs to keep them contiguous.
        sort_within_rank : bool
            Sorts protein IDs within a specific rank when `True`.

    Returns
    -------
        List[List[str]]
            List of per-rank ID lists where index `i` corresponds to rank `i`.

    Raises
    ------
        ValueError
            If `world_size < 1`, or if `ensure_non_empty=True` and
            `len(ids) < world_size`.
        RuntimeError
            If `ensure_non_empty=True` and partitioning still produces an empty
            rank shard.
    """
    ids = list(ids)
    if world_size < 1:
        raise ValueError(f"Invalid world_size={world_size}, must be >= 1")
    if not ids:
        return [[] for _ in range(world_size)]
    if ensure_non_empty and len(ids) < world_size:
        raise ValueError(
            f"Cannot assign non-empty train shards: n_ids={len(ids)} < world_size={world_size}")

    weights = weights or {}
    groups = groups or {}
    items: List[Tuple[str, float, str]] = []
    for protein_id in ids:
        # workload estimate for this protein ID
        raw = float(weights.get(protein_id, 1.0))
        weight = raw if (math.isfinite(raw) and raw > 0) else 1.0
        items.append((protein_id, weight, groups.get(protein_id, "")))

    # sort by largest workloads first, then keep group locality
    items.sort(key=lambda x: (-x[1], x[2], x[0]))

    parts = [[] for _ in range(world_size)]
    loads = [0.0 for _ in range(world_size)]

    for protein_id, load, _ in items:
        # pick rank with smallest current load, break tie by protein ID
        r = min(range(world_size), key=lambda k: (loads[k], len(parts[k]), k))
        parts[r].append(protein_id)
        loads[r] += load

    # optionally sort wuthin each rank
    if sort_within_rank:
        for r in range(world_size):
            parts[r].sort(key=lambda protein_id: (
                groups.get(protein_id, ""), protein_id))

    if ensure_non_empty and any(len(p) == 0 for p in parts):
        raise RuntimeError("Weighted partition produced an empty rank shard")

    return parts
