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


def build_kfold_splits(ids: List[str], n_folds: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    """
    Build deterministic K-fold train/validation splits over IDs.

    Parameters
    ----------
        ids : List[str]
            IDs to partition.
        n_folds : int
            Number of folds, must be >= 2 and <= len(ids).
        seed : int
            Seed used to shuffle IDs before assignment.

    Returns
    -------
        List[Tuple[List[str], List[str]]]
            A list of `(train_ids, val_ids)` pairs for each fold.
    """
    ids = list(ids)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(ids) < n_folds:
        raise ValueError(
            f"n_folds={n_folds} cannot exceed number of IDs={len(ids)}"
        )

    rng = random.Random(seed)
    rng.shuffle(ids)

    fold_bins: List[List[str]] = [[] for _ in range(n_folds)]
    for idx, protein_id in enumerate(ids):
        fold_bins[idx % n_folds].append(protein_id)

    out: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(n_folds):
        val_ids = list(fold_bins[fold_idx])
        train_ids = [
            protein_id
            for idx, members in enumerate(fold_bins)
            if idx != fold_idx
            for protein_id in members
        ]
        out.append((train_ids, val_ids))
    return out


def build_grouped_kfold_splits(
    ids: List[str],
    n_folds: int,
    seed: int,
    groups: Dict[str, str]
) -> List[Tuple[List[str], List[str]]]:
    """
    Build deterministic K-fold splits while keeping groups intact.

    Any ID missing in `groups` is treated as a singleton group keyed by the ID.
    """
    ids = list(ids)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(ids) == 0:
        raise ValueError("ids cannot be empty")

    # build group -> member IDs while preserving the order seen in `ids`
    group_to_ids: Dict[str, List[str]] = {}
    for protein_id in ids:
        group_id = str(groups.get(protein_id, protein_id))
        group_to_ids.setdefault(group_id, []).append(protein_id)

    if len(group_to_ids) < n_folds:
        raise ValueError(
            f"n_folds={n_folds} cannot exceed number of groups={len(group_to_ids)}"
        )

    rng = random.Random(seed)
    group_items = list(group_to_ids.items())
    rng.shuffle(group_items)
    # assign larger groups first for better fold balance
    group_items.sort(key=lambda item: -len(item[1]))

    fold_groups: List[Set[str]] = [set() for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]
    for group_id, members in group_items:
        fold_idx = min(
            range(n_folds),
            key=lambda k: (fold_sizes[k], len(fold_groups[k]), k)
        )
        fold_groups[fold_idx].add(group_id)
        fold_sizes[fold_idx] += len(members)

    if any(len(member_groups) == 0 for member_groups in fold_groups):
        raise RuntimeError("Grouped K-fold assignment produced an empty fold")

    out: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(n_folds):
        val_group_ids = fold_groups[fold_idx]
        val_ids = [
            protein_id
            for protein_id in ids
            if str(groups.get(protein_id, protein_id)) in val_group_ids
        ]
        train_ids = [
            protein_id
            for protein_id in ids
            if str(groups.get(protein_id, protein_id)) not in val_group_ids
        ]
        if len(val_ids) == 0 or len(train_ids) == 0:
            raise RuntimeError(
                f"Fold {fold_idx + 1} has empty split (train={len(train_ids)}, val={len(val_ids)})"
            )
        out.append((train_ids, val_ids))
    return out


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


def shuffle_ids_by_group(
    ids: List[str],
    seed: int,
    groups: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Shuffle group order while keeping IDs within each group contiguous.


    Parameters
    ----------
        ids : List[str]
            Protein IDs to sort.
        seed : int
            Number used to seed the random number generator for shuffling.
        groups : Optional[Dict[str, str]]
            Optional group dictionary mapping how to group protein IDs to keep them contiguous.

    Returns
    -------
        List[str]
        Sorted list of protein IDs optionally grouped together in contiguous manner.
    """
    ids = list(ids)
    if len(ids) <= 1:
        return ids

    groups = groups or {}
    groups_to_ids: Dict[str, List[str]] = {}
    for protein_id in ids:
        group_id = str(groups.get(protein_id, ""))
        groups_to_ids.setdefault(group_id, []).append(protein_id)

    rng = random.Random(seed)
    group_ids = sorted(groups_to_ids.keys())
    rng.shuffle(group_ids)

    out: List[str] = []
    for group_id in group_ids:
        out.extend(groups_to_ids[group_id])
    return out


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
