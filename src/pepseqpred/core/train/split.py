"""split.py

Dataset splitting helpers for PepSeqPred training.

Provides utilities to split protein IDs into train/validation sets and to shard
ID lists across DDP ranks for parallel trials.
"""

import math
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set, Mapping, Sequence

import torch


SPLIT_STRATEGIES: Tuple[str, ...] = ("size-balanced", "label-stratified")


def _positive_rate(pos_residues: int, valid_residues: int) -> float | None:
    if valid_residues <= 0:
        return None
    return float(pos_residues / valid_residues)


def _zero_stats() -> Dict[str, int]:
    return {
        "protein_count": 0,
        "valid_residues": 0,
        "positive_residues": 0,
        "negative_residues": 0,
    }


def _add_stats(left: Mapping[str, int], right: Mapping[str, int]) -> Dict[str, int]:
    return {
        "protein_count": int(left.get("protein_count", 0)) + int(right.get("protein_count", 0)),
        "valid_residues": int(left.get("valid_residues", 0)) + int(right.get("valid_residues", 0)),
        "positive_residues": int(left.get("positive_residues", 0)) + int(right.get("positive_residues", 0)),
        "negative_residues": int(left.get("negative_residues", 0)) + int(right.get("negative_residues", 0)),
    }


def count_label_tensor_support(labels: torch.Tensor) -> Dict[str, int]:
    """
    Count valid, positive, and negative residues using ProteinDataset masking semantics.

    Binary labels mark all residues valid. Three-column labels use columns
    [definite epitope, uncertain, not epitope] and exclude uncertain residues.
    """
    if not torch.is_tensor(labels):
        raise TypeError(f"labels must be a torch.Tensor, not {type(labels)}")
    y = labels.detach().cpu()
    if y.dim() == 2 and y.size(1) == 3:
        def_col = y[:, 0].float()
        unc_col = y[:, 1].float()
        valid_mask = unc_col == 0
        valid = int(valid_mask.sum().item())
        pos = int(((def_col == 1) & valid_mask).sum().item())
        neg = int(valid - pos)
    else:
        y_flat = y.view(-1)
        valid = int(y_flat.numel())
        pos = int((y_flat == 1).sum().item())
        neg = int(valid - pos)
    return {
        "valid_residues": valid,
        "positive_residues": pos,
        "negative_residues": neg,
    }


def build_label_support_by_id(
    ids: Sequence[str],
    label_index: Mapping[str, Path | str],
) -> Dict[str, Dict[str, Any]]:
    """
    Build residue label-support summaries for selected protein IDs.

    Label shards are loaded once per shard path. Missing IDs are retained with
    zero support and a non-OK status so split reports can account for them.
    """
    ids_ordered = [str(protein_id) for protein_id in ids]
    ids_by_shard: Dict[Path, List[str]] = {}
    support_by_id: Dict[str, Dict[str, Any]] = {}

    for protein_id in ids_ordered:
        shard_raw = label_index.get(protein_id)
        if shard_raw is None:
            support_by_id[protein_id] = {
                "protein_id": protein_id,
                "valid_residues": 0,
                "positive_residues": 0,
                "negative_residues": 0,
                "label_shard": None,
                "status": "missing_label",
            }
            continue
        ids_by_shard.setdefault(Path(shard_raw), []).append(protein_id)

    for shard_path in sorted(ids_by_shard.keys(), key=lambda p: str(p)):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "labels" not in payload:
            raise TypeError(
                f"Label shard {shard_path} must be a dict with 'labels' key"
            )
        labels_obj = payload["labels"]
        if not isinstance(labels_obj, dict):
            raise TypeError(f"'labels' in {shard_path} must be a dict")

        for protein_id in ids_by_shard[shard_path]:
            labels = labels_obj.get(protein_id)
            if labels is None:
                support_by_id[protein_id] = {
                    "protein_id": protein_id,
                    "valid_residues": 0,
                    "positive_residues": 0,
                    "negative_residues": 0,
                    "label_shard": str(shard_path),
                    "status": "missing_label_tensor",
                }
                continue
            counts = count_label_tensor_support(labels)
            support_by_id[protein_id] = {
                "protein_id": protein_id,
                "valid_residues": int(counts["valid_residues"]),
                "positive_residues": int(counts["positive_residues"]),
                "negative_residues": int(counts["negative_residues"]),
                "label_shard": str(shard_path),
                "status": "ok",
            }
        del payload

    return support_by_id


def _support_stats_for_ids(
    ids: Sequence[str],
    support_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    stats = _zero_stats()
    stats["protein_count"] = int(len(ids))
    for protein_id in ids:
        support = support_by_id.get(str(protein_id), {})
        stats["valid_residues"] += int(support.get("valid_residues", 0))
        stats["positive_residues"] += int(support.get("positive_residues", 0))
        stats["negative_residues"] += int(support.get("negative_residues", 0))
    return stats


def _build_group_items(
    ids: Sequence[str],
    groups: Mapping[str, str],
    support_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    group_to_ids: Dict[str, List[str]] = {}
    for protein_id in ids:
        group_id = str(groups.get(str(protein_id), str(protein_id)))
        group_to_ids.setdefault(group_id, []).append(str(protein_id))

    items: List[Dict[str, Any]] = []
    for group_id, members in group_to_ids.items():
        stats = _support_stats_for_ids(members, support_by_id)
        items.append({
            "group_id": group_id,
            "ids": list(members),
            "stats": stats,
        })
    return items


def _balance_score(
    stats: Mapping[str, int],
    target: Mapping[str, float],
    overall_rate: float | None,
    total: Mapping[str, int],
) -> float:
    protein_total = max(float(total.get("protein_count", 0)), 1.0)
    valid_total = max(float(total.get("valid_residues", 0)), 1.0)
    pos_total = max(float(total.get("positive_residues", 0)), 1.0)
    neg_total = max(float(total.get("negative_residues", 0)), 1.0)

    protein_err = abs(float(stats.get("protein_count", 0)) - float(target["protein_count"])) / protein_total
    valid_err = abs(float(stats.get("valid_residues", 0)) - float(target["valid_residues"])) / valid_total
    pos_err = abs(float(stats.get("positive_residues", 0)) - float(target["positive_residues"])) / pos_total
    neg_err = abs(float(stats.get("negative_residues", 0)) - float(target["negative_residues"])) / neg_total

    rate_err = 0.0
    valid = int(stats.get("valid_residues", 0))
    if overall_rate is not None and valid > 0:
        rate = float(stats.get("positive_residues", 0) / valid)
        rate_err = abs(rate - overall_rate)

    empty_valid_penalty = (
        0.25 if float(target["valid_residues"]) > 0.0 and valid == 0 else 0.0
    )
    return (
        protein_err
        + (0.5 * valid_err)
        + (2.0 * pos_err)
        + (2.0 * neg_err)
        + (3.0 * rate_err)
        + empty_valid_penalty
    )


def _group_sort_key(
    item: Mapping[str, Any],
    overall_rate: float | None,
) -> Tuple[float, int, int, str]:
    stats = item["stats"]
    valid = int(stats["valid_residues"])
    rate = (
        float(stats["positive_residues"] / valid)
        if valid > 0
        else (overall_rate if overall_rate is not None else 0.0)
    )
    rate_delta = abs(rate - overall_rate) if overall_rate is not None else 0.0
    return (
        -rate_delta,
        -int(stats["valid_residues"]),
        -int(stats["protein_count"]),
        str(item["group_id"]),
    )


def split_ids_label_stratified(
    ids: List[str],
    val_frac: float,
    seed: int,
    groups: Dict[str, str],
    support_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Split IDs while keeping groups intact and balancing residue label support.

    This is an opt-in alternative to `split_ids_grouped`; it uses per-protein
    valid/positive/negative residue counts instead of group size alone.
    """
    ids = [str(protein_id) for protein_id in ids]
    if val_frac < 0.0 or val_frac > 1.0:
        return ids, []
    if val_frac == 0.0:
        return ids, []
    if val_frac == 1.0:
        return [], ids
    if len(ids) == 0:
        return [], []

    group_items = _build_group_items(ids, groups, support_by_id)
    if len(group_items) == 0:
        return [], []

    total = _support_stats_for_ids(ids, support_by_id)
    overall_rate = _positive_rate(
        total["positive_residues"], total["valid_residues"])
    target = {
        "protein_count": float(int(len(ids) * val_frac)),
        "valid_residues": float(total["valid_residues"] * val_frac),
        "positive_residues": float(total["positive_residues"] * val_frac),
        "negative_residues": float(total["negative_residues"] * val_frac),
    }
    if target["protein_count"] <= 0.0:
        return ids, []

    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: _group_sort_key(item, overall_rate))

    val_groups: Set[str] = set()
    val_stats = _zero_stats()
    remaining = list(group_items)

    while remaining:
        current_score = _balance_score(val_stats, target, overall_rate, total)
        candidates: List[Tuple[float, int, int, str, int, Dict[str, Any]]] = []
        for idx, item in enumerate(remaining):
            if len(val_groups) + 1 >= len(group_items):
                continue
            next_stats = _add_stats(val_stats, item["stats"])
            score = _balance_score(next_stats, target, overall_rate, total)
            candidates.append((
                score,
                abs(int(next_stats["protein_count"]) - int(target["protein_count"])),
                -int(next_stats["valid_residues"]),
                str(item["group_id"]),
                idx,
                item,
            ))

        if len(candidates) == 0:
            break

        best_score, _protein_err, _neg_valid, _group_id, idx, best_item = min(
            candidates,
            key=lambda row: row[:5],
        )
        should_take = (
            int(val_stats["protein_count"]) < int(target["protein_count"])
            or best_score < current_score
        )
        if not should_take:
            break

        val_groups.add(str(best_item["group_id"]))
        val_stats = _add_stats(val_stats, best_item["stats"])
        remaining.pop(idx)

    val_ids = [
        protein_id
        for protein_id in ids
        if str(groups.get(protein_id, protein_id)) in val_groups
    ]
    train_ids = [
        protein_id
        for protein_id in ids
        if str(groups.get(protein_id, protein_id)) not in val_groups
    ]
    return train_ids, val_ids


def build_label_stratified_kfold_splits(
    ids: List[str],
    n_folds: int,
    seed: int,
    groups: Dict[str, str],
    support_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Tuple[List[str], List[str]]]:
    """Build K-fold splits with group integrity and residue label-support balance."""
    ids = [str(protein_id) for protein_id in ids]
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(ids) == 0:
        raise ValueError("ids cannot be empty")

    group_items = _build_group_items(ids, groups, support_by_id)
    if len(group_items) < n_folds:
        raise ValueError(
            f"n_folds={n_folds} cannot exceed number of groups={len(group_items)}"
        )

    total = _support_stats_for_ids(ids, support_by_id)
    overall_rate = _positive_rate(
        total["positive_residues"], total["valid_residues"])
    target = {
        "protein_count": float(total["protein_count"] / n_folds),
        "valid_residues": float(total["valid_residues"] / n_folds),
        "positive_residues": float(total["positive_residues"] / n_folds),
        "negative_residues": float(total["negative_residues"] / n_folds),
    }

    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: _group_sort_key(item, overall_rate))

    fold_groups: List[Set[str]] = [set() for _ in range(n_folds)]
    fold_stats: List[Dict[str, int]] = [_zero_stats() for _ in range(n_folds)]

    for item in group_items:
        candidates = []
        for fold_idx in range(n_folds):
            next_stats = list(fold_stats)
            next_stats[fold_idx] = _add_stats(fold_stats[fold_idx], item["stats"])
            score = sum(
                _balance_score(stats, target, overall_rate, total)
                for stats in next_stats
            )
            candidates.append((
                score,
                int(fold_stats[fold_idx]["protein_count"]),
                len(fold_groups[fold_idx]),
                fold_idx,
            ))
        fold_idx = min(candidates, key=lambda row: row)[3]
        fold_groups[fold_idx].add(str(item["group_id"]))
        fold_stats[fold_idx] = _add_stats(fold_stats[fold_idx], item["stats"])

    if any(len(member_groups) == 0 for member_groups in fold_groups):
        raise RuntimeError("Label-stratified grouped K-fold assignment produced an empty fold")

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


def summarize_split_ids(
    ids: Sequence[str],
    support_by_id: Mapping[str, Mapping[str, Any]],
    families_by_id: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Summarize protein and residue support for a train/validation split."""
    ids = [str(protein_id) for protein_id in ids]
    stats = _support_stats_for_ids(ids, support_by_id)
    family_counts: Dict[str, int] = {}
    shard_counts: Dict[str, int] = {}
    families_by_id = families_by_id or {}
    for protein_id in ids:
        family = str(families_by_id.get(protein_id, "__unavailable__"))
        family_counts[family] = family_counts.get(family, 0) + 1
        support = support_by_id.get(protein_id, {})
        shard = support.get("label_shard")
        shard_key = str(shard) if shard is not None else "__missing_label__"
        shard_counts[shard_key] = shard_counts.get(shard_key, 0) + 1

    return {
        "protein_count": int(stats["protein_count"]),
        "valid_residues": int(stats["valid_residues"]),
        "positive_residues": int(stats["positive_residues"]),
        "negative_residues": int(stats["negative_residues"]),
        "positive_rate": _positive_rate(
            stats["positive_residues"], stats["valid_residues"]),
        "family_counts": dict(sorted(family_counts.items())),
        "label_shard_counts": dict(sorted(shard_counts.items())),
        "pathogen_counts": {"__unavailable__": int(stats["protein_count"])},
    }


def build_split_report(
    run_splits: Sequence[Mapping[str, Any]],
    support_by_id: Mapping[str, Mapping[str, Any]],
    families_by_id: Mapping[str, str],
    split_type: str,
    split_strategy: str,
) -> Dict[str, Any]:
    """Build split report JSON payload for training and Optuna split plans."""
    all_ids: List[str] = []
    seen: Set[str] = set()
    entries: List[Dict[str, Any]] = []
    for split in run_splits:
        train_ids = [str(x) for x in split.get("train_ids", [])]
        val_ids = [str(x) for x in split.get("val_ids", [])]
        for protein_id in train_ids + val_ids:
            if protein_id not in seen:
                seen.add(protein_id)
                all_ids.append(protein_id)
        entries.append({
            "run_index": split.get("run_index"),
            "train_mode": split.get("train_mode"),
            "split_seed": split.get("split_seed"),
            "train_seed": split.get("train_seed"),
            "fold_index": split.get("fold_index"),
            "n_folds": split.get("n_folds"),
            "ensemble_set_index": split.get("ensemble_set_index"),
            "train": summarize_split_ids(train_ids, support_by_id, families_by_id),
            "validation": summarize_split_ids(val_ids, support_by_id, families_by_id),
        })

    return {
        "schema_version": 1,
        "split_type": str(split_type),
        "split_strategy": str(split_strategy),
        "pathogen_metadata_status": "unavailable",
        "all_ids": summarize_split_ids(all_ids, support_by_id, families_by_id),
        "runs": entries,
    }


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
