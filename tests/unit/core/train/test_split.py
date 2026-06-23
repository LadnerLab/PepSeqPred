import pytest
import torch
from pepseqpred.core.train.split import (
    build_label_support_by_id,
    build_label_stratified_kfold_splits,
    count_label_tensor_support,
    split_ids_label_stratified,
    split_ids_grouped,
    partition_ids_weighted,
    build_kfold_splits,
    build_grouped_kfold_splits,
)

pytestmark = pytest.mark.unit


def test_split_ids_grouped_keeps_groups_intact():
    ids = ["a1", "a2", "b1", "b2", "c1"]
    groups = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"}
    train_ids, val_ids = split_ids_grouped(
        ids, val_frac=0.4, seed=7, groups=groups)

    train_groups = {groups[i] for i in train_ids}
    val_groups = {groups[i] for i in val_ids}
    assert train_groups.isdisjoint(val_groups)


def test_label_support_counts_binary_tristate_and_missing(tmp_path):
    shard_path = tmp_path / "labels.pt"
    torch.save(
        {
            "labels": {
                "binary": torch.tensor([1, 0, 0, 1], dtype=torch.float32),
                "tristate": torch.tensor(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=torch.uint8,
                ),
                "all_uncertain": torch.tensor(
                    [[0, 1, 0], [0, 1, 0]],
                    dtype=torch.uint8,
                ),
            }
        },
        shard_path,
    )

    assert count_label_tensor_support(
        torch.tensor([1, 0, 0], dtype=torch.float32)
    ) == {
        "valid_residues": 3,
        "positive_residues": 1,
        "negative_residues": 2,
    }

    support = build_label_support_by_id(
        ["binary", "tristate", "all_uncertain", "missing"],
        {
            "binary": shard_path,
            "tristate": shard_path,
            "all_uncertain": shard_path,
        },
    )

    assert support["binary"]["status"] == "ok"
    assert support["binary"]["valid_residues"] == 4
    assert support["binary"]["positive_residues"] == 2
    assert support["binary"]["negative_residues"] == 2
    assert support["tristate"]["valid_residues"] == 2
    assert support["tristate"]["positive_residues"] == 1
    assert support["tristate"]["negative_residues"] == 1
    assert support["all_uncertain"]["valid_residues"] == 0
    assert support["missing"]["status"] == "missing_label"


def test_label_stratified_holdout_keeps_groups_and_balances_positive_rate():
    ids = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]
    groups = {protein_id: protein_id[0].upper() for protein_id in ids}
    support = {
        protein_id: {
            "valid_residues": 10,
            "positive_residues": 10 if groups[protein_id] in {"A", "C"} else 0,
            "negative_residues": 0 if groups[protein_id] in {"A", "C"} else 10,
            "label_shard": "labels.pt",
            "status": "ok",
        }
        for protein_id in ids
    }

    size_train, size_val = split_ids_grouped(
        ids, val_frac=0.5, seed=0, groups=groups)
    strat_train, strat_val = split_ids_label_stratified(
        ids, val_frac=0.5, seed=0, groups=groups, support_by_id=support)
    strat_train_2, strat_val_2 = split_ids_label_stratified(
        ids, val_frac=0.5, seed=0, groups=groups, support_by_id=support)

    def _pos_rate(split_ids):
        pos = sum(support[protein_id]["positive_residues"] for protein_id in split_ids)
        valid = sum(support[protein_id]["valid_residues"] for protein_id in split_ids)
        return pos / valid

    assert (strat_train, strat_val) == (strat_train_2, strat_val_2)
    assert {groups[i] for i in strat_train}.isdisjoint({groups[i] for i in strat_val})
    assert abs(_pos_rate(strat_val) - 0.5) < abs(_pos_rate(size_val) - 0.5)
    assert sorted(strat_train + strat_val) == sorted(ids)


def test_label_stratified_kfold_keeps_groups_and_is_deterministic():
    ids = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]
    groups = {protein_id: protein_id[0].upper() for protein_id in ids}
    support = {
        protein_id: {
            "valid_residues": 10,
            "positive_residues": 10 if groups[protein_id] in {"A", "C"} else 0,
            "negative_residues": 0 if groups[protein_id] in {"A", "C"} else 10,
            "label_shard": "labels.pt",
            "status": "ok",
        }
        for protein_id in ids
    }

    splits = build_label_stratified_kfold_splits(
        ids, n_folds=2, seed=5, groups=groups, support_by_id=support)
    splits_2 = build_label_stratified_kfold_splits(
        ids, n_folds=2, seed=5, groups=groups, support_by_id=support)

    assert splits == splits_2
    assert len(splits) == 2
    for train_ids, val_ids in splits:
        assert {groups[i] for i in train_ids}.isdisjoint({groups[i] for i in val_ids})
        assert sorted(train_ids + val_ids) == sorted(ids)
        pos = sum(support[protein_id]["positive_residues"] for protein_id in val_ids)
        valid = sum(support[protein_id]["valid_residues"] for protein_id in val_ids)
        assert pos / valid == pytest.approx(0.5)


def test_label_stratified_split_edge_cases():
    ids = ["a1", "b1"]
    groups = {"a1": "A", "b1": "B"}
    support = {
        protein_id: {
            "valid_residues": 0,
            "positive_residues": 0,
            "negative_residues": 0,
            "label_shard": "labels.pt",
            "status": "ok",
        }
        for protein_id in ids
    }

    assert split_ids_label_stratified(
        ids, val_frac=0.0, seed=1, groups=groups, support_by_id=support
    ) == (ids, [])
    assert split_ids_label_stratified(
        ids, val_frac=1.0, seed=1, groups=groups, support_by_id=support
    ) == ([], ids)
    with pytest.raises(ValueError, match="cannot exceed number of groups"):
        build_label_stratified_kfold_splits(
            ids, n_folds=3, seed=1, groups=groups, support_by_id=support)


def test_partition_ids_weighted_non_empty():
    ids = ["p1", "p2", "p3", "p4"]
    weights = {"p1": 100.0, "p2": 90.0, "p3": 10.0, "p4": 9.0}
    parts = partition_ids_weighted(
        ids, world_size=2, weights=weights, ensure_non_empty=True)
    assert len(parts) == 2
    assert all(len(p) > 0 for p in parts)
    assert sorted([x for part in parts for x in part]) == sorted(ids)


def test_build_kfold_splits_uses_all_ids_once_per_fold():
    ids = ["p1", "p2", "p3", "p4", "p5"]
    splits = build_kfold_splits(ids, n_folds=5, seed=11)

    assert len(splits) == 5
    for train_ids, val_ids in splits:
        assert len(val_ids) == 1
        assert len(train_ids) == 4
        assert set(train_ids).isdisjoint(set(val_ids))
        assert sorted(train_ids + val_ids) == sorted(ids)


def test_build_grouped_kfold_splits_keeps_groups_intact():
    ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    groups = {
        "a1": "A",
        "a2": "A",
        "b1": "B",
        "b2": "B",
        "c1": "C",
        "c2": "C",
    }
    splits = build_grouped_kfold_splits(ids, n_folds=3, seed=17, groups=groups)
    assert len(splits) == 3

    for train_ids, val_ids in splits:
        train_groups = {groups[i] for i in train_ids}
        val_groups = {groups[i] for i in val_ids}
        assert train_groups.isdisjoint(val_groups)
        assert sorted(train_ids + val_ids) == sorted(ids)
