import pytest
from pepseqpred.core.train.split import (
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
