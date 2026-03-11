import pytest
from pepseqpred.core.train.split import split_ids_grouped, partition_ids_weighted

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
