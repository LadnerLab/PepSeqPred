from pathlib import Path
import pytest
import torch
from pepseqpred.core.train.weights import pos_weight_from_label_shards

pytestmark = pytest.mark.unit


def _write_label_shard(path: Path, class_stats: dict) -> None:
    torch.save({"labels": {}, "class_stats": class_stats}, path)


def test_pos_weight_from_label_shards_accepts_neg_count(tmp_path: Path):
    shard = tmp_path / "labels.pt"
    _write_label_shard(
        shard, {"pos_count": 2, "neg_count": 6}
    )
    assert pos_weight_from_label_shards([shard]) == pytest.approx(3.0)


def test_pos_weight_from_label_shards_accepts_neg_counts(tmp_path: Path):
    shard = tmp_path / "labels.pt"
    _write_label_shard(
        shard, {"pos_count": 2, "neg_counts": 8}
    )
    assert pos_weight_from_label_shards([shard]) == pytest.approx(4.0)
