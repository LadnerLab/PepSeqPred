from pathlib import Path
import shutil
from contextlib import contextmanager
from uuid import uuid4
import pytest
import torch
import pepseqpred.core.train.weights as weights_mod
from pepseqpred.core.train.weights import (
    compute_pos_neg_counts,
    global_pos_weight,
    pos_weight_from_label_shards
)

pytestmark = pytest.mark.unit


def _write_label_shard(path: Path, class_stats: dict) -> None:
    torch.save({"labels": {}, "class_stats": class_stats}, path)


@contextmanager
def _scratch_dir():
    root = Path("localdata") / "tmp_pytest_weights"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_pos_weight_from_label_shards_accepts_neg_count():
    with _scratch_dir() as tmp_path:
        shard = tmp_path / "labels.pt"
        _write_label_shard(
            shard, {"pos_count": 2, "neg_count": 6}
        )
        assert pos_weight_from_label_shards([shard]) == pytest.approx(3.0)


def test_pos_weight_from_label_shards_accepts_neg_counts():
    with _scratch_dir() as tmp_path:
        shard = tmp_path / "labels.pt"
        _write_label_shard(
            shard, {"pos_count": 2, "neg_counts": 8}
        )
        assert pos_weight_from_label_shards([shard]) == pytest.approx(4.0)


def test_compute_pos_neg_counts_with_and_without_masks():
    loader = [
        (
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
        ),
        (
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
        )
    ]

    pos, neg = compute_pos_neg_counts(loader)
    assert pos == 3
    assert neg == 3


def test_global_pos_weight_without_ddp_uses_safe_denominator():
    assert global_pos_weight(local_pos=0, local_neg=9, ddp=None) == pytest.approx(9.0)


def test_global_pos_weight_with_ddp_all_reduce(monkeypatch):
    original_tensor = torch.tensor

    def _cpu_tensor(data, device=None, **kwargs):
        _ = device
        return original_tensor(data, device=torch.device("cpu"), **kwargs)

    def _all_reduce(t, op):
        assert op == weights_mod.dist.ReduceOp.SUM
        t[0] += 3
        t[1] += 7

    monkeypatch.setattr(weights_mod.torch, "tensor", _cpu_tensor)
    monkeypatch.setattr(weights_mod.dist, "all_reduce", _all_reduce)

    out = global_pos_weight(local_pos=2, local_neg=3, ddp={"rank": 0})
    assert out == pytest.approx(2.0)


def test_pos_weight_from_label_shards_aggregates_multiple_shards():
    with _scratch_dir() as tmp_path:
        shard_a = tmp_path / "labels_a.pt"
        shard_b = tmp_path / "labels_b.pt"
        _write_label_shard(
            shard_a, {"pos_count": 2, "neg_count": 6}
        )
        _write_label_shard(
            shard_b, {"pos_count": 1, "neg_counts": 5}
        )
        assert pos_weight_from_label_shards([shard_a, shard_b]) == pytest.approx(11 / 3)


def test_pos_weight_from_label_shards_handles_zero_total_pos():
    with _scratch_dir() as tmp_path:
        shard = tmp_path / "labels.pt"
        _write_label_shard(
            shard, {"pos_count": 0, "neg_count": 7}
        )
        assert pos_weight_from_label_shards([shard]) == pytest.approx(7.0)


def test_pos_weight_from_label_shards_missing_class_stats_raises():
    with _scratch_dir() as tmp_path:
        shard = tmp_path / "labels.pt"
        torch.save({"labels": {}}, shard)
        with pytest.raises(ValueError, match="missing class_stats"):
            pos_weight_from_label_shards([shard])


def test_pos_weight_from_label_shards_missing_negative_key_raises():
    with _scratch_dir() as tmp_path:
        shard = tmp_path / "labels.pt"
        _write_label_shard(
            shard, {"pos_count": 2}
        )
        with pytest.raises(ValueError, match="missing negative count key"):
            pos_weight_from_label_shards([shard])
