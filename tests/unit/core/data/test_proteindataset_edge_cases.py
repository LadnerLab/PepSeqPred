from pathlib import Path
import pytest
import torch
from pepseqpred.core.data.proteindataset import (
    ProteinDataset,
    _build_embedding_index,
    _build_label_index,
    _slice_ids_contiguous
)

pytestmark = pytest.mark.unit


def _write_embedding(path: Path, length: int = 5, dim: int = 4) -> None:
    torch.save(torch.randn(length, dim, dtype=torch.float32), path)


def _write_label_shard(path: Path, labels: dict[str, torch.Tensor]) -> None:
    pos = sum(int((y == 1).sum().item()) for y in labels.values())
    neg = sum(int((y == 0).sum().item()) for y in labels.values())
    torch.save(
        {
            "labels": labels,
            "class_stats": {"pos_count": pos, "neg_count": neg}
        },
        path
    )


def test_slice_ids_contiguous_balanced_chunks():
    ids = ["a", "b", "c", "d", "e", "f", "g"]
    assert _slice_ids_contiguous(ids, worker_id=0, num_workers=3) == [
        "a", "b", "c"]
    assert _slice_ids_contiguous(ids, worker_id=1, num_workers=3) == ["d", "e"]
    assert _slice_ids_contiguous(ids, worker_id=2, num_workers=3) == ["f", "g"]


def test_build_label_index_requires_labels_key(tmp_path: Path):
    bad_shard = tmp_path / "bad.pt"
    torch.save({"not_labels": {}}, bad_shard)

    with pytest.raises(TypeError, match="'labels' key"):
        _build_label_index([bad_shard])


def test_build_embedding_index_detects_duplicate_ids(tmp_path: Path):
    d1 = tmp_path / "emb1"
    d2 = tmp_path / "emb2"
    d1.mkdir()
    d2.mkdir()

    _write_embedding(d1 / "P001-111.pt")
    _write_embedding(d2 / "P001-111.pt")

    with pytest.raises(ValueError, match="Duplicate embedding"):
        _build_embedding_index([d1, d2])


def test_dataset_rejects_invalid_label_cache_mode():
    with pytest.raises(ValueError, match="label_cache_mode"):
        ProteinDataset(
            embedding_dirs=[],
            label_shards=[],
            embedding_index={},
            label_index={},
            label_cache_mode="invalid"  # type: ignore[arg-type]
        )


def test_dataset_raises_on_embedding_label_length_mismatch(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    shard = tmp_path / "labels.pt"

    _write_embedding(emb_dir / "P001-111.pt", length=5, dim=4)
    _write_label_shard(shard, {"P001": torch.tensor(
        [1, 0, 1, 0], dtype=torch.float32)})

    ds = ProteinDataset(
        embedding_dirs=[emb_dir],
        label_shards=[shard],
        window_size=None,
        stride=1
    )

    with pytest.raises(ValueError, match="same size at dim 0"):
        list(ds)


def test_label_cache_mode_all_matches_current_outputs(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    shard1 = tmp_path / "labels_1.pt"
    shard2 = tmp_path / "labels_2.pt"

    _write_embedding(emb_dir / "P001-111.pt", length=5, dim=4)
    _write_embedding(emb_dir / "P002-222.pt", length=4, dim=4)

    _write_label_shard(shard1, {"P001": torch.tensor(
        [1, 0, 1, 0, 0], dtype=torch.float32)})
    _write_label_shard(shard2, {"P002": torch.tensor(
        [0, 1, 0, 1], dtype=torch.float32)})

    ds_current = ProteinDataset(
        embedding_dirs=[emb_dir],
        label_shards=[shard1, shard2],
        window_size=None,
        stride=1,
        label_cache_mode="current"
    )
    ds_all = ProteinDataset(
        embedding_dirs=[emb_dir],
        label_shards=[shard1, shard2],
        window_size=None,
        stride=1,
        label_cache_mode="all"
    )

    out_current = list(ds_current)
    out_all = list(ds_all)

    assert len(out_current) == len(out_all) == 2
    for (x1, y1, m1), (x2, y2, m2) in zip(out_current, out_all):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
        assert torch.equal(m1, m2)
