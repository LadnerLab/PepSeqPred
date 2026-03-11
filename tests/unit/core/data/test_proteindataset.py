from pathlib import Path
import pytest
import torch
from pepseqpred.core.data.proteindataset import ProteinDataset, _iter_windows, pad_collate

pytestmark = pytest.mark.unit


def test_iter_windows_disabled():
    assert list(_iter_windows(
        length=5, window_size=None, stride=1)) == [(0, 5)]


def test_pad_collate_shapes():
    x1 = torch.ones(2, 3)
    y1 = torch.tensor([1.0, 0.0])
    m1 = torch.tensor([1, 1])

    x2 = torch.ones(4, 3)
    y2 = torch.tensor([1.0, 0.0, 1.0, 0.0])
    m2 = torch.tensor([1, 1, 1, 0])

    x, y, m = pad_collate([(x1, y1, m1), (x2, y2, m2)])
    assert x.shape == (2, 4, 3)
    assert y.shape == (2, 4)
    assert m.shape == (2, 4)


def test_dataset_masks_uncertain_and_padding(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    shard = tmp_path / "labels.pt"

    torch.save(torch.randn(5, 4), emb_dir / "P001-111.pt")
    labels = {
        "P001": torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
            dtype=torch.uint8
        )
    }
    torch.save({"labels": labels, "class_stats": {
               "pos_count": 2, "neg_counts": 2}}, shard)

    ds = ProteinDataset(
        embedding_dirs=[emb_dir],
        label_shards=[shard],
        window_size=4,
        stride=4,
        collapse_labels=True,
        pad_last_window=True
    )
    samples = list(ds)
    assert len(samples) == 2
    _, _, m0 = samples[0]
    _, _, m1 = samples[1]
    assert m0.tolist() == [1, 0, 1, 1]
    assert m1.tolist() == [1, 0, 0, 0]
