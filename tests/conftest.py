from pathlib import Path
import pytest
import torch


@pytest.fixture
def training_artifacts(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    label_shard = tmp_path / "labels_000.pt"

    labels = {}
    pos = 0
    neg = 0

    for protein_id, family in [
        ("P001", "111"), ("P002", "111"), ("P003", "222"), ("P004", "222")
    ]:
        x = torch.randn(6, 4, dtype=torch.float32)
        torch.save(x, emb_dir / f"{protein_id}-{family}.pt")

        y = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
        labels[protein_id] = y
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    payload = {
        "labels": labels,
        "class_stats": {
            "pos_count": pos,
            "neg_count": neg
        }
    }
    torch.save(payload, label_shard)

    return {"embedding_dir": emb_dir, "label_shard": label_shard}
