import math
import random
import warnings
from pathlib import Path
import numpy as np
import pytest
import torch
import pepseqpred.core.train.ddp as ddp_mod
from pepseqpred.core.train.embedding import infer_emb_dim
from pepseqpred.core.train.metrics import compute_eval_metrics
from pepseqpred.core.train.seed import set_all_seeds

pytestmark = pytest.mark.unit


def test_compute_eval_metrics_happy_and_single_class():
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1])
    y_prob = torch.tensor([0.1, 0.9, 0.7, 0.8], dtype=torch.float32)

    out = compute_eval_metrics(y_true, y_pred, y_prob)
    assert {"precision", "recall", "f1", "mcc", "auc", "pr_auc", "auc10"}.issubset(
        out
    )

    y_true_2 = torch.tensor([1, 1, 1, 1])
    y_pred_2 = torch.tensor([1, 1, 1, 1])
    y_prob_2 = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        out_2 = compute_eval_metrics(y_true_2, y_pred_2, y_prob_2)
    assert not caught
    assert math.isnan(out_2["auc"])
    assert out_2["pr_auc"] == pytest.approx(1.0)
    assert math.isnan(out_2["auc10"])


def test_set_all_seeds_reproducible():
    set_all_seeds(123)
    a = torch.rand(3)
    b = np.random.rand(3)
    c = random.random()

    set_all_seeds(123)
    a_2 = torch.rand(3)
    b_2 = np.random.rand(3)
    c_2 = random.random()

    assert torch.allclose(a, a_2)
    assert np.allclose(b, b_2)
    assert c == c_2


def test_infer_emb_dim_success_and_errors(tmp_path: Path):
    emb = tmp_path / "x.pt"
    torch.save(torch.randn(5, 7), emb)
    assert infer_emb_dim({"P1": emb}) == 7

    with pytest.raises(ValueError, match="No embedding found"):
        infer_emb_dim({})

    bad = tmp_path / "bad.pt"
    torch.save(torch.randn(5), bad)
    with pytest.raises(ValueError, match="Expected embedding tensor"):
        infer_emb_dim({"P2": bad})


def test_ddp_helpers_no_ddp(monkeypatch):
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: False)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: False)

    t = torch.tensor([1.0, 2.0])
    out = ddp_mod.ddp_all_reduce_sum(t.clone())
    assert torch.equal(out, t)

    gathered, sizes = ddp_mod.ddp_gather_all_1d(
        torch.tensor([1, 2, 3]), torch.device("cpu")
    )
    assert sizes == [3]
    assert torch.equal(gathered[0], torch.tensor([1, 2, 3]))


def test_init_ddp_enabled(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("PEPSEQPRED_DDP_TIMEOUT_MIN", "5")

    calls = {}

    monkeypatch.setattr(
        ddp_mod.dist,
        "init_process_group",
        lambda backend, timeout: calls.update({"backend": backend})
    )
    monkeypatch.setattr(ddp_mod.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(ddp_mod.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        ddp_mod.torch.cuda,
        "set_device",
        lambda idx: calls.update({"local_rank": idx})
    )

    out = ddp_mod.init_ddp()
    assert out == {"rank": 2, "world_size": 4, "local_rank": 1}
    assert calls["backend"] == "nccl"
    assert calls["local_rank"] == 1
