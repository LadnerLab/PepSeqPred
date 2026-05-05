import math
import random
import warnings
from datetime import timedelta
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


def test_init_ddp_no_rank_returns_none(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    assert ddp_mod.init_ddp() is None


def test_init_ddp_invalid_timeout_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("PEPSEQPRED_DDP_TIMEOUT_MIN", "not-an-int")

    calls = {}

    def _init_process_group(backend, timeout):
        calls["backend"] = backend
        calls["timeout"] = timeout

    monkeypatch.setattr(ddp_mod.dist, "init_process_group", _init_process_group)
    monkeypatch.setattr(ddp_mod.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(ddp_mod.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(ddp_mod.torch.cuda, "set_device", lambda _idx: None)

    out = ddp_mod.init_ddp()
    assert out == {"rank": 1, "world_size": 2, "local_rank": 0}
    assert calls["backend"] == "nccl"
    assert calls["timeout"] == timedelta(minutes=60)


def test_ddp_rank_world_and_all_reduce_enabled(monkeypatch):
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(ddp_mod.dist, "get_world_size", lambda: 5)

    def _all_reduce(t, op):
        assert op == ddp_mod.dist.ReduceOp.SUM
        t += 10

    monkeypatch.setattr(ddp_mod.dist, "all_reduce", _all_reduce)

    assert ddp_mod.ddp_rank() == 3
    assert ddp_mod._ddp_world() == 5

    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = ddp_mod.ddp_all_reduce_sum(t)
    assert torch.equal(out, torch.tensor([11.0, 12.0]))


def test_ddp_gather_all_1d_enabled(monkeypatch):
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "get_world_size", lambda: 3)

    def _all_gather(out_list, in_tensor):
        if in_tensor.dtype == torch.long and in_tensor.numel() == 1:
            for dst, value in zip(out_list, [2, 4, 1]):
                dst.fill_(value)
            return
        payloads = [
            torch.tensor([9, 8, 0, 0], dtype=in_tensor.dtype),
            torch.tensor([7, 6, 5, 4], dtype=in_tensor.dtype),
            torch.tensor([3, 0, 0, 0], dtype=in_tensor.dtype)
        ]
        for dst, src in zip(out_list, payloads):
            dst.copy_(src)

    monkeypatch.setattr(ddp_mod.dist, "all_gather", _all_gather)

    gathered, sizes = ddp_mod.ddp_gather_all_1d(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.device("cpu")
    )

    assert sizes == [2, 4, 1]
    assert len(gathered) == 3
    assert gathered[0].tolist() == [9, 8, 0, 0]
    assert gathered[1].tolist() == [7, 6, 5, 4]
    assert gathered[2].tolist() == [3, 0, 0, 0]


def test_ddp_gather_all_1d_rejects_invalid_gathered_sizes(monkeypatch):
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "get_world_size", lambda: 2)

    def _all_gather(out_list, in_tensor):
        if in_tensor.dtype == torch.long and in_tensor.numel() == 1:
            out_list[0].fill_(2)
            out_list[1].fill_(10**12)
            return
        raise AssertionError("Payload gather should not execute after size validation failure")

    monkeypatch.setattr(ddp_mod.dist, "all_gather", _all_gather)

    with pytest.raises(RuntimeError, match="Invalid gathered tensor sizes"):
        ddp_mod.ddp_gather_all_1d(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.device("cpu")
        )
