import logging
import pytest
import torch
import torch.nn as nn
import pepseqpred.core.train.trainer as trainer_mod
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.train.trainer import Trainer, TrainerConfig

pytestmark = pytest.mark.unit


class BadShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Intentionally wrong shape (B, L+1) instead of (B, L)
        return torch.zeros((x.size(0), x.size(1) + 1), device=x.device) + (self.p * 0.0)


class ConstantLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.size(0), x.size(1)), device=x.device) + self.p


def _make_trainer(model: nn.Module) -> Trainer:
    return Trainer(
        model=model,
        train_loader=[],
        logger=logging.getLogger("test_trainer"),
        val_loader=None,
        config=TrainerConfig(epochs=1, batch_size=2, device="cpu")
    )


def test_batch_step_zero_mask_returns_zero_n(monkeypatch):
    model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False
    )
    trainer = _make_trainer(model)

    x = torch.randn(1, 3, 4)
    y = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    mask = torch.zeros((1, 3), dtype=torch.long)
    step_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    monkeypatch.setattr(trainer.optimizer, "step", _step)

    out = trainer._batch_step((x, y, mask), train=True)
    assert out["n"] == 0
    assert out["loss"] == pytest.approx(0.0, abs=1e-12)
    assert out["global_valid_count"] == 0
    assert out["optimizer_step"] is False
    assert step_calls == 0
    assert all(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() == 0
        for parameter in model.parameters()
    )


def test_batch_step_zero_local_mask_steps_when_global_valid(monkeypatch):
    model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False
    )
    trainer = _make_trainer(model)

    x = torch.randn(1, 3, 4)
    y = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    mask = torch.zeros((1, 3), dtype=torch.long)
    step_calls = 0

    def _step():
        nonlocal step_calls
        step_calls += 1

    def _global_valid(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[0] = 1
        out[1] = 2
        return out

    monkeypatch.setattr(trainer.optimizer, "step", _step)
    monkeypatch.setattr(trainer_mod, "ddp_all_reduce_sum", _global_valid)

    out = trainer._batch_step((x, y, mask), train=True)
    assert out["n"] == 0
    assert out["loss"] == pytest.approx(0.0, abs=1e-12)
    assert out["global_valid_count"] == 1
    assert out["optimizer_step"] is True
    assert step_calls == 1


def test_batch_step_scales_training_loss_by_global_valid_count(monkeypatch):
    model = ConstantLogitModel()
    trainer = _make_trainer(model)

    x = torch.randn(1, 2, 4)
    y = torch.zeros((1, 2), dtype=torch.float32)
    mask = torch.tensor([[1, 0]], dtype=torch.long)

    def _step():
        return None

    def _global_stats(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[0] = 4
        out[1] = 2
        return out

    monkeypatch.setattr(trainer.optimizer, "step", _step)
    monkeypatch.setattr(trainer_mod, "ddp_all_reduce_sum", _global_stats)

    out = trainer._batch_step((x, y, mask), train=True)
    assert out["n"] == 1
    assert out["global_valid_count"] == 4
    assert out["optimizer_step"] is True
    assert model.p.grad.item() == pytest.approx(0.25, abs=1e-12)


def test_batch_step_rejects_invalid_y_dim():
    model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False
    )
    trainer = _make_trainer(model)

    x = torch.randn(1, 3, 4)
    y_bad = torch.randn(1, 3, 1)

    with pytest.raises(ValueError, match="Expected y_onehot shape"):
        trainer._batch_step((x, y_bad), train=False)


def test_batch_step_rejects_logit_shape_mismatch():
    trainer = _make_trainer(BadShapeModel())

    x = torch.randn(2, 4, 4)
    y = torch.zeros((2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="Expected logits shape"):
        trainer._batch_step((x, y), train=False)
