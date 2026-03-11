import logging
import pytest
import torch
import torch.nn as nn
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


def _make_trainer(model: nn.Module) -> Trainer:
    return Trainer(
        model=model,
        train_loader=[],
        logger=logging.getLogger("test_trainer"),
        val_loader=None,
        config=TrainerConfig(epochs=1, batch_size=2, device="cpu")
    )


def test_batch_step_zero_mask_returns_zero_n():
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

    out = trainer._batch_step((x, y, mask), train=True)
    assert out["n"] == 0
    assert out["loss"] == pytest.approx(0.0, abs=1e-12)


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
