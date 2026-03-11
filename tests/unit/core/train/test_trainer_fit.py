import logging
from pathlib import Path
import optuna
import pytest
import torch
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.train.trainer import Trainer, TrainerConfig

pytestmark = pytest.mark.unit


def _make_batches(
    n_batches: int = 2,
    batch_size: int = 2,
    length: int = 6,
    emb_dim: int = 4,
    mask_value: int = 1
):
    batches = []
    base_y = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32).repeat(
        batch_size, 1
    )
    for _ in range(n_batches):
        x = torch.randn(batch_size, length, emb_dim)
        y = base_y.clone()
        m = torch.full((batch_size, length), mask_value, dtype=torch.long)
        batches.append((x, y, m))
    return batches


def _make_trainer(train_loader, val_loader=None, emb_dim: int = 4, epochs: int = 2):
    model = PepSeqFFNN(
        emb_dim=emb_dim,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False
    )
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logging.getLogger("trainer_fit_test"),
        config=TrainerConfig(
            epochs=epochs, batch_size=2, learning_rate=1e-2, device="cpu"
        )
    )


def test_fit_with_validation_saves_checkpoint(tmp_path: Path):
    trainer = _make_trainer(
        _make_batches(), _make_batches(), emb_dim=4, epochs=2)
    summary = trainer.fit(save_dir=tmp_path, score_key="loss")

    assert summary["best_epoch"] >= 0
    assert (tmp_path / "fully_connected.pt").exists()
    assert isinstance(summary["best_metrics"], dict)


def test_fit_without_validation_saves_no_val_checkpoint(tmp_path: Path):
    trainer = _make_trainer(_make_batches(), None, emb_dim=4, epochs=1)
    summary = trainer.fit(save_dir=tmp_path, score_key="loss")

    assert summary["best_epoch"] == -1
    assert (tmp_path / "fully_connected_no_val.pt").exists()


def test_run_epoch_eval_no_valid_residues():
    train_loader = _make_batches(n_batches=1, mask_value=1)
    val_loader = _make_batches(n_batches=1, mask_value=0)
    trainer = _make_trainer(train_loader, val_loader, emb_dim=4, epochs=1)

    out = trainer._run_epoch(0, train=False)
    assert out["eval_metrics"]["threshold_status"] == "no_valid_residues"


class _AlwaysPruneTrial:
    def report(self, value, step):
        _ = (value, step)

    def should_prune(self):
        return True


def test_fit_optuna_prune_path(tmp_path: Path):
    trainer = _make_trainer(
        _make_batches(), _make_batches(), emb_dim=4, epochs=2)
    with pytest.raises(optuna.TrialPruned):
        trainer.fit_optuna(save_dir=tmp_path,
                           trial=_AlwaysPruneTrial(), score_key="f1")
