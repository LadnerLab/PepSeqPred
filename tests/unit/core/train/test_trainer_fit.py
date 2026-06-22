import logging
from pathlib import Path
import optuna
import pytest
import torch
import torch.nn as nn
from pepseqpred.core.models.ffnn import PepSeqFFNN
import pepseqpred.core.train.trainer as trainer_mod
from pepseqpred.core.train.trainer import (
    Trainer,
    TrainerConfig,
    ValidationCurveArtifactConfig
)

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


def test_make_zero_valid_dummy_batch_supports_masked_and_unmasked_batches():
    trainer = _make_trainer(_make_batches(n_batches=1), epochs=1)
    x, y, mask = _make_batches(n_batches=1)[0]

    masked_dummy = trainer._make_zero_valid_dummy_batch((x, y, mask))
    unmasked_dummy = trainer._make_zero_valid_dummy_batch((x, y))

    for dummy_x, dummy_y, dummy_mask in (masked_dummy, unmasked_dummy):
        assert dummy_x.shape == (1, 1, x.size(-1))
        assert dummy_y.shape == (1, 1)
        assert dummy_mask.shape == (1, 1)
        assert torch.count_nonzero(dummy_x).item() == 0
        assert torch.count_nonzero(dummy_y).item() == 0
        assert torch.count_nonzero(dummy_mask).item() == 0


def test_synchronized_training_batches_yields_dummy_after_local_exhaustion(monkeypatch):
    batches = _make_batches(n_batches=1)
    trainer = _make_trainer(batches, epochs=1)
    reduced_states = iter(((2, 2), (1, 2), (0, 2)))
    local_states = []

    def _reduce_active(tensor: torch.Tensor) -> torch.Tensor:
        local_states.append(tuple(int(x) for x in tensor.tolist()))
        active, world = next(reduced_states)
        return torch.tensor(
            [active, world], device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(trainer_mod, "ddp_all_reduce_sum", _reduce_active)

    synchronized = list(trainer._synchronized_training_batches(batches))

    assert local_states == [(1, 1), (0, 1), (0, 1)]
    assert len(synchronized) == 2
    assert synchronized[0][0] is batches[0]
    assert synchronized[0][1] is True
    dummy_batch, is_real = synchronized[1]
    assert is_real is False
    assert dummy_batch[0].shape == (1, 1, 4)
    assert torch.count_nonzero(dummy_batch[2]).item() == 0


def test_synchronized_training_batches_rejects_initial_empty_rank(monkeypatch):
    trainer = _make_trainer([], epochs=1)

    def _one_active_rank(tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor([1, 2], device=tensor.device, dtype=tensor.dtype)

    monkeypatch.setattr(trainer_mod, "ddp_all_reduce_sum", _one_active_rank)

    with pytest.raises(RuntimeError, match="no initial training batch"):
        list(trainer._synchronized_training_batches([]))


def test_run_epoch_train_reports_single_rank_synchronization():
    batches = _make_batches(n_batches=2)
    trainer = _make_trainer(batches, epochs=1)

    out = trainer._run_epoch(0, train=True)

    assert out["synchronized_steps"] == 2
    assert out["optimizer_steps"] == 2
    assert out["zero_valid_steps"] == 0
    assert out["real_batches"] == 2
    assert out["dummy_batches"] == 0
    assert out["sync_summary"] == {
        "synchronized_steps": 2,
        "optimizer_steps": 2,
        "zero_valid_steps": 0,
        "dummy_batch_fraction": 0.0,
        "per_rank": [{"rank": 0, "real_batches": 2, "dummy_batches": 0}],
    }


def test_fit_with_validation_saves_checkpoint(tmp_path: Path):
    trainer = _make_trainer(
        _make_batches(), _make_batches(), emb_dim=4, epochs=2)
    summary = trainer.fit(save_dir=tmp_path, score_key="loss")

    assert summary["best_epoch"] >= 0
    assert (tmp_path / "fully_connected.pt").exists()
    assert isinstance(summary["best_metrics"], dict)
    metrics = summary["best_metrics"]
    assert metrics["threshold_policy"] == "max-recall-min-precision"
    assert metrics["threshold_min_precision"] == pytest.approx(0.25)
    assert isinstance(metrics["threshold_grid"], list)


def test_fit_without_validation_saves_no_val_checkpoint(tmp_path: Path):
    trainer = _make_trainer(_make_batches(), None, emb_dim=4, epochs=1)
    summary = trainer.fit(save_dir=tmp_path, score_key="loss")

    assert summary["best_epoch"] == -1
    assert (tmp_path / "fully_connected_no_val.pt").exists()


def test_fit_with_validation_curve_artifacts(tmp_path: Path):
    trainer = _make_trainer(
        _make_batches(), _make_batches(), emb_dim=4, epochs=1)
    summary = trainer.fit(
        save_dir=tmp_path,
        score_key="loss",
        val_curve_artifacts=ValidationCurveArtifactConfig(
            max_points=32,
            plot_formats=("png",),
            output_subdir="validation_curves"
        )
    )

    assert summary["best_epoch"] >= 0
    curves_dir = tmp_path / "validation_curves"
    assert (curves_dir / "epoch_0000_curves.json").exists()

    roc_plot = curves_dir / "epoch_0000_roc_auc.png"
    pr_plot = curves_dir / "epoch_0000_pr_auc.png"
    if roc_plot.exists() or pr_plot.exists():
        assert roc_plot.exists()
        assert pr_plot.exists()


def test_run_epoch_eval_no_valid_residues():
    train_loader = _make_batches(n_batches=1, mask_value=1)
    val_loader = _make_batches(n_batches=1, mask_value=0)
    trainer = _make_trainer(train_loader, val_loader, emb_dim=4, epochs=1)

    out = trainer._run_epoch(0, train=False)
    assert out["eval_metrics"]["threshold_status"] == "no_valid_residues"
    assert out["eval_metrics"]["threshold_policy"] == "max-recall-min-precision"


def test_run_epoch_eval_records_configured_threshold_policy():
    train_loader = _make_batches(n_batches=1, mask_value=1)
    val_loader = _make_batches(n_batches=1, mask_value=1)
    trainer = _make_trainer(train_loader, val_loader, emb_dim=4, epochs=1)
    trainer.config.threshold_policy = "fixed"
    trainer.config.threshold_fixed_value = 0.5

    out = trainer._run_epoch(0, train=False)
    metrics = out["eval_metrics"]
    assert metrics["threshold_policy"] == "fixed"
    assert metrics["threshold"] == pytest.approx(0.5)
    assert metrics["threshold_status"] == "ok"


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


def test_run_epoch_eval_uses_wrapped_module_for_ddp_like_models(monkeypatch):
    class _FakeDDP(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, _x):
            raise RuntimeError("DDP wrapper forward should not be used during eval")

    monkeypatch.setattr(trainer_mod, "TorchDDP", _FakeDDP)

    base_model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(8,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False
    )
    wrapped_model = _FakeDDP(base_model)
    trainer = Trainer(
        model=wrapped_model,
        train_loader=_make_batches(n_batches=1),
        val_loader=_make_batches(n_batches=1),
        logger=logging.getLogger("trainer_eval_ddp_like_test"),
        config=TrainerConfig(epochs=1, batch_size=2, learning_rate=1e-2, device="cpu")
    )

    out = trainer._run_epoch(0, train=False)
    assert "eval_metrics" in out
