import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda" # should only train using GPUs (but can be changed to "cpu")

class Trainer:
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, 
                 logger: logging.Logger, 
                 val_loader: Optional[DataLoader] = None, 
                 config: TrainerConfig = TrainerConfig(), 
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # default to Adam optimizer (may change later)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config.learning_rate, 
                                          weight_decay=self.config.weight_decay)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.class_weights = class_weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # inital for reproduceability
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("trainer_init", 
                         extra={"extra": {
                             "device": self.device, 
                             "epochs": self.config.epochs, 
                             "batch_size": self.config.batch_size, 
                             "learning_rate": self.config.learning_rate, 
                             "weight_decay": self.config.weight_decay, 
                             "num_params": num_params, 
                             "has_val_loader": self.val_loader is not None, 
                             "has_class_weights": self.class_weights is not None
                         }})

    def _batch_step(self, batch: torch.Tensor, train: bool = True) -> Dict[str, Any]:
        X, y_onehot = batch # (B, L, E), (B, 3)
        X = X.to(self.device)
        y_onehot = y_onehot.to(self.device)

        # convert one-hot to class indices {0, 1, 2} for loss calculation
        y = y_onehot.argmax(dim=-1)

        logits = self.model(X)
        loss = self.criterion(logits, y)

        # simple accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        acc = correct / total

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.debug("batch_step_complete", 
                          extra={"extra": {
                              "mode": "train" if train else "val", 
                              "batch_size": total, 
                              "batch_loss": loss.item(), 
                              "batch_acc": acc
                          }})

        return {"loss": loss.item(), 
                "correct": correct, 
                "n": total}

    def _run_epoch(self, epoch: int, train: bool = True) -> Dict[str, float]:
        loader = self.train_loader if train else self.val_loader
        if loader is None:
            return {"loss": float("nan"), "acc": float("nan")}

        self.logger.info("epoch_start", 
                         extra={"extra": {
                             "epoch": epoch, 
                             "mode": "train" if train else "val", 
                             "num_batches": len(loader)
                         }})

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            with torch.set_grad_enabled(train):
                out = self._batch_step(batch, train=train)

            total_loss += out["loss"] * out["n"]
            total_correct += out["correct"]
            total_samples += out["n"]

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        self.logger.info("epoch_complete", 
                         extra={"extra": {
                             "epoch": epoch, 
                             "mode": "train" if train else "val", 
                             "avg_loss": avg_loss, 
                             "avg_acc": avg_acc, 
                             "total_samples": total_samples
                         }})

        return {"loss": avg_loss, "acc": avg_acc}

    def fit(self, save_dir: Optional[Path | str] = None) -> None:
        best_val_loss = float("inf")
        save_dir = Path(save_dir) if save_dir is not None else None

        self.logger.info("training_started", 
                         extra={"extra": {
                             "epochs": self.config.epochs, 
                             "save_dir": str(save_dir) if save_dir is not None else None
                         }})

        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(epoch, train=True)

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._run_epoch(epoch, train=False)

            if save_dir is not None and (self.val_loader is not None or val_metrics):
                metric_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
                if metric_loss < best_val_loss:
                    best_val_loss = metric_loss
                    self._save_checkpoint(save_dir / "best_model.pt", epoch, best_val_loss)

        self.logger.info("training_done", 
                         extra={"extra": {
                             "best_val_loss": best_val_loss
                         }})

    def _save_checkpoint(self, path: Path | str, epoch: int, loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"model_state_dict": self.model.state_dict(), 
                 "optim_state_dict": self.optimizer.state_dict(), 
                 "epoch": epoch, 
                 "config": self.config.__dict__, 
                 "best_loss": loss}
        torch.save(state, path)

        self.logger.info("checkpoint_saved", 
                         extra={"extra": {
                             "checkpoint_path": str(path), 
                             "epoch": epoch, 
                             "loss": loss
                         }})
