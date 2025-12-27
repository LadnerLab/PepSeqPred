import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, 
                             matthews_corrcoef, 
                             roc_auc_score, 
                             roc_curve, 
                             auc)
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

def count_classes(loader: DataLoader, num_classes: int = 3) -> List[int]:
    """
    Counts total number of peptides per class.

    Parameters
    ----------
        loader : DataLoader
            The DataLoader containing one-hot encoded targets to sum.
        num_classes : int
            Number of classes to sum. Default is 3.

    Returns
    -------
        counts : List[int]
            List containing the sum of each class.
    """
    counts = [0] * num_classes
    for _, y_onehot in loader:
        y = y_onehot.argmax(dim=-1)
        for c in range(num_classes):
            counts[c] += int((y == c).sum().item())
    return counts

def compute_eval_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor) -> Dict[str, Any]:
    """
    Computes evaluation metrics given true lables, predicted labels, and predicted probabilities.

    Parameters
    ----------
        y_true : Tensor
            True class labels.
        y_pred : Tensor
            Predicted class labels.
        y_prob : Tensor
            Predicted class probabilities.

    Returns
    -------
        metrics : Dict[str, Any]
            Dictionary of evaluation metrics.
    """
    metrics: Dict[str, Any] = {}

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_precision"] = float(precision)
    metrics["macro_recall"] = float(recall)
    metrics["macro_f1"] = float(f1)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # ROC AUC (one-vs-rest)
    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        auc_per_class = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None)
        metrics["auc_macro"] = auc_macro
        metrics["auc_per_class"] = [float(x) for x in auc_per_class]

    except Exception:
        metrics["auc_macro"] = float("nan")
        metrics["auc_per_class"] = [float("nan")] * 3

    # AUC10 calculation
    def auc10_binary(y_true_bin, y_score) -> float:
        fpr, tpr = roc_curve(y_true_bin, y_score)
        mask = fpr <= 0.10
        if mask.sum() < 2:
            return float("nan")
        return float(auc(fpr[mask], tpr[mask]) / 0.10)
    
    auc10s = []
    for c in range(3):
        y_bin = (y_true == c).astype(np.int32)
        auc10s.append(auc10_binary(y_bin, y_prob[:, c]))

    metrics["auc10_per_class"] = [float(x) for x in auc10s]
    metrics["auc10_macro"] = float(np.nanmean(auc10s))

    return metrics

@dataclass
class TrainerConfig:
    """Configuration dataclass used to configure the model training."""
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda" # should only train using GPUs (but can be changed to "cpu")

class Trainer:
    """
    Trainer class used to facilitate model training. Can take in most types of neural networks as input for training.

    Parameters
    ----------
        model : Module
            A PyTorch Module class which is the neural network to be trained.
        train_loader : DataLoader
            The model training data.
        logger : Logger
            Logger used to log progress and errors.
        val_loader : DataLoader or None
            Optional model evaluation data.
        config : TrainerConfig
            Neural network specific parameters to tweak.
        class_weights : Tensor or None
            Optional weights per class in the case of class imbalance.
    """
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

        # inital log for reproduceability
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("trainer_init", 
                         extra={"extra": {
                             "device": str(self.device), 
                             "epochs": self.config.epochs, 
                             "batch_size": self.config.batch_size, 
                             "learning_rate": self.config.learning_rate, 
                             "weight_decay": self.config.weight_decay, 
                             "num_params": num_params, 
                             "has_val_loader": self.val_loader is not None, 
                             "class_weights": self.class_weights.tolist()
                         }})
        
        # log class distributions
        train_counts = count_classes(train_loader)
        val_counts = count_classes(val_loader) if val_loader is not None else None
        self.logger.info("class_distribution", 
                         extra={"extra": {
                             "train_counts": train_counts, 
                             "val_counts": val_counts
                         }})

    def _batch_step(self, batch: torch.Tensor, train: bool = True) -> Dict[str, Any]:
        """Steps through a batch to train and optimize the model."""
        X, y_onehot = batch # (B, L, E), (B, 3)
        non_blocking = (self.device.type == "cuda")
        X = X.to(self.device, non_blocking=non_blocking)
        y_onehot = y_onehot.to(self.device, non_blocking=non_blocking)

        # validate targets
        if y_onehot.dim() != 2 or y_onehot.size(-1) != 3:
            raise ValueError(f"Expected y_onehot shape (B, 3), got {tuple(y_onehot.shape)}")
        row_sums = y_onehot.sum(dim=-1)
        if not torch.all((row_sums == 1) | (row_sums == 1.0)):
            raise ValueError("Targets must be one-hot encoded per sample")

        # convert one-hot to class indices {0, 1, 2} for loss calculation
        y = y_onehot.argmax(dim=-1)

        # get logits to calculate loss and validate shape
        logits = self.model(X)
        if logits.dim() != 2 or logits.size(-1) != 3:
            raise ValueError(f"Expected logits shape (B, 3), got {tuple(logits.shape)}")
        # calculate loss and validate for NaNs
        loss = self.criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss caught: {loss.item()}")

        # simple accuracy
        preds = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        correct = (preds == y).sum().item()
        total = y.size(0)
        acc = correct / total

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.logger.debug("batch_step_complete", 
                          extra={"extra": {
                              "mode": "train" if train else "val", 
                              "batch_size": total, 
                              "batch_loss": loss.item(), 
                              "batch_acc": acc
                          }})

        return {"loss": float(loss.item()), 
                "correct": int(correct), 
                "n": int(total), 
                "y": y.detach().cpu(),         # (B,)
                "preds": preds.detach().cpu(), # (B,)
                "probs": probs.detach().cpu()} # (B, 3)

    def _run_epoch(self, epoch: int, train: bool = True) -> Dict[str, float]:
        """Runs one complete epoch (training step) from start to finish."""
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

        # set up confusion matrix to see eval stats
        cm = None
        if not train:
            cm = torch.zeros((3, 3), dtype=torch.int64) # rows true, cols pred

        all_y: List[torch.Tensor] = []
        all_preds: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []

        # use inference mode for eval
        ctx = torch.enable_grad() if train else torch.inference_mode()
        for batch in loader:
            with ctx:
                out = self._batch_step(batch, train=train)

            # collect data for metrics
            if not train:
                all_y.append(out["y"])
                all_preds.append(out["preds"])
                all_probs.append(out["probs"])

            total_loss += out["loss"] * out["n"]
            total_correct += out["correct"]
            total_samples += out["n"]

            # build confusion matrix for detailed eval logging
            if cm is not None:
                yt = out["y"]
                yp = out["preds"]
                for t, p in zip(yt.tolist(), yp.tolist()):
                    cm[t, p] += 1

                # compute eval metrics
                y_true = torch.cat(all_y, dim=0).numpy()
                y_pred = torch.cat(all_preds, dim=0).numpy()
                y_prob = torch.cat(all_probs, dim=0).numpy()
                eval_metrics = compute_eval_metrics(y_true, y_pred, y_prob)

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
        
        # log confusion matrix
        if cm is not None:
            per_class_acc = cm.diag().float() / cm.sum(dim=1).clamp_min(1).float()
            balanced_acc = float(per_class_acc.mean().item())
            self.logger.info("val_confusion_matrix", 
                             extra={"extra": {
                                "epoch": epoch,
                                "confusion_matrix": cm.tolist(),
                                "per_class_acc": per_class_acc.tolist(),
                                "balanced_acc": balanced_acc,
                                "macro_precision": eval_metrics["macro_precision"],
                                "macro_recall": eval_metrics["macro_recall"],
                                "macro_f1": eval_metrics["macro_f1"],
                                "mcc": eval_metrics["mcc"],
                                "auc_macro": eval_metrics["auc_macro"],
                                "auc_per_class": eval_metrics["auc_per_class"],
                                "auc10_macro": eval_metrics["auc10_macro"],
                                "auc10_per_class": eval_metrics["auc10_per_class"],
                             }})

        return {"loss": avg_loss, "acc": avg_acc, "eval_metrics": eval_metrics if not train else None}

    def fit(self, save_dir: Optional[Path | str] = None) -> None:
        """
        Fits a neural network model to the data provided.

        Parameters
        ----------
            save_dir : Path or str or None
                An optional path to a directory to save the best performing model to.
        """
        best_val_loss = float("inf")
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

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

            if save_dir is not None:
                metric_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
                if metric_loss < best_val_loss:
                    best_val_loss = metric_loss
                    self._save_checkpoint(save_dir / "best_model.pt", epoch, best_val_loss, val_metrics["eval_metrics"])

        self.logger.info("training_done", 
                         extra={"extra": {
                             "best_val_loss": best_val_loss
                         }})

    def _save_checkpoint(self, path: Path | str, epoch: int, loss: float, metrics: Dict[str, Any]) -> None:
        """Saves model checkpoint to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"model_state_dict": self.model.state_dict(), 
                 "optim_state_dict": self.optimizer.state_dict(), 
                 "epoch": epoch, 
                 "config": self.config.__dict__, 
                 "best_loss": loss, 
                 "metrics": metrics}
        torch.save(state, path)

        self.logger.info("checkpoint_saved", 
                         extra={"extra": {
                             "checkpoint_path": str(path), 
                             "epoch": epoch, 
                             "loss": loss
                         }})
