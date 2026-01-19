import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
from sklearn.metrics import (precision_recall_fscore_support, 
                             matthews_corrcoef, 
                             roc_auc_score, 
                             roc_curve, 
                             auc)
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

def count_classes(loader: DataLoader) -> List[int]:
    """
    Counts total number of peptides per class.

    Parameters
    ----------
        loader : DataLoader
            The DataLoader containing one-hot encoded targets to sum.

    Returns
    -------
        counts : List[int]
            List containing the sum of each class.
    """
    neg, pos = 0, 0
    for _, y in loader:
        y = y.view(-1)
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())
    return [neg, pos]

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

    # calculate precesion, recall, f1, and mcc
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # ROC AUC
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    except Exception:
        metrics["auc"] = float("nan")

    # AUC10 calculation]
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        mask = fpr <= 0.10
        if mask.sum() >= 2:
            metrics["auc10"] = float(auc(fpr[mask], tpr[mask]) / 0.10)
        
        else:
            metrics["auc10"] = float("nan")
    
    except Exception:
        metrics["auc10"] = float("nan")

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
    pos_weight: Optional[float] = None

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
    """
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, 
                 logger: logging.Logger, 
                 val_loader: Optional[DataLoader] = None, 
                 config: TrainerConfig = TrainerConfig()):
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
        
        pos_weight = None
        if self.config.pos_weight is not None:
            pos_weight = torch.tensor([self.config.pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


        # inital log for reproduceability
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("trainer_init", 
                         extra={"extra": {
                             "device": str(self.device), 
                             "seed": torch.initial_seed(),  
                             "epochs": self.config.epochs, 
                             "batch_size": self.config.batch_size, 
                             "learning_rate": self.config.learning_rate, 
                             "weight_decay": self.config.weight_decay, 
                             "num_params": num_params, 
                             "has_val_loader": self.val_loader is not None, 
                             "pos_weight": self.config.pos_weight
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
        X, y = batch
        non_blocking = (self.device.type == "cuda")
        X = X.to(self.device, non_blocking=non_blocking)
        y = y.to(self.device, non_blocking=non_blocking).float()

        # validate targets
        if y.dim() != 2:
            raise ValueError(f"Expected y_onehot shape (B, L), got {tuple(y.shape)}")

        # get logits to calculate loss and validate shape
        logits = self.model(X)
        if logits.shape != y.shape:
            raise ValueError(f"Expected logits shape {tuple(y.shape)}, got {tuple(logits.shape)}")

        # calculate loss
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits) # (B, L)
        preds = (probs >= 0.5).to(torch.long) # (B, L)
        y_int = y.to(torch.long)

        # simple accuracy calculation for batch step
        correct = (preds == y_int).sum().item()
        total = y.numel()
        acc = correct / total

        # optimize model in training batch
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
                "y": y_int.view(-1).detach().cpu(), 
                "preds": preds.view(-1).detach().cpu(), 
                "probs": probs.view(-1).detach().cpu()}

    def _run_epoch(self, epoch: int, train: bool = True) -> Dict[str, Any]:
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
            cm = torch.zeros((2, 2), dtype=torch.int64) # rows true, cols pred

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

                idx = yt * 2 + yp
                cm += torch.bincount(idx, minlength=4).view(2, 2)

        # compute eval metrics
        if not train:
            y_true = torch.cat(all_y, dim=0).numpy()
            y_pred = torch.cat(all_preds, dim=0).numpy()
            y_prob = torch.cat(all_probs, dim=0).numpy()

            eval_metrics = compute_eval_metrics(y_true, y_pred, y_prob)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        # log confusion matrix
        if cm is not None:
            per_class_acc = cm.diag().float() / cm.sum(dim=1).clamp_min(1).float()
            balanced_acc = float(per_class_acc.mean().item())
            eval_metrics["per_class_acc"] = [float(x) for x in per_class_acc.tolist()]
            eval_metrics["res_balanced_acc"] = balanced_acc
            self.logger.info("res_confusion_matrix", 
                             extra={"extra": {
                                "epoch": epoch,
                                "confusion_matrix": cm.tolist(),
                                "balanced_acc": balanced_acc,
                                "per_class_acc": per_class_acc.tolist(),
                                "precision": eval_metrics["precision"],
                                "recall": eval_metrics["recall"],
                                "f1": eval_metrics["f1"],
                                "mcc": eval_metrics["mcc"],
                                "auc": eval_metrics["auc"],
                                "auc10": eval_metrics["auc10"]
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
        best_metrics: Dict[str, Any] = {}
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("training_started", 
                         extra={"extra": {
                             "epochs": self.config.epochs, 
                             "save_dir": str(save_dir) if save_dir is not None else None
                         }})

        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(epoch, train=True)
            self.logger.info("train_epoch_summary", 
                             extra={"extra": {
                                 "epoch": epoch, 
                                 "train_loss": float(train_metrics["loss"]), 
                                 "train_acc": float(train_metrics["acc"])
                             }})

            eval_out = None
            if self.val_loader is not None:
                eval_out = self._run_epoch(epoch, train=False)
                self.logger.info("eval_epoch_summary", 
                    extra={"extra": {
                        "epoch": epoch, 
                        "eval_loss": float(eval_out["loss"]),
                        "eval_acc": float(eval_out["acc"]),
                        "precision": float(eval_out["eval_metrics"]["precision"]),
                        "recall": float(eval_out["eval_metrics"]["recall"]),
                        "f1": float(eval_out["eval_metrics"]["f1"]),
                        "mcc": float(eval_out["eval_metrics"]["mcc"]),
                        "auc": float(eval_out["eval_metrics"]["auc"]),
                        "auc10": float(eval_out["eval_metrics"]["auc10"])
                    }})

                # save from validated model only
                if save_dir is not None:
                    metric_loss = eval_out["loss"]
                    if metric_loss < best_val_loss:
                        best_val_loss = metric_loss
                        best_metrics = eval_out["eval_metrics"]
                        self._save_checkpoint(save_dir / "best_model.pt", epoch, best_val_loss, metrics=best_metrics)

        # final check to save a model if no validation set 
        if self.val_loader is None and save_dir is not None:
            self._save_checkpoint(save_dir / "best_model_no_val.pt", self.config.epochs - 1, float(train_metrics["loss"]))

        self.logger.info("training_done", 
                         extra={"extra": {
                             "best_loss": best_val_loss if self.val_loader else train_metrics["loss"], 
                             "best_metrics": best_metrics if self.val_loader else None
                         }})

    def _save_checkpoint(self, path: Path | str, epoch: int, loss: float, metrics: Optional[Dict[str, Any]] = None) -> None:
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
        
    def fit_optuna(self, 
                   save_dir: Optional[Path | str] = None, 
                   trial: Optional[optuna.trial.Trial] = None, 
                   score_key: str = "f1") -> Tuple[float, int, float, Dict[str, Any]]:
        """
        Fits model using Optuna for hyperparameter optimization.
        """
        best_val_loss = float("inf")
        best_val_loss_at_score = float("inf")
        best_score = float("-inf")
        best_epoch = -1
        best_metrics: Dict[str, Any] = {}

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("training_started", 
                         extra={"extra":{
                             "epochs": self.config.epochs, 
                             "save_dir": str(save_dir) if save_dir else None
                         }})
        
        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(epoch, train=True)
            self.logger.info("train_epoch_summary", 
                             extra={"extra": {
                                 "epoch": epoch, 
                                 "train_loss": train_metrics["loss"], 
                                 "train_acc": train_metrics["acc"]
                             }})

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._run_epoch(epoch, train=False)

            if val_metrics is None:
                continue

            val_loss = float(val_metrics["loss"])
            metrics = val_metrics.get("eval_metrics", {})
            score = metrics.get(score_key, float("nan"))

            # report intermediate score for pruning
            if trial is not None:
                if np.isfinite(score):
                    trial.report(score, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
            # track overall best loss
            best_val_loss = min(best_val_loss, val_loss)

            # track best loss, score, and metrics for Optuna
            if np.isfinite(score) and score > best_score:
                best_val_loss_at_score = float(val_loss)
                best_score = float(score)
                best_epoch = int(epoch)

                best_metrics = dict(metrics)
                best_metrics["best_val_loss_overall"] = best_val_loss
                best_metrics["best_val_loss_at_score"] = best_val_loss_at_score
                best_metrics["best_epoch"] = best_epoch
                best_metrics["best_score_key"] = score_key
                best_metrics["best_score_value"] = best_score
                    
                # save by score metric (F1, MCC, AUC, etc.)
                if save_dir is not None:
                    self._save_checkpoint(save_dir / "best_model_by_score.pt", epoch, best_val_loss_at_score, best_metrics)

        self.logger.info("training_done", 
                         extra={"extra": {
                             "best_val_loss_at_score": best_val_loss_at_score, 
                             "best_val_loss_overall": best_val_loss, 
                             f"best_{score_key}_score": best_score, 
                             "best_epoch": best_epoch
                         }})
        
        return float(best_score), int(best_epoch), float(best_val_loss_at_score), best_metrics
