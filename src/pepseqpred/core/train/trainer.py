"""trainer.py

Training loop utilities for PepSeqPred models.

Defines a configurable Trainer with support for masked residue loss, evaluation
metrics, threshold selection, checkpointing, and optional Optuna tuning.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import optuna
from .ddp import ddp_rank, ddp_all_reduce_sum, ddp_gather_all_1d
from .metrics import compute_eval_metrics
from .threshold import find_threshold_max_recall_min_precision


@dataclass
class TrainerConfig:
    """Configuration dataclass used to configure the model training."""
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    # should only train using GPUs (but can be changed to "cpu")
    device: str = "cuda"
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

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # default to Adam optimizer (may change later)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)

        pos_weight = None
        if self.config.pos_weight is not None:
            pos_weight = torch.tensor(
                [self.config.pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none")

        # inital log for reproduceability
        num_params = sum(p.numel() for p in self.model.parameters())
        if ddp_rank() == 0:
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

    def _batch_step(self, batch: torch.Tensor, train: bool = True) -> Dict[str, Any]:
        """Steps through a batch to train and optimize the model."""
        # ensure we grab mask when applicable
        if len(batch) == 2:
            X, y = batch
            mask = None
        else:
            X, y, mask = batch

        non_blocking = (self.device.type == "cuda")
        X = X.to(self.device, non_blocking=non_blocking)
        y = y.to(self.device, non_blocking=non_blocking).float()
        if mask is not None:
            mask = mask.to(self.device, non_blocking=non_blocking)

        # validate targets
        if y.dim() != 2:
            raise ValueError(
                f"Expected y_onehot shape (B, L), got {tuple(y.shape)}")

        # get logits to calculate loss and validate shape
        logits = self.model(X)
        if logits.shape != y.shape:
            raise ValueError(
                f"Expected logits shape {tuple(y.shape)}, got {tuple(logits.shape)}")
        # calculate loss
        loss_raw = self.criterion(logits, y)
        if mask is not None:
            if mask.shape != y.shape:
                raise ValueError(
                    f"Expected mask shape {tuple(y.shape)}, got {tuple(mask.shape)}")
            denom = mask.float().sum()
            if denom.item() == 0.0:
                loss = loss_raw.sum() * 0.0
                if train:
                    return {"loss": 0.0, "n": 0}
            else:
                loss = (loss_raw * mask.float()).sum() / denom
        else:
            loss = loss_raw.mean()

        # optimize model in training batch
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            n = int(mask.sum().item() if mask is not None else int(y.numel()))
            return {"loss": float(loss.item()), "n": n}

        probs = torch.sigmoid(logits)  # (B, L)
        y_flat = y.to(torch.long).view(-1)
        probs_flat = probs.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            y_flat = y_flat[mask_flat]
            probs_flat = probs_flat[mask_flat]
        return {"loss": float(loss.item()),
                "n": int(y_flat.numel()),
                "y": y_flat.detach().cpu(),
                "probs": probs_flat.detach().cpu()}

    def _run_epoch(self, epoch: int, train: bool = True) -> Dict[str, Any]:
        """Runs one complete epoch (training step) from start to finish."""
        loader = self.train_loader if train else self.val_loader
        if loader is None:
            return {"loss": float("nan"), "acc": float("nan")}

        if ddp_rank() == 0:
            self.logger.info("epoch_start",
                             extra={"extra": {
                                 "epoch": epoch,
                                 "mode": "train" if train else "val"
                             }})

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        total_pos = 0
        total_neg = 0

        all_y: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []

        # use inference mode for eval
        ctx = torch.enable_grad() if train else torch.inference_mode()
        for batch in loader:
            with ctx:
                out = self._batch_step(batch, train=train)

            # collect data for metrics
            if not train and out["n"] > 0:
                all_y.append(out["y"])
                all_probs.append(out["probs"])

            total_loss += out["loss"] * out["n"]
            total_samples += out["n"]

            # track class counts from masked residues
            if not train:
                y_flat = out["y"]
            else:
                # reuse labels from batch without extra forward pass
                if len(batch) == 2:
                    _, y = batch
                    mask = None
                else:
                    _, y, mask = batch

                y = y.view(-1)
                if mask is not None:
                    mask = mask.view(-1).bool()
                    y = y[mask]
                y_flat = y

            if y_flat.numel() > 0:
                total_pos += int((y_flat == 1).sum().item())
                total_neg += int((y_flat == 0).sum().item())

        # compute average loss (global across ranks)
        loss_sum = torch.tensor(
            [total_loss], device=self.device, dtype=torch.float64)
        n_sum = torch.tensor(
            [total_samples], device=self.device, dtype=torch.float64)
        loss_sum = ddp_all_reduce_sum(loss_sum)
        n_sum = ddp_all_reduce_sum(n_sum)
        total_samples = int(n_sum.item())
        avg_loss = float((loss_sum / n_sum.clamp_min(1)).item())

        # reduce pos/neg across rank for reporting
        pos_sum = torch.tensor(
            [total_pos], device=self.device, dtype=torch.float64)
        neg_sum = torch.tensor(
            [total_neg], device=self.device, dtype=torch.float64)
        pos_sum = ddp_all_reduce_sum(pos_sum)
        neg_sum = ddp_all_reduce_sum(neg_sum)
        total_pos = int(pos_sum.item())
        total_neg = int(neg_sum.item())

        # compute eval metrics
        cm = None
        eval_metrics = None
        if not train:
            # all ranks must participate in gathers to avoid DDP divergence.
            if len(all_y) > 0:
                y_local = torch.cat(all_y, dim=0).to(self.device)
                p_local = torch.cat(all_probs, dim=0).to(self.device)
            else:
                y_local = torch.empty(
                    0, device=self.device, dtype=torch.long)
                p_local = torch.empty(
                    0, device=self.device, dtype=torch.float32)

            # gather predictions across ranks for global metrics
            y_list, y_sizes = ddp_gather_all_1d(y_local, self.device)
            p_list, _ = ddp_gather_all_1d(p_local, self.device)

            if ddp_rank() != 0:
                # non-zero ranks skip metric computation
                eval_metrics = None
                avg_acc = float("nan")
            else:
                ys, ps = [], []
                for i, size in enumerate(y_sizes):
                    if size > 0:
                        ys.append(y_list[i][:size].cpu())
                        ps.append(p_list[i][:size].cpu())
                y_true = torch.cat(ys, dim=0).numpy() if len(
                    ys) > 0 else np.array([])
                y_prob = torch.cat(ps, dim=0).numpy() if len(
                    ps) > 0 else np.array([])

                # guard against invalid or non-existent residues globally
                if y_true.size == 0:
                    eval_metrics = {
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "f1": float("nan"),
                        "mcc": float("nan"),
                        "auc": float("nan"),
                        "auc10": float("nan"),
                        "pr_auc": float("nan"),
                        "threshold": float("nan"),
                        "threshold_status": "no_valid_residues",
                        "threshold_min_precision": 0.25
                    }
                    avg_acc = float("nan")
                else:
                    # compute predictions at most optimal threshold calculated
                    thresh_out = find_threshold_max_recall_min_precision(
                        y_true, y_prob, min_precision=0.25, num_thresholds=999)
                    best_thresh = float(thresh_out["threshold"])
                    y_pred = (y_prob >= best_thresh).astype(np.int64)

                    total_correct = int((y_pred == y_true).sum())
                    avg_acc = total_correct / max(total_samples, 1)

                    # build confusion matrix from true and predicted values
                    # rows true, cols pred
                    cm = torch.zeros((2, 2), dtype=torch.int64)
                    cm[0, 0] = int(((y_true == 0) & (y_pred == 0)).sum())
                    cm[0, 1] = int(((y_true == 0) & (y_pred == 1)).sum())
                    cm[1, 0] = int(((y_true == 1) & (y_pred == 0)).sum())
                    cm[1, 1] = int(((y_true == 1) & (y_pred == 1)).sum())

                    eval_metrics = compute_eval_metrics(y_true, y_pred, y_prob)
                    eval_metrics["threshold"] = best_thresh
                    eval_metrics["threshold_status"] = thresh_out["status"]
                    eval_metrics["threshold_min_precision"] = thresh_out["min_precision"]

        # log confusion matrix
        if (not train and cm is not None
                and eval_metrics is not None and ddp_rank() == 0):
            per_class_acc = cm.diag().float() / cm.sum(dim=1).clamp_min(1).float()
            balanced_acc = float(per_class_acc.mean())
            eval_metrics["per_class_acc"] = [
                float(x) for x in per_class_acc.tolist()]
            eval_metrics["res_balanced_acc"] = balanced_acc

            self.logger.info("res_confusion_matrix",
                             extra={"extra": {
                                 "epoch": epoch,
                                 "confusion_matrix": cm.tolist(),
                                 "balanced_acc": balanced_acc,
                                 "per_class_acc": per_class_acc.tolist(),
                                 "threshold": eval_metrics["threshold"],
                                 "threshold_status": eval_metrics["threshold_status"],
                                 "threshold_min_precision": eval_metrics["threshold_min_precision"],
                                 "precision": eval_metrics["precision"],
                                 "recall": eval_metrics["recall"],
                                 "f1": eval_metrics["f1"],
                                 "mcc": eval_metrics["mcc"],
                                 "auc": eval_metrics["auc"],
                                 "auc10": eval_metrics["auc10"],
                                 "pr_auc": eval_metrics["pr_auc"]
                             }})

        # handle training vs eval output
        out = {"loss": avg_loss, "n_residues": total_samples,
               "pos_residues": total_pos, "neg_residues": total_neg}
        if not train:
            out["acc"] = avg_acc
            out["eval_metrics"] = eval_metrics
        return out

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

        if ddp_rank() == 0:
            self.logger.info("training_started",
                             extra={"extra": {
                                 "epochs": self.config.epochs,
                                 "save_dir": str(save_dir) if save_dir is not None else None
                             }})

        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(epoch, train=True)
            if ddp_rank() == 0:
                self.logger.info("train_epoch_summary",
                                 extra={"extra": {
                                     "epoch": epoch,
                                     "train_loss": float(train_metrics["loss"]),
                                     "train_residues": int(train_metrics["n_residues"]),
                                     "train_pos_residues": int(train_metrics["pos_residues"]),
                                     "train_neg_residues": int(train_metrics["neg_residues"])
                                 }})

            eval_out = None
            if self.val_loader is not None:
                eval_out = self._run_epoch(epoch, train=False)
                if ddp_rank() == 0 and eval_out["eval_metrics"] is not None:
                    self.logger.info("eval_epoch_summary",
                                     extra={"extra": {
                                         "epoch": epoch,
                                         "eval_loss": float(eval_out["loss"]),
                                         "overall_acc_at_threshold": float(eval_out["acc"]),
                                         "eval_residues": int(eval_out["n_residues"]),
                                         "eval_pos_residues": int(eval_out["pos_residues"]),
                                         "eval_neg_residues": int(eval_out["neg_residues"]),
                                         "precision": float(eval_out["eval_metrics"]["precision"]),
                                         "recall": float(eval_out["eval_metrics"]["recall"]),
                                         "f1": float(eval_out["eval_metrics"]["f1"]),
                                         "mcc": float(eval_out["eval_metrics"]["mcc"]),
                                         "auc": float(eval_out["eval_metrics"]["auc"]),
                                         "auc10": float(eval_out["eval_metrics"]["auc10"]),
                                         "pr_auc": float(eval_out["eval_metrics"]["pr_auc"])
                                     }})

                # save from validated model only
                if save_dir is not None:
                    metric_loss = eval_out["loss"]
                    if metric_loss < best_val_loss:
                        best_val_loss = metric_loss
                        best_metrics = eval_out["eval_metrics"]
                        self._save_checkpoint(
                            save_dir / "best_model.pt", epoch, best_val_loss, metrics=best_metrics)

        # final check to save a model if no validation set
        if self.val_loader is None and save_dir is not None:
            self._save_checkpoint(save_dir / "best_model_no_val.pt",
                                  self.config.epochs - 1, float(train_metrics["loss"]))

        if ddp_rank() == 0:
            self.logger.info("training_done",
                             extra={"extra": {
                                 "best_loss": best_val_loss if self.val_loader is not None else train_metrics["loss"],
                                 "best_metrics": best_metrics if self.val_loader is not None else None
                             }})

    def _save_checkpoint(self, path: Path | str, epoch: int, loss: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Saves model checkpoint to path."""
        if ddp_rank() != 0:
            return

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

        Parameters
        ----------
            save_dir : Path or str
                Directory to save the model state to. Default is `None`.

            trial : Trial
                The current state of the Optuna trial used to optimize model hyperparameters. Default is `None`.

            score_key : str
                The metric used to score the model during a trial. Can be any accepted metric like `"f1"`, `"mcc"`, `"recall"`, `"precision"`, `"auc"`, or `"pr_auc"`.

        Returns
        -------
            best_score : int
                Best score achieved through hyperparameter tuning determined by `score_key`.

            best_epoch : int
                Best performing epoch by score.

            best_val_loss_at_score : float
                The best (lowest) loss acheived at the best score. 

            best_metrics : Dict[str, Any]
                All other metrics generated during evaluation step.
        """
        # nonzero ranks never touch Optuna APIs
        if ddp_rank() != 0:
            trial = None

        best_val_loss = float("inf")
        best_val_loss_at_score = float("inf")
        best_score = float("-inf")
        best_epoch = -1
        best_metrics: Dict[str, Any] = {}

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if ddp_rank() == 0:
            self.logger.info("training_started",
                             extra={"extra": {
                                 "epochs": self.config.epochs,
                                 "save_dir": str(save_dir) if save_dir else None
                             }})

        for epoch in range(self.config.epochs):
            train_metrics = self._run_epoch(epoch, train=True)
            if ddp_rank() == 0:
                self.logger.info("train_epoch_summary",
                                 extra={"extra": {
                                     "epoch": epoch,
                                     "train_loss": float(train_metrics["loss"]),
                                     "train_residues": int(train_metrics["n_residues"]),
                                     "train_pos_residues": int(train_metrics["pos_residues"]),
                                     "train_neg_residues": int(train_metrics["neg_residues"])
                                 }})

            val_metrics = self._run_epoch(
                epoch, train=False) if self.val_loader is not None else None

            metrics = {}
            score = float("nan")
            if val_metrics is not None:
                metrics = val_metrics.get("eval_metrics") or {}
                score = metrics.get(score_key, float("nan"))

            # broadcast pruning across ranks
            should_prune = False
            if trial is not None and ddp_rank() == 0 and val_metrics is not None:
                if score_key in {"precision", "recall", "f1", "mcc"}:
                    if metrics.get("threshold_status", "ok") != "ok":
                        should_prune = True
                if np.isfinite(score):
                    trial.report(score, step=epoch)
                    if trial.should_prune():
                        should_prune = True

            # synchronize pruning decision across ranks
            prune_flag = torch.tensor(
                [1 if should_prune else 0],
                device=self.device,
                dtype=torch.int32
            )
            prune_flag = ddp_all_reduce_sum(prune_flag)
            if prune_flag.item() > 0:
                if ddp_rank() == 0 and trial is not None:
                    raise optuna.TrialPruned()
                break  # exit epoch loop on nonzero ranks

            if val_metrics is None:
                continue

            # track overall best loss
            val_loss = float(val_metrics["loss"])
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
                    self._save_checkpoint(
                        save_dir / "best_model_by_score.pt", epoch, best_val_loss_at_score, best_metrics)

        if ddp_rank() == 0:
            self.logger.info("training_done",
                             extra={"extra": {
                                 "best_val_loss_at_score": best_val_loss_at_score,
                                 "best_val_loss_overall": best_val_loss,
                                 f"best_{score_key}_score": best_score,
                                 "best_epoch": best_epoch
                             }})

        return float(best_score), int(best_epoch), float(best_val_loss_at_score), best_metrics
