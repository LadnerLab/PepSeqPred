import argparse
import json
import time
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import optuna
from torch.utils.data import DataLoader, random_split
from pipelineio.logger import setup_logger
from pipelineio.peptidedataset import PeptideDataset
from nn.models.ffnn import PepSeqFFNN
from nn.train.trainer import Trainer, TrainerConfig
from typing import Any, Dict, Tuple


def compute_pos_weight(loader: DataLoader) -> float:
    """
    Calculates the weight of each class.

    Parameters
    ----------
        loader : DataLoader
            Model training data with likely imbalanced classes.

    Returns
    -------
        float
            The ratio of (uncertain + not epitopes) vs. definite epitopes.
            If no positive samples are present, 1.0 is returned.
    """
    neg, pos = 0, 0
    for _, y in loader:
        y = y.view(-1)
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    # if no bias, return 1.0
    if pos == 0:
        return 1.0

    return float(neg / pos)


def set_all_seeds(seed: int) -> None:
    """Sets seed value for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_hidden_sizes(trial: optuna.trial.Trial,
                       depth_min: int,
                       depth_max: int,
                       width_min: int,
                       width_max: int,
                       mode: str) -> Tuple[Tuple[int, ...], Tuple[float, ...], int, int]:
    """
    Builds hidden architecture and dropouts for PepSeqPredFFNN.

    Parameters
    ----------
        depth_min : int

        depth_max : int

        width_min : int

        width_max : int

        mode : str
            `"flat"`: all layers have the same width.\n
            `"bottleneck"`: monotonically decrease widths.\n
            `"pyramid"`: monotonically increase widths.

    Returns
    -------

    """
    depth = trial.suggest_int("depth", depth_min, depth_max)
    step = trial.suggest_categorical("width_step", [16, 32, 64])
    base = trial.suggest_int("base_width", width_min, width_max, step=step)

    # determine overall architecture
    if mode == "flat":
        sizes = [base] * depth

    else:
        ratio = trial.suggest_float("shape_ratio", 0.60, 0.95)
        sizes = []
        for i in range(depth):
            if mode == "bottleneck":
                w = int(round(base * (ratio ** i)))

            else:
                w = int(round(base * (ratio ** (depth - 1 - i))))

            w = max(width_min, min(width_max, w))
            sizes.append(w)

    dropout = trial.suggest_float("dropout", 0.0, 0.25)
    dropouts = [dropout] * depth

    return tuple(sizes), tuple(dropouts), depth, base


def append_csv_row(csv_path: Path | str, row: Dict[str, Any]) -> None:
    """
    Appends a new row to CSV runs file. Will create and add header if necessary.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])
    df_new.to_csv(csv_path,
                  mode="a",
                  header=(not csv_path.exists()),
                  index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna tuning CLI for PepSeqPredFFNN.")
    parser.add_argument("input_data",
                        type=Path,
                        help="Path to .pt file containing PeptideDataset")
    parser.add_argument("--study-name",
                        type=str,
                        default="ffnn_optuna",
                        help="Optuna study name")
    parser.add_argument("--storage",
                        type=str,
                        default="",
                        help="Optuna storage URL. Example sqlite:////scratch/$USER/optuna/study.db")
    parser.add_argument("--n-trials",
                        type=int,
                        default=40,
                        help="Number of Optuna trials to run")
    parser.add_argument("--epochs",
                        type=int,
                        default=8,
                        help="Epochs per trial")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Base seed")
    parser.add_argument("--metric",
                        type=str,
                        default="f1",
                        choices=["precision", "recall", "f1", "mcc", "auc"],
                        help="Metric to maximize")
    parser.add_argument("--val-frac",
                        type=float,
                        default=0.2,
                        help="Validation fraction")
    parser.add_argument("--subset",
                        type=int,
                        default=0,
                        help="If > 0, use only first N samples from dataset")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="DataLoader workers")
    parser.add_argument("--save-path",
                        type=Path,
                        default=Path("checkpoints/ffnn_optuna"),
                        help="Directory to save best model and artifacts")
    parser.add_argument("--csv-path",
                        type=Path,
                        default=Path("optuna_trials.csv"),
                        help="CSV output path for trial results")
    parser.add_argument("--arch-mode",
                        type=str,
                        default="flat",
                        choices=["flat", "bottleneck", "pyramid"],
                        help="Architecture family")
    parser.add_argument("--depth-min",
                        type=int,
                        default=2,
                        help="Min hidden layers")
    parser.add_argument("--depth-max",
                        type=int,
                        default=6,
                        help="Max hidden layers")
    parser.add_argument("--width-min",
                        type=int,
                        default=64,
                        help="Min hidden width")
    parser.add_argument("--width-max",
                        type=int,
                        default=512,
                        help="Max hidden width")
    parser.add_argument("--batch-sizes",
                        type=str,
                        default="32,64,128",
                        help="Comma separated batch sizes")
    parser.add_argument("--lr-min",
                        type=float,
                        default=1e-4,
                        help="Min learning rate")
    parser.add_argument("--lr-max",
                        type=float,
                        default=3e-3,
                        help="Max learning rate")
    parser.add_argument("--wd-min",
                        type=float,
                        default=1e-8,
                        help="Min weight decay")
    parser.add_argument("--wd-max",
                        type=float,
                        default=1e-2,
                        help="Max weight decay")
    parser.add_argument("--use-pos-weight",
                        action="store_true",
                        help="Use positive weight to handle positive vs. negative class imbalances.")
    parser.add_argument("--pruner-warmup",
                        type=int,
                        default=2,
                        help="Epochs to wait before pruning can happen")
    parser.add_argument("--timeout-s",
                        type=int,
                        default=0,
                        help="Optional wall timeout in seconds for study.optimize")
    args = parser.parse_args()

    args.save_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(json_lines=True, json_indent=2,
                          name="optuna_train_ffnn")

    set_all_seeds(args.seed)

    data = PeptideDataset.load(args.input_data)
    if args.subset > 0:
        n = min(args.subset, len(data))
        data = PeptideDataset(embeddings=data.embeddings[:n],
                              targets=data.targets[:n],
                              code_names=data.code_names[:n],
                              protein_ids=data.protein_ids[:n],
                              peptides=data.peptides[:n],
                              align_starts=data.align_starts[:n],
                              align_stops=data.align_stops[:n])

    # split data into training and val sets
    n_total = len(data)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val

    # generator used for dataset split reproducibility
    gen = torch.Generator().manual_seed(args.seed)
    train_data, val_data = random_split(data, [n_train, n_val], generator=gen)

    # get batch sizes from args
    pin = torch.cuda.is_available()
    batch_sizes = [int(x.strip())
                   for x in args.batch_sizes.split(",") if x.strip()]
    if len(batch_sizes) == 0:
        raise ValueError("No batch sizes provided")

    # setup Optuna study
    emb_dim = data.embeddings.size(-1)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup)

    # store study results in SQLite DB if applicable
    if args.storage.strip():
        study = optuna.create_study(study_name=args.study_name,
                                    direction="maximize",
                                    pruner=pruner,
                                    storage=args.storage,
                                    load_if_exists=True)

    else:
        study = optuna.create_study(study_name=args.study_name,
                                    direction="maximize",
                                    pruner=pruner)

    best_ckpt_path = args.save_path / "best_model_by_score.pt"
    best_trial_json = args.save_path / "best_trial.json"

    def _objective(trial: optuna.trial.Trial) -> float:
        """Helper function to run a trial."""
        trial_seed = args.seed + trial.number
        set_all_seeds(trial_seed)

        # determine overall shape
        hidden_sizes, dropouts, depth, _ = build_hidden_sizes(
            trial=trial,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            width_min=args.width_min,
            width_max=args.width_max,
            mode=args.arch_mode
        )

        # determine special model parameters
        use_layer_norm = trial.suggest_categorical(
            "use_layer_norm", [True, False])
        use_residual = trial.suggest_categorical("use_residual", [True, False])

        lr = trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True)
        wd = trial.suggest_float(
            "weight_decay", args.wd_min, args.wd_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)

        # setup data loaders
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=pin)
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=pin)

        # build FFNN model
        model = PepSeqFFNN(emb_dim=emb_dim,
                           hidden_sizes=hidden_sizes,
                           dropouts=dropouts,
                           use_layer_norm=use_layer_norm,
                           use_residual=use_residual,
                           num_classes=1)

        # compute positive weight (like class weight)
        pos_weight = None
        if args.use_pos_weight:
            pos_weight = compute_pos_weight(train_loader)

        # setup config and trainer class
        config = TrainerConfig(epochs=args.epochs,
                               batch_size=batch_size,
                               learning_rate=lr,
                               weight_decay=wd,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               pos_weight=pos_weight)
        trial_dir = args.save_path / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(model=model,
                          train_loader=train_loader,
                          logger=logger,
                          val_loader=val_loader,
                          config=config)

        # start and time trial
        start = time.time()
        best_score, best_epoch, best_val_loss_at_score, best_metrics = trainer.fit_optuna(
            save_dir=trial_dir,
            trial=trial,
            score_key=args.metric
        )
        elapsed = time.time() - start

        # add new row to CSV file
        row: Dict[str, Any] = {
            "RunID": f"{args.study_name}_trial_{trial.number:04d}",
            "Timestamp": pd.Timestamp.utcnow().isoformat(),
            "ModelVersion": "ffnn_optuna",
            "NumParameters": int(sum(p.numel() for p in model.parameters())),
            "HiddenLayers": int(depth),
            "HiddenSizes": ",".join(str(x) for x in hidden_sizes),
            "Activation": "ReLU",
            "Dropout": float(dropouts[0]) if len(dropouts) else 0.0,
            "UseLayerNorm": bool(use_layer_norm),
            "UseResidual": bool(use_residual),
            "BatchSize": int(batch_size),
            "LearningRate": float(lr),
            "WeightDecay": float(wd),
            "PosWeight": float(pos_weight) if args.use_pos_weight else 0.0,
            "Epochs": int(args.epochs),
            "BestValLossAtScore": float(best_val_loss_at_score),
            "BestEpoch": int(best_epoch),
            "Precision": float(best_metrics.get("precision", float("nan"))),
            "Recall": float(best_metrics.get("recall", float("nan"))),
            "F1": float(best_metrics.get("f1", float("nan"))),
            "MCC": float(best_metrics.get("mcc", float("nan"))),
            "BalancedAcc": float(best_metrics.get("res_balanced_acc", float("nan"))),
            "AUC": float(best_metrics.get("auc", float("nan"))),
            "AUC10": float(best_metrics.get("auc10", float("nan"))),
            "Threshold": float(best_metrics.get("threshold", float("nan"))),
            "ThresholdStatus": str(best_metrics.get("threshold_status", "")),
            "ThresholdMinPrecision": float(best_metrics.get("threshold_min_precision", float("nan"))),
            "ScoreKey": args.metric,
            "ScoreValue": float(best_score),
            "ElapsedSec": float(elapsed),
            "Status": "OK"
        }
        append_csv_row(args.csv_path, row)

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("best_val_loss_at_score",
                            float(best_val_loss_at_score))
        trial.set_user_attr("best_metrics", best_metrics)

        return float(best_score)

    # run study
    timeout_s = None if args.timeout_s <= 0 else int(args.timeout_s)
    study.optimize(_objective, n_trials=args.n_trials, timeout=timeout_s)

    # get most optimal results
    best = study.best_trial
    best_payload = {"study_name": args.study_name,
                    "best_value": float(best.value),
                    "best_params": dict(best.params),
                    "best_user_attrs": dict(best.user_attrs),
                    "metric": args.metric}
    best_trial_json.write_text(json.dumps(best_payload, indent=2))

    best_trial_dir = Path(best.user_attrs["trial_dir"])
    src = best_trial_dir / "best_model_by_score.pt"
    if src.exists():
        best_ckpt_path.write_bytes(src.read_bytes())

    logger.info("best_results", extra={"extra": best_payload})


if __name__ == "__main__":
    main()
