import argparse
import json
import time
from pathlib import Path
import random
import os
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nn.train.pipelineio.logger import setup_logger
from nn.train.pipelineio.proteindataset import ProteinDataset, pad_collate
from nn.models.ffnn import PepSeqFFNN
from nn.train.trainer import Trainer, TrainerConfig


def _set_all_seeds(seed: int) -> None:
    """Sets seed value for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _init_ddp() -> Dict[str, Any] | None:
    """Initialize DDP if launched with srun. Returns rank info or None."""
    if "RANK" not in os.environ:
        return None

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": local_rank
    }


def _global_pos_weight(local_pos: int, local_neg: int, ddp: Dict[str, Any] | None) -> float:
    """Computes the global neg/pos across ranks if DDP is running."""
    if ddp is None:
        return float(local_neg / max(local_pos, 1))
    t = torch.tensor([local_pos, local_neg], device=torch.device("cuda"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    pos = int(t[0].item())
    neg = int(t[1].item())
    return float(neg / max(pos, 1))


def _infer_emb_dim(emb_index: Dict[str, Path]) -> int:
    """Infers the embedding dimension by loading first embedding from index. We expect embedding vectors to be of length 1281."""
    if not emb_index:
        raise ValueError("No embedding found in provided directories")
    first_path = next(iter(emb_index.values()))
    emb = torch.load(first_path, map_location="cpu", weights_only=True)
    if not isinstance(emb, torch.Tensor) or emb.dim() != 2:
        raise ValueError(
            f"Expected embedding tensor of shape (L, D) in {first_path}, got {type(emb)} with shape {getattr(emb, 'shape', None)}")
    return int(emb.size(1))


def _split_ids(ids: List[str], val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """Splits protein IDs into training and validation subsets."""
    if val_frac < 0.0 or val_frac > 1.0:
        return ids, []
    ids = list(ids)
    range_ = random.Random(seed)
    range_.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return ids[n_val:], ids[:n_val]


def _broadcast_params(params: Dict[str, Any], ddp: Dict[str, Any] | None) -> Dict[str, Any]:
    """Broadcasts params from rank 0 to all ranks if DDP is enabled."""
    if ddp is None:
        return params
    obj = [params]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def compute_pos_neg_counts(loader: DataLoader) -> Tuple[int, int]:
    """
    Calculates the positive and negative counts of each class.

    Parameters
    ----------
        loader : DataLoader
            Model training data with likely imbalanced classes.

    Returns
    -------
        Tuple[int, int]
            (pos_count, neg_count) over valid (masked) residues.
    """
    neg, pos = 0, 0
    for batch in loader:
        if len(batch) == 2:
            _, y = batch
            mask = None
        else:
            _, y, mask = batch
        y = y.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            y = y[mask]
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    return pos, neg


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
    parser.add_argument("--embedding-dirs",
                        nargs="+",
                        required=True,
                        type=Path,
                        help="One or more directories containing protein embeddings")
    parser.add_argument("--label-shards",
                        nargs="+",
                        required=True,
                        type=Path,
                        help="One or more label shard .pt files")
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
                        choices=["precision", "recall",
                                 "f1", "mcc", "auc", "pr_auc"],
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
    parser.add_argument("--calc-pos-weight",
                        dest="calc_pos_weight",
                        action="store_true",
                        help="Calculate positive weight to handle positive vs. negative class imbalances, note this may take a long time to run")
    parser.add_argument("--pos-weight",
                        dest="pos_weight",
                        action="store",
                        type=float,
                        default=None,
                        help="Optionally include a pre-calculated postive class weight")
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
    parser.add_argument("--window-size",
                        action="store",
                        dest="window_size",
                        type=int,
                        default=1000,
                        help="Window size for long protein sequences (<= 0 to disable)")
    parser.add_argument("--stride",
                        action="store",
                        dest="stride",
                        type=int,
                        default=900,
                        help="Stride between windows for long proteins")
    parser.add_argument("--no-collapse-labels",
                        dest="collapse_labels",
                        action="store_false",
                        help="Keep full label vectors when labels are (L, 3)")
    parser.add_argument("--no-pad-last-window",
                        dest="pad_last_window",
                        action="store_false",
                        help="Disable padding of final short window")
    parser.add_argument("--no-cache-label-shard",
                        dest="cache_current_label_shard",
                        action="store_false",
                        help="Reload label shard for each protein instead of caching")
    parser.add_argument("--no-drop-label-after-use",
                        dest="drop_label_after_use",
                        action="store_false",
                        help="Keep labels in memory after each protein is processed")
    args = parser.parse_args()

    args.save_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(json_lines=True, json_indent=2,
                          name="optuna_train_ffnn")

    ddp = _init_ddp()
    rank = ddp["rank"] if ddp is not None else 0
    world_size = ddp["world_size"] if ddp is not None else 1

    # log DDP info
    logger.info("ddp_init",
                extra={"extra": {
                    "ddp_enabled": ddp is not None,
                    "world_size": world_size,
                    "rank": rank,
                    "local_rank": ddp["local_rank"] if ddp is not None else 0
                }})

    # disable logging for other ranks when DDP enabled
    if ddp is not None and rank != 0:
        logger.disabled = True

    seed = args.seed
    _set_all_seeds(seed)

    embedding_dirs = list(args.embedding_dirs)
    label_shards = list(args.label_shards)
    if ddp is not None:
        pairs = list(zip(embedding_dirs, label_shards))
        if len(embedding_dirs) != len(label_shards):
            raise ValueError("Embedding dirs and label shards length mismatch")
        if len(embedding_dirs) == 0 or len(label_shards) == 0:
            raise ValueError(
                f"Rank {rank} for no shards, check shard counts vs. world size")
        pairs = pairs[rank::world_size]
        embedding_dirs = [p[0] for p in pairs]
        label_shards = [p[1] for p in pairs]

    base_dataset = ProteinDataset(
        embedding_dirs=embedding_dirs,
        label_shards=label_shards,
        window_size=args.window_size if args.window_size > 0 else None,
        stride=args.stride,
        collapse_labels=args.collapse_labels,
        pad_last_window=args.pad_last_window,
        return_meta=False,
        cache_current_label_shard=args.cache_current_label_shard,
        drop_label_after_use=args.drop_label_after_use
    )
    protein_ids = list(base_dataset.protein_ids)
    if args.subset > 0:
        protein_ids = protein_ids[:args.subset]

    train_ids, val_ids = _split_ids(protein_ids, args.val_frac, seed)

    train_data = ProteinDataset(
        embedding_dirs=embedding_dirs,
        label_shards=label_shards,
        protein_ids=train_ids,
        label_index=base_dataset.label_index,
        embedding_index=base_dataset.embedding_index,
        window_size=args.window_size if args.window_size > 0 else None,
        stride=args.stride,
        collapse_labels=args.collapse_labels,
        pad_last_window=args.pad_last_window,
        return_meta=False,
        cache_current_label_shard=args.cache_current_label_shard,
        drop_label_after_use=args.drop_label_after_use
    )

    val_data = None
    if len(val_ids) > 0:
        val_data = ProteinDataset(
            embedding_dirs=embedding_dirs,
            label_shards=label_shards,
            protein_ids=val_ids,
            label_index=base_dataset.label_index,
            embedding_index=base_dataset.embedding_index,
            window_size=args.window_size if args.window_size > 0 else None,
            stride=args.stride,
            collapse_labels=args.collapse_labels,
            pad_last_window=args.pad_last_window,
            return_meta=False,
            cache_current_label_shard=args.cache_current_label_shard,
            drop_label_after_use=args.drop_label_after_use
        )

    # get batch sizes from args
    pin = torch.cuda.is_available()
    batch_sizes = [int(x.strip())
                   for x in args.batch_sizes.split(",") if x.strip()]
    if len(batch_sizes) == 0:
        raise ValueError("No batch sizes provided")

    # setup Optuna study
    emb_dim = _infer_emb_dim(base_dataset.embedding_index)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup)

    # store study results in SQLite DB if applicable
    if rank == 0:
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

    def _sample_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Returns suggested parameters from Optuna optimization."""
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

        return {
            "hidden_sizes": hidden_sizes,
            "dropouts": dropouts,
            "depth": depth,
            "use_layer_norm": use_layer_norm,
            "use_residual": use_residual,
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": batch_size
        }

    def _run_trial(params: Dict[str, Any], trial: optuna.trial.Trial | None) -> float:
        trial_seed = int(params["trial_seed"])
        _set_all_seeds(trial_seed)

        hidden_sizes = tuple(params["hidden_sizes"])
        dropouts = tuple(params["dropouts"])
        depth = int(params["depth"])
        use_layer_norm = bool(params["use_layer_norm"])
        use_residual = bool(params["use_residual"])
        lr = float(params["learning_rate"])
        wd = float(params["weight_decay"])
        batch_size = int(params["batch_size"])

        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=pin,
                                  collate_fn=pad_collate)

        val_loader = None
        if val_data is not None:
            val_loader = DataLoader(val_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=pin,
                                    collate_fn=pad_collate)

        model = PepSeqFFNN(emb_dim=emb_dim,
                           hidden_sizes=hidden_sizes,
                           dropouts=dropouts,
                           use_layer_norm=use_layer_norm,
                           use_residual=use_residual,
                           num_classes=1)

        device = torch.device(f"cuda:{ddp['local_rank']}") if ddp is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if ddp is not None:
            model = DDP(model, device_ids=[
                        ddp["local_rank"]], output_device=ddp["local_rank"])

        # compute or store positive weight (like class weight)
        pos_weight = None
        if args.pos_weight is not None:
            pos_weight = float(args.pos_weight)
        elif args.calc_pos_weight:
            pos, neg = compute_pos_neg_counts(train_loader)
            pos_weight = _global_pos_weight(pos, neg, ddp)

        # setup config and trainer class
        config = TrainerConfig(epochs=args.epochs,
                               batch_size=batch_size,
                               learning_rate=lr,
                               weight_decay=wd,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               pos_weight=pos_weight)

        trial_dir = None
        if rank == 0 and trial is not None:
            trial_number = int(params["trial_number"])
            trial_dir = args.save_path / "trials" / f"trial_{trial_number:04d}"
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

        if rank == 0 and trial is not None:
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
                "PosWeight": float(pos_weight),
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
                "PR_AUC": float(best_metrics.get("pr_auc", float("nan"))),
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

    # run Optuna study
    timeout_s = None if args.timeout_s <= 0 else int(args.timeout_s)
    end_time = time.time() + timeout_s if timeout_s is not None else None

    for _ in range(args.n_trials):
        if rank == 0:
            if end_time is not None and time.time() >= end_time:
                params: Dict[str, Any] = {"_stop": True}
                trial = None
            else:
                trial = study.ask()
                params = _sample_params(trial)
                params["trial_number"] = int(trial.number)
                params["trial_seed"] = int(seed + trial.number)
                params["_stop"] = False
        else:
            trial = None
            params = {"_stop": False}

        params = _broadcast_params(params, ddp)
        if params.get("_stop"):
            break

        score = _run_trial(params, trial)
        if rank == 0:
            study.tell(trial, score)

    if rank == 0:
        # get most optimal results consolidated in rank 0
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

    if ddp is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
