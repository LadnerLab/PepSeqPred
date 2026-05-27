"""train_optuna_cli.py

This CLI is very similar to `train_cli.py`, except it optimizes for the best possible hyperparameters
within the user-defined ranges in the shell script `scripts/hpc/trainoptuna.sh`.

Handles end-to-end training, evaluation, and hyperparameter optimization of a PepSeqPred model head with the goal 
to predict the locations of antibody epitopes within a protein sequence downstream. The resulting model 
will make binary predictions: definite epitope or not epitope, it handles residues labeled uncertain through
a masking process.

The training module utilizes DistributedDataParallel (DDP) to train the model distributed across multiple 
GPUs. DDP runs one trial across N GPUs for maximum efficiency in trial runs. It is highly recommended you train 
this model using an HPC, training locally is often impossible due to time constraints and limited compute. 
For context, our models were often trained using 4 A100 GPUs and take anywhere from 3 to 5 hours. Given this
script is designed to train several models during one run to optimize for the best hyperparameters, we
recommend several hours to a few days are allocated per Optuna study session.

Usage
-----
>>> # from scripts/hpc/trainoptuna.sh (see shell script for CLI config)
>>> sbatch trainoptuna.sh /path/to/emb_shard_dir0 ... /path/to/emb_shard_dirN -- \\ 
                              /path/to/label_shard0.pt ... /path/to/label_shardN.pt
"""


import argparse
import json
import math
import time
import random
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.write import append_csv_row
from pepseqpred.core.data.proteindataset import ProteinDataset, pad_collate
from pepseqpred.core.models.factory import (
    MODEL_HEADS,
    PepSeqModelConfig,
    build_pepseq_model,
    model_config_to_dict,
    validate_model_config,
)
from pepseqpred.core.train.trainer import Trainer, TrainerConfig
from pepseqpred.core.train.ddp import init_ddp
from pepseqpred.core.train.split import (
    SPLIT_STRATEGIES,
    split_ids,
    split_ids_grouped,
    split_ids_label_stratified,
    build_label_support_by_id,
    build_split_report,
    partition_ids_weighted,
    sort_ids_for_locality
)
from pepseqpred.core.train.weights import (
    compute_pos_neg_counts,
    global_pos_neg_counts
)
from pepseqpred.core.train.threshold import THRESHOLD_POLICIES
from pepseqpred.core.train.embedding import infer_emb_dim
from pepseqpred.core.train.seed import set_all_seeds


def _broadcast_params(params: Dict[str, Any], ddp: Dict[str, Any] | None) -> Dict[str, Any]:
    """Broadcasts params from rank 0 to all ranks if DDP is enabled."""
    if ddp is None:
        return params
    obj = [params]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def _finite_or_none(value: Any) -> float | None:
    """Tries to convert number to float if finite, otherwise returns None."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _parse_int_choices(raw: str, flag: str) -> Tuple[int, ...]:
    """Parse a comma-separated list of integer choices."""
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as e:
            raise ValueError(f"{flag} must contain only integers") from e
    if len(values) < 1:
        raise ValueError(f"{flag} must include at least one value")
    return tuple(values)


def _split_summary_csv_fields(
    prefix: str,
    summary: Dict[str, Any] | None
) -> Dict[str, Any]:
    """Flattens split report summary fields for trial CSV artifacts."""
    summary = summary or {}
    return {
        f"{prefix}Proteins": _finite_or_none(summary.get("protein_count")),
        f"{prefix}ValidResidues": _finite_or_none(summary.get("valid_residues")),
        f"{prefix}PosResidues": _finite_or_none(summary.get("positive_residues")),
        f"{prefix}NegResidues": _finite_or_none(summary.get("negative_residues")),
        f"{prefix}PositiveRate": _finite_or_none(summary.get("positive_rate")),
    }


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
        trial : optuna.trial.Trial
            Optuna trial used to sample architecture and dropout hyperparameters.
        depth_min : int
            Minimum number of hidden layers to sample.
        depth_max : int
            Maximum number of hidden layers to sample.
        width_min : int
            Minimum width for any hidden layer.
        width_max : int
            Maximum width for any hidden layer.
        mode : str
            `"flat"`: all layers have the same width.
            `"bottleneck"`: monotonically decrease widths from base.
            `"pyramid"`: monotonically increase widths toward base.

    Returns
    -------
        Tuple[Tuple[int, ...], Tuple[float, ...], int, int]
            A 4-tuple of `(sizes, dropouts, depth, base)` where `sizes` are the sampled
            hidden layer widths, `dropouts` are per-layer dropout rates, `depth` is the
            number of layers, and `base` is the base width sampled before shaping.
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


def main() -> None:
    """Parses CLI arguments and runs Optuna study."""
    parser = argparse.ArgumentParser(
        description="Optuna tuning CLI for PepSeqPred model heads.")
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
    parser.add_argument("--model-head",
                        action="store",
                        dest="model_head",
                        type=str,
                        choices=list(MODEL_HEADS),
                        default="ffnn",
                        help="Classifier head to tune.")
    parser.add_argument("--threshold-policy",
                        type=str,
                        default="max-recall-min-precision",
                        choices=list(THRESHOLD_POLICIES),
                        help="Validation threshold selection policy.")
    parser.add_argument("--threshold-min-precision",
                        type=float,
                        default=0.25,
                        help="Minimum precision for max-recall-min-precision threshold selection.")
    parser.add_argument("--threshold-min-recall",
                        type=float,
                        default=0.80,
                        help="Minimum recall for min-recall-max-precision threshold selection.")
    parser.add_argument("--threshold-fixed-value",
                        type=float,
                        default=0.50,
                        help="Fixed threshold used when --threshold-policy=fixed.")
    parser.add_argument("--val-frac",
                        type=float,
                        default=0.2,
                        help="Validation fraction")
    parser.add_argument("--subset",
                        type=int,
                        default=0,
                        help="If > 0, use only first N samples from dataset")
    parser.add_argument("--split-type",
                        type=str,
                        default="id-family",
                        choices=["id", "id-family"],
                        help="Data partition type, use ID only or ID and taxonomic family.")
    parser.add_argument("--split-strategy",
                        type=str,
                        default="size-balanced",
                        choices=list(SPLIT_STRATEGIES),
                        help="Split assignment strategy. Default preserves existing size-balanced behavior.")
    parser.add_argument("--split-report-json",
                        type=Path,
                        default=None,
                        help="Optional split report JSON path. Defaults to <save-path>/split_report.json.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="DataLoader workers")
    parser.add_argument("--pos-weight",
                        dest="pos_weight",
                        action="store",
                        type=float,
                        default=None,
                        help="Optional manual positive class weight; omitted means compute from the current training split")
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
    parser.add_argument("--conv-channel-choices",
                        type=str,
                        default="32,64,128",
                        help="Comma-separated Conv1d channel choices used when --model-head=conv1d.")
    parser.add_argument("--conv-layers-min",
                        type=int,
                        default=1,
                        help="Minimum Conv1d layers used when --model-head=conv1d.")
    parser.add_argument("--conv-layers-max",
                        type=int,
                        default=3,
                        help="Maximum Conv1d layers used when --model-head=conv1d.")
    parser.add_argument("--conv-kernel-size-choices",
                        type=str,
                        default="3,5,9,15",
                        help="Comma-separated odd Conv1d kernel choices used when --model-head=conv1d.")
    parser.add_argument("--conv-dropout-min",
                        type=float,
                        default=0.0,
                        help="Minimum Conv1d dropout used when --model-head=conv1d.")
    parser.add_argument("--conv-dropout-max",
                        type=float,
                        default=0.25,
                        help="Maximum Conv1d dropout used when --model-head=conv1d.")
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
                        help="Training window size for long protein sequences (<= 0 to disable; validation uses full proteins)")
    parser.add_argument("--stride",
                        action="store",
                        dest="stride",
                        type=int,
                        default=900,
                        help="Stride between training windows for long proteins")
    parser.add_argument("--no-collapse-labels",
                        dest="collapse_labels",
                        action="store_false",
                        help="Keep full label vectors when labels are (L, 3)")
    parser.add_argument("--no-pad-last-window",
                        dest="pad_last_window",
                        action="store_false",
                        help="Disable padding of final short training window")
    parser.add_argument("--no-cache-label-shard",
                        dest="cache_current_label_shard",
                        action="store_false",
                        help="Reload label shard for each protein instead of caching")
    parser.add_argument("--no-drop-label-after-use",
                        dest="drop_label_after_use",
                        action="store_false",
                        help="Keep labels in memory after each protein is processed")
    args = parser.parse_args()
    if args.threshold_min_precision < 0.0 or args.threshold_min_precision > 1.0:
        raise ValueError("--threshold-min-precision must be in [0.0, 1.0]")
    if args.threshold_min_recall < 0.0 or args.threshold_min_recall > 1.0:
        raise ValueError("--threshold-min-recall must be in [0.0, 1.0]")
    if args.threshold_fixed_value <= 0.0 or args.threshold_fixed_value >= 1.0:
        raise ValueError("--threshold-fixed-value must be between (0.0, 1.0)")
    if args.conv_layers_min < 1:
        raise ValueError("--conv-layers-min must be >= 1")
    if args.conv_layers_max < args.conv_layers_min:
        raise ValueError("--conv-layers-max must be >= --conv-layers-min")
    if args.conv_dropout_min < 0.0 or args.conv_dropout_min > 1.0:
        raise ValueError("--conv-dropout-min must be in [0.0, 1.0]")
    if args.conv_dropout_max < args.conv_dropout_min or args.conv_dropout_max > 1.0:
        raise ValueError("--conv-dropout-max must be in [--conv-dropout-min, 1.0]")

    args.save_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(json_lines=True, json_indent=2,
                          name="train_optuna_cli")

    ddp = init_ddp()
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
    set_all_seeds(seed)

    embedding_dirs = list(args.embedding_dirs)
    label_shards = list(args.label_shards)
    if len(embedding_dirs) == 0 or len(label_shards) == 0:
        raise ValueError("No embedding dirs or label files provided")

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

    split_report_json = args.split_report_json or (
        args.save_path / "split_report.json")

    # partition data by ID + family or just ID
    family_groups: Dict[str, str] = {}
    missing_family_ids = 0
    for protein_id in protein_ids:
        family = base_dataset.embedding_family_by_id.get(protein_id)
        if family is None or str(family).strip() == "":
            family_groups[protein_id] = f"__missing_family__:{protein_id}"
            missing_family_ids += 1
        else:
            family_groups[protein_id] = str(family)
    split_groups = (
        family_groups
        if args.split_type == "id-family"
        else {protein_id: protein_id for protein_id in protein_ids}
    )

    if ddp is None or rank == 0:
        label_support_by_id = build_label_support_by_id(
            protein_ids,
            base_dataset.label_index,
        )
    else:
        label_support_by_id = {}
    if ddp is not None:
        obj = [label_support_by_id]
        dist.broadcast_object_list(obj, src=0)
        label_support_by_id = obj[0]

    if args.split_strategy == "label-stratified":
        train_ids_all, val_ids_all = split_ids_label_stratified(
            protein_ids,
            args.val_frac,
            seed,
            split_groups,
            label_support_by_id,
        )
    elif args.split_type == "id-family":
        train_ids_all, val_ids_all = split_ids_grouped(
            protein_ids, args.val_frac, seed, family_groups
        )
    else:
        train_ids_all, val_ids_all = split_ids(
            protein_ids, args.val_frac, seed
        )

    if args.split_type == "id-family":
        train_families = {family_groups[pid] for pid in train_ids_all}
        val_families = {family_groups[pid] for pid in val_ids_all}
        overlap = train_families & val_families
        if overlap:
            raise RuntimeError(
                f"Family leakage detected for split_type='id-family': n_overlap={len(overlap)}"
            )
    else:
        val_families = set()

    if len(train_ids_all) == 0:
        raise ValueError("Global split produced 0 train IDs")

    split_report_payload = None
    train_split_summary: Dict[str, Any] = {}
    val_split_summary: Dict[str, Any] = {}
    if rank == 0:
        split_report_payload = build_split_report(
            run_splits=[
                {
                    "run_index": 1,
                    "train_mode": "optuna-holdout",
                    "split_seed": int(seed),
                    "train_seed": int(seed),
                    "fold_index": None,
                    "n_folds": 1,
                    "ensemble_set_index": None,
                    "train_ids": train_ids_all,
                    "val_ids": val_ids_all,
                }
            ],
            support_by_id=label_support_by_id,
            families_by_id=family_groups,
            split_type=str(args.split_type),
            split_strategy=str(args.split_strategy),
        )
        split_report_json.parent.mkdir(parents=True, exist_ok=True)
        split_report_json.write_text(
            json.dumps(split_report_payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        split_entry = split_report_payload["runs"][0]
        train_split_summary = dict(split_entry["train"])
        val_split_summary = dict(split_entry["validation"])
        logger.info("family_split_summary",
                    extra={"extra": {
                        "split_type": args.split_type,
                        "split_strategy": args.split_strategy,
                        "split_report_json": str(split_report_json),
                        "train_ids": len(train_ids_all),
                        "val_ids": len(val_ids_all),
                        "val_families": len(val_families),
                        "missing_family_ids": missing_family_ids
                    }})

    # weight by embedding file size to reduce I/O
    id_weights: Dict[str, float] = {}
    for protein_id in protein_ids:
        emb_path = base_dataset.embedding_index.get(protein_id)
        if emb_path is None:
            id_weights[protein_id] = 1.0
            continue
        # use file size as weight
        try:
            id_weights[protein_id] = float(max(1, emb_path.stat().st_size))
        except OSError:
            id_weights[protein_id] = 1.0

    # generate protein ID groupings
    id_groups: Dict[str, str] = {
        protein_id: str(base_dataset.label_index.get(protein_id, ""))
        for protein_id in protein_ids
    }

    if ddp is None:
        train_ids = sort_ids_for_locality(train_ids_all, id_groups)
        val_ids = sort_ids_for_locality(val_ids_all, id_groups)
    else:
        # build train and validation sets on rank 0
        if rank == 0:
            payload = {
                "train_ids_by_rank": partition_ids_weighted(
                    train_ids_all,
                    world_size,
                    weights=id_weights,
                    groups=id_groups,
                    ensure_non_empty=True
                ),
                "val_ids_by_rank": partition_ids_weighted(
                    val_ids_all,
                    world_size,
                    weights=id_weights,
                    groups=id_groups,
                    ensure_non_empty=False
                )
            }
        else:
            payload = {}

        # broadcast payload across each rank
        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]

        train_ids = list(payload["train_ids_by_rank"][rank])
        val_ids = list(payload["val_ids_by_rank"][rank])

        # gather splits per rank
        per_rank = [None] * world_size
        dist.all_gather_object(per_rank, {
            "rank": rank,
            "train_ids": len(train_ids),
            "val_ids": len(val_ids)
        })

        if rank == 0:
            logger.info("partition_summary", extra={"extra": {
                "total_train_ids": len(train_ids_all),
                "total_val_ids": len(val_ids_all),
                "per_rank": per_rank
            }})

        if any(int(x["train_ids"]) == 0 for x in per_rank if x is not None):
            raise RuntimeError(
                "At least one rank received 0 train IDs after weighted partitioning")

    # shuffle train IDs per best deep learning practces
    rng = random.Random(seed)
    rng.shuffle(train_ids)
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
    if len(val_ids) > 0 or ddp is not None:
        val_data = ProteinDataset(
            embedding_dirs=embedding_dirs,
            label_shards=label_shards,
            protein_ids=val_ids,
            label_index=base_dataset.label_index,
            embedding_index=base_dataset.embedding_index,
            window_size=None,
            stride=1,
            collapse_labels=args.collapse_labels,
            pad_last_window=False,
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
    conv_channel_choices = _parse_int_choices(
        args.conv_channel_choices,
        "--conv-channel-choices",
    )
    conv_kernel_size_choices = _parse_int_choices(
        args.conv_kernel_size_choices,
        "--conv-kernel-size-choices",
    )
    bad_kernels = [x for x in conv_kernel_size_choices if x < 1 or x % 2 != 1]
    if bad_kernels:
        raise ValueError(
            "--conv-kernel-size-choices must contain only positive odd integers"
        )

    # resolve positive class weight once from the current training split
    pos_weight_source = "cli"
    train_pos_residues = None
    train_neg_residues = None
    if args.pos_weight is not None:
        resolved_pos_weight = float(args.pos_weight)
    else:
        count_loader_kwargs = {
            "batch_size": batch_sizes[0],
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": pin,
            "collate_fn": pad_collate
        }
        if args.num_workers > 0:
            count_loader_kwargs["multiprocessing_context"] = "spawn"
            count_loader_kwargs["prefetch_factor"] = 4

        count_loader = DataLoader(train_data, **count_loader_kwargs)
        local_pos, local_neg = compute_pos_neg_counts(count_loader)
        train_pos_residues, train_neg_residues = global_pos_neg_counts(
            local_pos,
            local_neg,
            ddp
        )
        resolved_pos_weight = float(train_neg_residues / max(train_pos_residues, 1))
        pos_weight_source = "train_loader"

    if rank == 0:
        logger.info(
            "pos_weight_resolved",
            extra={
                "extra": {
                    "source": pos_weight_source,
                    "train_pos_residues": train_pos_residues,
                    "train_neg_residues": train_neg_residues,
                    "pos_weight": float(resolved_pos_weight)
                }
            }
        )

    # setup Optuna study
    emb_dim = infer_emb_dim(base_dataset.embedding_index)
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
        if args.model_head == "conv1d":
            conv_channels = trial.suggest_categorical(
                "conv_channels",
                list(conv_channel_choices),
            )
            conv_layers = trial.suggest_int(
                "conv_layers",
                args.conv_layers_min,
                args.conv_layers_max,
            )
            conv_kernel_size = trial.suggest_categorical(
                "conv_kernel_size",
                list(conv_kernel_size_choices),
            )
            conv_dropout = trial.suggest_float(
                "conv_dropout",
                args.conv_dropout_min,
                args.conv_dropout_max,
            )
        else:
            conv_channels = int(conv_channel_choices[0])
            conv_layers = int(args.conv_layers_min)
            conv_kernel_size = int(conv_kernel_size_choices[0])
            conv_dropout = float(args.conv_dropout_min)

        return {
            "model_head": str(args.model_head),
            "hidden_sizes": hidden_sizes,
            "dropouts": dropouts,
            "depth": depth,
            "use_layer_norm": use_layer_norm,
            "use_residual": use_residual,
            "conv_channels": int(conv_channels),
            "conv_layers": int(conv_layers),
            "conv_kernel_size": int(conv_kernel_size),
            "conv_dropout": float(conv_dropout),
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": batch_size
        }

    def _run_trial(params: Dict[str, Any], trial: optuna.trial.Trial | None) -> float:
        trial_seed = int(params["trial_seed"])
        set_all_seeds(trial_seed)

        hidden_sizes = tuple(params["hidden_sizes"])
        dropouts = tuple(params["dropouts"])
        depth = int(params["depth"])
        use_layer_norm = bool(params["use_layer_norm"])
        use_residual = bool(params["use_residual"])
        conv_channels = int(params["conv_channels"])
        conv_layers = int(params["conv_layers"])
        conv_kernel_size = int(params["conv_kernel_size"])
        conv_dropout = float(params["conv_dropout"])
        lr = float(params["learning_rate"])
        wd = float(params["weight_decay"])
        batch_size = int(params["batch_size"])

        # handle loader worker creation in multi-rank process
        loader_kwargs = {"batch_size": batch_size,
                         "shuffle": False,
                         "num_workers": args.num_workers,
                         "pin_memory": pin,
                         "collate_fn": pad_collate}
        if args.num_workers > 0:
            loader_kwargs["multiprocessing_context"] = "spawn"
            # reduce worker respawn overhead
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(train_data, **loader_kwargs)
        val_loader = DataLoader(
            val_data, **loader_kwargs) if val_data is not None else None

        model_config = validate_model_config(
            PepSeqModelConfig(
                emb_dim=emb_dim,
                hidden_sizes=hidden_sizes,
                dropouts=dropouts,
                num_classes=1,
                use_layer_norm=use_layer_norm,
                use_residual=use_residual,
                model_head=str(params["model_head"]),
                conv_channels=conv_channels,
                conv_layers=conv_layers,
                conv_kernel_size=conv_kernel_size,
                conv_dropout=conv_dropout,
            )
        )
        model = build_pepseq_model(model_config)

        device = torch.device(f"cuda:{ddp['local_rank']}") if ddp is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if ddp is not None:
            model = DDP(model, device_ids=[
                        ddp["local_rank"]], output_device=ddp["local_rank"])

        # use the train-split positive weight resolved before trial search
        pos_weight = resolved_pos_weight

        # setup config and trainer class
        config = TrainerConfig(epochs=args.epochs,
                               batch_size=batch_size,
                               learning_rate=lr,
                               weight_decay=wd,
                               device="cuda" if torch.cuda.is_available() else "cpu",
                               pos_weight=pos_weight,
                               threshold_policy=args.threshold_policy,
                               threshold_min_precision=args.threshold_min_precision,
                               threshold_min_recall=args.threshold_min_recall,
                               threshold_fixed_value=args.threshold_fixed_value)

        trial_dir = None
        if rank == 0 and trial is not None:
            trial_number = int(params["trial_number"])
            trial_dir = args.save_path / "trials" / f"trial_{trial_number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(model=model,
                          train_loader=train_loader,
                          logger=logger,
                          val_loader=val_loader,
                          config=config,
                          model_config=model_config)

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
                "ModelVersion": "train_optuna",
                "ModelHead": str(model_config.model_head),
                "NumParameters": int(sum(p.numel() for p in model.parameters())),
                "HiddenLayers": int(depth),
                "HiddenSizes": ",".join(str(x) for x in hidden_sizes),
                "Activation": "ReLU",
                "Dropout": float(dropouts[0]) if len(dropouts) else 0.0,
                "UseLayerNorm": bool(use_layer_norm),
                "UseResidual": bool(use_residual),
                "ConvChannels": int(model_config.conv_channels),
                "ConvLayers": int(model_config.conv_layers),
                "ConvKernelSize": int(model_config.conv_kernel_size),
                "ConvDropout": float(model_config.conv_dropout),
                "BatchSize": int(batch_size),
                "LearningRate": float(lr),
                "WeightDecay": float(wd),
                "PosWeight": float(pos_weight) if pos_weight else 1.0,
                "Epochs": int(args.epochs),
                "SplitStrategy": str(args.split_strategy),
                "SplitReportJson": str(split_report_json),
                **_split_summary_csv_fields("Train", train_split_summary),
                **_split_summary_csv_fields("Val", val_split_summary),
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
                "ThresholdPolicy": str(best_metrics.get("threshold_policy", args.threshold_policy)),
                "ThresholdStatus": str(best_metrics.get("threshold_status", "")),
                "ThresholdMinPrecision": float(best_metrics.get("threshold_min_precision", float("nan"))),
                "ThresholdMinRecall": float(best_metrics.get("threshold_min_recall", float("nan"))),
                "ThresholdFixedValue": float(best_metrics.get("threshold_fixed_value", float("nan"))),
                "ThresholdPredPosFrac": float(best_metrics.get("threshold_pred_pos_frac", float("nan"))),
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
            trial.set_user_attr("model_config", model_config_to_dict(model_config))

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

        # manually handle trial prunining in loop
        trial_state = optuna.trial.TrialState.COMPLETE
        score = float("nan")
        try:
            score = _run_trial(params, trial)
        except optuna.TrialPruned:
            trial_state = optuna.trial.TrialState.PRUNED

        if rank == 0:
            # study.tell(trial, score)
            if trial_state == optuna.trial.TrialState.PRUNED:
                study.tell(trial, state=trial_state)
            else:
                study.tell(trial, score)

    if rank == 0:
        # get most optimal results consolidated in rank 0
        best = study.best_trial
        best_payload = {"study_name": args.study_name,
                        "best_value": float(best.value),
                        "best_params": dict(best.params),
                        "best_user_attrs": dict(best.user_attrs),
                        "metric": args.metric,
                        "model_head": str(args.model_head),
                        "split_strategy": str(args.split_strategy),
                        "split_report_json": str(split_report_json),
                        "threshold_policy": str(args.threshold_policy),
                        "threshold_min_precision": float(args.threshold_min_precision),
                        "threshold_min_recall": float(args.threshold_min_recall),
                        "threshold_fixed_value": float(args.threshold_fixed_value)}
        best_trial_json.write_text(json.dumps(best_payload, indent=2))

        best_trial_dir = Path(best.user_attrs["trial_dir"])
        src = next(
            (
                path for path in (
                    best_trial_dir / "best_model_by_score.pt",
                    best_trial_dir / "fully_connected_by_score.pt",
                )
                if path.exists()
            ),
            None,
        )
        if src is not None:
            best_ckpt_path.write_bytes(src.read_bytes())
        else:
            logger.warning(
                "best_checkpoint_missing",
                extra={"extra": {"trial_dir": str(best_trial_dir)}}
            )

        logger.info("best_results", extra={"extra": best_payload})

    if ddp is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
