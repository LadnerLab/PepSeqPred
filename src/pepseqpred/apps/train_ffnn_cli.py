"""train_ffnn_cli.py

Handles end-to-end training and evaluation of a PepSeqPredFFNN with the goal to predict the locations
of antibody epitopes within a protein sequence downstream. The resulting model will make binary predictions:
definite epitope or not epitope, it handles residues labeled uncertain through a masking process.

The training module utilizes DistributedDataParallel to train the model distributed across multiple GPUs.
It is highly recommended you train this model using an HPC, training locally is often impossible due to time 
constraints and limited compute. For context, our models were often trained using 4 A100 GPUs and take anywhere
from 3 to 5 hours.

Usage
-----
>>> # from scripts/hpc/trainffnn.sh (see shell script for CLI config)
>>> sbatch trainffnn.sh /path/to/emb_shard_dir0 ... /path/to/emb_shard_dirN -- \\ 
                        /path/to/label_shard0.pt ... /path/to/label_shardN.pt
"""

import argparse
from pathlib import Path
from typing import Dict
import random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.data.proteindataset import ProteinDataset, pad_collate
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.train.trainer import Trainer, TrainerConfig
from pepseqpred.core.train.ddp import init_ddp
from pepseqpred.core.train.split import (
    split_ids,
    split_ids_grouped,
    partition_ids_weighted,
    sort_ids_for_locality
)
from pepseqpred.core.train.weights import compute_pos_neg_counts, global_pos_weight
from pepseqpred.core.train.embedding import infer_emb_dim
from pepseqpred.core.train.seed import set_all_seeds


def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Train FFNN program."""
    parser = argparse.ArgumentParser(
        description="Train PepSeqPred FFNN on protein ESM-2 embeddings for binary residue-level epitope prediction.")
    parser.add_argument("--embedding-dirs",
                        nargs="+",
                        required=True,
                        type=Path,
                        help="One or more directories containing per-protein embeddings")
    parser.add_argument("--label-shards",
                        nargs="+",
                        required=True,
                        type=Path,
                        help="One or more label shard .pt files containing a 'labels' dictionary")
    parser.add_argument("--epochs",
                        action="store",
                        dest="epochs",
                        type=int,
                        default=10,
                        help="Number of epochs (training cycles)")
    parser.add_argument("--seed",
                        action="store",
                        dest="seed",
                        type=int,
                        default=42,
                        help="Seed for random split reproducibility")
    parser.add_argument("--batch-size",
                        action="store",
                        dest="batch_size",
                        type=int,
                        default=64,
                        help="Training batch size per epoch")
    parser.add_argument("--lr",
                        action="store",
                        dest="lr",
                        type=float,
                        default=1e-3,
                        help="Model learning rate")
    parser.add_argument("--wd",
                        action="store",
                        dest="weight_decay",
                        type=float,
                        default=0.0,
                        help="Model training weight decay to prevent overfitting by shrinking model weights during training")
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
                        action="store",
                        dest="save_path",
                        type=Path,
                        default=Path("checkpoints/ffnn_v1"),
                        help="Path to save the best model to")
    parser.add_argument("--val-frac",
                        action="store",
                        dest="val_frac",
                        type=float,
                        default=0.2,
                        help="Fraction of input data used for validation")
    parser.add_argument("--subset",
                        action="store",
                        dest="subset",
                        type=int,
                        default=0,
                        help="If > 0, use only first N proteins from dataset")
    parser.add_argument("--split-type",
                        type=str,
                        default="id-family",
                        choices=["id", "id-family"],
                        help="Data partition type, use ID only or ID and taxonomic family.")
    parser.add_argument("--num-workers",
                        action="store",
                        dest="num_workers",
                        type=int,
                        default=0,
                        help="Number of worker threads for data loading")
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
    logger = setup_logger(json_lines=True,
                          json_indent=2,
                          name="train_ffnn_cli")

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

    # set random number seeds
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

    # parition data by ID + family or just ID
    if args.split_type == "id-family":
        family_groups: Dict[str, str] = {}
        missing_family_ids = 0

        for protein_id in protein_ids:
            family = base_dataset.embedding_family_by_id.get(protein_id)
            if family is None or str(family).strip() == "":
                # singleton group when family missing fallback
                family_groups[protein_id] = f"__missing_family__:{protein_id}"
                missing_family_ids += 1
            else:
                family_groups[protein_id] = str(family)

        train_ids_all, val_ids_all = split_ids_grouped(
            protein_ids, args.val_frac, seed, family_groups
        )

        train_families = {family_groups[pid] for pid in train_ids_all}
        val_families = {family_groups[pid] for pid in val_ids_all}
        overlap = train_families & val_families
        if overlap:
            raise RuntimeError(
                f"Family leakage detected for split_type='id-family': n_overlap={len(overlap)}"
            )

        if rank == 0:
            logger.info("family_split_summary",
                        extra={"extra": {
                            "split_type": args.split_type,
                            "train_ids": len(train_ids_all),
                            "val_ids": len(val_ids_all),
                            "val_families": len(val_families),
                            "missing_family_ids": missing_family_ids
                        }})
    else:
        train_ids_all, val_ids_all = split_ids(
            protein_ids, args.val_frac, seed
        )

    if len(train_ids_all) == 0:
        raise ValueError("Global split produced 0 train IDs")

    # estimate relative workload without tensor I/O by using embedding file size.
    id_weights: Dict[str, float] = {}
    for protein_id in protein_ids:
        emb_path = base_dataset.embedding_index.get(protein_id)
        if emb_path is None:
            id_weights[protein_id] = 1.0
            continue
        try:
            id_weights[protein_id] = float(max(1, emb_path.stat().st_size))
        except OSError:
            id_weights[protein_id] = 1.0

    id_groups: Dict[str, str] = {
        protein_id: str(base_dataset.label_index.get(protein_id, ""))
        for protein_id in protein_ids
    }

    if ddp is None:
        train_ids = sort_ids_for_locality(train_ids_all, id_groups)
        val_ids = sort_ids_for_locality(val_ids_all, id_groups)
    else:
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

        obj = [payload]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]

        train_ids = list(payload["train_ids_by_rank"][rank])
        val_ids = list(payload["val_ids_by_rank"][rank])

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

        if any(int(x["train_ids"]) == 0 for x in per_rank):
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
            window_size=args.window_size if args.window_size > 0 else None,
            stride=args.stride,
            collapse_labels=args.collapse_labels,
            pad_last_window=args.pad_last_window,
            return_meta=False,
            cache_current_label_shard=args.cache_current_label_shard,
            drop_label_after_use=args.drop_label_after_use
        )

    # set up data loaders
    pin = torch.cuda.is_available()  # pin memory depending on if CUDA available
    # handle loader worker creation in multi-rank process
    loader_kwargs = {"batch_size": args.batch_size,
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

    # compute or store positive weight (like class weight)
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = float(args.pos_weight)
    elif args.calc_pos_weight:
        pos, neg = compute_pos_neg_counts(train_loader)
        pos_weight = global_pos_weight(pos, neg, ddp)

    # build our FFNN model
    emb_dim = infer_emb_dim(base_dataset.embedding_index)
    model = PepSeqFFNN(emb_dim=emb_dim,
                       hidden_sizes=(150, 120, 45),
                       dropouts=(0.1, 0.1, 0.1),
                       use_layer_norm=False,
                       use_residual=False,
                       num_classes=1)

    if ddp is not None:
        device = torch.device(f"cuda:{ddp['local_rank']}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if ddp is not None:
        model = DDP(model, device_ids=[
                    ddp["local_rank"]], output_device=ddp["local_rank"])

    # setup config and train
    config = TrainerConfig(epochs=args.epochs,
                           batch_size=args.batch_size,
                           learning_rate=args.lr,
                           weight_decay=args.weight_decay,
                           device="cuda" if torch.cuda.is_available() else "cpu",
                           pos_weight=pos_weight)
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      logger=logger,
                      val_loader=val_loader,
                      config=config)

    # run training, only save if rank 0 or single rank run
    save_dir = args.save_path if (ddp is None or rank == 0) else None
    trainer.fit(save_dir=save_dir)

    if ddp is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
