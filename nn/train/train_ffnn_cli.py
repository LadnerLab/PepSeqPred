import argparse
from pathlib import Path
import os
import random
from typing import Dict, List, Tuple, Any
import numpy as np
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

    # set random number seeds
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

    # set up data loaders
    pin = torch.cuda.is_available()  # pin memory depending on if CUDA available
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=pin,
                              collate_fn=pad_collate)
    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(val_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=pin,
                                collate_fn=pad_collate)

    # compute or store positive weight (like class weight)
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = float(args.pos_weight)
    elif args.calc_pos_weight:
        pos, neg = compute_pos_neg_counts(train_loader)
        pos_weight = _global_pos_weight(pos, neg, ddp)

    # build our FFNN model
    emb_dim = _infer_emb_dim(base_dataset.embedding_index)
    model = PepSeqFFNN(emb_dim=emb_dim,
                       hidden_sizes=(150, 120, 45),
                       dropouts=(0.1, 0.1, 0.1),
                       use_layer_norm=True,
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
