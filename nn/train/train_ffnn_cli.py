"""train_ffnn_cli.py

This module is designed to train a FFNN to predict if a residue from a peptide is likely to be or not be an epitope.

***Usage TBD
"""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pipelineio.logger import setup_logger
from nn.train.pipelineio.proteindataset import PeptideDataset
from nn.models.ffnn import PepSeqFFNN
from nn.train.trainer import Trainer, TrainerConfig


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
            The ratio of uncertain + not epitopes vs. definite epitopes.
            If no bias is present (i.e., pos = neg = 0), 1.0 is returned
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


def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Train FFNN program."""
    parser = argparse.ArgumentParser(
        description="Train PepSeqPred FFNN on peptide-level ESM-2 embeddings for binary residue-level epitope prediction. Class 0, 1, and 2 from legacy labeling are collapsed into epitope vs not epitope.")
    parser.add_argument("input_data",
                        type=Path,
                        help="Path to .pt file containing PeptideDataset")
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
    parser.add_argument("--use-pos-weight",
                        dest="use_pos_weight",
                        action="store_true",
                        help="Use positive weight to handle positive vs. negative class imbalances.")
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
                        help="If > 0, use only first N samples from dataset")
    parser.add_argument("--num-workers",
                        action="store",
                        dest="num_workers",
                        type=int,
                        default=0,
                        help="Number of worker threads for data loading")

    args = parser.parse_args()
    logger = setup_logger(json_lines=True,
                          json_indent=2,
                          name="train_ffnn_cli")

    # set random number seeds
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

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
    gen = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(data, [n_train, n_val], generator=gen)

    # set up data loaders
    pin = torch.cuda.is_available()  # pin memory depending on if CUDA available
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=pin)
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=pin)

    # compute positive weight (like class weight)
    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(train_loader)

    # build our FFNN model
    emb_dim = data.embeddings.size(-1)
    model = PepSeqFFNN(emb_dim=emb_dim,
                       hidden_sizes=(128, 128, 128, 128),
                       dropouts=(0.1, 0.1, 0.1, 0.1),
                       use_layer_norm=True,
                       use_residual=True,
                       num_classes=1)

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

    # run training
    trainer.fit(save_dir=args.save_path)


if __name__ == "__main__":
    main()
