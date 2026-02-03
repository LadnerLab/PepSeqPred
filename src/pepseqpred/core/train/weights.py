from typing import Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


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


def global_pos_weight(local_pos: int, local_neg: int, ddp: Dict[str, Any] | None) -> float:
    """Computes the global neg/pos across ranks if DDP is running."""
    if ddp is None:
        return float(local_neg / max(local_pos, 1))
    t = torch.tensor([local_pos, local_neg], device=torch.device("cuda"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    pos = int(t[0].item())
    neg = int(t[1].item())
    return float(neg / max(pos, 1))
