"""ddp.py

Distributed training helpers for PepSeqPred.

Provides DDP initialization plus small utilities for rank-aware reductions and
gathering variable-length 1D tensors across processes.
"""

import os
from datetime import timedelta
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist


def init_ddp() -> Dict[str, Any] | None:
    """
    Initialize DDP if launched with srun. Sets timeout duration in minutes.

    Returns
    -------
        Dict[str, Any] | None
            Rank info dict with keys `rank`, `world_size`, and `local_rank` if DDP is enabled,
            otherwise `None`.
    """
    if "RANK" not in os.environ:
        return None

    timeout_min_raw = os.environ.get("PEPSEQPRED_DDP_TIMEOUT_MIN", "60")
    try:
        timeout_min = max(1, int(timeout_min_raw))
    except ValueError:
        timeout_min = 60  # minutes
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=timeout_min))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": local_rank
    }


def _ddp_enabled() -> bool:
    """Check if DDP is enabled for parallelism."""
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    """
    Return the rank of the current process, or 0 if DDP is not enabled.

    Returns
    -------
        int
            Rank of the current process.
    """
    return dist.get_rank() if _ddp_enabled() else 0


def _ddp_world() -> int:
    """Returns world size if DDP enabled, else 1."""
    return dist.get_world_size() if _ddp_enabled() else 1


def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """
    Sum-reduce a tensor across ranks if DDP is enabled.

    Parameters
    ----------
        t : torch.Tensor
            Tensor to reduce in-place.

    Returns
    -------
        torch.Tensor
            The reduced tensor (same object as input).
    """
    if _ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def ddp_gather_all_1d(t: torch.Tensor, device: torch.device) -> Tuple[List[torch.Tensor], List[int]]:
    """
    All-gather 1D tensor across all ranks with padding to max length.
    Returns a list of gathered tensors and the original sizes.

    Parameters
    ----------
        t : torch.Tensor
            1D tensor to gather across ranks.
        device : torch.device
            Device to allocate intermediate buffers on.

    Returns
    -------
        Tuple[List[torch.Tensor], List[int]]
        -----------------------------------
            A tuple of `(gathered, sizes)` where `gathered` is the list of padded tensors
            from each rank and `sizes` are the original lengths per rank.
    """
    if not _ddp_enabled():
        return [t], [int(t.numel())]

    sizes = torch.tensor([t.numel()], device=device, dtype=torch.long)
    size_list = [torch.zeros_like(sizes) for _ in range(_ddp_world())]
    dist.all_gather(size_list, sizes)
    sizes_int = [int(s.item()) for s in size_list]
    max_size = max(sizes_int) if sizes_int else int(t.numel())

    padded = torch.zeros(max_size, device=device, dtype=t.dtype)
    if t.numel() > 0:
        padded[:t.numel()] = t

    gathered = [torch.zeros_like(padded) for _ in range(_ddp_world())]
    dist.all_gather(gathered, padded)
    return gathered, sizes_int
