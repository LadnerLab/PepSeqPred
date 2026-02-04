"""ddp.py

Distributed training helpers for PepSeqPred.

Provides DDP initialization plus small utilities for rank-aware reductions and
gathering variable-length 1D tensors across processes.
"""

import os
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist


def init_ddp() -> Dict[str, Any] | None:
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


def _ddp_enabled() -> bool:
    """Check if DDP is enabled for parallelism."""
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    """Returns rank of current process, else 0 if DDP not enabled."""
    return dist.get_rank() if _ddp_enabled() else 0


def _ddp_world() -> int:
    """Returns world size if DDP enabled, else 1."""
    return dist.get_world_size() if _ddp_enabled() else 1


def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """Returns a reduced sum of the tensor if DDP enabled."""
    if _ddp_enabled():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def ddp_gather_all_1d(t: torch.Tensor, device: torch.device) -> Tuple[List[torch.Tensor], List[int]]:
    """
    All-gather 1D tensor across all ranks with padding to max length.
    Returns a list of gathered tensors and the original sizes.
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
