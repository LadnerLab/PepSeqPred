"""seed.py

Seeding utilities for deterministic training runs.

Provides a helper to seed Python, NumPy, and PyTorch ranges.
"""

import random
import torch
import numpy as np


def set_all_seeds(seed: int) -> None:
    """
    Set seed values for all random number generators.

    Parameters
    ----------
        seed : int
            Seed value used for Python, NumPy, and PyTorch RNGs.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
