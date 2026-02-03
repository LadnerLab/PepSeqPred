import random
import torch
import numpy as np


def set_all_seeds(seed: int) -> None:
    """Sets seed value for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
