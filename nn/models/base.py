from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class PepSeqClassifierBase(nn.Module, ABC):
    """
    Base class for peptide-level epitope classifications in PepSeqPred. Contains the `forward` abstract
    method that must be overridden in all subclasses.

    Parameters
    ----------
        emb_dim : int
            The dimension of each embedding vector. Default is `1281` using the `esm2_t33_650M_UR50D` encoder.
        num_classes : int
            The number of output classes. Default is `3` where each class represents the probability of a peptide
            containing an epitope, uncertain about containing an epitope, and not containing an epitope.
    """
    def __init__(self, emb_dim: int = 1281, num_classes: int = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Abstract `forward` method to be implemented in any subclass."""
        raise NotImplementedError
