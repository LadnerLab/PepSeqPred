from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class PepSeqClassifierBase(nn.Module, ABC):
    """
    Base class for peptide-level epitope classifications in PepSeqPred. Contains the `forward` abstract
    method that must be overridden in all subclasses.
    """

    def __init__(self, emb_dim: int = 1281, num_classes: int = 1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Abstract `forward` method to be implemented in any subclass."""
        raise NotImplementedError
