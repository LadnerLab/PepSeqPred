from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn

class PepSeqClassifierBase(nn.Module, ABC):
    def __init__(self, emb_dim: int = 1281, num_classes: int = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError
    
    def predict_proba(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(X, mask=mask)
        return torch.softmax(logits, dim=-1)
