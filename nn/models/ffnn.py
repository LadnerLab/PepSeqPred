import torch
import torch.nn as nn
from .base import PepSeqClassifierBase
from typing import Iterable, Sequence

class PepSeqFFNN(PepSeqClassifierBase):
    def __init__(self, emb_dim: int = 1281, 
                 hidden_sizes: Sequence[int] = (150, 120, 45), 
                 dropouts: Sequence[float] = (0.2, 0.2, 0.2), 
                 num_classes: int = 3):
        super().__init__(emb_dim=emb_dim, num_classes=num_classes)

        if len(hidden_sizes) != len(dropouts):
            raise ValueError("hidden_sizes and dropouts must be the same size")

        layers: Iterable[nn.Module] = []
        in_features = emb_dim

        # arbitrary depth FFNN (default is 3 layers)
        for hidden_size, p in zip(hidden_sizes, dropouts):
            layers += [
                nn.Linear(in_features, hidden_size), 
                nn.ReLU(), 
                nn.Dropout(p)
            ]
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))

        self.ff_model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(f"Expected shape (B, L, E), got {X.shape}")
        
        if X.size(-1) != self.emb_dim:
            raise ValueError(f"Expected emb_dim {self.emb_dim}, got {X.size(-1)}")
        
        # predicting peptide contains epitope, not if each residue is an epitope
        pooled = X.mean(dim=1) # mean pool: (B, L, E) --> (B, E)
        logits = self.ff_model(pooled)
        return logits
