import torch
import torch.nn as nn
from .base import PepSeqClassifierBase
from typing import Iterable, Sequence

class PepSeqFFNN(PepSeqClassifierBase):
    """
    Implements PepSeqClassifierBase as a modular FFNN.

    Parameters
    ----------
        emb_dim : int
            The dimension of each embedding vector. Default is `1281` using the `esm2_t33_650M_UR50D` encoder.
        hidden_sizes : Sequence[int]
            A sequence of integers denoting the size of the hidden layers. This can be larger than 3 hidden layers.
        dropouts : Sequence[float]
            A sequence containing the dropout rates at each layer. This can be larger than 3 dropouts but must be the same size as hidden_sizes.
        num_classes : int
            The number of output classes. Default is `3` where each class represents the probability of a peptide
            containing an epitope, uncertain about containing an epitope, and not containing an epitope.
    """
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
        """
        Forward pass for peptide-level epitope classification.

        Parameters
        ----------
            X : Tensor
                Input tensor of shape (B, L, E), where B is the batch size, 
                L is the peptide length, and E is the embedding dimension.

        Returns
        -------
            Tensor
                Logits of shape (B, C), where C is the number of output classes (usually 3).

        Notes
        -----
        This model performs mean pooling acorss the sequence length dimension 
        to convert residue-level embeddings into a single fixed size peptide 
        representation. This representation is then passed through this 
        feed-forward neural network to produce class logits. Predictions are 
        made at the peptide level rather than at the individual residue level.
        """
        if X.dim() != 3:
            raise ValueError(f"Expected shape (B, L, E), got {X.shape}")
        
        if X.size(-1) != self.emb_dim:
            raise ValueError(f"Expected emb_dim {self.emb_dim}, got {X.size(-1)}")
        
        # predicting peptide contains epitope, not if each residue is an epitope
        pooled = X.mean(dim=1) # mean pool: (B, L, E) --> (B, E)
        logits = self.ff_model(pooled)
        return logits
