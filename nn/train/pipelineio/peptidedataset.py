from dataclasses import dataclass
from pathlib import Path
import contextlib
from typing import List, Dict, Any, Tuple
import torch
import torch.serialization as ts
from torch.utils.data import Dataset

@dataclass
class PeptideDataset(Dataset):
    """
    PyTorch Dataset for peptide level epitope classification.

    This dataset stores fixed length peptide embeddings along with
    associated metadata and classification targets. Each sample
    corresponds to a single peptide extracted from a parent protein
    and is labeled for epitope presence at the peptide level.

    Parameters
    ----------
        embeddings : Tensor or List[Tensor]
            Peptide embeddings of shape (N, L, D) or a list of tensors
            each of shape (L, D), where N is the number of peptides,
            L is the peptide length, and D is the embedding dimension.
        targets : Tensor or List[Tensor]
            Target labels of shape (N, C) or a list of tensors of shape (C,),
            where C is the number of output classes.
        code_names : List[str]
            Unique identifiers for each peptide instance.
        protein_ids : List[str]
            Identifiers of the parent proteins from which peptides were derived.
        peptides : List[str]
            Amino acid sequences of each peptide.
        align_starts : List[int]
            Start indices of each peptide within the parent protein sequence.
        align_stops : List[int]
            Stop indices of each peptide within the parent protein sequence.
    """
    embeddings: torch.Tensor | List[torch.Tensor]
    targets: torch.Tensor | List[torch.Tensor] # peptide level targets
    code_names: List[str]
    protein_ids: List[str]
    peptides: List[str]
    align_starts: List[int]
    align_stops: List[int]

    def __post_init__(self) -> None:
        """Normalize input data into tensor form."""
        # normalize embeddings to a single tensor (N, L, D)
        if isinstance(self.embeddings, list):
            self.embeddings = torch.stack(self.embeddings, dim=0)

        # normalize targets to a single tensor
        if isinstance(self.targets, list):
            self.targets = torch.stack(self.targets, dim=0)

    # ----- Dataset API -----
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.targets.size(0)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single embeddings sample and its target label."""
        X = self.embeddings[idx] # (L, D)
        y_pep = self.targets[idx] # (3,)
        y_res = y_pep.unsqueeze(0).repeat(X.size(0), 1) # (L, 3)
        return X, y_res
    
    # ----- Convenient props -----
    @property
    def n_samples(self) -> int:
        """Number of peptide samples in the dataset."""
        return self.targets.size(0)
    
    @property
    def peptide_len(self) -> int:
        """Length of each peptide sequence."""
        if self.embeddings:
            return self.embeddings.size(1)
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataset into a serializable dictionary."""
        return {"embeddings": self.embeddings, 
                "targets": self.targets, 
                "code_names": self.code_names, 
                "protein_ids": self.protein_ids, 
                "peptides": self.peptides, 
                "align_starts": self.align_starts, 
                "align_stops": self.align_stops}
    
    def save(self, save_path: Path | str) -> None:
        """Saves the dataset to disk."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "PeptideDataset":
        """Load a PeptideDataset from disk."""
        if hasattr(ts, "safe_globals"):
            ctx = ts.safe_globals([cls])
        else:
            if hasattr(ts, "add_safe_globals"):
                ts.add_safe_globals([cls])
            ctx = contextlib.nullcontext()

        with ctx:
            obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            return cls(**obj)
        
        raise TypeError(f"Unrecognized object in {path}. "
                        f"Expected dict payload, got {type(obj)}.")
