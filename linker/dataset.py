from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.serialization as ts
from torch.utils.data import Dataset

@dataclass
class PeptideDataset(Dataset):
    embeddings: torch.Tensor | List[torch.Tensor]
    targets: torch.Tensor | List[torch.Tensor]
    code_names: List[str]
    protein_ids: List[str]
    peptides: List[str]
    align_starts: List[int]
    align_stops: List[int]

    def __post_init__(self) -> None:
        # normalize embeddings to a single tensor (N, L, D)
        if isinstance(self.embeddings, list):
            self.embeddings = torch.stack(self.embeddings, dim=0)

        # normalize targets to a single tensor
        if isinstance(self.targets, list):
            self.targets = torch.stack(self.targets, dim=0)

    # ----- Dataset API -----
    def __len__(self) -> int:
        return self.targets.size(0)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.embeddings[idx] # (L, D)
        y = self.targets[idx] # (3,)
        return X, y
    
    # ----- Convenient props -----
    @property
    def n_samples(self) -> int:
        return self.targets.size(0)
    
    @property
    def peptide_len(self) -> int:
        if self.embeddings:
            return self.embeddings[0].size(1)
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {"embeddings": self.embeddings, 
                "targets": self.targets, 
                "code_names": self.code_names, 
                "protein_ids": self.protein_ids, 
                "peptides": self.peptides, 
                "align_starts": self.align_starts, 
                "align_stops": self.align_stops}
    
    def save(self, save_path: Path | str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "PeptideDataset":
        with ts.safe_globals([cls]):
            obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            return cls(**obj)
        
        raise TypeError(f"Unrecognized object in {path}. "
                        f"Expected dict payload, got {type(obj)}.")
