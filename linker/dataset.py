from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.serialization as ts

@dataclass
class PeptideDataset:
    embeddings: torch.Tensor | List[torch.Tensor]
    targets: List[torch.Tensor]
    code_names: List[str]
    protein_ids: List[str]
    peptides: List[str]
    align_starts: List[int]
    align_stops: List[int]

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
        torch.save(self, save_path)

    @classmethod
    def load(cls, path: Path | str) -> "PeptideDataset":
        with ts.safe_globals([cls]):
            obj = torch.load(path, map_location="cpu")
        
        if not isinstance(obj, cls):
            raise TypeError(f"Object in {path} is not of type {cls.__name__}")
        return obj
