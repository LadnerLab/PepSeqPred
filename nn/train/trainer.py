from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda" # should only train using GPUs (but can be changed to "cpu")

class Trainer:
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: Optional[DataLoader] = None, 
                 config: TrainerConfig = TrainerConfig(), 
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config.learning_rate, 
                                          weight_decay=self.config.weight_decay)
        self.criterion = nn.CrossEntropyLoss(weight=self.config.weight_decay)

    def _batch_step(self, batch: torch.Tensor, train: bool = True) -> Dict[str, Any]:
        pass

    def _run_epoch(self, epoch: int, train: bool = True) -> Dict[str, float]:
        pass

    def fit(self, save_dir: Optional[Path | str] = None) -> None:
        pass

    def _save_checkpoints(self, path: Path | str, epoch: int) -> None:
        pass
