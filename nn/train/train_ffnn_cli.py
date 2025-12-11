import json
import argparse
from pathlib import Path
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from linker.dataset import PeptideDataset
from nn.models.ffnn import PepSeqFFNN
from nn.train.trainer import Trainer, TrainerConfig

def setup_logger(log_level: str = "INFO", json_lines: bool = False) -> logging.Logger:
    """
    Creates and sets up a configured logger for this CLI.

    Parameters
    ----------
        log_level : str
            Minimum level for logs. Default is "INFO".
        json_lines : bool
            When True, formats logs as a JSON object. Default is False (`logging` library default format).

    Returns
    -------
        logging.Logger
            Logger named `train_ffnn_cli` with a stream handler attached.
    """
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            payload = {"timestamp": datetime.now().isoformat(), 
                       "level": record.levelname, 
                       "message": record.getMessage(), 
                       "logger": record.name, 
                       "where": f"{record.pathname}:{record.lineno}"}
            
            # add all detailed logs using "extra" kwargs
            if hasattr(record, "extra") and isinstance(record.extra, dict):
                payload.update(record.extra)
            
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2)
    
    # create named logger and reset any inherited handlers to avoid duplication
    logger = logging.getLogger("train_ffnn_cli")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers[:] = [] # avoid duplicate handlers

    # choose formatter style
    stream_formatter = JSONFormatter() if json_lines else logging.Formatter("%(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PepSeqPred FFNN on peptide-level ESM-2 embeddings to predict likelihood of peptide containing vs not containing vs uncertain about containing epitopes across the infectome.")
    parser.add_argument("input_data", 
                        type=Path, 
                        help="Path to .pt file containing PeptideDataset")
    parser.add_argument("--epochs", 
                        action="store", 
                        dest="epochs", 
                        type=int, 
                        default=10, 
                        help="Number of epochs (training cycles)")
    parser.add_argument("--batch-size", 
                        action="store", 
                        dest="batch_size", 
                        type=int, 
                        default=64, 
                        help="Training batch size per epoch")
    parser.add_argument("--lr", 
                        action="store", 
                        dest="lr", 
                        type=float, 
                        default=1e-3, 
                        help="Model learning rate")
    parser.add_argument("--weight-decay", 
                        action="store", 
                        dest="weight_decay", 
                        type=float, 
                        default=0.0, 
                        help="Model training weight decay to prevent overfitting by shrinking model weights during training")
    parser.add_argument("--save-path", 
                        action="store", 
                        dest="save_path", 
                        type=Path, 
                        default=Path("checkpoints/ffnn_v1"))
    parser.add_argument("--val-frac", 
                        action="store", 
                        dest="val_frac", 
                        type=float, 
                        default=0.2, 
                        help="Fraction of input data used for validation")

    args = parser.parse_args()
    logger = setup_logger(json_lines=True)

    data = PeptideDataset.load(args.input_data)

    # split data into training and val sets
    n_total = len(data)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_data, val_data = random_split(data, [n_train, n_val])

    # set up data loaders
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=0, 
                              pin_memory=True)
    val_loader = DataLoader(val_loader, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=0, 
                            pin_memory=True)
    
    # build our FFNN model
    emb_dim = data.embeddings.size(-1)
    model = PepSeqFFNN(emb_dim=emb_dim, num_classes=3)

    # setup config and train
    config = TrainerConfig(epochs=args.epochs, 
                           batch_size=args.batch_size, 
                           learning_rate=args.lr, 
                           weight_decay=args.weight_decay, 
                           device="cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model=model, 
                      train_loader=train_loader, 
                      logger=logger, 
                      val_loader=val_loader, 
                      config=config, 
                      class_weights=None) # change to tensor of shape (3,) if we want to weight each class
    
    # run training
    trainer.fit(save_dir=args.save_path)

if __name__ == "__main__":
    main()
