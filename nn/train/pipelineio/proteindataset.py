from pathlib import Path
from typing import List, Dict, Iterator, Iterable, Tuple, Sequence, Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from torch.utils.data import IterableDataset


def _build_embedding_index(embedding_dirs: List[Path]) -> Dict[str, Path]:
    """Builds protein_id --> embedding_path from multiple shard directories."""
    index: Dict[str, Path] = {}
    for emb_dir in embedding_dirs:
        for pt_path in emb_dir.glob("*.pt"):
            index[pt_path.stem] = pt_path
    return index


def _build_label_index(label_shards: List[Path | str]) -> Dict[str, Path]:
    """Builds protein_id --> shard_path index by scanning 'labels' keys."""
    index: Dict[str, Path] = {}
    for shard_path in label_shards:
        shard_path = Path(shard_path)
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(shard, dict) or "labels" not in shard:
            raise TypeError(
                f"Label shard {shard_path} must be a dict with 'labels' key, not type {type(shard)}")
        labels = shard["labels"]
        if not isinstance(labels, dict):
            raise TypeError(
                f"'labels' in {shard_path} must be a dict, not type {type(labels)}")
        for protein_id in labels.keys():
            index[str(protein_id)] = Path(shard_path)
        del shard
    return index


def _iter_windows(length: int, window_size: Optional[int] = 1000, stride: Optional[int] = 900) -> Iterable[Tuple[int, int]]:
    """Generates iterable of stop and end positions for long protein sequences."""
    if window_size is None or window_size <= 0:
        yield 0, length
        return
    if stride is None or stride <= 0:
        raise ValueError("stride must be > 0")

    start = 0
    while start < length:
        end = min(start + window_size, length)
        yield start, end
        if end == length:
            break
        start += stride


def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads the embeddings, labels, and masks with default values to meet minimum Tensor size requirement.

    Parameters
    ----------
        batch : Tuple[Tensor, Tensor, Tensor]
            xs : Tensor
                The embeddings per batch.
            ys : Tensor
                The labels per batch.
            masks : Tensor
                The masks per batch.
    """
    xs, ys, masks = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=0.0)
    m_pad = pad_sequence(masks, batch_first=True, padding_value=0)
    return x_pad, y_pad, m_pad


class ProteinDataset(IterableDataset):
    """
    Streaming dataset that yields residue-level training examples from full proteins.

    This dataset is designed for large scale embedding collections where each protein embedding
    is stored as its own .pt file, and labels are stored in one or more shard files.

    Data layout assumptions
    -----------------------
    Embeddings:
        One .pt file per protein, stored under one or more directories.
        The filename stem must equal protein_id.
        File contents should be a tensor of shape (L, D).

    Labels:
        One or more shard files.
        Each shard must be a dict with a "labels" key.
        shard["labels"] must be a dict mapping protein_id to a label tensor.
        Labels can be either:
            shape (L,) for binary labels
            shape (L, 3) for [def epitope, uncertain, not epitope]

    What this dataset yields
    ------------------------
    Each yielded sample corresponds to either:
        the full protein (when window_size is None or <= 0),
        or a window into the protein (when window_size is provided).

    The dataset also produces a mask tensor that can be used to ignore:
        uncertain residues (when labels are (L, 3)),
        padded residues (when pad_last_window is enabled),
        any other residues you want to exclude downstream via masking.

    Parameters
    ----------
    embedding_dirs : List[Path or str]
        Directories containing per protein embedding .pt files.
    label_shards : List[Path or str]
        Paths to label shard .pt files.
    protein_ids : Sequence[str] or None
        Optional subset of protein IDs to iterate. If None, uses the intersection
        of IDs present in embedding index and label index.
    label_index : Dict[str, Path] or None
        Optional precomputed protein_id to shard_path index.
        Passing this avoids scanning all label shards at initialization.
    embedding_index : Dict[str, Path] or None
        Optional precomputed protein_id to embedding_path index.
        Passing this avoids scanning embedding directories at initialization.
    window_size : int or None
        If provided, yields windows of this length from each protein.
        If None, yields the full protein tensor.
    stride : int or None
        Step size between windows. Only used when window_size is provided.
    collapse_labels : bool
        If True and labels are (L, 3), collapse to binary y = def epitope column.
        If False, keep full label vectors.
    pad_last_window : bool
        If True and windowing is enabled, pads the final short window to window_size.
        Padding is reflected in the returned mask (0 on padded positions).
    return_meta : bool
        If True, yields (X, y, mask, protein_id, start, end) for each window.
        If False, yields (X, y, mask).
    delete_embedding_files : bool
        If True, intended to support cleanup after reading.
    cache_current_label_shard : bool
        If True, keeps the currently used label shard loaded on CPU and only reloads
        when a protein moves to a different shard. This can reduce IO overhead.
    drop_label_after_use : bool
        If True, intended to support deleting labels from the cached shard after use.

    Notes
    -----
    This is an IterableDataset, so random access and __len__ are not defined.
    Use DataLoader with num_workers as appropriate for your IO throughput.
    """

    def __init__(
        self,
        embedding_dirs: List[Path | str],
        label_shards: List[Path | str],
        protein_ids: Optional[Sequence[str]] = None,
        label_index: Optional[Dict[str, Path]] = None,
        embedding_index: Optional[Dict[str, Path]] = None,
        window_size: Optional[int] = 1000,
        stride: Optional[int] = 900,
        collapse_labels: bool = True,
        pad_last_window: bool = True,
        return_meta: bool = False,
        delete_embedding_files: bool = False,
        cache_current_label_shard: bool = True,
        drop_label_after_use: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dirs = [Path(p) for p in embedding_dirs]
        self.label_shards = [Path(p) for p in label_shards]
        self.window_size = window_size
        self.stride = stride
        self.collapse_labels = collapse_labels
        self.pad_last_window = pad_last_window
        self.return_meta = return_meta
        self.delete_embedding_files = delete_embedding_files
        self.cache_current_label_shard = cache_current_label_shard
        self.drop_label_after_use = drop_label_after_use

        # ensure embedding index exists for fast lookup
        if embedding_index is None:
            self.embedding_index = _build_embedding_index(self.embedding_dirs)
        else:
            self.embedding_index = {str(k): Path(v)
                                    for k, v in embedding_index.items()}

        # ensure label index exists for fast lookup
        if label_index is None:
            self.label_index = _build_label_index(self.label_shards)
        else:
            self.label_index = {str(k): Path(v)
                                for k, v in label_index.items()}

        if protein_ids is None:
            # intersect embeddings and labels
            emb_ids = set(self.embedding_index.keys())
            lbl_ids = set(self.label_index.keys())
            self.protein_ids = sorted(emb_ids & lbl_ids)
        else:
            self.protein_ids = protein_ids

    def __iter__(self) -> Iterator:
        current_shard_path: Optional[Path] = None
        current_payload: Optional[Dict[str, torch.Tensor]] = None
        current_labels: Optional[Dict[str, torch.Tensor]] = None

        for protein_id in self.protein_ids:
            emb_path = self.embedding_index.get(protein_id)
            shard_path = self.label_index.get(protein_id)
            if emb_path is None or shard_path is None:
                continue

            if not self.cache_current_label_shard or (shard_path != current_shard_path):
                current_payload = torch.load(
                    shard_path, map_location="cpu", weights_only=False)
                if not isinstance(current_payload, dict) or "labels" not in current_payload:
                    raise TypeError(
                        f"Label shard {shard_path} must be a dict with 'labels' key")
                current_labels = current_payload["labels"]
                if not isinstance(current_labels, dict):
                    raise TypeError(
                        f"'labels' in {shard_path} must be a dict, not type {type(current_labels)}")
                current_shard_path = shard_path

            if current_labels is None or str(protein_id) not in current_labels:
                continue

            X = torch.load(emb_path, map_location="cpu", weights_only=False)
            y_full = current_labels[str(protein_id)]

            if X.size(0) != y_full.size(0):
                raise ValueError(
                    f"Embeddings and labels must be the same size at dim 0, instead got X.size(0)={X.size(0)} and y_full.size(0)={y_full.size(0)}")

            # build uncertain residue exclusion mask
            if y_full.dim() == 2 and y_full.size(1) == 3:
                # columns: [def epitope, uncertain, not epitope]
                def_col = y_full[:, 0].float()
                unc_col = y_full[:, 1].float()
                # not_col = y_full[:, 2].float() (not needed)

                # mask residues that are not uncertain
                mask_valid = (unc_col == 0).long()

                # collapse to binary when requested
                if self.collapse_labels:
                    y = def_col
                else:
                    y = y_full.float()
            else:
                # no uncertainty, can treat all residues the same
                y = y_full.float()
                mask_valid = torch.ones(y.size(0), dtype=torch.long)

            L = int(X.size(0))
            for start, end in _iter_windows(L, self.window_size, self.stride):
                Xw = X[start:end]
                yw = y[start:end]
                mv = mask_valid[start:end]

                if self.pad_last_window and self.window_size and (end - start) < self.window_size:
                    pad_len = self.window_size - (end - start)
                    if Xw.dim() == 2:
                        Xw = pad(Xw, (0, 0, 0, pad_len))
                    else:
                        Xw = pad(Xw, (0, pad_len))
                    # pad valid labels and mask
                    yw = pad(yw, (0, pad_len))
                    mv = pad(mv, (0, pad_len))

                    # 1 for real residues, 0 for pad
                    pad_mask = torch.cat(
                        [torch.ones(end - start), torch.zeros(pad_len)]).long()
                else:
                    pad_mask = torch.ones(end - start).long()

                # final mask excludes uncertain and padding
                mask = (mv * pad_mask).long()

                if self.return_meta:
                    yield Xw, yw, mask, protein_id, start, end
                else:
                    yield Xw, yw, mask

            # release large tensors, labels, and embeddings to keep memory usage low
            del X
            del y_full
            del y
            del mask_valid

            if self.drop_label_after_use and current_labels is not None:
                current_labels.pop(str(protein_id), None)

            if self.delete_embedding_files:
                try:
                    emb_path.unlink()
                except OSError:
                    pass
