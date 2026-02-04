"""esm2.py

ESM-2 embedding utilities for PepSeqPred.

This module normalizes protein sequences, batches inputs by a token budget to
reduce padding, and computes per-residue embeddings for both short and long
proteins (using sliding windows for long sequences). It also appends sequence
length as a feature and writes per-sequence `.pt` embeddings plus a CSV index
describing stored artifacts.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Iterable, Iterator, List, Tuple
import esm
import torch
import numpy as np
import pandas as pd


def clean_seq(seq: str) -> str:
    """
    Normalize a protein sequence and drop any odd characters.

    Parameters
    ----------
        seq : str
            Raw sequence as a string.

    Returns
    -------
        str
            Cleaned sequence without any odd characters or whitespace.
    """
    seq = (seq or "").upper().strip()
    allowed = set("ACDEFGHIKLMNPQRSTVWYBZXUO")
    return "".join([aa for aa in seq if aa in allowed])


def token_packed_batches(pairs: Iterable[Tuple[str, str]],
                         max_tokens: int,
                         max_tokens_per_seq: int = 1022) -> Iterator[List[Tuple[str, str]]]:
    """
    Yields batches of (id, seq) pairs such that the sum of per sequence token counts in each batch stays
    under a token budget. This reduces padding and keeps the model efficiently filled when sequence lengths
    vary. 

    Parameters
    ----------
        pairs : Iterable[Tuple[str, str]]
            Iterable of (id, seq) pairs. For best efficiency, ensure these are sorted by ascending sequence
            length before calling.
        max_tokens : int
            Total token budget for one batch. 1022 multiplied by the batch size is a good choice for ESM-2.
        max_tokens_per_seq : int
            Maximum tokens used per sequence. Default is 1022.

    Yields
    ------
        Iterator[List[Tuple[str, str]]]
            An iterator of a list of (id, seq) pairs whose items combined token count is within the budget.

    Notes
    -----
    The token budget should reflect the model limit minus special tokens. For ESM2, the effective residue
    budget is typically 1022 per sequence. When using windowed encoding for long sequences, this function 
    still governs batching of the original full sequences.
    """
    batch = []
    used = 0
    for pid, seq in pairs:
        # tokens we expect to use for this sequence
        need = min(len(seq), max_tokens_per_seq)

        # if adding seq exceeds budget and batch not empty, yield current batch and start new one
        if used and (used + need) > max_tokens:
            yield batch
            batch, used = [], 0

        # add to batch and update total
        batch.append((pid, seq))
        used += need

    # flush out remaining items
    if batch:
        yield batch


def compute_window_embedding(token: torch.Tensor,
                             model: torch.nn.Module,
                             layer: int,
                             device: str,
                             window_size: int = 1000,
                             stride: int = 900) -> torch.Tensor:
    """
    Compute per residue ESM representations for a long sequence using sliding windows.

    Parameters
    ----------
        token : torch.Tensor
            Tokenized sequence including CLS and EOS, shape 1 x T.
        model : torch.nn.Module
            ESM model in eval mode.
        layer : int
            Representation layer index to extract.
        device : str
            Device string `cpu` or `cuda`.
        window_size : int
            Number of residue tokens to include per window. Default is 1000.
        stride : int
            Step between window starts in residue tokens. Overlap equals `window_size` minus `stride`.
            Default is 900.

    Returns
    -------
        torch.Tensor
            An array of shape (L, D+1) where L is the number of residues (excluding CLS and EOS), and D+1 is
            the embedding dimension for the chosen layer plus an additional column for the protein sequence length.
    """
    seq_len = token.size(1) - 2  # remove CLS and EOS tokens

    if device.startswith("cuda"):
        token = token.to(device, non_blocking=True)
    cls_token = token[:, :1]
    eos_token = token[:, -1:]

    # determine embedding dimension (either simply or by running model)
    D = getattr(model, "embed_dim", None)
    if D is None:
        with torch.inference_mode():
            probe_end = min(window_size, seq_len)
            probe_tokens = torch.cat(
                [cls_token, token[:, 1:1 + probe_end], eos_token], dim=1)
            if device.startswith("cuda"):
                probe_tokens = probe_tokens.to(device, non_blocking=True)
            probe_out = model(probe_tokens, repr_layers=[
                              layer], return_contacts=False)
            D = int(probe_out["representations"][layer].shape[-1])
            del probe_out, probe_tokens

    # account for sums of parallel counter of contributions per position
    full = torch.zeros((seq_len, D), device="cpu")
    counts = torch.zeros((seq_len,), device="cpu", dtype=torch.int32)

    # start window position using the stride
    start_positions = list(range(0, seq_len, stride))

    # when GPUs are not available
    if not device.startswith("cuda"):
        with torch.inference_mode():
            for start in start_positions:
                # compute inclusive end index for residue span
                end = min(start + window_size, seq_len)
                window_len = end - start

                # rebuild window with CLS and EOS tokens around slice
                window_tokens = torch.cat(
                    [cls_token, token[:, 1 + start:1 + end], eos_token],
                    dim=1)
                output = model(window_tokens, repr_layers=[
                               layer], return_contacts=False)
                rep = output["representations"][layer][0, 1:1 + window_len]

                # store segment and window slide count
                full[start:end] += rep
                counts[start:end] += 1

                # clear from memory manually
                del output, rep, window_tokens

                # exit early once we covered tail of the sequence
                if end == seq_len:
                    break

        counts = counts.clamp_min(1).unsqueeze(1).to(full.dtype)
        return full / counts

    # when GPUs are available
    autocast_ctx = torch.amp.autocast
    with torch.inference_mode(), autocast_ctx("cuda", enabled=True):
        starts_list = list(start_positions)
        i = 0
        while i < len(starts_list):
            batch_starts = starts_list[i:i + 1]
            windows = []
            spans = []
            for start in batch_starts:
                # compute exclusive end index for residue span
                end = min(start + window_size, seq_len)
                # rebuild window with CLS and EOS tokens around slice
                window_tokens = torch.cat(
                    [cls_token, token[:, 1 + start:1 + end], eos_token], dim=1).to(device, non_blocking=True)

                # store window tokens and spans
                windows.append(window_tokens)
                spans.append((start, end))
            batch_tokens = torch.cat(windows, dim=0)
            # manually free list
            del windows

            output = model(batch_tokens, repr_layers=[
                           layer], return_contacts=False)
            result = output["representations"][layer]

            # overlap-ad on device
            for batch, (start, end) in enumerate(spans):
                window_len = end - start
                rep = result[batch, 1:1 +
                             window_len].to("cpu", non_blocking=True)
                full[start:end] += rep
                counts[start:end] += 1

            # manually free memory
            del output, result, batch_tokens, spans

            # increment by windows per call
            i += 1

    return (full / counts.clamp_min(1).unsqueeze(1).to(full.dtype))


def append_seq_len(res_vec: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Appends the protein sequence length to the end of a results vector embedding generated by ESM-2.

    Parameters
    ----------
        res_vec : np.ndarray
            Result vector embeddings generated by passing a protein sequence through ESM-2.
        seq_len : int
            Pre-computed protein sequence length.

    Returns
    -------
        np.ndarray
            The updated embedding array with the protein sequence length appended.
    """
    col = np.full((res_vec.shape[0], 1), float(seq_len), dtype=res_vec.dtype)
    return np.concatenate([res_vec, col], axis=1)


def esm_embeddings_from_fasta(fasta_df: pd.DataFrame,
                              id_col: str = "ID",
                              seq_col: str = "Sequence",
                              model_name: str = "esm2_t33_650M_UR50D",
                              max_tokens: int = 1022,
                              batch_size: int = 8,
                              per_seq_dir: Path | str = "esm2_per_seq",
                              index_csv_path: Path | str = "esm2_seq_index.csv",
                              logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate per residue ESM embeddings for sequences in a DataFrame and write outputs.

    Parameters
    ----------
        fasta_df : pandas.DataFrame
            Must contain `id_col` and `seq_col` with valid sequences.
        id_col : str
            Column name for record IDs used as .pt filenames.
        seq_col : str
            Column name for sequences.
        model_name : str
            Name of the `esm.pretrained` model factory to call.
        max_tokens : int
            The maximum number of tokens in the model's context window excluding start 
            and end tokens.
        batch_size : int
            Number of sequences processed together when possible.
        per_seq_dir : str or Path
            The directoyy to store .pt files.
        index_csv_path : str or Path
            Output CSV with metadata about stored arrays.
        logger : logging.Logger or None
            Logger to use. If None, uses esm_cli logger.

    Returns
    -------
        Tuple[pd.DataFrame, List[str]]
            `(index_df, failed_seqs)` where `index_df` summarizes embeddings and `failed_seqs`
            lists ids that could not be processed.
    """
    # set up logger and start timer
    logger = logger or logging.getLogger("esm_cli")
    index_records = []
    t0 = time.perf_counter()

    # set up directories to save results
    per_seq_dir.mkdir(parents=True, exist_ok=True)

    # clean sequences
    df = fasta_df[[id_col, seq_col]].dropna().copy()
    df[seq_col] = df[seq_col].map(clean_seq)
    df = df[df[seq_col].str.len() > 0].reset_index(drop=True)
    logger.info("embedding_start", extra={
                "extra": {"total_sequences": len(df)}})

    # load ESM embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # choose representation layer based on model size
    layer = 33 if "t33" in model_name else 30 if "t30" in model_name else 12 if "t12" in model_name else 6
    logger.info("model_loaded", extra={"extra": {
        "device": device,
        "first_param_device": next(model.parameters()).device.type,
        "layer": layer,
        "max_tokens": max_tokens,
        "model_name": model_name
    }})

    # prepare batches
    ids = df[id_col].tolist()
    seqs = df[seq_col].tolist()
    seq_pairs = list(zip(ids, seqs))

    failed_seqs = []

    # process in batches
    total = len(seq_pairs)
    done = 0
    for batch in token_packed_batches(seq_pairs, max_tokens=(max_tokens * batch_size)):
        b_start = time.perf_counter()

        # convert batch of id - seq pairs into model tokens
        data = [(id, seq) for id, seq in batch]
        _labels, strings, tokens = batch_converter(data)

        # compute token lengths and determine which exceed model window
        token_lens = [len(s) for s in strings]
        short_idxs = [i for i, len_ in enumerate(
            token_lens) if len_ + 2 <= max_tokens]
        long_idxs = [i for i in range(len(batch)) if i not in short_idxs]

        logger.info("batch_start", extra={"extra": {
            "batch_size": len(batch),
            "num_short": len(short_idxs),
            "num_longs": len(long_idxs)
        }})
        with torch.no_grad():
            # handle short sequences first
            if short_idxs:
                batch_tokens = tokens[short_idxs].to(device)
                output = model(batch_tokens, repr_layers=[
                               layer], return_contacts=False)
                reprs = output["representations"][layer]

                # for each item in the short slice, keep per-residue (L, D) embeddings
                for j, batch_idx in enumerate(short_idxs):
                    len_ = token_lens[batch_idx]
                    per_residue = reprs[j, 1:1 + len_]  # L by D
                    # per-residue embeddings
                    res_vec = per_residue.cpu().numpy().astype(np.float32)  # (L, D)
                    seq_id = batch[batch_idx][0]

                    # append sequence length column
                    res_vec = append_seq_len(res_vec, len_)  # (L, D+1)

                    # save as .pt per sequence
                    torch.save(torch.from_numpy(res_vec),
                               per_seq_dir / f"{seq_id}.pt")

                    # save logs per-residue embeddings for short sequences
                    index_records.append({"id": seq_id,
                                          # L
                                          "length": int(res_vec.shape[0]),
                                          # D
                                          "embed_dim": int(res_vec.shape[1]),
                                          # entire sequence length
                                          "original_seq_len": int(len_),
                                          "handle": "short",
                                          "model": model_name,
                                          "storage": ".pt",
                                          "path": str(per_seq_dir / f"{seq_id}.pt")})

            # handle long sequences
            for batch_idx in long_idxs:
                protein_id, protein_seq = batch[batch_idx]
                try:
                    single_token = tokens[batch_idx:batch_idx + 1]
                    # default uses sliding window to cover entirety of long sequence
                    res_vec = compute_window_embedding(
                        # (L, D)
                        single_token, model, layer, device).cpu().numpy().astype(np.float32)

                    # append sequence length column
                    res_vec = append_seq_len(
                        res_vec, len(protein_seq))  # (L, D+1)

                    # save logs per-residue embeddings for sliding window long sequences
                    index_records.append({"id": protein_id,
                                          "length": int(res_vec.shape[0]),
                                          "embed_dim": int(res_vec.shape[1]),
                                          "original_seq_len": len(protein_seq),
                                          "handle": "long",
                                          "model": model_name,
                                          "storage": ".pt",
                                          "path": str(per_seq_dir / f"{protein_id}.pt")})

                    torch.save(torch.from_numpy(res_vec),
                               per_seq_dir / f"{protein_id}.pt")
                except Exception as e:
                    logger.error("sequence_failed", extra={"extra": {
                        "id": protein_id, "error": repr(e)
                    }})
                    failed_seqs.append(protein_id)

        # make regular progress updates
        done += len(batch)
        logger.info("batch_done", extra={"extra": {
            "done": done,
            "total": total,
            "percent": round(100.0 * done / total, 2),
            "batch_duration_s": round(time.perf_counter() - b_start, 3)
        }})

    # metadata to describe embeddings
    index_df = pd.DataFrame(index_records, columns=[
        "id", "length", "embed_dim", "original_seq_len", "handle", "model"
    ])

    base = Path(per_seq_dir).resolve()
    index_df["file_path"] = index_df["id"].astype(
        str).map(lambda s: str(base / f"{s}.pt"))
    index_df.to_csv(index_csv_path, index=False)

    # summarize in log
    elapsed = time.perf_counter() - t0
    ok_count = len(index_records)
    by_handle = index_df["handle"].value_counts().to_dict()
    logger.info("embedding_done", extra={"extra": {
        "ok": ok_count,
        "failed": len(failed_seqs),
        "handled_short": int(by_handle.get("short", 0)),
        "handled_long": int(by_handle.get("long", 0)),
        "total_duration_s": round(elapsed, 3),
        "artifacts_path": str(os.path.abspath(per_seq_dir)),
        "index_csv_path": str(os.path.abspath(index_csv_path))
    }})

    return index_df, failed_seqs
