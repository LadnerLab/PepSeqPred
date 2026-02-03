from typing import Tuple, Dict, Any
import esm
import torch
import numpy as np
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.embeddings.esm2 import clean_seq, compute_window_embedding, append_seq_len


CLASS_NAMES = ["Def epitope", "Uncertain", "Not epitope"]


def _find_peptide_start_stop(protein_seq: str, peptide: str) -> Tuple[int, int]:
    """
    Find the peptide start and stop indices within an overall protein sequence if they were not provided.

    Parameters
    ----------
        protein_seq : str
            The entire protein sequence used to generate the peptide.
        peptide : str
            The peptide as a string which should be a subset of the overall protein sequence.

    Returns
    -------
        start : int
            Start index of the peptide.
        stop : int
            Stop index of the peptide.

    Raises
    ------
        ValueError
            If peptide is not found in the protein sequence.
    """
    start = protein_seq.find(peptide)
    stop = len(peptide) + start
    if start < 0 or stop > len(protein_seq):
        raise ValueError(
            "Peptide not found as a contiguous substring in the provided protein sequence")
    return start, stop


def infer_emb_dim_from_state(state: Dict[str, Any]) -> int:
    """
    Infers the embedding dimension from the model state.

    Parameters
    ----------
        state : Dict[str, Any]
            The current model state as a dict.

    Returns
    -------
        int
            The embedding dimension (usually 1281).

    Raises
    ------
        ValueError
            If no 2D weight tensors were found to infer embedding dimension.
    """
    for _, v in state.items():
        if torch.is_tensor(v) and v.dim() == 2:
            return int(v.shape[1])
    raise ValueError(
        "Could not infer embedding dimension from model checkpoint, no 2D weight tensors found")


def _embed_protein_seq(protein_seq: str,
                       esm_model: torch.nn.Module,
                       layer: int,
                       batch_converter: esm.data.BatchConverter,
                       device: str,
                       max_tokens: int = 1022) -> torch.Tensor:
    """
    Generates the embedding for an entire protein sequence.

    Parameters
    ----------
        protein_seq : str
            The overall protein sequence the peptide was generated from.
        esm_model : torch.nn.Module
            The model used to generate ESM-2 training embeddings from. This 
            model must match the training model. Default is esm2_t33_650M_UR50D.
        layer : int
            The model layer to extract embeddings from.
        batch_converter : esm.data.BatchConverter
            The ESM batch converter for tokenizing protein sequences.
        device : str
            The device to run the embedding model on (`"cpu"` or `"cuda"`).
        max_tokens : int
            Maximum number of tokens the ESM model can fit in its context window. Default is 1022.

    Returns
    -------
        Tensor
            The entire protein sequence's embeddings as a tensor of shape (seq_len, emb_dim).
    """
    # get batched tokens using one sequence passed
    pairs = [("query", protein_seq)]
    _, _, batch_tokens = batch_converter(pairs)

    seq_len = len(protein_seq)
    token_len = batch_tokens.size(1)

    with torch.inference_mode():
        if token_len <= (max_tokens + 2):
            if device.startswith("cuda") and torch.cuda.is_available():
                batch_tokens = batch_tokens.to(device, non_blocking=True)
            out = esm_model(batch_tokens, repr_layers=[
                            layer], return_contacts=False)
            rep = out["representations"][layer][0, 1:1 + seq_len].to("cpu")

        else:
            rep = compute_window_embedding(
                batch_tokens, esm_model, layer, device)

    rep_np = rep.numpy().astype(np.float32)
    rep_np = append_seq_len(rep_np, seq_len)

    return torch.from_numpy(rep_np)


def predict(psp_model: PepSeqFFNN,
            esm_model: torch.nn.Module,
            layer: int,
            batch_converter: esm.data.BatchConverter,
            protein_seq: str,
            peptide: str,
            max_tokens: int,
            device: str) -> Dict[str, Any]:
    """
    Predicts the epitope class probabilities for a given peptide and the protein sequence it was generated from.

    Parameters
    ----------
        psp_model : PepSeqFFNN
            The trained PepSeqFFNN model to make predictions.
        esm_model : torch.nn.Module
            The model used to generate ESM-2 embeddings from. This model must match the training model.
        layer : int
            The model layer to extract embeddings from.
        batch_converter : esm.data.BatchConverter
            The ESM batch converter for tokenizing protein sequences.
        protein_seq : str
            The entire protein sequence used to derive the peptide.
        peptide : str
            The peptide sequence that may or may not contain epitopes.
        max_tokens : int
            Maximum number of tokens the ESM model can fit in its context window. Default is 1022.
        device : str
            The device to run the embedding model on (`"cpu"` or `"cuda"`).

    Returns
    -------
        Dict[str, Any]
            A dictionary containing the peptide, protein sequence, predicted probabilities for each class, 
            predicted class name, predicted class index, and the largest predicted probability.
    """
    protein_seq = clean_seq(protein_seq)
    peptide = clean_seq(peptide)

    align_start, align_stop = _find_peptide_start_stop(protein_seq, peptide)

    # generate embedding and get peptide-level embedding
    protein_emb = _embed_protein_seq(
        protein_seq, esm_model, layer, batch_converter, device, max_tokens)
    peptide_emb = protein_emb[align_start:align_stop, :]
    X = peptide_emb.unsqueeze(0).to(device, non_blocking=True)

    # generate probability outputs (predictions)
    with torch.inference_mode():
        logits = psp_model(X)  # (1, 3)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    pred_idx = int(logits.argmax(dim=-1).item())

    return {"peptide": peptide,
            "protein": protein_seq,
            "p_def_epitope": probs[0],
            "p_uncertain": probs[1],
            "p_not_epitope": probs[2],
            "pred_class": CLASS_NAMES[pred_idx],
            "pred_index": pred_idx,
            "max_prob": max(probs)}
