from dataclasses import dataclass
import math
import re
from typing import Tuple, Dict, Any, Mapping, List
import esm
import torch
import numpy as np
from pepseqpred.core.models.ffnn import PepSeqFFNN
from pepseqpred.core.embeddings.esm2 import clean_seq, compute_window_embedding, append_seq_len

_HIDDEN_LAYER_RE = re.compile(r"^ff_model\.(\d+)\.linear\.weight$")
_OUTPUT_LAYER_RE = re.compile(r"^ff_model\.(\d+)\.weight$")
_LAYER_NORM_RE = re.compile(r"^ff_model\.\d+\.layer_norm\.weight$")
_SKIP_LINEAR_RE = re.compile(r"^ff_model\.\d+\.skip\.weight$")


@dataclass
class FFNNModelConfig:
    emb_dim: int
    hidden_sizes: Tuple[int, ...]
    dropouts: Tuple[float, ...]
    num_classes: int
    use_layer_norm: bool
    use_residual: bool


def normalize_state_dict_keys(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Strips DDP `module.` prefix so checkpoints load in non-DDP inference."""
    out_dict: Dict[str, Any] = {}
    for k, v in state.items():
        key = str(k)
        if key.startswith("module."):
            key = key[len("module."):]
        out_dict[key] = v
    return out_dict


def infer_model_config_from_state(state: Mapping[str, Any]) -> FFNNModelConfig:
    """Infer PepSeqFFNN architecture directly from checkpoint `model_state_dict`."""
    state = normalize_state_dict_keys(state)

    # get hidden layers and sizes
    hidden_layers: List[Tuple[int, int, int]] = []
    for key, tensor in state.items():
        match_ = _HIDDEN_LAYER_RE.match(key)
        if match_ is None:
            continue
        if not torch.is_tensor(tensor) or tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensor at {key}")

        idx = int(match_.group(1))
        out_dim = int(tensor.shape[0])
        in_dim = int(tensor.shape[1])
        hidden_layers.append((idx, in_dim, out_dim))

    hidden_layers.sort(key=lambda x: x[0])
    hidden_sizes = tuple(out_dim for _, _, out_dim in hidden_layers)

    # get output layers and number of classes
    output_layers: List[Tuple[int, int, int]] = []
    for key, tensor in state.items():
        match_ = _OUTPUT_LAYER_RE.match(key)
        if match_ is None:
            continue
        if not torch.is_tensor(tensor) or tensor.dim() != 2:
            continue

        idx = int(match_.group(1))
        num_classes = int(tensor.shape[0])
        in_dim = int(tensor.shape[1])
        output_layers.append((idx, num_classes, in_dim))

    if len(output_layers) < 1:
        raise ValueError(
            "Could not infer output layer from checkpoint state_dict")

    output_layers.sort(key=lambda x: x[0])
    _, num_classes, out_in_dim = output_layers[-1]

    # validate hidden layers make sense (previous output layer should match current input layer)
    if len(hidden_layers) > 0:
        emb_dim = hidden_layers[0][1]
        for i in range(1, len(hidden_layers)):
            prev_out = hidden_layers[i - 1][2]
            curr_in = hidden_layers[i][1]
            if prev_out != curr_in:
                raise ValueError(
                    f"Hidden layer mismatch: prev_out={prev_out}, curr_in={curr_in}")
        if hidden_layers[-1][2] != out_in_dim:
            raise ValueError(
                f"Output in_features {out_in_dim} does not match last hidden size {hidden_layers[-1][2]}")
    else:
        emb_dim = out_in_dim

    # infer if using layer normalization
    use_layer_norm = any(_LAYER_NORM_RE.match(k) for k in state.keys())

    # try to infer if residuals used (ambiguous case)
    has_skip_linear = any(_SKIP_LINEAR_RE.match(k) for k in state.keys())
    if has_skip_linear:
        use_residual = True
    else:
        if len(hidden_layers) > 0 and all(in_dim == out_dim for _, in_dim, out_dim in hidden_layers):
            raise ValueError(
                "Cannot infer use_residual from state_dict when all hidden layers are width-preserving and no skip weights exist")
        use_residual = False

    # dropout is not stored in state_dict
    dropouts = tuple(0.0 for _ in hidden_sizes)

    return FFNNModelConfig(
        emb_dim=emb_dim,
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        num_classes=num_classes,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual
    )


def build_model_from_checkpoint(checkpoint: Mapping[str, Any], device: str = "cpu") -> Tuple[PepSeqFFNN, FFNNModelConfig]:
    """Builds and loads model exactly from training checkpoint."""
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected checkpoint dict containing 'model_state_dict'")

    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("'model_state_dict' must be a mapping")

    state = normalize_state_dict_keys(state)
    config = infer_model_config_from_state(state)

    if config.num_classes != 1:
        raise ValueError(
            f"Inference expects binary residue model (num_classes=1), got {config.num_classes}")

    model = PepSeqFFNN(
        emb_dim=config.emb_dim,
        hidden_sizes=config.hidden_sizes,
        dropouts=config.dropouts,
        num_classes=config.num_classes,
        use_layer_norm=config.use_layer_norm,
        use_residual=config.use_residual
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    return model, config


def infer_decision_threshold(checkpoint: Mapping[str, Any], default: float = 0.5) -> float:
    """Use validation threshold saved by training if applicable."""
    metrics = checkpoint.get("metrics", None) if isinstance(
        checkpoint, Mapping) else None
    if not isinstance(metrics, Mapping):
        return default

    thresh = metrics.get("threshold", None)
    if thresh is None:
        return default

    try:
        thresh = float(thresh)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(thresh) or thresh <= 0.0 or thresh >= 1.0:
        return default

    return thresh


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


def predict_protein(psp_model: PepSeqFFNN,
                    esm_model: torch.nn.Module,
                    layer: int,
                    batch_converter: esm.data.BatchConverter,
                    protein_seq: str,
                    max_tokens: int,
                    device: str,
                    threshold: float = 0.5) -> Dict[str, Any]:
    """
    Predict residue-level binary epitope mask for one protein sequence
    """
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in (0.0, 1.0)")

    protein_seq = clean_seq(protein_seq)
    if not protein_seq:
        raise ValueError("Protein sequence is empty after cleaning")

    # generate protein sequence residue embeddings
    protein_emb = _embed_protein_seq(
        protein_seq, esm_model, layer, batch_converter, device, max_tokens)
    non_blocking = device.startswith("cuda") and torch.cuda.is_available()
    X = protein_emb.unsqueeze(0).to(device, non_blocking=non_blocking)

    # start inference
    with torch.inference_mode():
        logits = psp_model(X)  # (1, L)
        if logits.dim() != 2 or logits.size(0) != 1:
            raise ValueError(
                f"Expected logits shape (1, L), got {tuple(logits.shape)}")
        # get probabiity scores then assign binary values (1=def epitope, 0=not epitope)
        probs = torch.sigmoid(logits)[0].detach().cpu()
        mask = (probs >= threshold).to(torch.int64)

    # build binary string
    binary_mask = "".join("1" if int(x) == 1 else "0" for x in mask.tolist())

    return {
        "binary_mask": binary_mask,
        "length": int(mask.numel()),
        "n_epitope": int(mask.sum().item()),
        "frac_epitope": float(mask.float().mean().item()),
        "p_epitope_mean": float(probs.mean().item()),
        "p_epitope_max": float(probs.max().item()),
        "threshold": threshold
    }
