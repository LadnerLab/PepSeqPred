from dataclasses import dataclass
import math
import re
from typing import Tuple, Dict, Any, Mapping, List, Optional, Sequence
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
    """
    Strips DDP `module.` prefix so checkpoints load in non-DDP inference.

    Parameters
    ----------
        state : Mapping[str, Any]
            State dictionary obtained from saved trained model checkpoint.

    Returns
    -------
        Dict[str, Any]
            The normalized state dictionary with the `module.` prefix removed.
    """
    out_dict: Dict[str, Any] = {}
    for k, v in state.items():
        key = str(k)
        if key.startswith("module."):
            key = key[len("module."):]
        out_dict[key] = v
    return out_dict


def infer_model_config_from_state(state: Mapping[str, Any]) -> FFNNModelConfig:
    """
    Infer PepSeqFFNN architecture directly from checkpoint `model_state_dict`.

    Parameters
    ----------
        state : Mapping[str, Any]
            State dictionary obtained from saved trained model checkpoint.

    Returns
    -------
        FFNNModelConfig
            A fully populated model configuration dataclass to be used in model building.

    Raises
    ------
        ValueError
            When value in the itemized state is not a tensor or not a 2D tensor. When the output layer could not be inferred from the `state_dict`. When there is a hidden layer mismatch (prev != curr). When output `in_features` does not match the last hidden layer size. When all hidden sizes are the same, you cannot infer if residuals were used.
    """
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


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: str = "cpu",
    model_config: Optional[FFNNModelConfig] = None
) -> Tuple[PepSeqFFNN, FFNNModelConfig, str]:
    """
    Builds and loads model exactly from training checkpoint.

    Parameters
    ----------
        checkpoint : Mapping[str, Any]
            State dictionary obtained from saved trained model checkpoint.
        device : str
            Device type to use for building model checkpoint, default is `"cpu"`, `"cuda"` if accepted if GPUs are available.
        model_config : FFNNModelConfig or None
            A fully populated model configuration dataclass to be used in model building.

    Returns
    -------
        PepSeqFFNN
            Fully populated PepSeqPred model ready to use in inference.
        FFNNModelConfig
            Model configuration dataclass returned for logging/tracking purposes.
        str
            The model configuration source, either `"cli"` if passed explicitly or `"state_dict"` if inferred from the state dictionary.

    Raises
    ------
        ValueError
            If `checkpoint` is not of type `Mapping` or `"model_state_dict"` is not a key in `checkpoint`. If `state` is not of type `Mapping`. If the number of output classes is not 1 (binary), if the embedding dimension is <= 0, or if the number hidden layer sizes is not the same as the number of dropouts. If the checkpoint weights are incompatible with the model architecture provided. 
    """
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected checkpoint dict containing 'model_state_dict'")

    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("'model_state_dict' must be a mapping")
    state = normalize_state_dict_keys(state)

    # can either pass config directly or infer from the state dict
    if model_config is not None:
        config = model_config
        config_src = "cli"
    else:
        config = infer_model_config_from_state(state)
        config_src = "state_dict"

    # validate config
    if config.num_classes != 1:
        raise ValueError(
            f"Inference expects binary residue model (num_classes=1), got {config.num_classes}")
    if config.emb_dim <= 0:
        raise ValueError("emb_dim must be > 0")
    if len(config.hidden_sizes) != len(config.dropouts):
        raise ValueError("hidden_sizes and dropouts must have the same length")

    model = PepSeqFFNN(
        emb_dim=config.emb_dim,
        hidden_sizes=config.hidden_sizes,
        dropouts=config.dropouts,
        num_classes=config.num_classes,
        use_layer_norm=config.use_layer_norm,
        use_residual=config.use_residual
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(
            "Checkpoint weights are incompatible with the provided model architecture. "
            "Check --emb-dim/--hidden-sizes/--dropouts/--use-layer-norm/--use-residual"
        ) from e
    model.eval().to(device)
    return model, config, config_src


def infer_decision_threshold(checkpoint: Mapping[str, Any], default: float = 0.5) -> float:
    """
    Use validation threshold saved by training if applicable.

    Parameters
    ----------
        checkpoint : Mapping[str, Any]
            State dictionary obtained from saved trained model checkpoint.
        default : float
            Default threshold to use in the event threshold cannot be inferred from the checkpoint. Default threshold is `0.5`.

    Returns
    -------
        float
            Either inferred or default threshold value.
    """
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


def _build_prediction_payload(
    probs: torch.Tensor,
    mask: torch.Tensor,
    threshold: float
) -> Dict[str, Any]:
    """Builds prediction payload from already computed probabilities and mask."""
    if probs.dim() != 1:
        raise ValueError(
            f"Expected probs shape (L,), got {tuple(probs.shape)}")
    if mask.dim() != 1:
        raise ValueError(f"Expected mask shape (L,), got {tuple(mask.shape)}")
    if probs.numel() != mask.numel():
        raise ValueError(
            f"Expected matching probs/mask lengths, got probs={probs.numel()} mask={mask.numel()}"
        )

    # build binary string
    binary_mask = "".join("1" if int(x) == 1 else "0" for x in mask.tolist())
    return {
        "binary_mask": binary_mask,
        "length": int(mask.numel()),
        "n_epitopes": int(mask.sum().item()),
        "frac_epitope": float(mask.float().mean().item()),
        "p_epitope_mean": float(probs.mean().item()),
        "p_epitope_max": float(probs.max().item()),
        "threshold": float(threshold)
    }


def embed_protein_seq(protein_seq: str,
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

    Raises
    ------
        ValueError
            If the cleaned protein sequence is empty.
    """
    protein_seq = clean_seq(protein_seq)
    if not protein_seq:
        raise ValueError("Protein sequence is empty after cleaning")

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


def predict_member_probabilities_from_embedding(
    psp_model: PepSeqFFNN,
    protein_emb: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Predicts residue-level epitope probabilities from a precomputed protein embedding.

    Parameters
    ----------
        psp_model : PepSeqFFNN
            The PepSeqPred model to use for predicting member probabilities.
        protein_emb : torch.Tensor
            Protein embedding to pass through model.
        device : str
            Device type: `"cuda"` for GPUs, otherwise `"cpu"`.
    Returns
    -------
        torch.Tensor
            The predicted probabilities of each residue being an epitope.

    Raises
    ------
        ValueError
            If the protein embedding does not have shape `(L, E)` or the model
            logits are not returned as shape `(1, L)`.
    """
    if protein_emb.dim() != 2:
        raise ValueError(
            f"Expected protein embedding shape (L, E), got {tuple(protein_emb.shape)}"
        )
    non_blocking = device.startswith("cuda") and torch.cuda.is_available()
    X = protein_emb.unsqueeze(0).to(device, non_blocking=non_blocking)

    # start inference
    with torch.inference_mode():
        logits = psp_model(X)  # (1, L)
        if logits.dim() != 2 or logits.size(0) != 1:
            raise ValueError(
                f"Expected logits shape (1, L), got {tuple(logits.shape)}")
        return torch.sigmoid(logits)[0].detach().cpu()


def predict_from_embedding(psp_model: PepSeqFFNN,
                           protein_emb: torch.Tensor,
                           device: str,
                           threshold: float = 0.5) -> Dict[str, Any]:
    """
    Predict residue-level binary epitope mask from precomputed protein embedding.

    Parameters
    ----------
        psp_model : PepSeqFFNN
            The PepSeqPred model to use for predicting member probabilities.
        protein_emb : torch.Tensor
            Protein embedding to pass through model.
        device : str
            Device type: `"cuda"` for GPUs, otherwise `"cpu"`.
        threshold : float
            The threshold between `0.0` and `1.0`, that determine the cutoff for non-epitope vs definite epitope. Default is `0.5`.

    Returns
    -------
        Dict[str, Any]
            Prediction payload from computed probabilities and mask.

    Raises
    ------
        ValueError
            If `threshold` is outside `(0.0, 1.0)`, or if downstream probability
            prediction/payload construction receives invalid tensor shapes.
    """
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in (0.0, 1.0)")

    probs = predict_member_probabilities_from_embedding(
        psp_model=psp_model,
        protein_emb=protein_emb,
        device=device
    )
    mask = (probs >= threshold).to(torch.int64)
    return _build_prediction_payload(probs=probs, mask=mask, threshold=threshold)


def predict_ensemble_from_embedding(
    psp_models: Sequence[PepSeqFFNN],
    protein_emb: torch.Tensor,
    device: str,
    thresholds: Sequence[float]
) -> Dict[str, Any]:
    """
    Predict residue-level mask using strict majority vote across model members.

    Parameters
    ----------
        psp_models : Sequence[PepSeqFFNN]
            A sequence of PepSeqPred models to use for majority vote prediction of member probabilities.
        protein_emb : torch.Tensor
            Protein embedding to pass through models.
        device : str
            Device type: `"cuda"` for GPUs, otherwise `"cpu"`.
        thresholds : Sequence[float]
            A sequence of thresholds between `0.0` and `1.0`, that determine the cutoff for non-epitope vs definite epitope. Default is `0.5`.

    Returns
    -------
        Dict[str, Any]
            Complete prediction payload based on each model fold prediction.

    Raises
    ------
        ValueError
            If no models are provided, if number of models and thresholds differs,
            if any threshold is outside `(0.0, 1.0)`, if member probability shapes
            are invalid, or if member output lengths are inconsistent.
    """
    if len(psp_models) < 1:
        raise ValueError(
            "At least one model is required for ensemble prediction")
    if len(psp_models) != len(thresholds):
        raise ValueError(
            f"Model/threshold length mismatch: models={len(psp_models)} thresholds={len(thresholds)}"
        )

    member_probs: List[torch.Tensor] = []
    member_masks: List[torch.Tensor] = []
    for model, threshold in zip(psp_models, thresholds):
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("each threshold must be in (0.0, 1.0)")
        probs = predict_member_probabilities_from_embedding(
            psp_model=model,
            protein_emb=protein_emb,
            device=device
        )
        if probs.dim() != 1:
            raise ValueError(
                f"Expected member probs shape (L,), got {tuple(probs.shape)}"
            )
        member_probs.append(probs)
        member_masks.append((probs >= threshold).to(torch.int64))

    seq_lengths = {int(mask.numel()) for mask in member_masks}
    if len(seq_lengths) != 1:
        raise ValueError(
            "All ensemble members must produce the same sequence length")

    vote_sum = torch.stack(member_masks, dim=0).sum(dim=0)
    n_members = int(len(member_masks))
    votes_needed = int((n_members // 2) + 1)
    majority_mask = (vote_sum >= votes_needed).to(torch.int64)
    mean_probs = torch.stack(member_probs, dim=0).mean(dim=0)
    out = _build_prediction_payload(
        probs=mean_probs,
        mask=majority_mask,
        threshold=float("nan")
    )
    out["n_members"] = n_members
    out["votes_needed"] = votes_needed
    out["member_thresholds"] = [float(x) for x in thresholds]
    return out


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

    Parameters
    ----------
        psp_model : PepSeqFFNN
            The PepSeqPred model to use for predicting member probabilities.
        esm_model : torch.nn.Module
            The ESM-2 model to use for embedding generation.
        layer : int
            Which layer of the ESM-2 model to use for embedding generation.
        batch_converter : esm.data.BatchConverter
            ESM-2 batch converter for converting raw protein sequences to embeddings.
        protein_seq : str
            Protein sequence to embed then pass through PepSeqPred.
        max_token : int
            The maximum amount of tokens to account for per pass.
        device : str
            Device type: `"cuda"` for GPUs, otherwise `"cpu"`.
        threshold : float
            The threshold between `0.0` and `1.0`, that determine the cutoff for non-epitope vs definite epitope. Default is `0.5`.

    Returns
    -------
        Dict[str, Any]
            Prediction payload from computed probabilities and mask.

    Raises
    ------
        ValueError
            If the threshold is outside (`0.0, 1.0)`.
    """
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in (0.0, 1.0)")

    protein_emb = embed_protein_seq(
        protein_seq=protein_seq,
        esm_model=esm_model,
        layer=layer,
        batch_converter=batch_converter,
        device=device,
        max_tokens=max_tokens
    )
    return predict_from_embedding(
        psp_model=psp_model,
        protein_emb=protein_emb,
        device=device,
        threshold=threshold
    )
