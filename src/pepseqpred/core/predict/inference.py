import math
import re
from typing import Tuple, Dict, Any, Mapping, List, Optional, Sequence
import esm
import torch
import numpy as np
from pepseqpred.core.models.factory import (
    MODEL_HEAD_FFNN,
    PepSeqModelConfig,
    build_pepseq_model,
    model_config_from_mapping,
    validate_model_config
)
from pepseqpred.core.embeddings.esm2 import (
    apply_seq_len_feature,
    clean_seq,
    compute_window_embedding
)
from pepseqpred.core.data.seq_len_feature import (
    SEQ_LEN_FEATURE_AUTO,
    SEQ_LEN_FEATURE_NONE,
    model_seq_len_feature_to_embedding,
    normalize_prediction_seq_len_feature
)

_HIDDEN_LAYER_RE = re.compile(r"^ff_model\.(\d+)\.linear\.weight$")
_OUTPUT_LAYER_RE = re.compile(r"^ff_model\.(\d+)\.weight$")
_LAYER_NORM_RE = re.compile(r"^ff_model\.\d+\.layer_norm\.weight$")
_SKIP_LINEAR_RE = re.compile(r"^ff_model\.\d+\.skip\.weight$")


FFNNModelConfig = PepSeqModelConfig


def resolve_prediction_seq_len_feature(
    seq_len_feature: str | None,
    model_config: PepSeqModelConfig,
) -> str:
    """Resolve prediction embedding length-feature mode from override/config."""
    mode = normalize_prediction_seq_len_feature(seq_len_feature)
    if mode == SEQ_LEN_FEATURE_AUTO:
        return model_seq_len_feature_to_embedding(
            getattr(model_config, "seq_len_feature", None))
    return mode


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


def infer_model_config_from_state(state: Mapping[str, Any]) -> PepSeqModelConfig:
    """
    Infer PepSeqFFNN architecture directly from checkpoint `model_state_dict`.

    Parameters
    ----------
        state : Mapping[str, Any]
            State dictionary obtained from saved trained model checkpoint.

    Returns
    -------
        PepSeqModelConfig
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

    return PepSeqModelConfig(
        emb_dim=emb_dim,
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        num_classes=num_classes,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual,
        model_head=MODEL_HEAD_FFNN,
    )


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: str = "cpu",
    model_config: Optional[PepSeqModelConfig] = None
) -> Tuple[torch.nn.Module, PepSeqModelConfig, str]:
    """
    Builds and loads model exactly from training checkpoint.

    Parameters
    ----------
        checkpoint : Mapping[str, Any]
            State dictionary obtained from saved trained model checkpoint.
        device : str
            Device type to use for building model checkpoint, default is `"cpu"`, `"cuda"` if accepted if GPUs are available.
        model_config : PepSeqModelConfig or None
            A fully populated model configuration dataclass to be used in model building.

    Returns
    -------
        torch.nn.Module
            Fully populated PepSeqPred model ready to use in inference.
        PepSeqModelConfig
            Model configuration dataclass returned for logging/tracking purposes.
        str
            The model configuration source: `"cli"`, `"checkpoint"`, or `"state_dict"`.

    Raises
    ------
        ValueError
            If `checkpoint` is not of type `Mapping` or `"model_state_dict"` is not a key in `checkpoint`. If `state` is not of type `Mapping`. If the model config is invalid or if the checkpoint weights are incompatible with the model architecture provided.
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
        config = validate_model_config(model_config)
        config_src = "cli"
    elif isinstance(checkpoint.get("model_config"), Mapping):
        config = model_config_from_mapping(checkpoint["model_config"])
        config_src = "checkpoint"
    else:
        config = infer_model_config_from_state(state)
        config_src = "state_dict"

    model = build_pepseq_model(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(
            "Checkpoint weights are incompatible with the provided model architecture. "
            "Check --model-head and architecture flags."
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
                      max_tokens: int = 1022,
                      seq_len_feature: str | None = SEQ_LEN_FEATURE_NONE) -> torch.Tensor:
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
        seq_len_feature : str or None
            Sequence-length feature mode: `"none"`, `"raw"`, or `"inverse"`.

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
    rep_np = apply_seq_len_feature(
        rep_np,
        seq_len=seq_len,
        seq_len_feature=seq_len_feature,
    )

    return torch.from_numpy(rep_np)


def predict_member_probabilities_from_embedding(
    psp_model: torch.nn.Module,
    protein_emb: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Predicts residue-level epitope probabilities from a precomputed protein embedding.

    Parameters
    ----------
        psp_model : torch.nn.Module
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


def predict_from_embedding(psp_model: torch.nn.Module,
                           protein_emb: torch.Tensor,
                           device: str,
                           threshold: float = 0.5) -> Dict[str, Any]:
    """
    Predict residue-level binary epitope mask from precomputed protein embedding.

    Parameters
    ----------
        psp_model : torch.nn.Module
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
    psp_models: Sequence[torch.nn.Module],
    protein_emb: torch.Tensor,
    device: str,
    thresholds: Sequence[float],
    aggregation: str = "majority",
    ensemble_threshold: float | None = None
) -> Dict[str, Any]:
    """
    Predict residue-level mask from an ensemble of model members.

    Parameters
    ----------
        psp_models : Sequence[torch.nn.Module]
            A sequence of PepSeqPred models to use for majority vote prediction of member probabilities.
        protein_emb : torch.Tensor
            Protein embedding to pass through models.
        device : str
            Device type: `"cuda"` for GPUs, otherwise `"cpu"`.
        thresholds : Sequence[float]
            A sequence of thresholds between `0.0` and `1.0`, that determine the cutoff for non-epitope vs definite epitope. Default is `0.5`.
        aggregation : str
            Ensemble aggregation rule. `"majority"` thresholds each member and
            applies strict majority vote. `"mean-prob"` thresholds the mean
            probability across members.
        ensemble_threshold : float or None
            Threshold used for `"mean-prob"` aggregation. If omitted, the mean
            of member thresholds is used.

    Returns
    -------
        Dict[str, Any]
            Complete prediction payload based on each model fold prediction.

    Raises
    ------
        ValueError
            If no models are provided, if number of models and thresholds differs,
            if any threshold is outside `(0.0, 1.0)`, if aggregation is invalid,
            if member probability shapes are invalid, or if member output lengths
            are inconsistent.
    """
    aggregation = str(aggregation).strip().lower()
    if aggregation not in {"majority", "mean-prob"}:
        raise ValueError("aggregation must be one of: majority, mean-prob")
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

    n_members = int(len(member_masks))
    mean_probs = torch.stack(member_probs, dim=0).mean(dim=0)
    votes_needed: int | None = None
    if aggregation == "majority":
        vote_sum = torch.stack(member_masks, dim=0).sum(dim=0)
        votes_needed = int((n_members // 2) + 1)
        mask = (vote_sum >= votes_needed).to(torch.int64)
        payload_threshold = float("nan")
        resolved_ensemble_threshold = None
    else:
        resolved_ensemble_threshold = (
            float(ensemble_threshold)
            if ensemble_threshold is not None
            else float(sum(float(x) for x in thresholds) / len(thresholds))
        )
        if resolved_ensemble_threshold <= 0.0 or resolved_ensemble_threshold >= 1.0:
            raise ValueError("ensemble_threshold must be in (0.0, 1.0)")
        mask = (mean_probs >= resolved_ensemble_threshold).to(torch.int64)
        payload_threshold = float(resolved_ensemble_threshold)

    out = _build_prediction_payload(
        probs=mean_probs,
        mask=mask,
        threshold=payload_threshold
    )
    out["n_members"] = n_members
    out["votes_needed"] = votes_needed
    out["member_thresholds"] = [float(x) for x in thresholds]
    out["ensemble_aggregation"] = aggregation
    out["ensemble_threshold"] = (
        float(resolved_ensemble_threshold)
        if resolved_ensemble_threshold is not None
        else None
    )
    return out


def predict_protein(psp_model: torch.nn.Module,
                    esm_model: torch.nn.Module,
                    layer: int,
                    batch_converter: esm.data.BatchConverter,
                    protein_seq: str,
                    max_tokens: int,
                    device: str,
                    threshold: float = 0.5,
                    seq_len_feature: str | None = SEQ_LEN_FEATURE_NONE) -> Dict[str, Any]:
    """
    Predict residue-level binary epitope mask for one protein sequence

    Parameters
    ----------
        psp_model : torch.nn.Module
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
        seq_len_feature : str or None
            Sequence-length feature mode used for generated embeddings.

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
        max_tokens=max_tokens,
        seq_len_feature=seq_len_feature,
    )
    return predict_from_embedding(
        psp_model=psp_model,
        protein_emb=protein_emb,
        device=device,
        threshold=threshold
    )
