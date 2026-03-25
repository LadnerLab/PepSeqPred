"""eval_ffnn_cli.py

Evaluate trained PepSeqPred FFNN checkpoints or ensemble manifests on labeled
embedding shards.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.read import parse_float_csv, parse_int_csv
from pepseqpred.core.data.proteindataset import ProteinDataset, pad_collate
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    infer_decision_threshold,
    predict_member_probabilities_from_embedding
)
from pepseqpred.core.train.metrics import compute_eval_metrics


@dataclass(frozen=True)
class EvaluationMember:
    checkpoint: Path
    threshold: float | None
    fold_index: int | None
    member_index: int | None


def _build_cli_model_config(args: argparse.Namespace) -> FFNNModelConfig | None:
    """Builds the model configuration from explicit CLI architecture flags."""
    any_explicit = any(
        arg is not None
        for arg in (
            args.emb_dim,
            args.hidden_sizes,
            args.dropouts,
            args.use_layer_norm,
            args.use_residual,
            args.num_classes
        )
    )
    if not any_explicit:
        return None

    missing = []
    if args.emb_dim is None:
        missing.append("--emb-dim")
    if args.hidden_sizes is None:
        missing.append("--hidden-sizes")
    if args.dropouts is None:
        missing.append("--dropouts")
    if args.use_layer_norm is None:
        missing.append("--use-layer-norm/--no-use-layer-norm")
    if args.use_residual is None:
        missing.append("--use-residual/--no-use-residual")
    if missing:
        raise ValueError(
            "When using explicit architecture flags, provide all required values: "
            + ", ".join(missing)
        )

    hidden_sizes = parse_int_csv(args.hidden_sizes, "--hidden-sizes")
    dropouts = parse_float_csv(args.dropouts, "--dropouts")
    if len(hidden_sizes) != len(dropouts):
        raise ValueError(
            "--hidden-sizes and --dropouts must be the same length")

    num_classes = int(args.num_classes) if args.num_classes is not None else 1
    return FFNNModelConfig(
        emb_dim=int(args.emb_dim),
        hidden_sizes=tuple(hidden_sizes),
        dropouts=tuple(dropouts),
        num_classes=num_classes,
        use_layer_norm=bool(args.use_layer_norm),
        use_residual=bool(args.use_residual)
    )


def _as_optional_int(value: Any) -> int | None:
    """Parses optional integer-like value."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_threshold(value: Any) -> float | None:
    """Parses optional valid threshold in (0.0, 1.0)."""
    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        return None
    return threshold


def _resolve_member_checkpoint_path(raw_path: Any, manifest_path: Path) -> Path | None:
    """Resolves member checkpoint path relative to manifest if needed."""
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _member_sort_key(member: EvaluationMember) -> Tuple[int, int, str]:
    """Sort key for deterministic member ordering."""
    fold_index = member.fold_index if member.fold_index is not None else int(
        1e9)
    member_index = (
        member.member_index if member.member_index is not None else int(1e9)
    )
    return (fold_index, member_index, str(member.checkpoint))


def _resolve_manifest_members(
    manifest_path: Path,
    ensemble_set_index: int
) -> Tuple[List[EvaluationMember], Dict[str, Any]]:
    """Resolves valid members from schema v1/v2 ensemble manifests."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in ensemble manifest: {manifest_path}"
        ) from e
    if not isinstance(payload, Mapping):
        raise ValueError("Ensemble manifest must be a JSON object")

    schema_version = _as_optional_int(payload.get("schema_version"))
    selected_set: Mapping[str, Any] | None = None
    members_raw: Any = None

    if schema_version == 2:
        sets_raw = payload.get("sets")
        if not isinstance(sets_raw, list):
            raise ValueError(
                "schema_version=2 manifest must contain 'sets' list")
        for entry in sets_raw:
            if not isinstance(entry, Mapping):
                continue
            if _as_optional_int(entry.get("set_index")) == ensemble_set_index:
                selected_set = entry
                break
        if selected_set is None:
            raise ValueError(
                f"Requested --ensemble-set-index={ensemble_set_index} not found in manifest"
            )
        members_raw = selected_set.get("members")
    else:
        selected_set = payload
        members_raw = payload.get("members")

    if not isinstance(members_raw, list):
        raise ValueError("Manifest must contain a 'members' list")

    members: List[EvaluationMember] = []
    for member in members_raw:
        if not isinstance(member, Mapping):
            continue
        status = str(member.get("status", "")).strip().upper()
        checkpoint_path = _resolve_member_checkpoint_path(
            member.get("checkpoint"), manifest_path
        )
        if status != "OK" or checkpoint_path is None:
            continue
        members.append(
            EvaluationMember(
                checkpoint=checkpoint_path,
                threshold=_as_optional_threshold(member.get("threshold")),
                fold_index=_as_optional_int(member.get("fold_index")),
                member_index=_as_optional_int(member.get("member_index"))
            )
        )

    if not members:
        raise ValueError(
            "No valid ensemble members found (requires status=OK and checkpoint path)"
        )

    members_sorted = sorted(members, key=_member_sort_key)
    meta: Dict[str, Any] = {
        "schema_version": schema_version,
        "set_index": (
            _as_optional_int(selected_set.get("set_index"))
            if isinstance(selected_set, Mapping)
            else None
        ),
        "n_members_total": len(members_raw),
        "n_members_valid": len(members_sorted)
    }
    return members_sorted, meta


def _resolve_evaluation_members(
    model_artifact: Path,
    ensemble_set_index: int,
    k_folds: int | None
) -> Tuple[str, List[EvaluationMember], Dict[str, Any]]:
    """Resolves members from either a checkpoint path or ensemble manifest."""
    suffix = model_artifact.suffix.lower()
    if suffix == ".pt":
        members = [
            EvaluationMember(
                checkpoint=model_artifact,
                threshold=None,
                fold_index=1,
                member_index=1
            )
        ]
        mode = "single-checkpoint"
        meta = {
            "schema_version": None,
            "set_index": None,
            "n_members_total": 1,
            "n_members_valid": 1
        }
    elif suffix == ".json":
        members, meta = _resolve_manifest_members(
            manifest_path=model_artifact, ensemble_set_index=ensemble_set_index
        )
        mode = "ensemble-manifest"
    else:
        raise ValueError(
            f"Unsupported model artifact type '{model_artifact.suffix}'. "
            "Expected .pt checkpoint or .json ensemble manifest."
        )

    if k_folds is not None:
        if k_folds < 1:
            raise ValueError("--k-folds must be >= 1")
        if k_folds > len(members):
            raise ValueError(
                f"--k-folds={k_folds} exceeds available valid members={len(members)}"
            )
        members = members[:k_folds]

    return mode, members, meta


def _sanitize_for_json(value: Any) -> Any:
    """Recursively converts non-finite floats to None for strict JSON output."""
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _evaluate_dataset(
    psp_models: Sequence[torch.nn.Module],
    thresholds: Sequence[float],
    eval_loader: DataLoader,
    device: str,
    logger: Any
) -> Dict[str, Any]:
    """Evaluates one model or an ensemble on the given eval dataset loader."""
    n_members = int(len(psp_models))
    if n_members < 1:
        raise ValueError("At least one model is required for evaluation")
    if len(thresholds) != n_members:
        raise ValueError(
            f"Model/threshold length mismatch: models={n_members} thresholds={len(thresholds)}"
        )

    votes_needed = int((n_members // 2) + 1)

    all_true: List[torch.Tensor] = []
    all_pred: List[torch.Tensor] = []
    all_prob: List[torch.Tensor] = []

    processed = 0
    zero_valid = 0
    total_valid = 0
    total_pos = 0
    total_neg = 0
    total_pred_pos = 0
    total_pred_neg = 0

    for batch_idx, batch in enumerate(eval_loader, start=1):
        X, y, mask = batch
        batch_size = int(X.size(0))

        for i in range(batch_size):
            processed += 1
            protein_emb = X[i]
            y_row = y[i].to(torch.int64)
            valid_mask = mask[i].bool()
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                zero_valid += 1
                continue

            if n_members == 1:
                probs = predict_member_probabilities_from_embedding(
                    psp_model=psp_models[0], protein_emb=protein_emb, device=device
                )
                pred = (probs >= float(thresholds[0])).to(torch.int64)
            else:
                member_probs: List[torch.Tensor] = []
                member_masks: List[torch.Tensor] = []
                for model, threshold in zip(psp_models, thresholds):
                    probs_member = predict_member_probabilities_from_embedding(
                        psp_model=model, protein_emb=protein_emb, device=device
                    )
                    member_probs.append(probs_member)
                    member_masks.append(
                        (probs_member >= threshold).to(torch.int64))
                vote_sum = torch.stack(member_masks, dim=0).sum(dim=0)
                pred = (vote_sum >= votes_needed).to(torch.int64)
                probs = torch.stack(member_probs, dim=0).mean(dim=0)

            y_true_valid = y_row[valid_mask].to(torch.int64).cpu()
            y_pred_valid = pred[valid_mask].to(torch.int64).cpu()
            y_prob_valid = probs[valid_mask].to(torch.float32).cpu()

            all_true.append(y_true_valid)
            all_pred.append(y_pred_valid)
            all_prob.append(y_prob_valid)

            total_valid += int(y_true_valid.numel())
            total_pos += int((y_true_valid == 1).sum().item())
            total_neg += int((y_true_valid == 0).sum().item())
            total_pred_pos += int((y_pred_valid == 1).sum().item())
            total_pred_neg += int((y_pred_valid == 0).sum().item())

        if batch_idx % 50 == 0:
            logger.info(
                "eval_progress",
                extra={
                    "extra": {
                        "batches": batch_idx,
                        "processed_proteins": processed,
                        "valid_residues": total_valid
                    }
                },
            )

    if not all_true:
        return {
            "processed_proteins": int(processed),
            "proteins_with_valid_residues": 0,
            "proteins_zero_valid_residues": int(zero_valid),
            "valid_residues": 0,
            "pos_residues": 0,
            "neg_residues": 0,
            "pred_pos_residues": 0,
            "pred_neg_residues": 0,
            "accuracy": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "votes_needed": votes_needed if n_members > 1 else None,
            "metrics": {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "mcc": float("nan"),
                "auc": float("nan"),
                "auc10": float("nan"),
                "pr_auc": float("nan"),
                "res_balanced_acc": float("nan"),
                "per_class_acc": [float("nan"), float("nan")]
            }
        }

    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    y_prob = torch.cat(all_prob, dim=0).numpy()

    metrics = compute_eval_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    cm = np.zeros((2, 2), dtype=np.int64)
    cm[0, 0] = int(((y_true == 0) & (y_pred == 0)).sum())
    cm[0, 1] = int(((y_true == 0) & (y_pred == 1)).sum())
    cm[1, 0] = int(((y_true == 1) & (y_pred == 0)).sum())
    cm[1, 1] = int(((y_true == 1) & (y_pred == 1)).sum())

    per_class_acc = np.array(
        [
            float(cm[0, 0]) / max(int(cm[0, 0] + cm[0, 1]), 1),
            float(cm[1, 1]) / max(int(cm[1, 0] + cm[1, 1]), 1),
        ],
        dtype=np.float64,
    )
    res_balanced_acc = float(per_class_acc.mean())
    metrics["per_class_acc"] = [float(x) for x in per_class_acc.tolist()]
    metrics["res_balanced_acc"] = res_balanced_acc

    return {
        "processed_proteins": int(processed),
        "proteins_with_valid_residues": int(processed - zero_valid),
        "proteins_zero_valid_residues": int(zero_valid),
        "valid_residues": int(total_valid),
        "pos_residues": int(total_pos),
        "neg_residues": int(total_neg),
        "pred_pos_residues": int(total_pred_pos),
        "pred_neg_residues": int(total_pred_neg),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size > 0 else float("nan"),
        "confusion_matrix": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
        "votes_needed": votes_needed if n_members > 1 else None,
        "metrics": metrics
    }


def main() -> None:
    """Handles CLI argument parsing and high-level evaluation flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PepSeqPred FFNN checkpoint(s) on labeled embedding shards."
        )
    )
    parser.add_argument(
        "model_artifact",
        type=Path,
        help="Path to model artifact (.pt checkpoint or ensemble manifest .json)."
    )
    parser.add_argument(
        "--embedding-dirs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more directories containing per-protein embeddings."
    )
    parser.add_argument(
        "--label-shards",
        nargs="+",
        required=True,
        type=Path,
        help="One or more label shard .pt files containing a 'labels' dictionary."
    )
    parser.add_argument(
        "--output-json",
        action="store",
        dest="output_json",
        type=Path,
        default=Path("ffnn_eval_summary.json"),
        help="Path to write evaluation summary JSON."
    )
    parser.add_argument(
        "--threshold",
        action="store",
        dest="threshold",
        type=float,
        default=None,
        help="Optional global threshold override in (0.0, 1.0)."
    )
    parser.add_argument(
        "--ensemble-set-index",
        action="store",
        dest="ensemble_set_index",
        type=int,
        default=1,
        help="Set index to use for schema v2 ensemble manifest evaluation."
    )
    parser.add_argument(
        "--k-folds",
        action="store",
        dest="k_folds",
        type=int,
        default=None,
        help="Optional number of ensemble members to evaluate (first K by fold/member order)."
    )
    parser.add_argument(
        "--subset",
        action="store",
        dest="subset",
        type=int,
        default=0,
        help="If > 0, evaluate only first N proteins by deterministic protein_id sort order."
    )
    parser.add_argument(
        "--batch-size",
        action="store",
        dest="batch_size",
        type=int,
        default=64,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--num-workers",
        action="store",
        dest="num_workers",
        type=int,
        default=0,
        help="Number of data loader workers."
    )
    parser.add_argument(
        "--label-cache-mode",
        action="store",
        dest="label_cache_mode",
        type=str,
        choices=["current", "all"],
        default="current",
        help="Label shard cache strategy used by ProteinDataset."
    )
    parser.add_argument(
        "--log-dir",
        action="store",
        dest="log_dir",
        type=Path,
        default=Path("logs"),
        help="Directory for logs."
    )
    parser.add_argument(
        "--log-level",
        action="store",
        dest="log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level choice."
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        dest="log_json",
        default=False,
        help="Emit logs as JSON lines for simple parsing."
    )
    parser.add_argument(
        "--emb-dim",
        action="store",
        dest="emb_dim",
        type=int,
        default=None,
        help="Explicit model embedding dimension used in training."
    )
    parser.add_argument(
        "--hidden-sizes",
        action="store",
        dest="hidden_sizes",
        type=str,
        default=None,
        help="Explicit hidden sizes CSV (e.g. 150,120,45)."
    )
    parser.add_argument(
        "--dropouts",
        action="store",
        dest="dropouts",
        type=str,
        default=None,
        help="Explicit dropouts CSV (e.g. 0.1,0.1,0.1)."
    )
    parser.add_argument(
        "--num-classes",
        action="store",
        dest="num_classes",
        type=int,
        default=None,
        help="Explicit output classes (binary=1)."
    )
    parser.add_argument(
        "--use-layer-norm",
        action="store_true",
        dest="use_layer_norm",
        help="Explicitly use layer normalization."
    )
    parser.add_argument(
        "--use-residual",
        action="store_true",
        dest="use_residual",
        help="Explicitly use residual connections."
    )
    parser.add_argument(
        "--no-use-layer-norm",
        action="store_false",
        dest="use_layer_norm",
        help="Explicitly disable layer normalization."
    )
    parser.add_argument(
        "--no-use-residual",
        action="store_false",
        dest="use_residual",
        help="Explicitly disable residual connections."
    )
    parser.set_defaults(use_layer_norm=None, use_residual=None)
    args = parser.parse_args()

    if args.ensemble_set_index < 1:
        raise ValueError("--ensemble-set-index must be >= 1")
    if args.k_folds is not None and args.k_folds < 1:
        raise ValueError("--k-folds must be >= 1")
    if args.threshold is not None and (args.threshold <= 0.0 or args.threshold >= 1.0):
        raise ValueError("--threshold must be between (0.0, 1.0)")
    if args.subset < 0:
        raise ValueError("--subset must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    json_indent = 2 if args.log_json else None
    logger = setup_logger(
        log_dir=args.log_dir,
        log_level=args.log_level,
        json_lines=args.log_json,
        json_indent=json_indent,
        name="eval_ffnn_cli"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_mode, member_specs, artifact_meta = _resolve_evaluation_members(
        model_artifact=args.model_artifact,
        ensemble_set_index=args.ensemble_set_index,
        k_folds=args.k_folds
    )

    cli_model_cfg = _build_cli_model_config(args)
    psp_models: List[torch.nn.Module] = []
    member_thresholds: List[float] = []
    member_model_cfgs: List[FFNNModelConfig] = []
    member_model_cfg_srcs: List[str] = []

    for spec in member_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")

        checkpoint = torch.load(
            spec.checkpoint, map_location="cpu", weights_only=True)
        psp_model, model_cfg, model_cfg_src = build_model_from_checkpoint(
            checkpoint=checkpoint, device=device, model_config=cli_model_cfg
        )
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else (
                spec.threshold
                if spec.threshold is not None
                else infer_decision_threshold(checkpoint)
            )
        )
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(
                f"Invalid threshold for member checkpoint: {spec.checkpoint}")

        psp_models.append(psp_model)
        member_thresholds.append(float(threshold))
        member_model_cfgs.append(model_cfg)
        member_model_cfg_srcs.append(model_cfg_src)

    if not psp_models:
        raise ValueError("No evaluation models were loaded")

    emb_dims = {int(cfg.emb_dim) for cfg in member_model_cfgs}
    if len(emb_dims) != 1:
        raise ValueError(
            "All ensemble members must share emb_dim for shared-embedding evaluation, "
            f"got {sorted(emb_dims)}"
        )

    base_dataset = ProteinDataset(
        embedding_dirs=args.embedding_dirs,
        label_shards=args.label_shards,
        protein_ids=None,
        window_size=None,
        stride=1,
        collapse_labels=True,
        pad_last_window=False,
        return_meta=False,
        cache_current_label_shard=True,
        drop_label_after_use=False,
        label_cache_mode=args.label_cache_mode
    )
    candidate_ids = list(base_dataset.protein_ids)
    if args.subset > 0:
        selected_ids = list(candidate_ids[: int(args.subset)])
    else:
        selected_ids = list(candidate_ids)
    if len(selected_ids) == 0:
        raise ValueError("No proteins available for evaluation")

    eval_dataset = ProteinDataset(
        embedding_dirs=args.embedding_dirs,
        label_shards=args.label_shards,
        protein_ids=selected_ids,
        label_index=base_dataset.label_index,
        embedding_index=base_dataset.embedding_index,
        window_size=None,
        stride=1,
        collapse_labels=True,
        pad_last_window=False,
        return_meta=False,
        cache_current_label_shard=True,
        drop_label_after_use=False,
        label_cache_mode=args.label_cache_mode
    )

    pin = torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin,
        "collate_fn": pad_collate
    }
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    eval_loader = DataLoader(eval_dataset, **loader_kwargs)

    first_cfg = member_model_cfgs[0]
    logger.info(
        "eval_init",
        extra={
            "extra": {
                "model_artifact": str(args.model_artifact),
                "artifact_mode": artifact_mode,
                "manifest_schema_version": artifact_meta.get("schema_version"),
                "ensemble_set_index": artifact_meta.get("set_index"),
                "requested_ensemble_set_index": int(args.ensemble_set_index),
                "k_folds": args.k_folds,
                "n_members": int(len(psp_models)),
                "threshold_mode": (
                    "global-override"
                    if args.threshold is not None
                    else "per-member-default"
                ),
                "threshold": float(member_thresholds[0]) if len(member_thresholds) == 1 else None,
                "member_thresholds": [float(x) for x in member_thresholds],
                "device": device,
                "model_cfg_src": (
                    str(member_model_cfg_srcs[0])
                    if len(set(member_model_cfg_srcs)) == 1
                    else "mixed"
                ),
                "emb_dim": int(first_cfg.emb_dim),
                "hidden_sizes": [int(x) for x in first_cfg.hidden_sizes],
                "use_layer_norm": bool(first_cfg.use_layer_norm),
                "use_residual": bool(first_cfg.use_residual),
                "num_classes": int(first_cfg.num_classes),
                "n_candidate_proteins": int(len(candidate_ids)),
                "n_selected_proteins": int(len(selected_ids))
            }
        }
    )

    eval_out = _evaluate_dataset(
        psp_models=psp_models,
        thresholds=member_thresholds,
        eval_loader=eval_loader,
        device=device,
        logger=logger
    )

    summary = {
        "model_artifact": str(args.model_artifact),
        "artifact_mode": artifact_mode,
        "manifest_schema_version": artifact_meta.get("schema_version"),
        "ensemble_set_index": artifact_meta.get("set_index"),
        "requested_ensemble_set_index": int(args.ensemble_set_index),
        "k_folds": args.k_folds,
        "n_members": int(len(psp_models)),
        "threshold_mode": (
            "global-override" if args.threshold is not None else "per-member-default"
        ),
        "threshold": float(member_thresholds[0]) if len(member_thresholds) == 1 else None,
        "member_thresholds": [float(x) for x in member_thresholds],
        "subset": int(args.subset),
        "n_candidate_proteins": int(len(candidate_ids)),
        "n_selected_proteins": int(len(selected_ids)),
        "embedding_dirs": [str(p) for p in args.embedding_dirs],
        "label_shards": [str(p) for p in args.label_shards],
        "device": device,
        "model_cfg_src": (
            str(member_model_cfg_srcs[0])
            if len(set(member_model_cfg_srcs)) == 1
            else "mixed"
        ),
        "emb_dim": int(first_cfg.emb_dim),
        "hidden_sizes": [int(x) for x in first_cfg.hidden_sizes],
        "use_layer_norm": bool(first_cfg.use_layer_norm),
        "use_residual": bool(first_cfg.use_residual),
        "num_classes": int(first_cfg.num_classes),
        "evaluation": eval_out
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_sanitize_for_json(summary), indent=2, allow_nan=False),
        encoding="utf-8"
    )

    logger.info(
        "eval_done",
        extra={
            "extra": {
                "output_json": str(args.output_json),
                "processed_proteins": int(eval_out["processed_proteins"]),
                "valid_residues": int(eval_out["valid_residues"]),
                "accuracy": eval_out["accuracy"],
                "metrics": eval_out["metrics"]
            }
        }
    )


if __name__ == "__main__":
    main()
