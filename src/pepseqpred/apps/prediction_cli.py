"""prediction_cli.py

Predict residue-level binary epitope masks for proteins from FASTA input.

Usage
-----
>>> # HPC usage with shell script
>>> sbatch predictepitope.sh <model artifact path> <input .fasta path> <output .fasta path>

>>> # single-model example
>>> python prediction_cli.py <checkpoint .pt path> <input .fasta path> --output-fasta <output .fasta path>

>>> # ensemble-manifest example
>>> python prediction_cli.py <ensemble_manifest.json> <input .fasta path> --k-folds 5
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple
import torch
import esm
from pepseqpred.core.io.logger import setup_logger
from pepseqpred.core.io.read import parse_int_csv, parse_float_csv
from pepseqpred.core.embeddings.esm2 import clean_seq
from pepseqpred.core.predict.inference import (
    FFNNModelConfig,
    build_model_from_checkpoint,
    embed_protein_seq,
    infer_decision_threshold,
    predict_ensemble_from_embedding,
    predict_from_embedding
)


@dataclass(frozen=True)
class PredictionMember:
    checkpoint: Path
    threshold: float | None
    fold_index: int | None
    member_index: int | None


def read_fasta_records(fasta_path: Path | str) -> Iterator[Tuple[str, str]]:
    """
    Yields (header, protein_sequence) from FASTA, header text is preserved.
    """
    header = None
    seq_lines = []
    with open(fasta_path, "r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, clean_seq("".join(seq_lines))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            yield header, clean_seq("".join(seq_lines))


def _build_cli_model_config(args: argparse.Namespace) -> FFNNModelConfig | None:
    """Builds the model configuration using CLI arguments."""
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
            "--hidden-sizes and --dropouts must be the same length"
        )
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


def _member_sort_key(member: PredictionMember) -> Tuple[int, int, str]:
    """Sort key for deterministic member selection."""
    fold_index = member.fold_index if member.fold_index is not None else int(
        1e9)
    member_index = member.member_index if member.member_index is not None else int(
        1e9)
    return (fold_index, member_index, str(member.checkpoint))


def _resolve_manifest_members(
    manifest_path: Path,
    ensemble_set_index: int
) -> Tuple[List[PredictionMember], Dict[str, Any]]:
    """Resolves valid members from schema v1/v2 ensemble manifests."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in ensemble manifest: {manifest_path}") from e
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

    members: List[PredictionMember] = []
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
            PredictionMember(
                checkpoint=checkpoint_path,
                threshold=_as_optional_threshold(member.get("threshold")),
                fold_index=_as_optional_int(member.get("fold_index")),
                member_index=_as_optional_int(member.get("member_index")),
            )
        )

    if not members:
        raise ValueError(
            "No valid ensemble members found (requires status=OK and checkpoint path)")

    members_sorted = sorted(members, key=_member_sort_key)
    meta: Dict[str, Any] = {
        "schema_version": schema_version,
        "set_index": _as_optional_int(selected_set.get("set_index")) if isinstance(selected_set, Mapping) else None,
        "n_members_total": len(members_raw),
        "n_members_valid": len(members_sorted),
    }
    return members_sorted, meta


def _resolve_prediction_members(
    model_artifact: Path,
    ensemble_set_index: int,
    k_folds: int | None
) -> Tuple[str, List[PredictionMember], Dict[str, Any]]:
    """Resolves prediction members from either a checkpoint path or ensemble manifest."""
    suffix = model_artifact.suffix.lower()
    if suffix == ".pt":
        members = [
            PredictionMember(
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
            "n_members_valid": 1,
        }
    elif suffix == ".json":
        members, meta = _resolve_manifest_members(
            manifest_path=model_artifact,
            ensemble_set_index=ensemble_set_index
        )
        mode = "ensemble-manifest"
    else:
        raise ValueError(
            f"Unsupported model artifact type '{model_artifact.suffix}'. Expected .pt checkpoint or .json ensemble manifest."
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


def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Prediction program."""
    parser = argparse.ArgumentParser(
        description="Predict residue-level binary epitope masks for proteins in FASTA.")
    parser.add_argument("model_artifact",
                        type=Path,
                        help="Path to model artifact (.pt checkpoint or ensemble manifest .json).")
    parser.add_argument("fasta_input",
                        type=Path,
                        help="Input FASTA with taxonomy/protein headers and protein sequences.")
    parser.add_argument("--output-fasta",
                        action="store",
                        dest="output_fasta",
                        type=Path,
                        default=Path("predictions.fasta"),
                        help="Output FASTA path for binary epitope masks."
                        )
    parser.add_argument("--threshold",
                        action="store",
                        dest="threshold",
                        type=float,
                        default=None,
                        help="Optional global threshold override within (0.0, 1.0).")
    parser.add_argument("--ensemble-set-index",
                        action="store",
                        dest="ensemble_set_index",
                        type=int,
                        default=1,
                        help="Set index to use when model artifact is schema v2 ensemble manifest.")
    parser.add_argument("--k-folds",
                        action="store",
                        dest="k_folds",
                        type=int,
                        default=None,
                        help="Optional number of ensemble members to use; selects first K by fold/member order.")
    parser.add_argument("--model-name",
                        action="store",
                        dest="model_name",
                        type=str,
                        default="esm2_t33_650M_UR50D",
                        help="ESM model name to generate embeddings.")
    parser.add_argument("--max-tokens",
                        action="store",
                        dest="max_tokens",
                        type=int,
                        default=1022,
                        help="ESM residue token budget excluding CLS and EOS.")
    parser.add_argument("--log-dir",
                        action="store",
                        dest="log_dir",
                        type=Path,
                        default=Path("logs"),
                        help="Directory for logs.")
    parser.add_argument("--log-level",
                        action="store",
                        dest="log_level",
                        type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level choice.")
    parser.add_argument("--log-json",
                        action="store_true",
                        dest="log_json",
                        default=False,
                        help="Emit logs as JSON lines for simple parsing.")
    parser.add_argument("--emb-dim",
                        action="store",
                        dest="emb_dim",
                        type=int,
                        default=None,
                        help="Explicit model embedding dimension used in training.")
    parser.add_argument("--hidden-sizes",
                        action="store",
                        dest="hidden_sizes",
                        type=str,
                        default=None,
                        help="Explicit hidden sizes CSV (e.g. 150,120,45).")
    parser.add_argument("--dropouts",
                        action="store",
                        dest="dropouts",
                        type=str,
                        default=None,
                        help="Explicit dropouts CSV (e.g. 0.1,0.1,0.1).")
    parser.add_argument("--num-classes",
                        action="store",
                        dest="num_classes",
                        type=int,
                        default=None,
                        help="Explicit output classes (binary=1).")
    parser.add_argument("--use-layer-norm",
                        action="store_true",
                        dest="use_layer_norm",
                        help="Explicit use layer normalization.")
    parser.add_argument("--use-residual",
                        action="store_true",
                        dest="use_residual",
                        help="Explicit use resdiuals.")
    parser.add_argument("--no-use-layer-norm",
                        action="store_false",
                        dest="use_layer_norm",
                        help="Explicit DO NOT use layer normalization.")
    parser.add_argument("--no-use-residual",
                        action="store_false",
                        dest="use_residual",
                        help="Explicit DO NOT use resdiuals.")
    parser.set_defaults(use_layer_norm=None, use_residual=None)
    args = parser.parse_args()

    if args.ensemble_set_index < 1:
        raise ValueError("--ensemble-set-index must be >= 1")
    if args.k_folds is not None and args.k_folds < 1:
        raise ValueError("--k-folds must be >= 1")
    if args.threshold is not None and (args.threshold <= 0.0 or args.threshold >= 1.0):
        raise ValueError("--threshold must be between (0.0, 1.0)")

    json_indent = 2 if args.log_json else None
    logger = setup_logger(log_dir=args.log_dir,
                          log_level=args.log_level,
                          json_lines=args.log_json,
                          json_indent=json_indent,
                          name="prediction_cli")

    # load ESM embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_model, alphabet = getattr(esm.pretrained, args.model_name)()
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    # set model layer to use (33 for default model)
    layer = esm_model.num_layers

    artifact_mode, member_specs, artifact_meta = _resolve_prediction_members(
        model_artifact=args.model_artifact,
        ensemble_set_index=args.ensemble_set_index,
        k_folds=args.k_folds,
    )

    # load all member models from disk
    cli_model_cfg = _build_cli_model_config(args)
    psp_models = []
    member_thresholds = []
    member_model_cfgs = []
    member_model_cfg_srcs = []
    for spec in member_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")

        checkpoint = torch.load(
            spec.checkpoint, map_location="cpu", weights_only=True)
        psp_model, model_cfg, model_cfg_src = build_model_from_checkpoint(
            checkpoint,
            device=device,
            model_config=cli_model_cfg
        )
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else (spec.threshold if spec.threshold is not None else infer_decision_threshold(checkpoint))
        )
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(
                f"Invalid threshold for member checkpoint: {spec.checkpoint}")

        psp_models.append(psp_model)
        member_thresholds.append(float(threshold))
        member_model_cfgs.append(model_cfg)
        member_model_cfg_srcs.append(model_cfg_src)

    if not psp_models:
        raise ValueError("No prediction models were loaded")

    emb_dims = {int(cfg.emb_dim) for cfg in member_model_cfgs}
    if len(emb_dims) != 1:
        raise ValueError(
            f"All ensemble members must share emb_dim for shared-embedding inference, got {sorted(emb_dims)}"
        )

    first_cfg = member_model_cfgs[0]
    logger.info("prediction_init",
                extra={"extra": {
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
                    "device": device,
                    "model_cfg_src": str(member_model_cfg_srcs[0]) if len(set(member_model_cfg_srcs)) == 1 else "mixed",
                    "emb_dim": int(first_cfg.emb_dim),
                    "hidden_sizes": [int(x) for x in first_cfg.hidden_sizes],
                    "use_layer_norm": bool(first_cfg.use_layer_norm),
                    "use_residual": bool(first_cfg.use_residual),
                    "num_classes": int(first_cfg.num_classes)
                }})

    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    failed = 0
    total_residues = 0
    total_epitopes = 0

    with open(args.output_fasta, "w", encoding="utf-8") as out_f:
        for header, protein_seq in read_fasta_records(args.fasta_input):
            try:
                protein_emb = embed_protein_seq(
                    protein_seq=protein_seq,
                    esm_model=esm_model,
                    layer=layer,
                    batch_converter=batch_converter,
                    device=device,
                    max_tokens=args.max_tokens
                )
                if len(psp_models) == 1:
                    pred = predict_from_embedding(
                        psp_model=psp_models[0],
                        protein_emb=protein_emb,
                        device=device,
                        threshold=float(member_thresholds[0])
                    )
                else:
                    pred = predict_ensemble_from_embedding(
                        psp_models=psp_models,
                        protein_emb=protein_emb,
                        device=device,
                        thresholds=member_thresholds
                    )
                out_f.write(f">{header}\n{pred['binary_mask']}\n")

                # increment progress
                processed += 1
                total_residues += int(pred["length"])
                total_epitopes += int(pred["n_epitopes"])

                # log every 10 proteins processed
                if processed % 10 == 0:
                    logger.info("prediction_progress",
                                extra={"extra": {
                                    "processed": processed,
                                    "failed": failed
                                }})

            except Exception as e:
                failed += 1
                logger.error("prediction_failed",
                             extra={"extra": {
                                 "header": header,
                                 "error": str(e)
                             }})

    logger.info("prediction_done",
                extra={"extra": {
                    "output_fasta": str(args.output_fasta),
                    "processed": processed,
                    "failed": failed,
                    "total_residues": total_residues,
                    "total_epitopes": total_epitopes
                }})


if __name__ == "__main__":
    main()
