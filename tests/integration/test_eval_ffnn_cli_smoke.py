import json
import sys
from pathlib import Path

import pytest
import torch

import pepseqpred.apps.evaluate_ffnn_cli as evaluate_ffnn_cli
from pepseqpred.core.models.factory import (
    PepSeqModelConfig,
    build_pepseq_model,
    model_config_to_dict,
)
from pepseqpred.core.models.ffnn import PepSeqFFNN

pytestmark = pytest.mark.integration


def _write_checkpoint(path: Path, threshold: float = 0.5) -> None:
    model = PepSeqFFNN(
        emb_dim=4,
        hidden_sizes=(3,),
        dropouts=(0.0,),
        use_layer_norm=False,
        use_residual=False,
        num_classes=1,
    )
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": {"threshold": float(threshold)},
        },
        path,
    )


def _write_conv_checkpoint(path: Path, threshold: float = 0.5) -> None:
    cfg = PepSeqModelConfig(
        emb_dim=4,
        hidden_sizes=(3,),
        dropouts=(0.0,),
        num_classes=1,
        use_layer_norm=False,
        use_residual=False,
        model_head="conv1d",
        conv_channels=3,
        conv_layers=1,
        conv_kernel_size=3,
        conv_dropout=0.0,
    )
    model = build_pepseq_model(cfg)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config_to_dict(cfg),
            "metrics": {"threshold": float(threshold)},
        },
        path,
    )


def test_eval_ffnn_cli_single_checkpoint_smoke(
    training_artifacts, tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "single.pt"
    _write_checkpoint(checkpoint, threshold=0.6)

    output_json = tmp_path / "eval_single.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_ffnn_cli.py",
            str(checkpoint),
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--output-json",
            str(output_json),
        ],
    )

    evaluate_ffnn_cli.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    eval_out = payload["evaluation"]
    metrics = eval_out["metrics"]

    assert payload["artifact_mode"] == "single-checkpoint"
    assert payload["n_members"] == 1
    assert payload["threshold"] == pytest.approx(0.6)
    assert payload["ensemble_aggregation"] == "majority"
    assert payload["ensemble_threshold"] is None
    assert eval_out["processed_proteins"] == 4
    assert eval_out["valid_residues"] == 24
    assert eval_out["pred_pos_residues"] == 0
    assert eval_out["ensemble_aggregation"] == "single-model"
    assert isinstance(eval_out["threshold_grid"], list)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_eval_ffnn_cli_loads_conv_checkpoint_metadata(
    training_artifacts, tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "conv_single.pt"
    _write_conv_checkpoint(checkpoint, threshold=0.4)

    output_json = tmp_path / "eval_conv.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_ffnn_cli.py",
            str(checkpoint),
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--output-json",
            str(output_json),
        ],
    )

    evaluate_ffnn_cli.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["model_cfg_src"] == "checkpoint"
    assert payload["model_head"] == "conv1d"
    assert payload["conv_channels"] == 3
    assert payload["evaluation"]["processed_proteins"] == 4


def test_eval_ffnn_cli_manifest_majority_vote_smoke(
    training_artifacts, tmp_path: Path, monkeypatch
):
    ckpt_1 = tmp_path / "fold_1.pt"
    ckpt_2 = tmp_path / "fold_2.pt"
    ckpt_3 = tmp_path / "fold_3.pt"
    _write_checkpoint(ckpt_1, threshold=0.6)
    _write_checkpoint(ckpt_2, threshold=0.6)
    _write_checkpoint(ckpt_3, threshold=0.4)

    manifest = {
        "schema_version": 1,
        "members": [
            {
                "member_index": 1,
                "fold_index": 1,
                "status": "OK",
                "checkpoint": str(ckpt_1),
                "threshold": 0.6,
            },
            {
                "member_index": 2,
                "fold_index": 2,
                "status": "OK",
                "checkpoint": str(ckpt_2),
                "threshold": 0.6,
            },
            {
                "member_index": 3,
                "fold_index": 3,
                "status": "OK",
                "checkpoint": str(ckpt_3),
                "threshold": 0.4,
            },
        ],
    }
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_json = tmp_path / "eval_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_ffnn_cli.py",
            str(manifest_path),
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--output-json",
            str(output_json),
        ],
    )

    evaluate_ffnn_cli.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    eval_out = payload["evaluation"]
    metrics = eval_out["metrics"]

    assert payload["artifact_mode"] == "ensemble-manifest"
    assert payload["n_members"] == 3
    assert payload["threshold"] is None
    assert payload["ensemble_aggregation"] == "majority"
    assert payload["ensemble_threshold"] is None
    assert payload["member_thresholds"] == [0.6, 0.6, 0.4]
    assert eval_out["votes_needed"] == 2
    assert eval_out["ensemble_aggregation"] == "majority"
    assert eval_out["ensemble_threshold"] is None
    assert isinstance(eval_out["threshold_grid"], list)
    assert eval_out["valid_residues"] == 24
    assert eval_out["pred_pos_residues"] == 0
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_eval_ffnn_cli_manifest_mean_prob_smoke(
    training_artifacts, tmp_path: Path, monkeypatch
):
    ckpt_1 = tmp_path / "fold_1.pt"
    ckpt_2 = tmp_path / "fold_2.pt"
    ckpt_3 = tmp_path / "fold_3.pt"
    _write_checkpoint(ckpt_1, threshold=0.6)
    _write_checkpoint(ckpt_2, threshold=0.6)
    _write_checkpoint(ckpt_3, threshold=0.4)

    manifest = {
        "schema_version": 1,
        "members": [
            {
                "member_index": 1,
                "fold_index": 1,
                "status": "OK",
                "checkpoint": str(ckpt_1),
                "threshold": 0.6,
            },
            {
                "member_index": 2,
                "fold_index": 2,
                "status": "OK",
                "checkpoint": str(ckpt_2),
                "threshold": 0.6,
            },
            {
                "member_index": 3,
                "fold_index": 3,
                "status": "OK",
                "checkpoint": str(ckpt_3),
                "threshold": 0.4,
            },
        ],
    }
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_json = tmp_path / "eval_manifest_mean_prob.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_ffnn_cli.py",
            str(manifest_path),
            "--embedding-dirs",
            str(training_artifacts["embedding_dir"]),
            "--label-shards",
            str(training_artifacts["label_shard"]),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--ensemble-aggregation",
            "mean-prob",
            "--ensemble-threshold",
            "0.49",
            "--output-json",
            str(output_json),
        ],
    )

    evaluate_ffnn_cli.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    eval_out = payload["evaluation"]
    metrics = eval_out["metrics"]

    assert payload["artifact_mode"] == "ensemble-manifest"
    assert payload["n_members"] == 3
    assert payload["threshold"] is None
    assert payload["ensemble_aggregation"] == "mean-prob"
    assert payload["ensemble_threshold"] == pytest.approx(0.49)
    assert payload["member_thresholds"] == [0.6, 0.6, 0.4]
    assert eval_out["votes_needed"] is None
    assert eval_out["ensemble_aggregation"] == "mean-prob"
    assert eval_out["ensemble_threshold"] == pytest.approx(0.49)
    assert isinstance(eval_out["threshold_grid"], list)
    assert eval_out["valid_residues"] == 24
    assert eval_out["pred_pos_residues"] == 24
    assert metrics["recall"] == pytest.approx(1.0)
