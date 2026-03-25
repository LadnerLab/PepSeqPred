import json
import sys
from pathlib import Path
import pytest
import torch
import pepseqpred.apps.evaluate_ffnn_cli as evaluate_ffnn_cli
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


def test_eval_ffnn_cli_single_checkpoint_smoke(training_artifacts, tmp_path: Path, monkeypatch):
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
    assert eval_out["processed_proteins"] == 4
    assert eval_out["valid_residues"] == 24
    assert eval_out["pred_pos_residues"] == 0
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_eval_ffnn_cli_manifest_majority_vote_smoke(training_artifacts, tmp_path: Path, monkeypatch):
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
    assert payload["member_thresholds"] == [0.6, 0.6, 0.4]
    assert eval_out["votes_needed"] == 2
    assert eval_out["valid_residues"] == 24
    assert eval_out["pred_pos_residues"] == 0
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
