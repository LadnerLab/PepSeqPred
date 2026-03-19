import json
from pathlib import Path
import pytest
from pepseqpred.apps.prediction_cli import _resolve_prediction_members

pytestmark = pytest.mark.unit


def test_resolve_prediction_members_single_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "single.pt"
    checkpoint.touch()

    mode, members, meta = _resolve_prediction_members(
        model_artifact=checkpoint,
        ensemble_set_index=1,
        k_folds=None
    )

    assert mode == "single-checkpoint"
    assert len(members) == 1
    assert members[0].checkpoint == checkpoint
    assert meta["n_members_valid"] == 1


def test_resolve_prediction_members_manifest_v1_filters_sorts_and_caps_k(tmp_path: Path):
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_skip = tmp_path / "skip.pt"
    ckpt_a.touch()
    ckpt_b.touch()
    ckpt_skip.touch()

    manifest = {
        "schema_version": 1,
        "members": [
            {"member_index": 2, "fold_index": 2, "status": "OK", "checkpoint": str(ckpt_b), "threshold": 0.6},
            {"member_index": 1, "fold_index": 1, "status": "OK", "checkpoint": "a.pt", "threshold": 0.4},
            {"member_index": 3, "fold_index": 3, "status": "FAIL", "checkpoint": str(ckpt_skip), "threshold": 0.5},
        ]
    }
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    mode, members, meta = _resolve_prediction_members(
        model_artifact=manifest_path,
        ensemble_set_index=1,
        k_folds=1
    )

    assert mode == "ensemble-manifest"
    assert len(members) == 1
    assert members[0].checkpoint == ckpt_a.resolve()
    assert members[0].fold_index == 1
    assert meta["n_members_total"] == 3
    assert meta["n_members_valid"] == 2


def test_resolve_prediction_members_manifest_v2_selects_set(tmp_path: Path):
    ckpt_1 = tmp_path / "set1.pt"
    ckpt_2 = tmp_path / "set2.pt"
    ckpt_1.touch()
    ckpt_2.touch()
    manifest = {
        "schema_version": 2,
        "sets": [
            {"set_index": 1, "members": [{"status": "OK", "checkpoint": str(ckpt_1), "fold_index": 1, "member_index": 1}]},
            {"set_index": 2, "members": [{"status": "OK", "checkpoint": str(ckpt_2), "fold_index": 1, "member_index": 1}]},
        ]
    }
    manifest_path = tmp_path / "ensemble_root.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, members, meta = _resolve_prediction_members(
        model_artifact=manifest_path,
        ensemble_set_index=2,
        k_folds=None
    )

    assert len(members) == 1
    assert members[0].checkpoint == ckpt_2
    assert meta["set_index"] == 2


def test_resolve_prediction_members_manifest_v2_invalid_set_raises(tmp_path: Path):
    ckpt = tmp_path / "set1.pt"
    ckpt.touch()
    manifest = {
        "schema_version": 2,
        "sets": [
            {"set_index": 1, "members": [{"status": "OK", "checkpoint": str(ckpt), "fold_index": 1, "member_index": 1}]}
        ]
    }
    manifest_path = tmp_path / "ensemble_root.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ensemble-set-index"):
        _resolve_prediction_members(
            model_artifact=manifest_path,
            ensemble_set_index=2,
            k_folds=None
        )


def test_resolve_prediction_members_k_folds_exceeds_members_raises(tmp_path: Path):
    ckpt = tmp_path / "single.pt"
    ckpt.touch()

    with pytest.raises(ValueError, match="exceeds available valid members"):
        _resolve_prediction_members(
            model_artifact=ckpt,
            ensemble_set_index=1,
            k_folds=2
        )
