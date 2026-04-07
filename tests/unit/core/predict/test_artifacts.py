import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from pepseqpred.core.predict.artifacts import (
    PredictionMember,
    _as_optional_int,
    _as_optional_threshold,
    _member_sort_key,
    _resolve_manifest_members,
    _resolve_member_checkpoint_path,
    resolve_prediction_members
)

pytestmark = pytest.mark.unit


@contextmanager
def _scratch_dir():
    root = Path("localdata") / "tmp_pytest_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_artifact_scalar_parsers_and_path_resolution():
    with _scratch_dir() as tmp_path:

        assert _as_optional_int(None) is None
        assert _as_optional_int("7") == 7
        assert _as_optional_int("bad") is None

        assert _as_optional_threshold(None) is None
        assert _as_optional_threshold("0.4") == pytest.approx(0.4)
        assert _as_optional_threshold("bad") is None
        assert _as_optional_threshold(float("nan")) is None
        assert _as_optional_threshold(float("inf")) is None
        assert _as_optional_threshold(0.0) is None
        assert _as_optional_threshold(1.0) is None

        manifest_path = tmp_path / "manifest.json"
        rel = _resolve_member_checkpoint_path("models/m.pt", manifest_path)
        assert rel == (tmp_path / "models" / "m.pt").resolve()

        abs_ckpt = (tmp_path / "abs.pt").resolve()
        resolved = _resolve_member_checkpoint_path(abs_ckpt, manifest_path)
        assert resolved == abs_ckpt

        assert _resolve_member_checkpoint_path(None, manifest_path) is None


def test_member_sort_key_puts_missing_indices_last():
    with _scratch_dir() as tmp_path:

        member_with_idx = PredictionMember(
            checkpoint=(tmp_path / "a.pt"),
            threshold=0.4,
            fold_index=1,
            member_index=1
        )
        member_no_idx = PredictionMember(
            checkpoint=(tmp_path / "b.pt"),
            threshold=0.4,
            fold_index=None,
            member_index=None
        )
        ordered = sorted([member_no_idx, member_with_idx], key=_member_sort_key)
        assert ordered[0] == member_with_idx
        assert ordered[1] == member_no_idx


def test_resolve_manifest_members_v1_filters_and_sorts():
    with _scratch_dir() as tmp_path:

        ckpt_a = tmp_path / "a.pt"
        ckpt_b = tmp_path / "b.pt"
        ckpt_a.touch()
        ckpt_b.touch()

        manifest = {
            "schema_version": 1,
            "members": [
                {"status": "OK", "checkpoint": "b.pt", "fold_index": 2, "member_index": 2, "threshold": "0.7"},
                {"status": "ok", "checkpoint": str(ckpt_a.resolve()), "fold_index": "1", "member_index": "1", "threshold": "bad"},
                {"status": "FAIL", "checkpoint": str(ckpt_a.resolve()), "fold_index": 3, "member_index": 3, "threshold": 0.5},
                {"status": "OK", "checkpoint": None, "fold_index": 4, "member_index": 4, "threshold": 0.5},
                "skip-non-mapping"
            ]
        }
        manifest_path = tmp_path / "manifest_v1.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        members, meta = _resolve_manifest_members(manifest_path, ensemble_set_index=1)

        assert [m.checkpoint for m in members] == [ckpt_a.resolve(), ckpt_b.resolve()]
        assert members[0].threshold is None
        assert members[1].threshold == pytest.approx(0.7)
        assert meta == {
            "schema_version": 1,
            "set_index": None,
            "n_members_total": 5,
            "n_members_valid": 2
        }


def test_resolve_manifest_members_v2_selects_requested_set():
    with _scratch_dir() as tmp_path:

        set1_ckpt = tmp_path / "set1.pt"
        set2_ckpt = tmp_path / "set2.pt"
        set1_ckpt.touch()
        set2_ckpt.touch()

        manifest = {
            "schema_version": 2,
            "sets": [
                {"set_index": 1, "members": [{"status": "OK", "checkpoint": str(set1_ckpt.resolve())}]},
                {"set_index": "2", "members": [{"status": "OK", "checkpoint": str(set2_ckpt.resolve()), "threshold": 0.55}]}
            ]
        }
        manifest_path = tmp_path / "manifest_v2.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        members, meta = _resolve_manifest_members(manifest_path, ensemble_set_index=2)
        assert len(members) == 1
        assert members[0].checkpoint == set2_ckpt.resolve()
        assert members[0].threshold == pytest.approx(0.55)
        assert meta["schema_version"] == 2
        assert meta["set_index"] == 2


def test_resolve_manifest_members_invalid_shapes_raise():
    with _scratch_dir() as tmp_path:

        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _resolve_manifest_members(bad_json, ensemble_set_index=1)

        non_object = tmp_path / "non_object.json"
        non_object.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            _resolve_manifest_members(non_object, ensemble_set_index=1)

        schema2_bad_sets = tmp_path / "schema2_bad_sets.json"
        schema2_bad_sets.write_text(
            json.dumps({"schema_version": 2, "sets": {"bad": True}}),
            encoding="utf-8"
        )
        with pytest.raises(ValueError, match="'sets' list"):
            _resolve_manifest_members(schema2_bad_sets, ensemble_set_index=1)

        schema2_missing_set = tmp_path / "schema2_missing_set.json"
        schema2_missing_set.write_text(
            json.dumps({"schema_version": 2, "sets": [{"set_index": 1, "members": []}]}),
            encoding="utf-8"
        )
        with pytest.raises(ValueError, match="ensemble_set_index=2"):
            _resolve_manifest_members(schema2_missing_set, ensemble_set_index=2)

        members_not_list = tmp_path / "members_not_list.json"
        members_not_list.write_text(
            json.dumps({"schema_version": 1, "members": {"not": "a-list"}}),
            encoding="utf-8"
        )
        with pytest.raises(ValueError, match="'members' list"):
            _resolve_manifest_members(members_not_list, ensemble_set_index=1)

        no_valid_members = tmp_path / "no_valid_members.json"
        no_valid_members.write_text(
            json.dumps({"schema_version": 1, "members": [{"status": "FAIL", "checkpoint": "x.pt"}]}),
            encoding="utf-8"
        )
        with pytest.raises(ValueError, match="No valid ensemble members"):
            _resolve_manifest_members(no_valid_members, ensemble_set_index=1)


def test_resolve_prediction_members_single_and_manifest_kfolds():
    with _scratch_dir() as tmp_path:

        single = tmp_path / "single.pt"
        single.touch()

        mode, members, meta = resolve_prediction_members(single)
        assert mode == "single-checkpoint"
        assert len(members) == 1
        assert members[0].checkpoint == single
        assert meta["n_members_valid"] == 1

        ckpt_a = tmp_path / "a.pt"
        ckpt_b = tmp_path / "b.pt"
        ckpt_a.touch()
        ckpt_b.touch()
        manifest = {
            "schema_version": 1,
            "members": [
                {"status": "OK", "checkpoint": str(ckpt_a.resolve()), "fold_index": 1, "member_index": 1},
                {"status": "OK", "checkpoint": str(ckpt_b.resolve()), "fold_index": 2, "member_index": 1}
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        mode, members, _ = resolve_prediction_members(manifest_path, k_folds=1)
        assert mode == "ensemble-manifest"
        assert len(members) == 1
        assert members[0].checkpoint == ckpt_a.resolve()


def test_resolve_prediction_members_invalid_inputs_raise():
    with _scratch_dir() as tmp_path:

        unsupported = tmp_path / "model.bin"
        unsupported.touch()
        with pytest.raises(ValueError, match="Unsupported model artifact type"):
            resolve_prediction_members(unsupported)

        single = tmp_path / "single.pt"
        single.touch()
        with pytest.raises(ValueError, match="k_folds must be >= 1"):
            resolve_prediction_members(single, k_folds=0)
        with pytest.raises(ValueError, match="exceeds valid member count"):
            resolve_prediction_members(single, k_folds=2)
