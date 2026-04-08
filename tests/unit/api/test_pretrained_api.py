import pytest
import torch

import pepseqpred.api.predictor as predictor_mod
from pepseqpred.api import (
    PepSeqPredictor,
    list_pretrained_models,
    load_pretrained_predictor,
)
from pepseqpred.api.pretrainedregistry import (
    canonicalize_pretrained_model_id,
    open_pretrained_model_artifact,
)
from pepseqpred.core.predict.artifacts import resolve_prediction_members

pytestmark = pytest.mark.unit


class _FakeAlphabet:
    def get_batch_converter(self):
        return lambda pairs: ([], [], torch.zeros((len(pairs), 3), dtype=torch.long))


class _FakeEsm:
    num_layers = 3

    def eval(self):
        return self

    def to(self, _device):
        return self


def test_pretrained_registry_alias_resolution_and_unknown_errors():
    assert canonicalize_pretrained_model_id(None) == "flagship2-v1"
    assert canonicalize_pretrained_model_id("default") == "flagship2-v1"
    assert canonicalize_pretrained_model_id("flagship2") == "flagship2-v1"
    assert canonicalize_pretrained_model_id("flagship1") == "flagship1-v1"
    assert canonicalize_pretrained_model_id("flagship1-v1") == "flagship1-v1"

    with pytest.raises(ValueError, match="Unknown pretrained model_id"):
        canonicalize_pretrained_model_id("missing-model")


def test_list_pretrained_models_returns_expected_catalog():
    models = list_pretrained_models()
    ids = [m.model_id for m in models]
    assert ids == ["flagship1-v1", "flagship2-v1"]

    defaults = [m.model_id for m in models if m.is_default]
    assert defaults == ["flagship2-v1"]

    alias_map = {m.model_id: set(m.aliases) for m in models}
    assert "flagship1" in alias_map["flagship1-v1"]
    assert {"flagship2", "default"}.issubset(alias_map["flagship2-v1"])


def test_pretrained_resources_exist_and_resolve_members():
    for model in list_pretrained_models():
        with open_pretrained_model_artifact(model.model_id) as (artifact_path, info):
            assert info.model_id == model.model_id
            assert artifact_path.exists()
            mode, members, _ = resolve_prediction_members(artifact_path)
            assert mode == "ensemble-manifest"
            assert len(members) == model.n_members
            assert all(member.checkpoint.exists() for member in members)


def test_from_pretrained_uses_registry_and_adds_pretrained_meta(monkeypatch):
    monkeypatch.setattr(
        predictor_mod,
        "_load_esm",
        lambda *_args, **_kwargs: (
            _FakeEsm(),
            _FakeAlphabet().get_batch_converter(),
            3,
        ),
    )
    predictor = PepSeqPredictor.from_pretrained(
        "flagship1-v1", device="cpu", k_folds=2)

    assert predictor.n_members == 2
    assert predictor.pretrained_meta["model_id"] == "flagship1-v1"
    assert predictor.pretrained_meta["expected_esm_model"] == "esm2_t33_650M_UR50D"
    assert predictor.pretrained_meta["provenance"]["source_set_index"] == 1

    result = predictor._payload_to_result(
        header="protein_1",
        sequence="ACD",
        payload={
            "binary_mask": "101",
            "length": 3,
            "n_epitopes": 2,
            "frac_epitope": 2 / 3,
            "p_epitope_mean": 0.4,
            "p_epitope_max": 0.9,
            "threshold": 0.5,
            "n_members": 2,
            "votes_needed": 2,
            "member_thresholds": [0.4, 0.5],
        },
        used_thresholds=(0.4, 0.5),
    )
    assert result.meta["pretrained"]["model_id"] == "flagship1-v1"


def test_from_pretrained_passthrough_to_from_artifact(monkeypatch):
    calls = {}

    def _fake_from_artifact(cls, model_artifact, **kwargs):
        calls["model_artifact"] = model_artifact
        calls.update(kwargs)
        return cls(
            psp_models=[object()],
            member_thresholds=[0.5],
            esm_model=_FakeEsm(),
            batch_converter=_FakeAlphabet().get_batch_converter(),
            layer=3,
            device="cpu",
            model_name="esm2_t33_650M_UR50D",
            max_tokens=int(kwargs.get("max_tokens", 1022)),
            artifact_mode="ensemble-manifest",
            artifact_meta={},
        )

    monkeypatch.setattr(
        predictor_mod.PepSeqPredictor,
        "from_artifact",
        classmethod(_fake_from_artifact),
    )

    out = PepSeqPredictor.from_pretrained(
        "flagship2",
        threshold=0.62,
        k_folds=3,
        max_tokens=777,
        device="cpu",
    )
    assert isinstance(out, PepSeqPredictor)
    assert calls["threshold"] == pytest.approx(0.62)
    assert calls["k_folds"] == 3
    assert calls["max_tokens"] == 777
    assert calls["device"] == "cpu"
    assert calls["model_name"] == "esm2_t33_650M_UR50D"


def test_load_pretrained_predictor_helper(monkeypatch):
    sentinel = object()
    captured = {}

    def _fake_from_pretrained(cls, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        predictor_mod.PepSeqPredictor,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    out = load_pretrained_predictor(
        model_id="flagship1", device="cpu", k_folds=1)
    assert out is sentinel
    assert captured["model_id"] == "flagship1"
    assert captured["device"] == "cpu"
    assert captured["k_folds"] == 1
