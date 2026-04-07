import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

import pepseqpred.api.predictor as predictor_mod
from pepseqpred.api.predictor import (
    PepSeqPredictor,
    _coerce_threshold,
    _load_esm,
    _normalize_sequence_input,
    _read_fasta_records,
    _resolve_device,
    predict_fasta
)
from pepseqpred.core.predict.artifacts import PredictionMember

pytestmark = pytest.mark.unit


@contextmanager
def _scratch_dir():
    root = Path("localdata") / "tmp_pytest_predictor_edges"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _FakeAlphabet:
    def get_batch_converter(self):
        return lambda pairs: ([], [], torch.zeros((1, 3), dtype=torch.long))


class _FakeEsm:
    num_layers = 3

    def eval(self):
        return self

    def to(self, _device):
        return self


def test_predictor_helper_input_normalization_and_device(monkeypatch):
    assert _coerce_threshold(None) is None
    assert _coerce_threshold(0.4) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="between"):
        _coerce_threshold(0.0)

    monkeypatch.setattr(predictor_mod.torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        _resolve_device("cuda")

    with pytest.raises(TypeError, match="not a single string"):
        _normalize_sequence_input("ACDE")
    assert _normalize_sequence_input({"p1": "AAA"}) == [("p1", "AAA")]
    assert _normalize_sequence_input([("p2", "BBB"), "CCC"]) == [("p2", "BBB"), (None, "CCC")]

    with _scratch_dir() as tmp_path:
        fasta = tmp_path / "x.fasta"
        fasta.write_text("\n>h1\nACD\n\n>h2\nEFG\n", encoding="utf-8")
        rows = list(_read_fasta_records(fasta))
        assert rows == [("h1", "ACD"), ("h2", "EFG")]


def test_load_esm_and_from_artifact_validation(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported ESM model name"):
        _load_esm("missing_model", "cpu")

    monkeypatch.setattr(
        predictor_mod.esm,
        "pretrained",
        SimpleNamespace(fake_model=lambda: (_FakeEsm(), _FakeAlphabet()))
    )
    esm_model, _batch_converter, layer = _load_esm("fake_model", "cpu")
    assert isinstance(esm_model, _FakeEsm)
    assert layer == 3

    with _scratch_dir() as tmp_path:
        model_artifact = tmp_path / "missing.pt"
        with pytest.raises(ValueError, match="threshold must be between"):
            PepSeqPredictor.from_artifact(model_artifact, threshold=1.2, model_name="fake_model", device="cpu")
        with pytest.raises(ValueError, match="ensemble_set_index must be >= 1"):
            PepSeqPredictor.from_artifact(model_artifact, ensemble_set_index=0, model_name="fake_model", device="cpu")
        with pytest.raises(ValueError, match="k_folds must be >= 1"):
            PepSeqPredictor.from_artifact(model_artifact, k_folds=0, model_name="fake_model", device="cpu")
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            PepSeqPredictor.from_artifact(model_artifact, max_tokens=0, model_name="fake_model", device="cpu")
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            PepSeqPredictor.from_artifact(model_artifact, model_name="fake_model", device="cpu")


def test_from_artifact_member_validation_and_emb_dim_errors(monkeypatch):
    monkeypatch.setattr(
        predictor_mod.esm,
        "pretrained",
        SimpleNamespace(fake_model=lambda: (_FakeEsm(), _FakeAlphabet()))
    )
    monkeypatch.setattr(predictor_mod, "_resolve_device", lambda d: "cpu")
    monkeypatch.setattr(predictor_mod, "_load_esm", lambda *args, **kwargs: (_FakeEsm(), _FakeAlphabet().get_batch_converter(), 3))

    with _scratch_dir() as tmp_path:
        artifact = tmp_path / "artifact.pt"
        artifact.touch()

        missing_ckpt = tmp_path / "missing_member.pt"
        monkeypatch.setattr(
            predictor_mod,
            "resolve_prediction_members",
            lambda **_kwargs: (
                "single-checkpoint",
                [PredictionMember(checkpoint=missing_ckpt, threshold=0.5, fold_index=1, member_index=1)],
                {}
            )
        )
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            PepSeqPredictor.from_artifact(artifact, model_name="fake_model", device="cpu")

        ckpt = tmp_path / "member.pt"
        ckpt.touch()
        monkeypatch.setattr(
            predictor_mod,
            "resolve_prediction_members",
            lambda **_kwargs: (
                "single-checkpoint",
                [PredictionMember(checkpoint=ckpt, threshold=1.2, fold_index=1, member_index=1)],
                {}
            )
        )
        monkeypatch.setattr(predictor_mod.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
        monkeypatch.setattr(
            predictor_mod,
            "build_model_from_checkpoint",
            lambda **_kwargs: (object(), SimpleNamespace(emb_dim=4), "state_dict")
        )
        with pytest.raises(ValueError, match="Invalid threshold"):
            PepSeqPredictor.from_artifact(artifact, model_name="fake_model", device="cpu")

        monkeypatch.setattr(
            predictor_mod,
            "resolve_prediction_members",
            lambda **_kwargs: ("single-checkpoint", [], {})
        )
        with pytest.raises(ValueError, match="No prediction models"):
            PepSeqPredictor.from_artifact(artifact, model_name="fake_model", device="cpu")

        ckpt_a = tmp_path / "a.pt"
        ckpt_b = tmp_path / "b.pt"
        ckpt_a.touch()
        ckpt_b.touch()
        monkeypatch.setattr(
            predictor_mod,
            "resolve_prediction_members",
            lambda **_kwargs: (
                "ensemble-manifest",
                [
                    PredictionMember(checkpoint=ckpt_a, threshold=0.5, fold_index=1, member_index=1),
                    PredictionMember(checkpoint=ckpt_b, threshold=0.5, fold_index=2, member_index=1),
                ],
                {}
            )
        )

        def _build_model_from_checkpoint(checkpoint, **_kwargs):
            token = checkpoint["token"]
            emb_dim = 4 if token == "a" else 5
            return object(), SimpleNamespace(emb_dim=emb_dim), "state_dict"

        monkeypatch.setattr(predictor_mod.torch, "load", lambda path, **_kwargs: {"token": path.stem})
        monkeypatch.setattr(predictor_mod, "build_model_from_checkpoint", _build_model_from_checkpoint)
        with pytest.raises(ValueError, match="share emb_dim"):
            PepSeqPredictor.from_artifact(artifact, model_name="fake_model", device="cpu")


def test_predictor_payload_and_wrapper_branches(monkeypatch):
    predictor = PepSeqPredictor(
        psp_models=[object(), object()],
        member_thresholds=[0.4, 0.6],
        esm_model=_FakeEsm(),
        batch_converter=_FakeAlphabet().get_batch_converter(),
        layer=3,
        device="cpu",
        model_name="fake_model",
        max_tokens=1022,
        artifact_mode="ensemble-manifest",
        artifact_meta={"set_index": 1}
    )

    assert predictor.artifact_mode == "ensemble-manifest"
    assert predictor.model_name == "fake_model"
    assert predictor.device == "cpu"
    assert predictor.n_members == 2
    assert predictor.artifact_meta["set_index"] == 1

    monkeypatch.setattr(
        predictor_mod,
        "predict_ensemble_from_embedding",
        lambda **_kwargs: {
            "binary_mask": "101",
            "length": 3,
            "n_epitopes": 2,
            "frac_epitope": 2 / 3,
            "p_epitope_mean": 0.6,
            "p_epitope_max": 0.9,
            "threshold": "not-a-float",
            "n_members": 2,
            "votes_needed": 2,
            "member_thresholds": [0.4, 0.6]
        }
    )
    payload, used = predictor._predict_from_embedding(
        protein_emb=torch.zeros((3, 4), dtype=torch.float32),
        threshold=0.7
    )
    assert used == (0.7, 0.7)
    result = predictor._payload_to_result(
        header="h1",
        sequence="ACD",
        payload=payload,
        used_thresholds=used
    )
    assert result.threshold is None
    assert result.member_thresholds == (0.4, 0.6)

    payload_no_member_thresholds = dict(payload)
    payload_no_member_thresholds.pop("member_thresholds")
    result2 = predictor._payload_to_result(
        header="h2",
        sequence="ACD",
        payload=payload_no_member_thresholds,
        used_thresholds=(0.8, 0.8)
    )
    assert result2.member_thresholds == (0.8, 0.8)

    monkeypatch.setattr(predictor_mod, "clean_seq", lambda _s: "")
    with pytest.raises(ValueError, match="empty after cleaning"):
        predictor.predict_sequence("ACD")

    class _FakeWrapperPredictor:
        def predict_fasta(self, **_kwargs):
            return ["no-output-file"]

        def write_fasta_predictions(self, **_kwargs):
            return ["with-output-file"]

    monkeypatch.setattr(predictor_mod, "load_predictor", lambda **_kwargs: _FakeWrapperPredictor())
    assert predict_fasta("x.pt", "in.fasta", output_fasta=None) == ["no-output-file"]
