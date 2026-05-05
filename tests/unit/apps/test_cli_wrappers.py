import argparse
import logging
from pathlib import Path
import pandas as pd
import pytest
import pepseqpred.apps.esm_cli as esm_cli
import pepseqpred.apps.labels_cli as labels_cli
import pepseqpred.apps.prepare_dataset_cli as prepare_dataset_cli
import pepseqpred.apps.preprocess_cli as preprocess_cli

pytestmark = pytest.mark.unit


def test_labels_cli_invokes_builder(monkeypatch, tmp_path: Path):
    captured = {}

    class DummyBuilder:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def build(self, save_path):
            captured["save_path"] = save_path
            return {}

    ns = argparse.Namespace(
        meta_path=tmp_path / "meta.tsv",
        save_path=tmp_path / "labels.pt",
        emb_dirs=[tmp_path / "emb"],
        restrict_to_embeddings=False,
        calc_pos_weight=True,
        embedding_key_delim="-"
    )

    monkeypatch.setattr(
        labels_cli.argparse.ArgumentParser, "parse_args", lambda self: ns
    )
    monkeypatch.setattr(
        labels_cli,
        "setup_logger",
        lambda **kwargs: logging.getLogger("labels_cli_test")
    )
    monkeypatch.setattr(labels_cli, "ProteinLabelBuilder", DummyBuilder)

    labels_cli.main()

    assert captured["init"]["meta_path"] == ns.meta_path
    assert captured["save_path"] == ns.save_path


def test_preprocess_cli_invokes_preprocess(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_preprocess(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame([{"CodeName": "x"}])

    ns = argparse.Namespace(
        meta_file=tmp_path / "meta.tsv",
        z_file=tmp_path / "z.tsv",
        fname_col="FullName",
        code_col="CodeName",
        is_epi_z_min=20.0,
        is_epi_min_subs=4,
        not_epi_z_max=10.0,
        not_epi_max_subs=0,
        subject_prefix="VW_",
        save=True
    )

    monkeypatch.setattr(
        preprocess_cli.argparse.ArgumentParser, "parse_args", lambda self: ns
    )
    monkeypatch.setattr(
        preprocess_cli,
        "setup_logger",
        lambda **kwargs: logging.getLogger("preprocess_cli_test")
    )
    monkeypatch.setattr(preprocess_cli, "preprocess", fake_preprocess)

    preprocess_cli.main()

    assert captured["kwargs"]["fname_col"] == "FullName"
    assert captured["kwargs"]["code_col"] == "CodeName"
    assert captured["kwargs"]["save_path"] is not None


def test_esm_cli_id_mode_invokes_embedding_pipeline(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_esm_embeddings_from_fasta(*args, **kwargs):
        captured["kwargs"] = kwargs
        return pd.DataFrame([{"id": "P1"}]), []

    ns = argparse.Namespace(
        log_dir=Path("logs"),
        log_level="INFO",
        log_json=False,
        per_seq_dir=Path("artifacts/pts"),
        idx_csv_path=Path("artifacts/index.csv"),
        out_dir=tmp_path,
        fasta_file=tmp_path / "in.fasta",
        metadata_file=None,
        metadata_name_col="Name",
        metadata_family_col="Family",
        id_col="ID",
        seq_col="Sequence",
        embedding_key_mode="id",
        key_delimiter="-",
        model_name="fake_model",
        max_tokens=16,
        batch_size=2,
        num_shards=1,
        shard_id=0
    )

    monkeypatch.setattr(esm_cli.argparse.ArgumentParser,
                        "parse_args", lambda self: ns)
    monkeypatch.setattr(
        esm_cli, "setup_logger", lambda **kwargs: logging.getLogger(
            "esm_cli_test")
    )
    monkeypatch.setattr(
        esm_cli,
        "read_fasta",
        lambda _p: pd.DataFrame([{"ID": "P1", "Sequence": "ACD"}])
    )
    monkeypatch.setattr(esm_cli, "esm_embeddings_from_fasta",
                        fake_esm_embeddings_from_fasta)
    monkeypatch.setattr(esm_cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(esm_cli.torch.cuda, "device_count", lambda: 0)

    esm_cli.main()

    assert captured["kwargs"]["key_mode"] == "id"
    assert captured["kwargs"]["id_col"] == "ID"


def test_esm_cli_id_family_requires_metadata(monkeypatch, tmp_path: Path):
    ns = argparse.Namespace(
        log_dir=Path("logs"),
        log_level="INFO",
        log_json=False,
        per_seq_dir=Path("artifacts/pts"),
        idx_csv_path=Path("artifacts/index.csv"),
        out_dir=tmp_path,
        fasta_file=tmp_path / "in.fasta",
        metadata_file=None,
        metadata_name_col="Name",
        metadata_family_col="Family",
        id_col="ID",
        seq_col="Sequence",
        embedding_key_mode="id-family",
        key_delimiter="-",
        model_name="fake_model",
        max_tokens=16,
        batch_size=2,
        num_shards=1,
        shard_id=0
    )

    monkeypatch.setattr(esm_cli.argparse.ArgumentParser,
                        "parse_args", lambda self: ns)
    monkeypatch.setattr(
        esm_cli, "setup_logger", lambda **kwargs: logging.getLogger(
            "esm_cli_test")
    )

    with pytest.raises(ValueError, match="Metadata file is required"):
        esm_cli.main()


def test_prepare_dataset_cli_invokes_adapter(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_prepare_dataset(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "prepared_targets_fasta": str(tmp_path / "prepared_targets.fasta"),
            "prepared_labels_metadata_tsv": str(tmp_path / "prepared_labels_metadata.tsv"),
            "prepared_embedding_metadata_tsv": str(tmp_path / "prepared_embedding_metadata.tsv"),
            "prepare_summary_json": str(tmp_path / "prepare_summary.json"),
            "n_targets": 2,
            "n_label_rows": 4,
            "n_label_proteins": 2,
        }

    ns = argparse.Namespace(
        meta_file=tmp_path / "meta.tsv",
        output_dir=tmp_path / "out",
        dataset_kind="cwp",
        protein_fasta=tmp_path / "proteins.faa",
        z_file=None,
        reactive_codes=tmp_path / "reactive.tsv",
        nonreactive_codes=tmp_path / "nonreactive.tsv",
        group_id_offset=None,
        is_epi_z_min=20.0,
        is_epi_min_subs=4,
        not_epi_z_max=10.0,
        not_epi_max_subs=0,
        subject_prefix="VW_",
        log_level="INFO",
        log_json=False,
    )

    monkeypatch.setattr(
        prepare_dataset_cli.argparse.ArgumentParser, "parse_args", lambda self: ns
    )
    monkeypatch.setattr(
        prepare_dataset_cli,
        "setup_logger",
        lambda **kwargs: logging.getLogger("prepare_dataset_cli_test")
    )
    monkeypatch.setattr(
        prepare_dataset_cli,
        "prepare_dataset",
        fake_prepare_dataset
    )

    prepare_dataset_cli.main()

    assert captured["kwargs"]["dataset_kind"] == "cwp"
    assert int(captured["kwargs"]["group_id_offset"]) == 100_000_000
