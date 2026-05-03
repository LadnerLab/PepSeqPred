import json
from pathlib import Path

import pandas as pd
import pytest

from pepseqpred.core.preprocess.preparedataset import (
    _build_group_numeric_map,
    prepare_dataset,
)

pytestmark = pytest.mark.unit


def _write_tsv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_code_list(path: Path, codes: list[str]) -> None:
    lines = ["Sequence name", *codes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_group_mapping_is_deterministic_and_disjoint_ranges():
    cwp_tokens = ["Cocci_id50_010", "Cocci_id50_001", "Cocci_id50_010"]
    bkp_tokens = ["BKP1_id70_200", "BKP1_id70_050"]

    cwp_map = _build_group_numeric_map(cwp_tokens, offset=100_000_000)
    bkp_map = _build_group_numeric_map(bkp_tokens, offset=200_000_000)

    assert cwp_map == {
        "Cocci_id50_001": 100000001,
        "Cocci_id50_010": 100000002,
    }
    assert bkp_map == {
        "BKP1_id70_050": 200000001,
        "BKP1_id70_200": 200000002,
    }
    assert max(cwp_map.values()) < min(bkp_map.values())


def test_prepare_cwp_outputs_and_drop_report(tmp_path: Path):
    meta_path = tmp_path / "cwp_meta.tsv"
    protein_fasta = tmp_path / "proteins.faa"
    reactive = tmp_path / "reactive.tsv"
    nonreactive = tmp_path / "nonreactive.tsv"
    out_dir = tmp_path / "prepared"

    _write_tsv(
        meta_path,
        [
            {
                "CodeName": "CWP_000001",
                "SequenceAccession": "A0A111",
                "Cluster50ID": "Cocci_id50_010",
                "StartIndex": 0,
                "StopIndex": 4,
                "PeptideSequence": "MNPQ",
            },
            {
                "CodeName": "CWP_000002",
                "SequenceAccession": "WP_200.1",
                "Cluster50ID": "Cocci_id50_001",
                "StartIndex": 1,
                "StopIndex": 5,
                "PeptideSequence": "QWER",
            },
            {
                "CodeName": "CWP_000003",
                "SequenceAccession": "MISSING_ACC",
                "Cluster50ID": "Cocci_id50_002",
                "StartIndex": 0,
                "StopIndex": 3,
                "PeptideSequence": "AAA",
            },
        ],
    )
    protein_fasta.write_text(
        ">tr|A0A111|A0A111_FAKE desc\nMNPQRS\n"
        ">WP_200.1 hypothetical protein\nAQWERT\n",
        encoding="utf-8",
    )
    _write_code_list(reactive, ["CWP_000001"])
    _write_code_list(nonreactive, ["CWP_000002", "CWP_000003"])

    summary = prepare_dataset(
        dataset_kind="cwp",
        meta_path=meta_path,
        output_dir=out_dir,
        protein_fasta=protein_fasta,
        reactive_codes=reactive,
        nonreactive_codes=nonreactive,
        group_id_offset=1000,
    )

    assert (out_dir / "prepared_targets.fasta").exists()
    assert (out_dir / "prepared_labels_metadata.tsv").exists()
    assert (out_dir / "prepared_embedding_metadata.tsv").exists()
    assert (out_dir / "prepare_summary.json").exists()
    assert int(summary["n_label_rows"]) == 2
    assert int(summary["n_label_proteins"]) == 2

    labels_df = pd.read_csv(out_dir / "prepared_labels_metadata.tsv", sep="\t")
    emb_df = pd.read_csv(out_dir / "prepared_embedding_metadata.tsv", sep="\t")
    assert labels_df["CodeName"].tolist() == ["CWP_000001", "CWP_000002"]

    fam_by_name = dict(zip(emb_df["Name"], emb_df["Family"]))
    name_c1 = labels_df.loc[labels_df["CodeName"]
                            == "CWP_000001", "FullName"].iloc[0]
    name_c2 = labels_df.loc[labels_df["CodeName"]
                            == "CWP_000002", "FullName"].iloc[0]
    assert int(fam_by_name[name_c2]) == 1001
    assert int(fam_by_name[name_c1]) == 1002

    payload = json.loads(
        (out_dir / "prepare_summary.json").read_text(encoding="utf-8"))
    assert int(payload["normalization"]["drop_counts"]
               ["missing_protein_sequence"]) == 1


def test_prepare_bkp_derives_peptide_and_align_fallback(tmp_path: Path):
    meta_path = tmp_path / "bkp_meta.tsv"
    protein_fasta = tmp_path / "proteins.faa"
    reactive = tmp_path / "reactive.tsv"
    nonreactive = tmp_path / "nonreactive.tsv"
    out_dir = tmp_path / "prepared"

    _write_tsv(
        meta_path,
        [
            {
                "CodeName": "BKP_000001",
                "SequenceAccession": "A0B123",
                "reClusterID_70": "BKP1_id70_200",
                "alignStart": "0.0",
                "alignStop": "4.0",
                "PeptideSequence": "",
            }
        ],
    )
    protein_fasta.write_text(
        ">tr|A0B123|A0B123_FAKE desc\nMNPQRS\n",
        encoding="utf-8",
    )
    _write_code_list(reactive, ["BKP_000001"])
    _write_code_list(nonreactive, [])

    summary = prepare_dataset(
        dataset_kind="bkp",
        meta_path=meta_path,
        output_dir=out_dir,
        protein_fasta=protein_fasta,
        reactive_codes=reactive,
        nonreactive_codes=nonreactive,
        group_id_offset=2000,
    )

    assert int(summary["n_label_rows"]) == 1
    labels_df = pd.read_csv(out_dir / "prepared_labels_metadata.tsv", sep="\t")
    emb_df = pd.read_csv(out_dir / "prepared_embedding_metadata.tsv", sep="\t")

    assert labels_df["Peptide"].iloc[0] == "MNPQ"
    assert int(labels_df["AlignStart"].iloc[0]) == 0
    assert int(labels_df["AlignStop"].iloc[0]) == 4
    assert int(emb_df["Family"].iloc[0]) == 2001


def test_prepare_pv1_reuses_preprocess_and_family_from_fullname(tmp_path: Path):
    pv1_meta = tmp_path / "pv1_meta.tsv"
    pv1_z = tmp_path / "pv1_z.tsv"
    pv1_fasta = tmp_path / "pv1_targets.fasta"
    out_dir = tmp_path / "prepared"

    _write_tsv(
        pv1_meta,
        [
            {
                "CodeName": "pep1",
                "Category": "SetCover",
                "SpeciesID": "1",
                "Species": "X",
                "Protein": "Y",
                "FullName": "ID=P001 AC=A1 OXX=11,22,33_0_4",
                "Peptide": "MNPQ",
                "Encoding": "enc",
            },
            {
                "CodeName": "pep2",
                "Category": "SetCover",
                "SpeciesID": "1",
                "Species": "X",
                "Protein": "Y",
                "FullName": "ID=P001 AC=A1 OXX=11,22,33_2_6",
                "Peptide": "PQRS",
                "Encoding": "enc",
            },
        ],
    )
    _write_tsv(
        pv1_z,
        [
            {"Sequence name": "pep1", "VW_001": 30.0, "VW_002": 0.0},
            {"Sequence name": "pep2", "VW_001": 1.0, "VW_002": 2.0},
        ],
    )
    pv1_fasta.write_text(
        ">ID=P001 AC=A1 OXX=11,22,33\nMNPQRS\n",
        encoding="utf-8",
    )

    summary = prepare_dataset(
        dataset_kind="pv1",
        meta_path=pv1_meta,
        output_dir=out_dir,
        protein_fasta=pv1_fasta,
        z_path=pv1_z,
        is_epitope_min_subjects=1,
    )

    assert int(summary["n_label_rows"]) == 2
    assert int(summary["n_label_proteins"]) == 1
    emb_df = pd.read_csv(out_dir / "prepared_embedding_metadata.tsv", sep="\t")
    assert int(emb_df["Family"].iloc[0]) == 33
