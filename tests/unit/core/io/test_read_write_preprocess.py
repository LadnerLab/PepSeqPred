import logging
from pathlib import Path
import pandas as pd
import pytest
from pepseqpred.core.io.read import read_fasta, read_metadata, read_zscores
from pepseqpred.core.io.write import append_csv_row
from pepseqpred.core.preprocess.pv1 import preprocess
from pepseqpred.core.preprocess.zscores import (
    apply_z_threshold,
    merge_zscores_metadata
)

pytestmark = pytest.mark.unit


def test_read_fasta_standard_and_full_name(tmp_path: Path):
    fasta = tmp_path / "in.fasta"
    fasta.write_text(
        ">ID=P001 AC=A1 OXX=11,22,33\nACDE\n>ID=P002 AC=A2 OXX=44,55,66\nFGH\n",
        encoding="utf-8"
    )

    df_std = read_fasta(fasta, full_name=False)
    assert list(df_std.columns) == ["ID", "AC", "OXX", "Sequence"]
    assert df_std["ID"].tolist() == ["P001", "P002"]
    assert df_std["Sequence"].tolist() == ["ACDE", "FGH"]

    df_full = read_fasta(fasta, full_name=True)
    assert list(df_full.columns) == ["FullName", "Sequence"]
    assert df_full["FullName"].iloc[0].startswith("ID=P001")


def test_read_fasta_bad_header_raises(tmp_path: Path):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text(">not pv1 style\nACD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Header does not match expected format"):
        read_fasta(fasta, full_name=False)


def test_read_metadata_extracts_align_indices_and_drops(tmp_path: Path):
    meta = tmp_path / "meta.tsv"
    pd.DataFrame(
        [
            {
                "CodeName": "pep1",
                "Category": "SetCover",
                "SpeciesID": "1",
                "Species": "X",
                "Protein": "Y",
                "FullName": "ID=P001 AC=A1 OXX=11,22,33_2_5",
                "Peptide": "ACD",
                "Encoding": "enc",
            },
            {
                "CodeName": "pep2",
                "Category": "Other",
                "SpeciesID": "1",
                "Species": "X",
                "Protein": "Y",
                "FullName": "ID=P002 AC=A2 OXX=44,55,66_1_3",
                "Peptide": "FGH",
                "Encoding": "enc"
            }
        ]
    ).to_csv(meta, sep="\t", index=False)

    out = read_metadata(
        meta,
        drop_cols=["Category", "SpeciesID", "Protein", "Encoding"]
    )
    assert len(out) == 1
    assert int(out["AlignStart"].iloc[0]) == 2
    assert int(out["AlignStop"].iloc[0]) == 5
    assert out["FullName"].iloc[0] == "ID=P001 AC=A1 OXX=11,22,33"


def test_read_zscores_rename(tmp_path: Path):
    z = tmp_path / "z.tsv"
    pd.DataFrame(
        [
            {"Sequence name": "pep1", "VW_001": 1.0, "VW_002": 2.0},
            {"Sequence name": "pep2", "VW_001": 3.0, "VW_002": 4.0}
        ]
    ).to_csv(z, sep="\t", index=False)

    out = read_zscores(z)
    assert "CodeName" in out.columns
    assert "Sequence name" not in out.columns


def test_append_csv_row_appends_without_duplicate_header(tmp_path: Path):
    csv_path = tmp_path / "runs" / "r.csv"
    append_csv_row(csv_path, {"a": 1, "b": "x"})
    append_csv_row(csv_path, {"a": 2, "b": "y"})

    df = pd.read_csv(csv_path)
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 2]


def test_apply_threshold_merge_and_preprocess(tmp_path: Path):
    z_df = pd.DataFrame(
        [
            {"CodeName": "pep1", "VW_001": 30, "VW_002": 25},
            {"CodeName": "pep2", "VW_001": 1, "VW_002": 2},
            {"CodeName": "pep3", "VW_001": 15, "VW_002": 5}
        ]
    )
    labeled = apply_z_threshold(
        z_df.copy(),
        is_epitope_z_min=20,
        is_epitope_min_subjects=1,
        not_epitope_z_max=10
    )
    assert labeled[["Def epitope", "Uncertain", "Not epitope"]].sum(
        axis=1).eq(1).all()

    meta_df = pd.DataFrame(
        {"CodeName": ["pep1", "pep2", "pep3"], "x": [1, 2, 3]})
    merged = merge_zscores_metadata(labeled, meta_df)
    assert {"Def epitope", "Uncertain", "Not epitope"}.issubset(merged.columns)

    meta_path = tmp_path / "meta.tsv"
    z_path = tmp_path / "z.tsv"
    save_path = tmp_path / "out.tsv"

    pd.DataFrame(
        [
            {
                "CodeName": "pep1",
                "Category": "SetCover",
                "SpeciesID": "1",
                "Species": "X",
                "Protein": "Y",
                "FullName": "ID=P001 AC=A1 OXX=11,22,33_2_5",
                "Peptide": "AAA",
                "Encoding": "enc"
            }
        ]
    ).to_csv(meta_path, sep="\t", index=False)
    pd.DataFrame(
        [{"Sequence name": "pep1", "VW_001": 50, "VW_002": 0}]
    ).to_csv(z_path, sep="\t", index=False)

    logger = logging.getLogger("test_preprocess")
    out = preprocess(meta_path, z_path, save_path=save_path, logger=logger)
    assert len(out) == 1
    assert save_path.exists()
