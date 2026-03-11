import logging
from pathlib import Path
import pandas as pd
import pytest
import torch
from pepseqpred.core.labels.builder import (
    ProteinLabelBuilder,
    parse_id_from_fullname,
    parse_taxonomy_from_fullname
)

pytestmark = pytest.mark.unit


def _write_embedding(path: Path, length: int = 6, dim: int = 4) -> None:
    torch.save(torch.randn(length, dim, dtype=torch.float32), path)


def _write_metadata(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_parse_helpers():
    full = "ID=P001 AC=A1 OXX=11,22,33"
    assert parse_id_from_fullname(full) == "P001"
    tax = parse_taxonomy_from_fullname(full)
    assert tax["protein_id"] == "P001"
    assert parse_taxonomy_from_fullname("bad") == {"fullname": "bad"}

    with pytest.raises(ValueError, match="Could not parse ID"):
        parse_id_from_fullname("not valid")


def test_builder_build_id_mode(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    _write_embedding(emb_dir / "P001.pt", length=6, dim=4)
    _write_embedding(emb_dir / "P002.pt", length=4, dim=4)

    meta = tmp_path / "meta.tsv"
    _write_metadata(
        meta,
        [
            {
                "CodeName": "pep1",
                "Peptide": "AAA",
                "AlignStart": 0,
                "AlignStop": 3,
                "Def epitope": 1,
                "Uncertain": 0,
                "Not epitope": 0,
                "FullName": "ID=P001 AC=A1 OXX=11,22,33"
            },
            {
                "CodeName": "pep2",
                "Peptide": "BBB",
                "AlignStart": 3,
                "AlignStop": 5,
                "Def epitope": 0,
                "Uncertain": 0,
                "Not epitope": 1,
                "FullName": "ID=P001 AC=A1 OXX=11,22,33"
            },
            {
                "CodeName": "pep3",
                "Peptide": "CCC",
                "AlignStart": 0,
                "AlignStop": 2,
                "Def epitope": 0,
                "Uncertain": 1,
                "Not epitope": 0,
                "FullName": "ID=P002 AC=A2 OXX=44,55,66"
            }
        ]
    )

    out_pt = tmp_path / "labels.pt"
    builder = ProteinLabelBuilder(
        meta_path=meta,
        emb_dirs=[emb_dir],
        logger=logging.getLogger("builder_test"),
        calc_pos_weight=True,
        embedding_key_delim=""
    )
    payload = builder.build(out_pt)

    assert out_pt.exists()
    assert set(payload.keys()) >= {"labels", "proteins", "class_stats"}
    p1 = payload["labels"]["P001"]
    assert p1.shape == (6, 3)
    assert p1[:3, 0].sum().item() == 3
    assert p1[3:5, 2].sum().item() == 2
    assert "tax_info" in payload["proteins"]["P001"]


def test_builder_restrict_to_embeddings_raises_if_empty(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    _write_embedding(emb_dir / "P001.pt")

    meta = tmp_path / "meta.tsv"
    _write_metadata(
        meta,
        [
            {
                "CodeName": "pepX",
                "Peptide": "AAA",
                "AlignStart": 0,
                "AlignStop": 2,
                "Def epitope": 1,
                "Uncertain": 0,
                "Not epitope": 0,
                "FullName": "ID=P999 AC=A9 OXX=1,2,3"
            }
        ]
    )

    with pytest.raises(ValueError, match="0 rows after --restrict-to-embeddings"):
        ProteinLabelBuilder(
            meta_path=meta,
            emb_dirs=[emb_dir],
            logger=logging.getLogger("builder_test"),
            restrict_to_embeddings=True
        )


def test_builder_id_family_duplicate_ids_raise(tmp_path: Path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    _write_embedding(emb_dir / "P001-111.pt")
    _write_embedding(emb_dir / "P001-222.pt")

    meta = tmp_path / "meta.tsv"
    _write_metadata(
        meta,
        [
            {
                "CodeName": "pep1",
                "Peptide": "AAA",
                "AlignStart": 0,
                "AlignStop": 2,
                "Def epitope": 1,
                "Uncertain": 0,
                "Not epitope": 0,
                "FullName": "ID=P001 AC=A1 OXX=11,22,33",
            }
        ]
    )

    with pytest.raises(ValueError, match="Duplicate ID-family embeddings"):
        ProteinLabelBuilder(
            meta_path=meta,
            emb_dirs=[emb_dir],
            logger=logging.getLogger("builder_test"),
            embedding_key_delim="-"
        )
