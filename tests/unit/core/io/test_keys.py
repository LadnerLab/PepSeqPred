from pathlib import Path
import pytest
from pepseqpred.core.io.keys import (
    build_emb_stem,
    build_id_to_family_from_metadata,
    normalize_family_value,
    parse_emb_stem,
    parse_family_from_oxx,
    parse_fullname
)

pytestmark = pytest.mark.unit


def test_parse_fullname_and_family():
    fullname = "ID=P001 AC=A1 OXX=10,20,333"
    protein_id, ac, oxx, family = parse_fullname(fullname)
    assert protein_id == "P001"
    assert ac == "A1"
    assert oxx == "10,20,333"
    assert family == "333"
    assert parse_family_from_oxx(oxx) == "333"


def test_normalize_family_value_variants():
    assert normalize_family_value(None) == ""
    assert normalize_family_value("nan") == ""
    assert normalize_family_value("123.0") == "123"
    assert normalize_family_value(" 456 ") == "456"


def test_build_and_parse_emb_stem_roundtrip():
    stem = build_emb_stem("P001", "111", delimiter="-")
    assert stem == "P001-111"

    protein_id, family, scheme = parse_emb_stem(stem, delimiter="-")
    assert protein_id == "P001"
    assert family == "111"
    assert scheme == "id-family"

    protein_id2, family2, scheme2 = parse_emb_stem("P002", delimiter="-")
    assert protein_id2 == "P002"
    assert family2 is None
    assert scheme2 == "id"


def test_build_emb_stem_rejects_non_numeric_family():
    with pytest.raises(ValueError, match="must be numeric"):
        build_emb_stem("P001", "abc")


def test_build_id_to_family_from_metadata_happy_path(tmp_path: Path):
    meta = tmp_path / "meta.tsv"
    meta.write_text(
        "Name\tFamily\n"
        "ID=P001 AC=A1 OXX=10,20,333\t333\n"
        "ID=P002 AC=A2 OXX=10,20,444\t444\n",
        encoding="utf-8"
    )

    mapping, duplicate_same = build_id_to_family_from_metadata(meta)
    assert mapping == {"P001": "333", "P002": "444"}
    assert duplicate_same == 0


def test_build_id_to_family_from_metadata_conflict_raises(tmp_path: Path):
    meta = tmp_path / "meta.tsv"
    meta.write_text(
        "Name\tFamily\n"
        "ID=P001 AC=A1 OXX=10,20,333\t333\n"
        "ID=P001 AC=A1 OXX=10,20,333\t444\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="conflicts"):
        build_id_to_family_from_metadata(meta)
