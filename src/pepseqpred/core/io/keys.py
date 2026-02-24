"""keys.py

Key parsing helpers for PepSeqPred embedding and taxonomy identifiers.
"""

import re
from typing import Tuple, Optional, Literal


FULLNAME_RE = re.compile(r"^ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")


def parse_family_from_oxx(oxx: str) -> str:
    tokens = [token.strip() for token in oxx.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"Could not parse viral family from OXX={oxx}")
    return tokens[-1]


def parse_fullname(fullname: str) -> Tuple[str, ...]:
    match_ = FULLNAME_RE.match(str(fullname).strip())
    if match_ is None:
        raise ValueError(f"Could not parse fullname: {fullname}")
    protein_id, ac, oxx = match_.groups()
    return protein_id, ac, oxx, parse_family_from_oxx(oxx)


def build_emb_stem(protein_id: str, viral_family: Optional[str] = None, delimiter: str = "-") -> str:
    protein_id = str(protein_id).strip()
    if viral_family is None or str(viral_family).strip() == "":
        return protein_id
    if delimiter == "":
        raise ValueError("Delimiter must not be empty for id-family keys")

    family = str(viral_family).strip()
    if not family.isdigit():
        raise ValueError(f"viral_family must be numeric, got {viral_family}")

    return f"{protein_id}{delimiter}{viral_family}"


def parse_emb_stem(stem: str, delimiter: str = "-") -> Tuple[str, str | None, Literal["id", "id-family"]]:
    if delimiter and delimiter in stem:
        maybe_id, maybe_family = stem.rsplit(delimiter, 1)
        if maybe_id and maybe_family.isdigit():
            return maybe_id, maybe_family, "id-family"

    return stem, None, "id"
