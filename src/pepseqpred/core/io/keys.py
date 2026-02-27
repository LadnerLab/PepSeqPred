"""keys.py

Key parsing helpers for PepSeqPred embedding and taxonomy identifiers.
"""

import re
from pathlib import Path
from typing import Tuple, Optional, Literal, Dict, Any
import pandas as pd


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


def normalize_family_value(raw: Any) -> str:
    if raw is None:
        return ""
    value = str(raw).strip()
    if value == "" or value.lower() == "nan":
        return ""
    # handle cases when raw value is float (should be int)
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def build_id_to_family_from_metadata(
    metadata_path: Path | str,
    name_col: str = "Name",
    family_col: str = "Family"
) -> Tuple[Dict[str, str], int]:
    meta_df = pd.read_csv(metadata_path, sep="\t", dtype=str)
    missing_cols = [col for col in [name_col, family_col]
                    if col not in meta_df.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata missing required columns: {missing_cols}"
            f"Available: {list(meta_df.columns)}"
        )

    mapping: Dict[str, str] = {}
    duplicate_same_family = 0
    parse_errors = []
    conflicts = []

    for index, (name_val, family_val) in enumerate(
        meta_df[[name_col, family_col]].itertuples(index=False, name=None),
        start=2
    ):
        raw_name = str(name_val).strip()
        try:
            protein_id, _ac, _oxx, _parsed_family = parse_fullname(
                raw_name
            )
        except ValueError:
            # keep track of errors for debugging
            if len(parse_errors) < 10:
                parse_errors.append((index, raw_name))
            continue

        family = normalize_family_value(family_val)
        # family value should an int
        if family != "" and not family.isdigit():
            raise ValueError(
                f"Invalid family '{family}' for protein_id={protein_id} "
                f"at metadata row {index}, ecpected numeric or empty"
            )

        # different protein IDs can have same family, but there should not be any familial conflicts (i.e., same protein ID already exists)
        prev = mapping.get(protein_id)
        if prev is None:
            mapping[protein_id] = family
        elif prev == family:
            duplicate_same_family += 1
        else:
            conflicts.append((protein_id, prev, family))

    if parse_errors:
        preview = ", ".join([
            f"row={r} name={n}" for r, n in parse_errors[:5]
        ])
        raise ValueError(
            f"Could not parse {len(parse_errors)} metadata Name rows, examples: {[preview]}"
        )
    if conflicts:
        preview = ", ".join([
            f"{pid} ({old} vs {new})" for pid, old, new in conflicts[:5]
        ])
        raise ValueError(
            f"Found {len(conflicts)} metadata ID --> family conflicts, examples: {preview}"
        )

    return mapping, duplicate_same_family


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
