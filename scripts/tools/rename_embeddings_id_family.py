"""rename_embeddings_id_family.py

Rename embedding files from `ID.pt` to `ID-family.pt` using metadata Name/Family columns.

Example
-------
Dry run:
    python scripts/tools/rename_embeddings_id_family.py \
        --metadata /scratch/$USER/data/fulldesign_2019-02-27_wGBKsw.metadata \
        --emb-root /scratch/$USER/esm2/artifacts/pts

Apply changes:
    python scripts/tools/rename_embeddings_id_family.py \
        --metadata /scratch/$USER/data/fulldesign_2019-02-27_wGBKsw.metadata \
        --emb-root /scratch/$USER/esm2/artifacts/pts \
        --apply
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


NAME_RE = re.compile(r"^ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")


def normalize_family(raw: str | None) -> str:
    """Normalize family token read from metadata."""
    if raw is None:
        return ""
    value = str(raw).strip()
    if value == "" or value.lower() == "nan":
        return ""
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def build_id_to_family(metadata_path: Path,
                       name_col: str,
                       family_col: str) -> Tuple[Dict[str, str], int, int]:
    """
    Parse metadata rows and build ID -> family mapping.

    Returns
    -------
    Tuple[Dict[str, str], int, int]
        Mapping, count of duplicate IDs with the same family, and count of rows
        where family was missing.
    """
    mapping: Dict[str, str] = {}
    duplicate_same_family = 0
    missing_family_rows = 0
    conflicts: List[Tuple[str, str, str]] = []
    parse_errors: List[Tuple[int, str]] = []
    parse_error_count = 0

    with metadata_path.open("r", encoding="utf-8", newline="") as meta_f:
        reader = csv.DictReader(meta_f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(
                f"Metadata file has no header row: {metadata_path}")
        missing_cols = [
            col for col in [name_col, family_col]
            if col not in set(reader.fieldnames)
        ]
        if len(missing_cols) > 0:
            raise ValueError(
                f"Metadata file missing required columns {missing_cols}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            fullname = str(row.get(name_col, "")).strip()
            if fullname == "":
                parse_error_count += 1
                if len(parse_errors) < 10:
                    parse_errors.append((row_idx, fullname))
                continue

            match_ = NAME_RE.match(fullname)
            if match_ is None:
                parse_error_count += 1
                if len(parse_errors) < 10:
                    parse_errors.append((row_idx, fullname))
                continue

            protein_id = match_.group(1)
            family = normalize_family(row.get(family_col))
            if family == "":
                # Missing families remain unmapped and will be reported by rename pass.
                missing_family_rows += 1
                continue
            if not family.isdigit():
                raise ValueError(
                    f"Invalid family value '{family}' for ID '{protein_id}' at row {row_idx}. "
                    "Expected numeric family."
                )

            prev = mapping.get(protein_id)
            if prev is None:
                mapping[protein_id] = family
            elif prev == family:
                duplicate_same_family += 1
            else:
                conflicts.append((protein_id, prev, family))

    if parse_error_count > 0:
        preview = ", ".join(
            [f"row={row_num} name='{name}'" for row_num,
                name in parse_errors[:5]]
        )
        raise ValueError(
            f"Found {parse_error_count} invalid metadata Name rows. Examples: {preview}"
        )

    if len(conflicts) > 0:
        preview = ", ".join(
            [f"{protein_id} ({old} vs {new})" for protein_id,
             old, new in conflicts[:5]]
        )
        raise ValueError(
            f"Found {len(conflicts)} ID->family conflicts in metadata. "
            f"Examples: {preview}"
        )

    return mapping, duplicate_same_family, missing_family_rows


def collect_shard_dirs(emb_root: Path | None, shard_dirs: Iterable[Path]) -> List[Path]:
    """Resolve shard directories from root and explicit --shard-dir args."""
    resolved: List[Path] = []

    if emb_root is not None:
        if not emb_root.exists():
            raise FileNotFoundError(
                f"Embedding root does not exist: {emb_root}")
        root_shards = sorted(
            [path for path in emb_root.glob("shard_*") if path.is_dir()])
        if len(root_shards) == 0:
            raise ValueError(f"No shard_* directories found under: {emb_root}")
        resolved.extend(root_shards)

    for shard_dir in shard_dirs:
        if not shard_dir.exists():
            raise FileNotFoundError(
                f"Shard directory does not exist: {shard_dir}")
        if not shard_dir.is_dir():
            raise ValueError(f"Expected directory, got file: {shard_dir}")
        resolved.append(shard_dir)

    deduped = sorted({path.resolve() for path in resolved})
    if len(deduped) == 0:
        raise ValueError("Provide --emb-root or at least one --shard-dir")
    return deduped


def rename_shard(
    shard_dir: Path,
    id_to_family: Dict[str, str],
    delimiter: str,
    apply_changes: bool,
    fail_on_missing_id: bool,
) -> Dict[str, int]:
    """Rename embeddings in one shard directory."""
    stats = {
        "total_pt": 0,
        "renamed": 0,
        "already": 0,
        "missing_id": 0,
        "collisions": 0,
    }

    missing_examples: List[str] = []
    collision_examples: List[Tuple[str, str]] = []

    for pt_path in sorted(shard_dir.glob("*.pt")):
        stats["total_pt"] += 1
        stem = pt_path.stem

        # Case 1: old format (stem is ID).
        if stem in id_to_family:
            family = id_to_family[stem]
            new_stem = f"{stem}{delimiter}{family}"
            new_path = pt_path.with_name(f"{new_stem}.pt")

            if new_path == pt_path:
                stats["already"] += 1
                continue
            if new_path.exists():
                stats["collisions"] += 1
                if len(collision_examples) < 10:
                    collision_examples.append((str(pt_path), str(new_path)))
                continue

            if apply_changes:
                pt_path.rename(new_path)
            stats["renamed"] += 1
            continue

        # Case 2: may already be renamed.
        if delimiter in stem:
            maybe_id, maybe_family = stem.rsplit(delimiter, 1)
            expected = id_to_family.get(maybe_id)
            if expected is not None and expected == maybe_family:
                stats["already"] += 1
                continue

        # Case 3: unknown stem.
        stats["missing_id"] += 1
        if len(missing_examples) < 10:
            missing_examples.append(stem)

    if len(collision_examples) > 0:
        print(
            f"[warn] {shard_dir}: destination collisions={len(collision_examples)}", file=sys.stderr)
        for src, dst in collision_examples:
            print(f"       src={src}", file=sys.stderr)
            print(f"       dst={dst}", file=sys.stderr)

    if len(missing_examples) > 0:
        msg = f"[warn] {shard_dir}: missing metadata IDs/families for {stats['missing_id']} files. examples={missing_examples}"
        if fail_on_missing_id:
            raise ValueError(msg)
        print(msg, file=sys.stderr)

    return stats


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Rename embedding files from ID.pt to ID-family.pt using metadata Name/Family columns."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata TSV containing Name and Family columns.",
    )
    parser.add_argument(
        "--name-col",
        type=str,
        default="Name",
        help="Metadata column containing full headers (default: Name).",
    )
    parser.add_argument(
        "--family-col",
        type=str,
        default="Family",
        help="Metadata column containing family values (default: Family).",
    )
    parser.add_argument(
        "--emb-root",
        type=Path,
        default=None,
        help="Root directory containing shard_* folders.",
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        dest="shard_dirs",
        type=Path,
        default=[],
        help="Embedding shard directory. Repeat for multiple shards.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="-",
        help="Delimiter between ID and family in output filenames.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames. Default is dry-run.",
    )
    parser.add_argument(
        "--fail-on-missing-id",
        action="store_true",
        help="Fail if any embedding stem cannot be mapped from metadata.",
    )
    args = parser.parse_args()

    if args.delimiter == "":
        raise ValueError("--delimiter must not be empty")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    shard_dirs = collect_shard_dirs(args.emb_root, args.shard_dirs)
    id_to_family, duplicate_same_family, missing_family_rows = build_id_to_family(
        metadata_path=args.metadata,
        name_col=args.name_col,
        family_col=args.family_col,
    )

    print(f"[info] Metadata IDs mapped: {len(id_to_family)}")
    print(f"[info] Duplicate IDs (same family): {duplicate_same_family}")
    print(f"[info] Metadata rows with missing family: {missing_family_rows}")
    print(f"[info] Shards: {len(shard_dirs)}")
    print(f"[info] Mode: {'apply' if args.apply else 'dry-run'}")

    totals = {
        "total_pt": 0,
        "renamed": 0,
        "already": 0,
        "missing_id": 0,
        "collisions": 0,
    }

    for shard_dir in shard_dirs:
        stats = rename_shard(
            shard_dir=shard_dir,
            id_to_family=id_to_family,
            delimiter=args.delimiter,
            apply_changes=args.apply,
            fail_on_missing_id=args.fail_on_missing_id,
        )
        for key in totals:
            totals[key] += int(stats[key])
        print(
            f"[info] {shard_dir} "
            f"total={stats['total_pt']} renamed={stats['renamed']} "
            f"already={stats['already']} missing_id={stats['missing_id']} "
            f"collisions={stats['collisions']}"
        )

    print(
        "[done] "
        f"total={totals['total_pt']} renamed={totals['renamed']} "
        f"already={totals['already']} missing_id={totals['missing_id']} "
        f"collisions={totals['collisions']}"
    )


if __name__ == "__main__":
    main()
