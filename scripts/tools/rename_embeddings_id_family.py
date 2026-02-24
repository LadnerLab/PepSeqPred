"""rename_embeddings_id_family.py

Rename embedding files from `ID.pt` to `ID-family.pt` using FASTA headers.

Example
-------
Dry run:
    python scripts/tools/rename_embeddings_id_family.py \
        --fasta /scratch/$USER/data/fulldesign_2019-02-27_wGBKsw.fasta \
        --emb-root /scratch/$USER/esm2/artifacts/pts

Apply changes:
    python scripts/tools/rename_embeddings_id_family.py \
        --fasta /scratch/$USER/data/fulldesign_2019-02-27_wGBKsw.fasta \
        --emb-root /scratch/$USER/esm2/artifacts/pts \
        --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HEADER_RE = re.compile(r"^>ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")


def parse_family(oxx: str) -> str:
    """Return the last non-empty OXX token."""
    tokens = [token.strip() for token in str(oxx).split(",") if token.strip()]
    if len(tokens) == 0:
        raise ValueError(f"Could not parse family from OXX='{oxx}'")
    return tokens[-1]


def build_id_to_family(fasta_path: Path) -> Tuple[Dict[str, str], int]:
    """
    Parse FASTA headers and build ID -> family mapping.

    Returns
    -------
    Tuple[Dict[str, str], int]
        Mapping plus count of duplicate IDs with the same family.
    """
    mapping: Dict[str, str] = {}
    duplicate_same_family = 0
    conflicts: List[Tuple[str, str, str]] = []

    with fasta_path.open("r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()
            if not line.startswith(">"):
                continue
            match_ = HEADER_RE.match(line)
            if match_ is None:
                raise ValueError(f"Header does not match expected format: '{line}'")

            protein_id = match_.group(1)
            family = parse_family(match_.group(3))

            prev = mapping.get(protein_id)
            if prev is None:
                mapping[protein_id] = family
            elif prev == family:
                duplicate_same_family += 1
            else:
                conflicts.append((protein_id, prev, family))

    if len(conflicts) > 0:
        preview = ", ".join(
            [f"{protein_id} ({old} vs {new})" for protein_id, old, new in conflicts[:5]]
        )
        raise ValueError(
            f"Found {len(conflicts)} ID->family conflicts in FASTA. "
            f"Examples: {preview}"
        )

    return mapping, duplicate_same_family


def collect_shard_dirs(emb_root: Path | None, shard_dirs: Iterable[Path]) -> List[Path]:
    """Resolve shard directories from root and explicit --shard-dir args."""
    resolved: List[Path] = []

    if emb_root is not None:
        if not emb_root.exists():
            raise FileNotFoundError(f"Embedding root does not exist: {emb_root}")
        root_shards = sorted([path for path in emb_root.glob("shard_*") if path.is_dir()])
        if len(root_shards) == 0:
            raise ValueError(f"No shard_* directories found under: {emb_root}")
        resolved.extend(root_shards)

    for shard_dir in shard_dirs:
        if not shard_dir.exists():
            raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")
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
        print(f"[warn] {shard_dir}: destination collisions={len(collision_examples)}", file=sys.stderr)
        for src, dst in collision_examples:
            print(f"       src={src}", file=sys.stderr)
            print(f"       dst={dst}", file=sys.stderr)

    if len(missing_examples) > 0:
        msg = f"[warn] {shard_dir}: missing FASTA IDs for {stats['missing_id']} files. examples={missing_examples}"
        if fail_on_missing_id:
            raise ValueError(msg)
        print(msg, file=sys.stderr)

    return stats


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Rename embedding files from ID.pt to ID-family.pt using FASTA OXX headers."
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Path to FASTA file with headers in format: >ID=... AC=... OXX=...",
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
        help="Fail if any embedding stem cannot be mapped from FASTA.",
    )
    args = parser.parse_args()

    if args.delimiter == "":
        raise ValueError("--delimiter must not be empty")
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")

    shard_dirs = collect_shard_dirs(args.emb_root, args.shard_dirs)
    id_to_family, duplicate_same_family = build_id_to_family(args.fasta)

    print(f"[info] FASTA IDs mapped: {len(id_to_family)}")
    print(f"[info] Duplicate IDs (same family): {duplicate_same_family}")
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
