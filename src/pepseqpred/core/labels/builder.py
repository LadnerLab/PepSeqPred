"""builder.py

Label building utilities for PepSeqPred peptide metadata.

Provides helpers to parse PV1-style metadata, map peptides to protein embeddings,
and build residue-level label tensors and peptide metadata for training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, cast
import re
import pandas as pd
import torch
from pepseqpred.core.io.keys import parse_emb_stem


FULLNAME_RE = re.compile(r"^ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")


def parse_id_from_fullname(fullname: str) -> str:
    """
    Parses ID part from fullname string. Example:
    ID=Q9DJD5_9PICO AC=Q9DJD5 OXX=12118,12110,12109,12058. ID would be:
    Q9DJD5_9PICO

    Parameters
    ----------
        fullname : str
            The fullname string from the column "FullName" in metadata file.

    Returns
    -------
        str
            The ID parsed from fullname string.
    """
    match_ = FULLNAME_RE.match(str(fullname).strip())
    if not match_:
        raise ValueError(f"Could not parse ID from fullname: '{fullname}'")
    return match_.group(1)


def parse_taxonomy_from_fullname(fullname: str) -> Dict[str, Any]:
    """
    Parses taxonomic data from the full name column in metadata file.

    Parameters
    ----------
        fullname : str
            The fullname string from the column "FullName" in metadata file.

    Returns
    -------
        Dict[str, Any]
            A dictionary containing the fullname, protein ID, AC, and OXX.
    """
    match_ = FULLNAME_RE.match(fullname.strip())
    if match_ is None:
        return {"fullname": fullname}
    return {
        "fullname": fullname,
        "protein_id": match_.group(1),
        "ac": match_.group(2),
        "oxx": match_.group(3)
    }


class ProteinLabelBuilder:
    """
    Builds dense residue labels per protein and stores peptide metadata.

    Labels are stored as (L, 3) with columns: [Def epitope, Uncertain, Not epitope].
    Overlap rule: def overrides not/uncertain, not overrides uncertain.

    Parameters
    ----------
        meta_path : Path or str
            Path to the .tsv peptide metadata file.
        emb_dirs : List[Path or str]
            List of the sharded embedding directories.
        logger : Logger
            The logger object to stream logs to output file for downstream program verification.
        restrict_to_embeddings : bool
            When `True`, restricts embedding search space to embeddings only located in the current shard. Default is `False`, which allows builder to use all directories provided.
        calc_pos_weight : bool
            When `True`, the weight of the positive class is calculated from generated label shards. Default is `False`.
        embedding_key_delim : str
            If set to the delimeter `"-"`, we assume the user has embeddings named in the ID-family.pt format for label generation. Default is `""`, where the embeddings are assumed to be the ID.pt format.

    Notes
    -----
    Only set `restrict_to_embeddings` to `True` if your machiner has a multi-core CPU, else keep `False`.
    """

    def __init__(self,
                 meta_path: Path | str,
                 emb_dirs: List[Path | str],
                 logger: logging.Logger,
                 restrict_to_embeddings: bool = False,
                 calc_pos_weight: bool = False,
                 embedding_key_delim: str = "") -> None:
        self.meta_path = Path(meta_path)
        self.emb_dirs = [Path(p) for p in emb_dirs]
        self.logger = logger
        self.restrict_to_embeddings = restrict_to_embeddings
        self.calc_pos_weight = calc_pos_weight

        self.embedding_key_delim = str(embedding_key_delim or "")
        if self.embedding_key_delim not in {"", "-"}:
            raise ValueError(
                "embedding_key_delim must be '' (ID.pt) or '-' (ID-family.pt)"
            )

        self.use_id_family = self.embedding_key_delim == "-"
        self._id_family_pt_by_id: Dict[str, Path] = {}
        if self.use_id_family:
            self._id_family_pt_by_id = self._build_id_family_lookup()
            if not self._id_family_pt_by_id:
                raise ValueError("No ID-family embeddings found in --emb-dir")

        self.df = pd.read_csv(self.meta_path, sep="\t")
        self.df["protein_id"] = self.df["FullName"].astype(
            str).map(parse_id_from_fullname)
        if self.restrict_to_embeddings:
            emb_ids = self._list_embedding_ids()
            before = len(self.df)
            self.df = self.df[self.df["protein_id"].isin(list(emb_ids))].copy()
            if self.df.empty:
                raise ValueError(
                    "0 rows after --restrict-to-embeddings. "
                    "Likely mismatch between embedding naming and --embedding-key-delim."
                )
            self.logger.info("filtered_to_embeddings", extra={"extra": {
                "rows_before": before,
                "rows_after": len(self.df),
                "num_embedding_ids": len(emb_ids)
            }})

    def _build_id_family_lookup(self) -> Dict[str, Path]:
        """Builds protein_id --> .pt path lookup for ID-family format."""
        index: Dict[str, Path] = {}
        for emb_dir in self.emb_dirs:
            for pt_path in emb_dir.glob("*.pt"):
                protein_id, _, scheme = parse_emb_stem(
                    pt_path.stem, delimiter=self.embedding_key_delim
                )
                if scheme != "id-family":
                    continue
                prev = index.get(protein_id)
                if prev is not None and prev != pt_path:
                    raise ValueError(
                        f"Duplicate ID-family embeddings for protein_id={protein_id}: {prev} vs {pt_path}"
                    )
                index[protein_id] = pt_path
        return index

    def _list_embedding_ids(self) -> set[str]:
        """Adds embedding IDs to single list."""
        if self.use_id_family:
            # ID-family look-up already built
            return set(self._id_family_pt_by_id.keys())
        ids: set[str] = set()
        for emb_dir in self.emb_dirs:
            for pt_path in emb_dir.glob("*.pt"):
                ids.add(pt_path.stem)
        return ids

    def _find_pt_path(self, protein_id: str) -> Path:
        """Finds .pt path given a protein ID."""
        if self.use_id_family:
            pt_path = self._id_family_pt_by_id.get(protein_id)
            if pt_path is not None:
                return pt_path
            raise FileNotFoundError(
                f"Missing ID-family embedding for '{protein_id}'. "
                f"Check --embedding-key-delim (expected '-') and emb dirs: {[str(d) for d in self.emb_dirs]}"
            )
        filename = f"{protein_id}.pt"
        for emb_dir in self.emb_dirs:
            pt_path = emb_dir / filename
            if pt_path.exists():
                return pt_path
        searched = [str(d) for d in self.emb_dirs]
        raise FileNotFoundError(
            f"Missing embedding file '{filename}' in embedding dirs: {searched}")

    def _load_embedding_length(self, protein_id: str) -> int:
        """Finds .pt path, loads embedding as tensor, and returns the length (number of amino acids)."""
        pt_path = self._find_pt_path(protein_id)
        embedding = torch.load(pt_path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor) or embedding.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor embedding for '{protein_id}', got {type(embedding)}")
        return int(embedding.size(0))

    @staticmethod
    def _row_label(row: pd.Series) -> str:
        """Returns the residue label given a row."""
        y_def = int(row["Def epitope"])
        y_unc = int(row["Uncertain"])
        y_not = int(row["Not epitope"])
        if (y_def + y_unc + y_not) != 1:
            raise ValueError(
                f"Invalid one-hot label for '{row.CodeName}': {y_def}, {y_unc}, {y_not}")
        if y_def == 1:
            return "def"
        if y_not == 1:
            return "not"
        return "uncertain"

    def _build_labels_for_protein(self, protein_id: str, group: pd.DataFrame) -> torch.Tensor:
        """Labels each residue as 'def epitope', 'uncertain', or 'not epitope'."""
        L = self._load_embedding_length(protein_id)
        def_mask = torch.zeros(L, dtype=torch.bool)
        not_mask = torch.zeros(L, dtype=torch.bool)

        for _, row in group.iterrows():
            start = int(row["AlignStart"])
            stop = int(row["AlignStop"])
            if start < 0 or stop <= start or stop > L:
                raise IndexError(
                    f"Invalid AlignStart/AlignStop for '{row.CodeName}': {start}, {stop}, L={L}")

            label = self._row_label(row)
            if label == "def":
                def_mask[start:stop] = True
            elif label == "not":
                not_mask[start:stop] = True

        labels = torch.zeros((L, 3), dtype=torch.uint8)
        labels[:, 1] = 1

        not_only = not_mask & ~def_mask
        labels[not_only, 1] = 0
        labels[not_only, 2] = 1

        labels[def_mask, 1] = 0
        labels[def_mask, 0] = 1

        return labels

    @staticmethod
    def _calculate_class_counts(labels: torch.Tensor) -> Tuple[int, int]:
        """Counts positive and negative residues for positive weight calculation."""
        if labels.dim() == 2 and labels.size(1) == 3:
            # handles definite, uncertain, and non-epitope classes
            yes_col = labels[:, 0].float()
            unc_col = labels[:, 1].float()
            not_col = labels[:, 2].float()
            mask = (unc_col == 0)
            pos = int(yes_col[mask].sum().item())
            neg = int(not_col[mask].sum().item())
        else:
            # binary labels: definite and non-epitopes
            y = labels.view(-1)
            pos = int((y == 1).sum().item())
            neg = int((y == 0).sum().item())

        return pos, neg

    def _build_peptides_for_protein(self, group: pd.DataFrame) -> List[Dict[str, Any]]:
        """Builds peptide dictionary from group of related peptides."""
        peptides: List[Dict[str, Any]] = []
        for _, row in group.iterrows():
            label = self._row_label(row)
            one_hot = [1, 0, 0] if label == "def" else (
                [0, 0, 1] if label == "not" else [0, 1, 0])
            peptides.append({
                "code_name": str(row["CodeName"]),
                "peptide": str(row["Peptide"]),
                "align_start": int(row["AlignStart"]),
                "align_stop": int(row["AlignStop"]),
                "label_onehot": one_hot,
                "fullname": str(row["FullName"])
            })
        return peptides

    def build(self, save_path: Path | str) -> Dict[str, Any]:
        """
        Main builder method that builds the labels per residue in each protein sequence.

        Parameters
        ----------
            save_path : Path or str
                The path to save the labels .pt file to.

        Returns
        -------
            Dict[str, Any]
                Returns the dictionary containing label data saved to .pt file for verfication. Optionally saves the definite and non-epitope class counts, and positive class weight if `calc_pos_weight` is set to `True`.
        """
        if self.df.empty:
            raise ValueError("No metadata rows available to build labels")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        labels_by_protein: Dict[str, torch.Tensor] = {}
        proteins: Dict[str, Dict[str, Any]] = {}
        total_pos, total_neg = 0, 0

        for protein_id, group in self.df.groupby("protein_id"):
            protein_id = str(protein_id)
            group_df = cast(pd.DataFrame, group)
            labels = self._build_labels_for_protein(protein_id, group_df)
            labels_by_protein[protein_id] = labels

            if self.calc_pos_weight:
                pos, neg = self._calculate_class_counts(labels)
                total_pos += pos
                total_neg += neg

            fullname = str(group_df.iloc[0]["FullName"])
            proteins[protein_id] = {
                "tax_info": parse_taxonomy_from_fullname(fullname),
                "peptides": self._build_peptides_for_protein(group_df)
            }

        if not labels_by_protein:
            raise ValueError(
                "Built zero protein labels, check metadata/embedding alignment"
            )

        payload = {
            "labels": labels_by_protein,
            "proteins": proteins
        }
        if self.calc_pos_weight:
            payload["class_stats"] = {
                "pos_count": int(total_pos),
                "neg_count": int(total_neg),
                "pos_weight": float(total_neg / max(1, total_pos))
            }
        torch.save(payload, save_path)

        self.logger.info("labels_built", extra={"extra": {
            "num_proteins": len(labels_by_protein),
            "save_path": str(save_path)
        }})

        if self.calc_pos_weight:
            self.logger.info("pos_weight_computed", extra={"extra": {
                "pos_count": int(total_pos),
                "neg_count": int(total_neg),
                "pos_weight": float(total_neg / max(1, total_pos)),
                "save_path": str(save_path)
            }})
        return payload
