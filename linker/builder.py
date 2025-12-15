import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from pipelineio.peptidedataset import PeptideDataset

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
    fullname_pattern = re.compile(r"^ID=([^\s]+)\s+AC=([^\s]+)\s+OXX=([^\s]+)\s*$")
    match_ = fullname_pattern.match(fullname)
    if not match_:
        raise ValueError(f"Could not parse ID from fullname: '{fullname}'")
    return match_.group(1)

def load_protein_embedding(pt_path: Path | str) -> torch.Tensor:
    """
    Loads the entire protein embedding (L, D + 1) from a .pt file. 
    
    Parameters
    ----------
        pt_path : Path or str
            Path to the .pt file containing entire protein embedding.

    Returns
    -------
        Tensor
            The contiguous embedding as a loaded tensor.

    Raises
    ------
        ValueError
            If embedding is not stored as a tensor, error is raised as unsupported .pt format.
    """
    embedding = torch.load(pt_path, map_location="cpu")
    if isinstance(embedding, torch.Tensor):
        return embedding.contiguous()
    else:
        raise ValueError(f"Unsupported .pt format in '{pt_path}'")
    
def load_one_hot_target(row: pd.Series) -> torch.Tensor:
    """
    Loads the one-hot encoded model targets from a DataFrame row.

    Parameters
    ----------
        row : Series
            A row from a Pandas DataFrame as type series.

    Returns
    -------
        Tensor
            A vector containing the one-hot encoded model targets.

    Raises
    ------
        ValueError
            If the vector values do not sum to 1.
    """
    y_def = int(row["Def epitope"])
    y_unc = int(row["Uncertain"])
    y_not = int(row["Not epitope"])

    vec = torch.tensor([y_def, y_unc, y_not], dtype=torch.float32)

    if vec.sum().item() != 1:
        raise ValueError(f"Row '{row.CodeName}' has invalid label combination: "
                         f"Def={y_def}, Uncertain={y_unc}, Not={y_not}")
    
    return vec

class PeptideDatasetBuilder:
    """
    Builds out the PeptideDataset dataclass for use in downstream model training.

    Parameters
    ----------
        meta_path : Path or str
            Path to the metadata TSV file.
        emb_dir : Path or str
            Path to the directory storing embeddings.
        logger : Logger
            Logger to keep track of progress and write errors.
    """
    def __init__(self, meta_path: Path | str, 
                 emb_dir: Path | str, 
                 logger: logging.Logger):
        self.meta_path = Path(meta_path)
        self.emb_dir = Path(emb_dir)
        self.df = pd.read_csv(meta_path, sep="\t")

        self._full_emb_cache: Dict[str, torch.Tensor] = {}

        self.logger = logger

    def _get_full_embedding(self, protein_id: str) -> torch.Tensor:
        """
        Adds protein embedding to cache if found, otherwise throws FileNotFoundError.
        """
        if protein_id not in self._full_emb_cache:
            pt_path = self.emb_dir / f"{protein_id}.pt"
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Missing embedding file: '{pt_path}'")
            self._full_emb_cache[protein_id] = load_protein_embedding(pt_path)

        return self._full_emb_cache[protein_id]
    
    def build(self, peptide_len: int = 30) -> PeptideDataset:
        """
        Builds the PeptideDataset dataclass for downstream model training.

        Parameters
        ----------
            peptide_len : int
                The expected length of a peptide for this dataset. We are expecting 30mers by default.

        Returns
        -------
            PeptideDataset
                A fully populated PeptideDataset object.

        Raises
        ------
            IndexError
                If parsed align_start and align_stop values are out of embedding bounds.
        """
        embeddings: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        code_names: List[str] = []
        protein_ids: List[str] = []
        peptides: List[str] = []
        align_starts: List[int] = []
        align_stops: List[int] = []

        t0 = time.perf_counter()
        self.logger.info("starting_build_loop", 
                         extra={"extra": {
                             "expected_peptide_len": peptide_len
                         }})

        for index, row in self.df.iterrows():
            code_name = str(row["CodeName"])
            align_start = int(row["AlignStart"])
            align_stop = int(row["AlignStop"])
            peptide_seq = str(row["Peptide"])

            fullname = str(row["FullName"])
            protein_id = parse_id_from_fullname(fullname)

            # skip missing embedding files
            try:
                full_emb = self._get_full_embedding(protein_id) # (L_full, D)
            except FileNotFoundError:
                continue
 
            L_full = full_emb.size(0)

            if align_start < 0 or align_stop > L_full:
                raise IndexError(f"Row {index} asks for slice [{align_start}:{align_stop}] "
                                 f"but full embedding length is {L_full} for protein '{protein_id}'")
            
            pep_emb = full_emb[align_start:align_stop] # (L_pep, D), usually (30, 1281)
            L_pep = pep_emb.size(0)
            parsed_pep_seq_len = len(peptide_seq)

            if L_pep != peptide_len or L_pep != parsed_pep_seq_len:
                # log warning then skip peptide
                self.logger.warning("length_mismatch", 
                                    extra={"extra": {
                                        "row_index": index,
                                        "protein_id": protein_id,
                                        "code_name": code_name,
                                        "align_start": align_start,
                                        "align_stop": align_stop,
                                        "peptide_len_from_param": peptide_len,
                                        "embedding_len": L_pep,
                                        "peptide_seq_len": parsed_pep_seq_len,
                                        "peptide_seq": peptide_seq,
                                    }})
                continue

            target_vec = load_one_hot_target(row)

            # build lists
            embeddings.append(pep_emb)
            targets.append(target_vec)
            code_names.append(code_name)
            protein_ids.append(protein_id)
            peptides.append(peptide_seq)
            align_starts.append(align_start)
            align_stops.append(align_stop)

        self.logger.info("build_loop_finished", 
                         extra={"extra": {
                             "total_duration_s": round(time.perf_counter() - t0, 3)
                         }})

        emb_tensor: torch.Tensor | List[torch.Tensor] = embeddings

        # stack all peptides into same tensor if they have same shape
        if all(emb.shape[0] == peptide_len for emb in embeddings):
            emb_tensor = torch.stack(embeddings, dim=0) # (N, L, D)

        return PeptideDataset(embeddings=emb_tensor, 
                                 targets=targets, 
                                 code_names=code_names, 
                                 protein_ids=protein_ids, 
                                 peptides=peptides, 
                                 align_starts=align_starts, 
                                 align_stops=align_stops)
