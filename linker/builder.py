import os
import re
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Optional
import pandas as pd
import numpy as np
import torch
from peptidedataset import PeptideDataset

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
        emb_dirs : List[Path or str]
            List of paths to the directories storing embeddings.
        logger : Logger
            Logger to keep track of progress and write errors.
    """
    def __init__(self, meta_path: Path | str, 
                 emb_dirs: List[Path | str], 
                 logger: logging.Logger):
        self.meta_path = Path(meta_path)
        self.emb_dirs = [Path(emb_dir) for emb_dir in emb_dirs]

        self.df = pd.read_csv(meta_path, sep="\t")
        self.df["protein_id"] = self.df["FullName"].astype(str).map(parse_id_from_fullname)

        self.logger = logger

    def _find_pt_path(self, protein_id: str) -> Path:
        """Searches for embedding on disk in one or more directories. Throws FileNotFoundError if missing."""
        filename = f"{protein_id}.pt"
        for emb_dir in self.emb_dirs:
            pt_path = emb_dir / filename
            if pt_path.exists():
                return pt_path
            
        searched_dirs = [str(emb_dir) for emb_dir in self.emb_dirs]
        raise FileNotFoundError(f"Missing embedding file '{filename}' in embedding directories: {searched_dirs}")

    def _get_full_embedding(self, protein_id: str) -> torch.Tensor:
        """Loads protein embedding from disk."""
        pt_path = self._find_pt_path(protein_id)
        return load_protein_embedding(pt_path)
    
    def _infer_D(self) -> int:
        """Infers D from one embedding and returns it, raises ValueError if unexpected dimension or RuntimeError if D cannot be inferred."""
        for protein_id in self.df["protein_id"].drop_duplicates().tolist():
            try:
                emb = self._get_full_embedding(protein_id)
            except FileNotFoundError:
                continue
            if emb.dim() != 2:
                raise ValueError(f"Expected 2D embedding (L, D) for '{protein_id}', got {tuple(emb.shape)}")
            emb_dim = int(emb.size(1))
            del emb
            return emb_dim
        raise RuntimeError("Could not infer D since no embeddings were found for any protein ID")
    
    def build(self, peptide_len: int = 30, memmap_dir: Optional[Path | str] = None, memmap_dtype: str = "float16") -> PeptideDataset:
        """
        Builds a PeptideDataset for downstream model training with bounded RAM usage.

        Parameters
        ----------
            peptide_len : int
                Expected peptide length (AlignStop - AlignStart) and expected 
                length of the peptide string in the metadata. Rows that do not 
                match are skipped. We are expecting 30mers by default.
            memmap_dir : Path or str or None
                Directory where temporary memmap files are created. If None, 
                defaults to the metadata file directory.
            memmap_dtype: str
                Floating-point dtype used for the stored embeddings (and targets in 
                this implementation). `float16` reduces disk footprint and write 
                bandwidth, but `float32` is also supported.

        Returns
        -------
            PeptideDataset
                Dataset containing: 
                embeddings: Tensor of shape (N, peptide_len, D)
                targets: Tensor of shape (N, 3) one-hot encoded labels plus 
                metadata lists aligned to the same row ordering.

        Raises
        ------
            RuntimeError
                If no peptides survive pass 1 filtering, or if pass 2 fails to 
                write all planned peptides.
            IndexError
                If a planned AlignStart/AlignStop slice is out of bounds for its 
                protein embedding during pass 2.
            ValueError
                If an embedding has unexpected dimensionality or embedding 
                dimension D.
        """
        memmap_dir = Path(memmap_dir) if memmap_dir is not None else self.meta_path.parent
        memmap_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        self.logger.info("starting_build_loop", 
                         extra={"extra": {
                             "expected_peptide_len": peptide_len
                         }})

        # start first pass, count valid peptides and collect metadata
        code_names: List[str] = []
        protein_ids: List[str] = []
        peptides: List[str] = []
        align_starts: List[int] = []
        align_stops: List[int] = []

        N = 0
        D = self._infer_D()

        records = []
        out_i = 0
        for index, row in self.df.iterrows():
            protein_id = str(row["protein_id"])
            code_name = str(row["CodeName"])
            align_start = int(row["AlignStart"])
            align_stop = int(row["AlignStop"])
            peptide_seq = str(row["Peptide"])

            L_pep = align_stop - align_start
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

            # record for pass 2
            records.append((out_i, protein_id, int(index), align_start, align_stop))

            # collect metadata only on first pass
            code_names.append(code_name)
            protein_ids.append(protein_id)
            peptides.append(peptide_seq)
            align_starts.append(align_start)
            align_stops.append(align_stop)

            out_i += 1

        N = out_i
        if N == 0:
            raise RuntimeError("Pass 1 found zero peptides after metadata filtering")

        self.logger.info("1st_pass_finished", 
                         extra={"extra": {
                             "valid_peptides": N, 
                             "peptide_len": peptide_len, 
                             "emb_dim": D, 
                             "duration_s": round(time.perf_counter() - t0, 3)
                         }})

        # start seconds pass, allocate memmap and write embeddings sequentially
        mm_name = f"peptides_L{peptide_len}_D{D}_N{N}_{os.getpid()}" # unique filename to prevent collisions
        emb_mm_path = memmap_dir / f"{mm_name}.emb.mm"
        tgt_mm_path = memmap_dir / f"{mm_name}.tgt.mm"

        np_dtype = np.float16 if memmap_dtype == "float16" else np.float32

        emb_mm = np.memmap(emb_mm_path, mode="w+", dtype=np_dtype, shape=(N, peptide_len, D))
        tgt_mm = np.memmap(tgt_mm_path, mode="w+", dtype=np.float32, shape=(N, 3))

        # setup records to line up data with pass 1
        by_protein = defaultdict(list)
        for out_i, protein_id, df_idx, a0, a1 in records:
            by_protein[protein_id].append((out_i, df_idx, a0, a1))

        # sanity check to avoid duplicate peptides
        written = np.zeros(N, dtype=np.bool_)
        for protein_id, items in by_protein.items():
            try:
                full_emb = self._get_full_embedding(protein_id) # (L_full, D)
            except FileNotFoundError:
                continue
            except Exception as e:
                self.logger.error(f"{e}: {protein_id}")
                continue

            if full_emb.dim() != 2 or int(full_emb.size(1)) != D:
                raise ValueError(f"Embedding dimension mismatch for '{protein_id}': expected={D}, got {tuple(full_emb.shape)}")
            
            L_full = int(full_emb.size(0))

            for out_i, df_idx, align_start, align_stop in items:
                if align_start < 0 or align_stop > L_full:
                    raise IndexError(f"Row {index} asks for slice [{align_start}:{align_stop}] "
                                    f"but full embedding length is {L_full} for protein '{protein_id}'")

                # copy slice without keeping full tensor
                pep_emb = full_emb[align_start:align_stop].contiguous()
                pep_emb = pep_emb.to(torch.float16 if memmap_dtype == "float16" else torch.float32)

                emb_mm[out_i, :, :] = pep_emb.cpu().numpy()
                tgt_vec = load_one_hot_target(self.df.loc[df_idx])
                tgt_mm[out_i, :] = tgt_vec.cpu().numpy()

                if written[out_i]:
                    raise RuntimeError(f"Duplicate write detected for out_i={out_i}")
                written[out_i] = True
            
            # manually delete full embedding
            del full_emb

        if not written.all():
            missing_peps = int((~written).sum())
            raise RuntimeError(f"Pass 2 did not write {missing_peps} planned peptides, check embeddings or filters")
        
        # flush disk
        emb_mm.flush()
        tgt_mm.flush()

        # wrap memmaps as tensors
        emb_tensor = torch.from_numpy(emb_mm)
        tgt_tensor = torch.from_numpy(tgt_mm)

        self.logger.info("build_loop_finished", 
                         extra={"extra": {
                             "total_duration_s": round(time.perf_counter() - t0, 3), 
                             "final_N": N, 
                             "memmap_dir": str(memmap_dir), 
                             "emb_memmap_path": str(emb_mm_path), 
                             "tgt_memmap_path": str(tgt_mm_path)
                         }})

        return PeptideDataset(embeddings=emb_tensor, 
                              targets=tgt_tensor, 
                              code_names=code_names, 
                              protein_ids=protein_ids, 
                              peptides=peptides, 
                              align_starts=align_starts, 
                              align_stops=align_stops)
