import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch

import pepseqpred.apps.esm_cli as esm_cli
import pepseqpred.apps.labels_cli as labels_cli
import pepseqpred.apps.train_ffnn_cli as train_cli
from pepseqpred.core.io.keys import parse_fullname
from pepseqpred.core.preprocess.preparedataset import prepare_dataset
from pepseqpred.core.train.split import split_ids_grouped

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class FakeAlphabet:
    def get_batch_converter(self):
        def _batch_converter(pairs):
            labels = [name for name, _seq in pairs]
            seqs = [seq for _name, seq in pairs]
            max_len = max((len(seq) for seq in seqs), default=0)
            tokens = torch.zeros((len(seqs), max_len + 2), dtype=torch.long)
            for i, seq in enumerate(seqs):
                seq_len = len(seq)
                tokens[i, 1:1 + seq_len] = 1
                tokens[i, 1 + seq_len] = 2
            return labels, seqs, tokens

        return _batch_converter


class FakeESMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch_tokens, repr_layers, return_contacts=False):
        _ = return_contacts
        batch_size, token_len = batch_tokens.shape
        rep_dim = 3  # append_seq_len => final emb dim=4
        reps = torch.ones((batch_size, token_len, rep_dim),
                          dtype=torch.float32)
        return {"representations": {repr_layers[0]: reps}}


def _write_code_list(path: Path, codes: list[str]) -> None:
    path.write_text("Sequence name\n" + "\n".join(codes) +
                    "\n", encoding="utf-8")


def _append_fasta_records(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("a", encoding="utf-8") as out_f:
        for header, seq in records:
            out_f.write(f">{header}\n{seq}\n")


def _build_pv1_inputs(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "pv1_meta.tsv"
    z = root / "pv1_z.tsv"
    fasta = root / "pv1_targets.fasta"

    pd.DataFrame(
        [
            {
                "CodeName": "pv1_pep_1",
                "Category": "SetCover",
                "SpeciesID": "1",
                "Species": "PV1",
                "Protein": "Prot",
                "FullName": "ID=PV1P001 AC=A1 OXX=11,22,301_0_4",
                "Peptide": "MNPQ",
                "Encoding": "enc",
            },
            {
                "CodeName": "pv1_pep_2",
                "Category": "SetCover",
                "SpeciesID": "1",
                "Species": "PV1",
                "Protein": "Prot",
                "FullName": "ID=PV1P001 AC=A1 OXX=11,22,301_2_6",
                "Peptide": "PQRS",
                "Encoding": "enc",
            },
        ]
    ).to_csv(meta, sep="\t", index=False)
    pd.DataFrame(
        [
            {"Sequence name": "pv1_pep_1", "VW_001": 30.0, "VW_002": 0.0},
            {"Sequence name": "pv1_pep_2", "VW_001": 1.0, "VW_002": 2.0},
        ]
    ).to_csv(z, sep="\t", index=False)
    fasta.write_text(
        ">ID=PV1P001 AC=A1 OXX=11,22,301\nMNPQRS\n",
        encoding="utf-8",
    )
    return meta, z, fasta


def _build_cwp_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "cwp_meta.tsv"
    reactive = root / "cwp_reactive.tsv"
    nonreactive = root / "cwp_nonreactive.tsv"
    fasta = root / "cwp_targets.faa"

    pd.DataFrame(
        [
            {
                "CodeName": "CWP_000001",
                "SequenceAccession": "A0CWP1",
                "Cluster50ID": "Cocci_id50_010",
                "StartIndex": 0,
                "StopIndex": 4,
                "PeptideSequence": "ACDE",
            },
            {
                "CodeName": "CWP_000002",
                "SequenceAccession": "A0CWP1",
                "Cluster50ID": "Cocci_id50_010",
                "StartIndex": 1,
                "StopIndex": 5,
                "PeptideSequence": "CDEF",
            },
        ]
    ).to_csv(meta, sep="\t", index=False)
    _write_code_list(reactive, ["CWP_000001"])
    _write_code_list(nonreactive, ["CWP_000002"])
    fasta.write_text(">tr|A0CWP1|A0CWP1_FAKE\nACDEFG\n", encoding="utf-8")
    return meta, reactive, nonreactive, fasta


def _build_bkp_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "bkp_meta.tsv"
    reactive = root / "bkp_reactive.tsv"
    nonreactive = root / "bkp_nonreactive.tsv"
    fasta = root / "bkp_targets.faa"

    pd.DataFrame(
        [
            {
                "CodeName": "BKP_000001",
                "SequenceAccession": "A0BKP1",
                "reClusterID_70": "BKP1_id70_200",
                "alignStart": "0.0",
                "alignStop": "4.0",
                "PeptideSequence": "WXYZ",
            },
            {
                "CodeName": "BKP_000002",
                "SequenceAccession": "A0BKP1",
                "reClusterID_70": "BKP1_id70_200",
                "alignStart": "1.0",
                "alignStop": "5.0",
                "PeptideSequence": "XYZA",
            },
        ]
    ).to_csv(meta, sep="\t", index=False)
    _write_code_list(reactive, ["BKP_000001"])
    _write_code_list(nonreactive, ["BKP_000002"])
    fasta.write_text(">tr|A0BKP1|A0BKP1_FAKE\nWXYZAB\n", encoding="utf-8")
    return meta, reactive, nonreactive, fasta


def test_prepare_dataset_multisource_pipeline_smoke(monkeypatch, tmp_path: Path):
    # Build three mini datasets.
    pv1_meta, pv1_z, pv1_fasta = _build_pv1_inputs(tmp_path / "pv1")
    cwp_meta, cwp_reactive, cwp_nonreactive, cwp_fasta = _build_cwp_inputs(
        tmp_path / "cwp")
    bkp_meta, bkp_reactive, bkp_nonreactive, bkp_fasta = _build_bkp_inputs(
        tmp_path / "bkp")

    out_pv1 = tmp_path / "out_pv1"
    out_cwp = tmp_path / "out_cwp"
    out_bkp = tmp_path / "out_bkp"

    prepare_dataset(
        dataset_kind="pv1",
        meta_path=pv1_meta,
        z_path=pv1_z,
        output_dir=out_pv1,
        protein_fasta=pv1_fasta,
        is_epitope_min_subjects=1,
    )
    prepare_dataset(
        dataset_kind="cwp",
        meta_path=cwp_meta,
        output_dir=out_cwp,
        protein_fasta=cwp_fasta,
        reactive_codes=cwp_reactive,
        nonreactive_codes=cwp_nonreactive,
        group_id_offset=100_000_000,
    )
    prepare_dataset(
        dataset_kind="bkp",
        meta_path=bkp_meta,
        output_dir=out_bkp,
        protein_fasta=bkp_fasta,
        reactive_codes=bkp_reactive,
        nonreactive_codes=bkp_nonreactive,
        group_id_offset=200_000_000,
    )

    # Combine prepared artifacts.
    combined_dir = tmp_path / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_fasta = combined_dir / "prepared_targets.fasta"
    combined_meta = combined_dir / "prepared_labels_metadata.tsv"
    combined_emb_meta = combined_dir / "prepared_embedding_metadata.tsv"
    combined_fasta.write_text("", encoding="utf-8")

    for source in [out_pv1, out_cwp, out_bkp]:
        recs = []
        header = None
        seq_lines = []
        for raw in (source / "prepared_targets.fasta").read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line == "":
                continue
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            recs.append((header, "".join(seq_lines)))
        _append_fasta_records(combined_fasta, recs)

    labels_df = pd.concat(
        [
            pd.read_csv(out_pv1 / "prepared_labels_metadata.tsv", sep="\t"),
            pd.read_csv(out_cwp / "prepared_labels_metadata.tsv", sep="\t"),
            pd.read_csv(out_bkp / "prepared_labels_metadata.tsv", sep="\t"),
        ],
        ignore_index=True,
    )
    labels_df.to_csv(combined_meta, sep="\t", index=False)

    emb_meta_df = pd.concat(
        [
            pd.read_csv(out_pv1 / "prepared_embedding_metadata.tsv", sep="\t"),
            pd.read_csv(out_cwp / "prepared_embedding_metadata.tsv", sep="\t"),
            pd.read_csv(out_bkp / "prepared_embedding_metadata.tsv", sep="\t"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["Name", "Family"])
    emb_meta_df.to_csv(combined_emb_meta, sep="\t", index=False)

    # Assert grouped split behavior: no family overlap between train/val IDs.
    id_to_family = {
        parse_fullname(str(name))[0]: str(int(family))
        for name, family in emb_meta_df[["Name", "Family"]].itertuples(index=False, name=None)
    }
    all_ids = sorted(id_to_family.keys())
    train_ids, val_ids = split_ids_grouped(
        all_ids,
        val_frac=0.34,
        seed=11,
        groups=id_to_family,
    )
    train_fams = {id_to_family[pid] for pid in train_ids}
    val_fams = {id_to_family[pid] for pid in val_ids}
    assert train_fams.isdisjoint(val_fams)

    # Run ESM CLI with fake model.
    fake_pretrained = types.SimpleNamespace(
        fake_model=lambda: (FakeESMModel(), FakeAlphabet())
    )
    monkeypatch.setattr(esm_cli.esm, "pretrained", fake_pretrained)
    monkeypatch.setattr(esm_cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(esm_cli.torch.cuda, "device_count", lambda: 0)

    embs_out = tmp_path / "esm_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "esm_cli.py",
            "--fasta-file",
            str(combined_fasta),
            "--metadata-file",
            str(combined_emb_meta),
            "--out-dir",
            str(embs_out),
            "--embedding-key-mode",
            "id-family",
            "--model-name",
            "fake_model",
            "--max-tokens",
            "16",
            "--batch-size",
            "4",
        ],
    )
    esm_cli.main()

    emb_dir = embs_out / "artifacts" / "pts"
    assert emb_dir.exists()

    # Run labels CLI against prepared metadata and generated embeddings.
    labels_pt = tmp_path / "labels.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "labels_cli.py",
            str(combined_meta),
            str(labels_pt),
            "--emb-dir",
            str(emb_dir),
            "--embedding-key-delim",
            "-",
            "--calc-pos-weight",
        ],
    )
    labels_cli.main()
    assert labels_pt.exists()

    # Train smoke run using grouped split (id-family) over all three datasets.
    save_dir = tmp_path / "train_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ffnn_cli.py",
            "--embedding-dirs",
            str(emb_dir),
            "--label-shards",
            str(labels_pt),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--hidden-sizes",
            "8",
            "--dropouts",
            "0.1",
            "--val-frac",
            "0.34",
            "--split-type",
            "id-family",
            "--split-seeds",
            "11",
            "--train-seeds",
            "101",
            "--save-path",
            str(save_dir),
            "--results-csv",
            str(save_dir / "runs.csv"),
        ],
    )
    train_cli.main()

    assert (save_dir / "runs.csv").exists()
    run_dirs = list(save_dir.glob("run_*"))
    assert run_dirs
    assert (run_dirs[0] / "fully_connected.pt").exists()
