<p align="center">
  <img src="https://github.com/LadnerLab/PepSeqPred/blob/main/PepSeqPred_logo_black.png?raw=1" alt="PepSeqPred logo" width="320">
</p>

[![PyPI version](https://img.shields.io/pypi/v/pepseqpred.svg)](https://pypi.org/project/pepseqpred/)
[![Python versions](https://img.shields.io/pypi/pyversions/pepseqpred.svg)](https://pypi.org/project/pepseqpred/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://github.com/LadnerLab/PepSeqPred/blob/main/LICENSE)
[![API: Pretrained](https://img.shields.io/badge/API-pretrained%20models-2ea44f.svg)](https://pypi.org/project/pepseqpred/)
[![API: Artifact Path](https://img.shields.io/badge/API-artifact%20path-1f6feb.svg)](https://pypi.org/project/pepseqpred/)

PepSeqPred predicts residue-level epitope locations for protein sequences.

## At A Glance

- ![Pretrained](https://img.shields.io/badge/pretrained-ready-2ea44f.svg) Use bundled models with `load_pretrained_predictor(...)`.
- ![Artifact](https://img.shields.io/badge/artifact-.pt%20or%20.json-1f6feb.svg) Use your own `.pt` or `.json` artifacts with `load_predictor(...)`.
- ![Output](https://img.shields.io/badge/output-binary%20mask-8b5cf6.svg) Get residue-aligned binary epitope masks via `result.binary_mask`.
- ![Device](https://img.shields.io/badge/device-auto-f59e0b.svg) Use `device="auto"` to select CUDA when available, otherwise CPU.

## Install

```bash
pip install pepseqpred
```

## Predict With A Bundled Pretrained Model

```python
from pepseqpred import load_pretrained_predictor

protein_seq = "ACDEFGHIKLMNPQRSTVWY"

predictor = load_pretrained_predictor(model_id="default", device="auto")
result = predictor.predict_sequence(protein_seq, header="example_protein")

print(result.binary_mask)     # e.g. 000001110000...
print(result.n_epitopes)      # number of residues predicted as epitope
print(result.frac_epitope)    # fraction of residues predicted as epitope
```

To inspect available bundled models:

```python
from pepseqpred import list_pretrained_models

for info in list_pretrained_models():
    print(info.model_id, info.aliases, info.is_default)
```

## Predict From Your Own Artifact Path

Use this when you have your own trained PepSeqPred artifact:
- single checkpoint: `.pt`
- ensemble manifest: `.json`

```python
from pepseqpred import load_predictor

predictor = load_predictor(
    model_artifact="path/to/ensemble_manifest.json",  # or path/to/model.pt
    device="auto"
)

result = predictor.predict_sequence("ACDEFGHIKLMNPQRSTVWY")
print(result.binary_mask)
```

## FASTA I/O

```python
results = predictor.predict_fasta("input.fasta")
predictor.write_fasta_predictions("input.fasta", "predicted_masks.fasta")
```

## Notes

- `device="auto"` uses CUDA if available, otherwise CPU.
- `result.binary_mask` is aligned to the cleaned protein sequence.
