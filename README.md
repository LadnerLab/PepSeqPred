<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="PepSeqPred_logo_white.png">
    <source media="(prefers-color-scheme: light)" srcset="PepSeqPred_logo_black.png">
    <img src="PepSeqPred_logo_black.png" alt="PepSeqPred logo" width="320">
  </picture>
</p>

[![PyPI version](https://img.shields.io/pypi/v/pepseqpred.svg)](https://pypi.org/project/pepseqpred/)
[![Python versions](https://img.shields.io/pypi/pyversions/pepseqpred.svg)](https://pypi.org/project/pepseqpred/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![PyPI README](https://img.shields.io/badge/docs-PyPI%20quickstart-2ea44f.svg)](README.pypi.md)
[![Developer README](https://img.shields.io/badge/docs-developer%20pipeline-1f6feb.svg)](README.md)

# PepSeqPred Developer README

This README is the developer-facing reference for the full PepSeqPred training and evaluation pipeline.

For lightweight inference usage and API quickstart, use [README.pypi.md](README.pypi.md).

## Scope

PepSeqPred supports two usage profiles:

- **PyPI quickstart profile (`pip install pepseqpred`)**: user-facing inference API with bundled pretrained artifacts and artifact-path inference helpers.
- **Repository developer profile (`pip install -e .[dev]`)**: full source tree for preprocessing, embeddings, label generation, FFNN training, Optuna tuning, prediction, evaluation, and HPC orchestration.

The repository profile is the source of truth for reproducing experiments end-to-end.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/pepseqpred/apps/` | CLI entrypoints for each pipeline stage |
| `src/pepseqpred/core/preprocess/` | Metadata and z-score preprocessing |
| `src/pepseqpred/core/embeddings/` | ESM-2 sequence embedding generation |
| `src/pepseqpred/core/labels/` | Residue-level label construction |
| `src/pepseqpred/core/data/` | Iterable dataset and windowing/padding logic |
| `src/pepseqpred/core/models/` | FFNN model definitions |
| `src/pepseqpred/core/train/` | DDP, splitting, metrics, thresholds, trainer, seeds, weights |
| `src/pepseqpred/core/predict/` | Checkpoint/manifest resolution and inference logic |
| `src/pepseqpred/core/io/` | FASTA/TSV readers, key parsing, logging, CSV appends |
| `src/pepseqpred/api/` | Stable Python inference API and pretrained registry |
| `scripts/hpc/` | SLURM wrappers for each production stage |
| `scripts/tools/` | Zipapp build tools and Cocci eval prep/compare tooling |
| `tests/` | Unit, integration, and e2e coverage |
| `envs/` | Conda environment specs for local and HPC |

## End-to-End Pipeline

```text
Stage 1  normalize dataset inputs (PV1/CWP/BKP) to a shared training contract
Stage 2  generate ESM-2 per-residue embeddings
Stage 3  build residue-level label shards
Stage 4  train FFNN (seeded or ensemble-kfold, DDP-aware)
Stage 5  optional Optuna tuning (DDP-aware)
Stage 6  predict residue masks from checkpoint/manifest
Stage 7  evaluate residue metrics (+ optional Cocci peptide compare)
```

## Stage Reference

### Stage 1: Multi-Dataset Prepare (PV1/CWP/BKP)

**CLI:** `pepseqpred-prepare-dataset` (`src/pepseqpred/apps/prepare_dataset_cli.py`)

This stage is the recommended entrypoint when training on one or more of:

- PV1 (human virome)
- CWP/Cocci (fungal)
- BKP (bacterial)

It normalizes source-specific metadata and FASTA headers into a shared PV1-compatible contract so downstream embedding, label generation, and training CLIs can be reused unchanged.

**Core module**

- `src/pepseqpred/core/preprocess/preparedataset.py`

**Required output contract per dataset**

- `prepared_targets.fasta`
- `prepared_labels_metadata.tsv`
- `prepared_embedding_metadata.tsv`
- `prepare_summary.json`

**PV1 inputs and command**

- metadata TSV
- z-score TSV
- protein FASTA

```bash
pepseqpred-prepare-dataset \
  localdata/PV1/PV1_meta_2020-11-23_cleaned.tsv \
  localdata/PV1/prepared \
  --dataset-kind pv1 \
  --protein-fasta localdata/PV1/PV1_targets.fasta \
  --z-file localdata/PV1/PV1_zscores.tsv
```

**CWP/Cocci inputs and command**

- metadata TSV
- protein FASTA
- reactive code list TSV
- non-reactive code list TSV

```bash
pepseqpred-prepare-dataset \
  localdata/Cocci/CWP_metadata.tsv \
  localdata/Cocci/prepared \
  --dataset-kind cwp \
  --protein-fasta localdata/Cocci/CWP_targets.faa \
  --reactive-codes localdata/Cocci/CWP_reactive_Z20N4.tsv \
  --nonreactive-codes localdata/Cocci/CWP_nonreactive_Z20N4.tsv
```

**BKP inputs and command**

- metadata TSV
- protein FASTA
- reactive code list TSV
- non-reactive code list TSV

```bash
pepseqpred-prepare-dataset \
  localdata/BKP/BKP_metadata.tsv \
  localdata/BKP/prepared \
  --dataset-kind bkp \
  --protein-fasta localdata/BKP/BKP.faa \
  --reactive-codes localdata/BKP/BKP_reactive_Z20N4.tsv \
  --nonreactive-codes localdata/BKP/BKP_nonreactive_Z20N4.tsv
```

**Dataset-specific grouping used for leakage-aware splitting (`--split-type id-family`)**

- PV1: family from PV1 `OXX`
- CWP/Cocci: `Cluster50ID` mapped to deterministic numeric IDs
- BKP: `reClusterID_70` mapped to deterministic numeric IDs

**Next stages after prepare**

- run `pepseqpred-esm` with `--embedding-key-mode id-family` and each dataset's `prepared_embedding_metadata.tsv`
- run `pepseqpred-labels` with `--embedding-key-delim -`
- train with `--split-type id-family`

### Stage 1 (Legacy): PV1 Z-Score Preprocess

**CLI:** `pepseqpred-preprocess` (`src/pepseqpred/apps/preprocess_cli.py`)

**Inputs**

- metadata TSV (PV1-style)
- z-score TSV

**Core modules**

- `core/preprocess/pv1.py`
- `core/preprocess/zscores.py`
- `core/io/read.py`

**Command**

```bash
pepseqpred-preprocess data/meta.tsv data/zscores.tsv --save
```

**Output**

- training-ready metadata TSV with `Def epitope`, `Uncertain`, `Not epitope`
- default filename pattern: `input_data_<is_epi_z>_<is_epi_min_subs>_<not_epi_z>_<not_epi_max_subs|all>.tsv`

### Stage 2: Generate ESM-2 Embeddings

**CLI:** `pepseqpred-esm` (`src/pepseqpred/apps/esm_cli.py`)

**Inputs**

- FASTA file
- optional metadata TSV for `id-family` naming mode

**Core modules**

- `core/embeddings/esm2.py`
- `core/io/read.py`
- `core/io/keys.py`

**Command**

```bash
pepseqpred-esm \
  --fasta-file data/targets.fasta \
  --out-dir localdata/esm2 \
  --embedding-key-mode id-family \
  --key-delimiter - \
  --model-name esm2_t33_650M_UR50D \
  --max-tokens 1022 \
  --batch-size 8
```

**Output**

- per-protein embedding files under `<out-dir>/artifacts/pts/*.pt`
- embedding index CSV under `<out-dir>/artifacts/*.csv`
- optional shard-specific outputs when `--num-shards > 1`

### Stage 3: Build Residue Labels

**CLI:** `pepseqpred-labels` (`src/pepseqpred/apps/labels_cli.py`)

**Inputs**

- preprocessed metadata TSV
- one or more embedding directories

**Core module**

- `core/labels/builder.py`

**Command**

```bash
pepseqpred-labels \
  data/input_data_20_4_10_all.tsv \
  localdata/labels/labels_shard_000.pt \
  --emb-dir localdata/esm2/artifacts/pts/shard_000 \
  --restrict-to-embeddings \
  --calc-pos-weight \
  --embedding-key-delim -
```

**Output**

- label shard `.pt` with protein label tensors and peptide metadata
- optional `class_stats` payload when `--calc-pos-weight` is enabled

### Stage 4: Train FFNN

**CLI:** `pepseqpred-train-ffnn` (`src/pepseqpred/apps/train_ffnn_cli.py`)

**Modes**

- `seeded`: split/train seed pairs define runs
- `ensemble-kfold`: K-fold members per set, optional multiple set seeds

**Core modules**

- `core/data/proteindataset.py`
- `core/models/ffnn.py`
- `core/train/{trainer,split,ddp,metrics,threshold,weights,seed,embedding}.py`

**Command (smoke)**

```bash
pepseqpred-train-ffnn \
  --embedding-dirs localdata/esm2/artifacts/pts/shard_000 \
  --label-shards localdata/labels/labels_shard_000.pt \
  --epochs 1 \
  --subset 100 \
  --save-path localdata/models/ffnn_smoke \
  --results-csv localdata/models/ffnn_smoke/runs.csv
```

**Outputs**

- run checkpoint(s), usually `fully_connected.pt`
- per-run CSV (`runs.csv` or `multi_run_results.csv`)
- aggregate `multi_run_summary.json`
- ensemble manifest JSON in `ensemble-kfold` mode

### Stage 5: Optuna Tuning (Optional)

**CLI:** `pepseqpred-train-ffnn-optuna` (`src/pepseqpred/apps/train_ffnn_optuna_cli.py`)

**Core modules**

- same data/model/train stack as Stage 4
- Optuna trial orchestration in app layer

**Command (smoke)**

```bash
pepseqpred-train-ffnn-optuna \
  --embedding-dirs localdata/esm2/artifacts/pts/shard_000 \
  --label-shards localdata/labels/labels_shard_000.pt \
  --n-trials 2 \
  --epochs 1 \
  --save-path localdata/models/optuna_smoke \
  --csv-path localdata/models/optuna_smoke/trials.csv
```

**Outputs**

- trial rows CSV
- study storage (if configured)
- per-trial checkpoints under `trials/trial_*`
- `best_trial.json` and copied best checkpoint

### Stage 6: Predict

**CLI:** `pepseqpred-predict` (`src/pepseqpred/apps/prediction_cli.py`)

**Accepted model artifact types**

- single checkpoint `.pt`
- ensemble manifest `.json` (schema v1 or v2)

**Core modules**

- `core/predict/artifacts.py`
- `core/predict/inference.py`

**Command**

```bash
pepseqpred-predict \
  localdata/models/run_001/fully_connected.pt \
  data/inference_targets.fasta \
  --output-fasta localdata/predictions/predictions.fasta
```

**Output**

- FASTA containing binary residue masks

### Stage 7: Evaluate

**CLI:** `pepseqpred-eval-ffnn` (`src/pepseqpred/apps/evaluate_ffnn_cli.py`)

**Capabilities**

- evaluate single checkpoint or ensemble manifest
- optional set auto-selection from `runs.csv`
- optional fold-level metrics and ROC/PR curves
- optional plot generation

**Core modules**

- `core/predict/artifacts.py`
- `core/predict/inference.py`
- `core/data/proteindataset.py`
- `core/train/metrics.py`

**Command**

```bash
pepseqpred-eval-ffnn \
  localdata/models/run_001/fully_connected.pt \
  --embedding-dirs localdata/esm2/artifacts/pts/shard_000 \
  --label-shards localdata/labels/labels_shard_000.pt \
  --output-json localdata/eval/ffnn_eval_summary.json
```

**Output**

- residue-level evaluation JSON
- optional fold payloads, curves, and plot files

## Inference API (`pepseqpred.api`)

The stable Python API is implemented in:

- `src/pepseqpred/api/predictor.py`
- `src/pepseqpred/api/pretrainedregistry.py`
- `src/pepseqpred/api/types.py`

Top-level exports (`import pepseqpred`):

- `load_pretrained_predictor`
- `list_pretrained_models`
- `load_predictor`
- `predict_sequence`
- `predict_fasta`
- `PepSeqPredictor`
- `PredictionResult`

Bundled pretrained registry currently includes:

- `flagship1-v1` (`alias: flagship1`)
- `flagship2-v1` (`aliases: flagship2, default`)

## Artifact Contracts

### Embedding `.pt`

- tensor shape: `(L, D+1)`
- `L`: residue count
- final feature column stores sequence length

### Label shard `.pt`

```python
{
  "labels": {"<protein_id>": Tensor[(L,3)] or Tensor[(L,)]},
  "proteins": {"<protein_id>": {"tax_info": {...}, "peptides": [...]}}
  # optional:
  "class_stats": {"pos_count": int, "neg_count": int, "pos_weight": float}
}
```

### Training checkpoint `.pt`

```python
{
  "model_state_dict": ..., 
  "optim_state_dict": ..., 
  "epoch": int,
  "config": {...},
  "best_loss": float,
  "metrics": {...}
}
```

### Ensemble manifest JSON

- schema v1: single set, `members` list
- schema v2: root `sets` list with `set_index`, each with `members`
- members are filtered by `status == "OK"` in predict/eval resolution

## CLI Reference

| CLI | File | Purpose |
| --- | --- | --- |
| `pepseqpred-prepare-dataset` | `apps/prepare_dataset_cli.py` | normalize PV1/CWP/BKP into shared training contract |
| `pepseqpred-preprocess` | `apps/preprocess_cli.py` | metadata + z-score preprocessing |
| `pepseqpred-esm` | `apps/esm_cli.py` | ESM-2 embedding generation |
| `pepseqpred-labels` | `apps/labels_cli.py` | residue label shard generation |
| `pepseqpred-train-ffnn` | `apps/train_ffnn_cli.py` | seeded or ensemble-kfold training |
| `pepseqpred-train-ffnn-optuna` | `apps/train_ffnn_optuna_cli.py` | Optuna tuning |
| `pepseqpred-predict` | `apps/prediction_cli.py` | FASTA inference from checkpoint/manifest |
| `pepseqpred-eval-ffnn` | `apps/evaluate_ffnn_cli.py` | residue-level evaluation |

## HPC Script Reference (`scripts/hpc`)

These wrappers are production-facing interfaces and should be treated as first-class entrypoints.

| Script | Stage | Default resources |
| --- | --- | --- |
| `generateembeddings.sh` | Embeddings | GPU, array `0-3`, `a100`, `2` CPU/GPU, `8G`/GPU, `01:00:00` |
| `generatelabels.sh` | Labels | CPU, `1` CPU, `16G`, `01:00:00` |
| `trainffnn.sh` | Train FFNN | GPU, `4xa100`, `20` CPU, `256G`, `12:00:00` |
| `trainffnnoptuna.sh` | Optuna | GPU, `4xa100`, `20` CPU, `448G`, `48:00:00` |
| `predictepitope.sh` | Predict | GPU, `a100`, `4` CPU, `32G`, `00:30:00` |
| `evaluateffnn.sh` | End-to-end eval pipeline | GPU, `a100`, `8` CPU, `128G`, `04:00:00` |
| `evalffnnsweep.sh` | Seeded eval batch submitter | wrapper script (calls `evaluateffnn.sh`) |
| `preprocessdata.sh` | Preprocess helper | local helper, not a SLURM script |

### Important HPC notes

- `evaluateffnn.sh` orchestrates prepare, embed, labels, predict, eval, and peptide compare stages with stage toggles (`RUN_PREP`, `RUN_EMBED`, `RUN_LABELS`, `RUN_PREDICT`, `RUN_EVAL`, `RUN_COMPARE`).
- `evaluateffnn.sh` and `evalffnnsweep.sh` depend on `scripts/tools/cocci_eval_pipeline.py`.
- HPC wrappers expect `.pyz` runtime artifacts in the working directory (for example `esm.pyz`, `train_ffnn.pyz`, `predict.pyz`).

## Zipapp and Tooling (`scripts/tools`)

| Tool | Purpose |
| --- | --- |
| `buildpyz.py` | build `.pyz` runtime apps from `src/pepseqpred` |
| `pyzapps.py` | registry of app target names to module entrypoints |
| `cocci_eval_pipeline.py` | Cocci-specific eval subset prep and peptide compare |
| `rename_embeddings_id_family.py` | rename `ID.pt` to `ID-family.pt` embeddings from metadata |

Build examples:

```bash
python scripts/tools/buildpyz.py --list
python scripts/tools/buildpyz.py esm
python scripts/tools/buildpyz.py all
```

## Testing Map

Test suites are organized as:

- `tests/unit/`: module-level behavior and edge cases
- `tests/integration/`: CLI-level smoke and interactions
- `tests/e2e/`: train-to-predict boundary validation

Representative coverage areas include:

- API registry and predictor behavior (`tests/unit/api/*`)
- dataset, embeddings, labels, predict, train internals (`tests/unit/core/*`)
- CLI parsers and eval selection/curve logic (`tests/unit/apps/*`)
- checkpoint/manifest prediction/evaluation smoke tests (`tests/integration/*`)

Run sequence:

```bash
ruff check .
pytest tests/unit
pytest tests/integration
pytest tests/e2e
```

## Environment and Setup

### Conda

```bash
conda env create -f envs/environment.local.yml
conda activate pepseqpred
pip install -e .[dev]
```

For GPU/HPC development:

```bash
conda env create -f envs/environment.hpc.yml
conda activate pepseqpred
pip install -e .[dev]
```

### Pip + venv

```bash
python -m venv .venv
. .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -e .[dev]
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

## Reproducibility and Safety Guardrails

When changing training or evaluation logic:

- preserve split semantics (`id` vs `id-family`)
- preserve seed handling and deterministic run planning
- avoid rank-dependent side effects in DDP code paths
- only write shared artifacts from intended rank
- avoid output path collisions between experiments
- prefer smoke tests over expensive full retraining during development

## Known Operational Notes

- `id-family` embedding key mode requires metadata family mapping.
- Label generation must align embedding naming with `--embedding-key-delim` (`""` for `ID.pt`, `-` for `ID-family.pt`).
- Prediction/evaluation threshold overrides must remain in `(0.0, 1.0)`.
- Ensemble manifests are resolved by valid `status=OK` members and optional `k_folds` truncation.
- For HPC pipelines, keep `.pyz` artifacts and shell scripts in the same execution directory unless intentionally using module imports.

## Contact

- GitHub issues: bug reports, feature requests, and development questions
- Maintainers: [Jeffrey Hoelzel](mailto:jmh2338@nau.edu), [Jason Ladner](mailto:jason.ladner@nau.edu)
