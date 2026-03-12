<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="PepSeqPred_logo_white.png">
    <source media="(prefers-color-scheme: light)" srcset="PepSeqPred_logo_black.png">
    <img src="PepSeqPred_logo_black.png" alt="PepSeqPred logo" width="320">
  </picture>
</p>

# Overview

PepSeqPred is a residue-level epitope prediction pipeline for protein workflows.  
It converts upstream assay and sequence data into training-ready artifacts, trains feed-forward neural network models on ESM-2 embeddings, and produces binary residue masks for downstream inference.

## About

PepSeqPred is designed for research workflows where you need to:
- map peptide-level signals to residue-level supervision,
- train reproducible epitope prediction models,
- run large jobs on HPCs with DistributedDataParallel (DDP),
- generate per-residue binary epitope predictions to develop new peptide libraries.

The pipeline is built around CLI entrypoints in `src/pepseqpred/apps/` and matching HPC scripts in `scripts/hpc/`.

## What Goes In / What Comes Out

Typical inputs:
- Metadata and reactivity tables (TSV) for preprocessing and label generation.
- Protein FASTA files for embedding generation and prediction.
- Optional metadata file for ID-family embedding key generation.

Typical outputs:
- Preprocessed model input table (TSV).
- Per-protein ESM-2 embedding shards (`.pt`) plus embedding index CSV.
- Residue-level label shards (`.pt`) including epitope / uncertain / non-epitope supervision.
- FFNN checkpoints and run summaries (for standard training or Optuna tuning).
- Predicted binary epitope mask FASTA files for inference.

## Pipeline Snapshot

```text
Preprocess data
  -> cleaned + labeled metadata TSV for downstream steps

Generate ESM-2 embeddings
  -> per-protein embedding .pt files (+ index CSV), often sharded

Generate residue-level labels
  -> label shard .pt files aligned to embedding keys/shards

Train FFNN (DDP)
  -> checkpoints + metrics artifacts

Predict epitopes
  -> output FASTA with binary residue mask predictions

Optional: Optuna hyperparameter tuning
  -> study storage + trial CSV + best-model artifacts
```

## Project Scope

This repository supports both:
- local development and validation of pipeline logic, and
- production-scale HPC execution for embedding generation, training, and tuning.

## Prerequisites

Software:
- At least Python `3.12` (required by project configuration and CI).
- `pip` and virtual environments (`venv`) or `conda`.
- `git` for cloning and contribution workflows.

Platform notes:
- Local development can run on CPU.
- GPU is highly recommended for embedding generation and required for practical training/tuning runtimes.
- HPC scripts in `scripts/hpc/` assume a SLURM-style environment and module-based setup (for example, `anaconda3` and `cuda`).

## Setup

### Option A: Conda Environment (recommended)

Local CPU-oriented environment:

```bash
conda env create -f envs/environment.local.yml
conda activate pepseqpred
pip install -e .[dev]
```

HPC/GPU-oriented environment:

```bash
conda env create -f envs/environment.hpc.yml
conda activate pepseqpred
pip install -e .[dev]
```

### Option B: Pip + Virtual Environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

CI-equivalent install shortcut:

```bash
pip install -r requirements.txt
```

## Verify Setup

Confirm package and CLI entrypoints:

```bash
python -c "import pepseqpred; print('pepseqpred import ok')"
pepseqpred-preprocess --help
pepseqpred-esm --help
pepseqpred-train-ffnn --help
pepseqpred-predict --help
```

Run required preflight checks before any development or pipeline usage:

```bash
ruff check .
pytest -m "unit or integration or e2e"
```

This repository expects all of the checks above to pass before you start development work or run pipeline stages.

## Pipeline Stages and Hardware Expectations

Run the main pipeline in this order:
1. Preprocess metadata/reactivity data.
2. Generate ESM-2 embeddings.
3. Generate residue-level labels.
4. Train FFNN model.
5. Predict epitopes on new FASTA input.

Optional branch:
1. Run Optuna tuning after label generation instead of fixed-parameter FFNN training.

### Hardware Matrix (default `scripts/hpc/` settings)

| Stage | Hardware target | Default SLURM request in repo scripts |
| --- | --- | --- |
| Preprocess | CPU only | Local shell helper (no fixed `#SBATCH` resources) |
| Embeddings | GPU | `a100` (1 GPU), `2` CPU/GPU, `8G`/GPU, `01:00:00`, array `0-3` |
| Labels | CPU only | `1` CPU, `16G` RAM, `01:00:00` |
| Train FFNN | Multi-GPU | `4x a100`, `20` CPU, `256G` RAM, `12:00:00` |
| Train FFNN Optuna | Multi-GPU | `4x a100`, `20` CPU, `448G` RAM, `48:00:00` |
| Predict | GPU | `a100` (1 GPU), `4` CPU, `32G` RAM, `00:30:00` |

These are baseline defaults from the current HPC scripts, not strict requirements for every dataset size. You will need to increase or decrease SLURM requests depending on the hardware available to you.

### Stage 1: Preprocess Metadata and Reactivity Inputs

Purpose:
- Merge metadata with z-score reactivity data and generate training-ready labels at preprocessing stage.

Required inputs:
- Metadata TSV (for example, PV1 metadata).
- Reactivity/z-score TSV.

Local CLI example:

```bash
pepseqpred-preprocess data/meta.tsv data/zscores.tsv --save
```

Expected outputs:
- A generated TSV in the working directory, named like `input_data_<is_epi_z>_<is_epi_min_subs>_<not_epi_z>_<not_epi_max_subs|all>.tsv`.

Expected hardware:
- CPU only; lightweight compared with downstream stages.

### Stage 2: Generate ESM-2 Embeddings

Purpose:
- Convert protein FASTA sequences into per-residue ESM-2 embeddings.

Required inputs:
- FASTA file.
- Metadata file when using `id-family` embedding keys (default mode).

Local CLI example:

```bash
pepseqpred-esm \
  --fasta-file data/targets.fasta \
  --metadata-file data/targets.metadata \
  --embedding-key-mode id-family \
  --key-delimiter - \
  --model-name esm2_t33_650M_UR50D \
  --max-tokens 1022 \
  --batch-size 8 \
  --out-dir localdata/esm2_run
```

HPC script example:

```bash
sbatch --export=ALL,IN_FASTA=/scratch/$USER/data/targets.fasta scripts/hpc/generateembeddings.sh
```

Expected outputs:
- Per-sequence `.pt` embeddings under `<out_dir>/artifacts/pts/` (or shard subfolders in sharded mode).
- Embedding index CSV under `<out_dir>/artifacts/`.

Expected hardware:
- GPU strongly recommended; this is the first expensive stage.

### Stage 3: Generate Residue-Level Label Shards

Purpose:
- Build dense residue labels aligned to generated embedding keys/shards.

Required inputs:
- Preprocessed metadata TSV.
- Embedding directory (or shard directories).

Local CLI example:

```bash
pepseqpred-labels \
  data/input_data_20_4_10_all.tsv \
  localdata/labels/labels_shard_000.pt \
  --emb-dir localdata/esm2_run/artifacts/pts/shard_000 \
  --restrict-to-embeddings \
  --calc-pos-weight \
  --embedding-key-delim -
```

HPC script examples:

```bash
sbatch --array=0-3 scripts/hpc/generatelabels.sh data/input_data_20_4_10_all.tsv /scratch/$USER/labels /scratch/$USER/esm2/artifacts/pts
```

Expected outputs:
- Label shard files such as `labels_shard_000.pt`.
- Optional positive class weight in the saved payload when `--calc-pos-weight` is used.

Expected hardware:
- CPU only; moderate memory.

### Stage 4: Train FFNN

Purpose:
- Train PepSeqPred FFNN on embedding shards and label shards.

Required inputs:
- One or more embedding shard directories.
- One or more label shard `.pt` files aligned to those embeddings.

Local smoke-test CLI example (small subset):

```bash
pepseqpred-train-ffnn \
  --embedding-dirs localdata/esm2_run/artifacts/pts/shard_000 \
  --label-shards localdata/labels/labels_shard_000.pt \
  --epochs 1 \
  --subset 100 \
  --save-path localdata/models/ffnn_smoke \
  --results-csv localdata/models/ffnn_smoke/runs.csv
```

HPC script example:

```bash
sbatch scripts/hpc/trainffnn.sh \
  /scratch/$USER/esm2/artifacts/pts/shard_000 /scratch/$USER/esm2/artifacts/pts/shard_001 /scratch/$USER/esm2/artifacts/pts/shard_002 /scratch/$USER/esm2/artifacts/pts/shard_003 \
  -- \
  /scratch/$USER/labels/labels_shard_000.pt /scratch/$USER/labels/labels_shard_001.pt /scratch/$USER/labels/labels_shard_002.pt /scratch/$USER/labels/labels_shard_003.pt
```

Expected outputs:
- Run directories under `--save-path` containing checkpoints (for example `fully_connected.pt`).
- Multi-run CSV summary (default `multi_run_results.csv`, or `--results-csv` path).
- Aggregated `multi_run_summary.json`.

Expected hardware:
- Practical training is multi-GPU/HPC-oriented.

### Stage 5 (Optional): Train FFNN with Optuna

Purpose:
- Run distributed hyperparameter optimization over FFNN architecture/training ranges.

Required inputs:
- Same embedding shard directories and label shard files used by training.

HPC script example:

```bash
sbatch scripts/hpc/trainffnnoptuna.sh \
  /scratch/$USER/esm2/artifacts/pts/shard_000 /scratch/$USER/esm2/artifacts/pts/shard_001 /scratch/$USER/esm2/artifacts/pts/shard_002 /scratch/$USER/esm2/artifacts/pts/shard_003 \
  -- \
  /scratch/$USER/labels/labels_shard_000.pt /scratch/$USER/labels/labels_shard_001.pt /scratch/$USER/labels/labels_shard_002.pt /scratch/$USER/labels/labels_shard_003.pt
```

Expected outputs:
- Trial metrics CSV (`--csv-path`).
- Optuna storage DB (`--storage`), SQLite on scratch.
- Trial checkpoint directories under `--save-path`.
- Best-trial metadata JSON under `--save-path`.

Expected hardware:
- Most expensive stage; budget multi-GPU runtime from several hours to days depending on `--n-trials`.

### Stage 6: Predict Residue-Level Epitopes

Purpose:
- Apply trained checkpoint to new FASTA input and emit binary residue masks.

Required inputs:
- Trained checkpoint `.pt`.
- Input FASTA file.

Local CLI example:

```bash
pepseqpred-predict \
  localdata/models/ffnn_smoke/run_001_split_11_train_101/fully_connected.pt \
  data/inference_targets.fasta \
  --output-fasta localdata/predictions/predictions.fasta \
  --model-name esm2_t33_650M_UR50D \
  --max-tokens 1022
```

HPC script example:

```bash
sbatch scripts/hpc/predictepitope.sh /scratch/$USER/models/ffnn_v1/run_001_split_11_train_101/fully_connected.pt /scratch/$USER/data/inference_targets.fasta /scratch/$USER/predictions/predictions.fasta
```

Expected outputs:
- Output FASTA with binary residue-level mask predictions.
- Prediction logs (console and optional log directory).

Expected hardware:
- Single GPU is recommended for throughput; CPU inference is possible but much slower.

### Stage Compatibility Notes

- Keep embedding key scheme consistent (`id` vs `id-family`) between embedding and label generation.
- Keep shard alignment explicit: embedding shard directories should map cleanly to label shard files.
- Use local smoke settings (`--subset`, low epochs, single shard) before submitting expensive HPC jobs.

## Reproducibility and Output Conventions

- Use a new output root per run/study to avoid accidental overwrite of prior artifacts.
- Keep preprocessing outputs, embeddings, labels, checkpoints, and predictions in separate subdirectories.
- Keep `split-seeds`, `train-seeds`, and core hyperparameters fixed when comparing experiments.
- Keep `split-type` and embedding key scheme (`id` vs `id-family`) unchanged within a single experiment.
- Do not mix artifacts from different preprocessing thresholds into one training run.

Suggested layout:

```text
localdata/
  runs/
    <run_name_or_date>/
      preprocess/
      embeddings/
      labels/
      models/
      predictions/
      logs/
```

## Troubleshooting

Common issues and fixes:

- `Metadata file is required for --embedding-key-mode='id-family'`:
  Provide `--metadata-file`, or use `--embedding-key-mode id` if that is your intended key scheme.
- `--embedding-key-delim must be '' or '-'`:
  Use `-` for `ID-family.pt` naming and empty delimiter for `ID.pt` naming.
- `No .pt files found in <emb_dir>` during label generation:
  Verify Stage 2 completed successfully and `--emb-dir` points to the directory that directly contains embedding `.pt` files.
- Label shard missing `class_stats` when training:
  Rebuild labels with `--calc-pos-weight`, or pass `--pos-weight` explicitly to training.
- `--hidden-sizes and --dropouts must be the same length`:
  Ensure both CSV lists have one value per hidden layer.
- Prediction threshold errors (`(0.0, 1.0)` required):
  Set `--threshold` strictly between `0` and `1`, or omit it to use checkpoint/default behavior.
- DDP or multi-GPU runs stall/hang:
  Confirm the requested GPU count matches `torchrun --nproc_per_node`, and validate with a small single-shard smoke test first.
- CUDA OOM:
  Reduce embedding `--batch-size`, reduce training `--batch-size`, or lower trial/search scope for Optuna.

## Pre-Run Checklist

Before starting a real run:

- Environment is activated and `pip install -e .[dev]` completed.
- Required preflight checks pass: `ruff check .` and `pytest -m "unit or integration or e2e"`.
- Stage input files exist and are from the intended experiment branch.
- Output paths are unique for this run and will not overwrite prior artifacts.
- Embedding key scheme/delimiter and split configuration are consistent across stages.
