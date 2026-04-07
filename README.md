<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="PepSeqPred_logo_white.png">
    <source media="(prefers-color-scheme: light)" srcset="PepSeqPred_logo_black.png">
    <img src="PepSeqPred_logo_black.png" alt="PepSeqPred logo" width="320">
  </picture>
</p>

# Overview

PepSeqPred is a residue-level epitope prediction pipeline for protein workflows.  
It converts upstream assay and sequence data into training-ready artifacts, trains feed-forward neural network models on ESM-2 embeddings, produces binary residue masks for downstream inference, and supports post-training evaluation workflows on held-out datasets.

## Quickstart

PepSeqPred can be installed via:
```bash
pip install pepseqpred
```

This API allows you to use any of the pretrained models in your own code, or load your own model(s) for downstream predictions.

Example usage:
```python
from pepseqpred import load_predictor

predictor = load_predictor("path/to/model.pt", device="cuda")
result = predictor.predict_sequence("ACDEFGHIKLMNP")

print(result.binary_mask, result.n_epitopes)
```

**Disclaimer:** The API is meant to be a simplified version of the overall codebase where you can build epitope inference into your existing/new workflows. To have more control over exactly how PepSeqPred works, feel free to fork this repository and ensure you adhere to the [GPL-3.0 license](LICENSE).

## About

PepSeqPred is designed for research workflows where you need to:
- map peptide-level signals to residue-level supervision,
- train reproducible epitope prediction models,
- run large jobs on HPCs with DistributedDataParallel (DDP),
- generate per-residue binary epitope predictions to develop new peptide libraries,
- evaluate trained checkpoints or ensemble manifests with residue-level and peptide-level metrics.

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
- FFNN evaluation summary JSON and optional per-peptide comparison CSV/JSON artifacts.

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

Evaluate trained FFNN
  -> residue-level metrics JSON and optional peptide-level comparison outputs

Optional: Optuna hyperparameter tuning
  -> study storage + trial CSV + best-model artifacts
```

## Project Scope

This repository supports both:
- local development and validation of pipeline logic, and
- production-scale HPC execution for embedding generation, training, and tuning.

## Maintainer Contact

- Use GitHub issues for normal development questions, bug reports, and feature requests.
- Use email for private or sensitive matters that should not be posted publicly.
- Maintainer contact: [Jeffrey Hoelzel](mailto:jmh2338@nau.edu) or [Jason Ladner](mailto:jason.ladner@nau.edu).

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
pepseqpred-eval-ffnn --help
```

Run required preflight checks before any development or pipeline usage:

```bash
ruff check .
pytest -m "unit or integration or e2e"
```

This repository expects all of the checks above to pass before you start development work or run pipeline stages.

## Build and Transfer HPC Runtime Artifacts (`.pyz` + SLURM Scripts)

The HPC shell scripts in `scripts/hpc/` execute zipapp files such as `esm.pyz`, `labels.pyz`, `train_ffnn.pyz`, `predict.pyz`, and `evaluate_ffnn.pyz`.

Build only the app(s) you need for your current stage, then copy those `.pyz` files plus the matching SLURM script(s) to HPC.

### 1. Build `.pyz` artifacts locally

List available zipapp targets:

```bash
python scripts/tools/buildpyz.py --list
```

Build one runtime app (recommended default):

```bash
python scripts/tools/buildpyz.py <app_name>
```

Examples:

```bash
python scripts/tools/buildpyz.py esm
python scripts/tools/buildpyz.py train_ffnn
```

Optional: build all apps:

```bash
python scripts/tools/buildpyz.py all
```

By default this writes artifacts to `dist/` as:
- versioned files: `<name>_<gitrev>.pyz`
- convenience copies: `<name>_latest.pyz`

### 2. Transfer required `.pyz` file(s) and shell script(s) to HPC

Each HPC script expects a plain `.pyz` filename in the same working directory:
- `generateembeddings.sh` -> `esm.pyz`
- `generatelabels.sh` -> `labels.pyz`
- `trainffnn.sh` -> `train_ffnn.pyz`
- `trainffnnoptuna.sh` -> `train_ffnn_optuna.pyz`
- `predictepitope.sh` -> `predict.pyz`
- `evaluateffnn.sh` -> `evaluate_ffnn.pyz` (or fallback module import)
- `preprocessdata.sh` -> `preprocess.pyz` (optional)

For the Cocci evaluation workflow in `evaluateffnn.sh`, also transfer:
- `scripts/tools/cocci_eval_pipeline.py` (called directly by the shell script for `prepare` and `compare` stages)

Example: transfer only embedding stage artifacts:

```bash
scp dist/esm_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/esm.pyz
scp scripts/hpc/generateembeddings.sh <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/
```

Example: transfer multiple stage artifacts:

```bash
scp dist/labels_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/labels.pyz
scp dist/train_ffnn_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/train_ffnn.pyz
scp scripts/hpc/generatelabels.sh scripts/hpc/trainffnn.sh <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/
```

Example: transfer evaluation artifacts:

```bash
scp dist/esm_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/esm.pyz
scp dist/labels_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/labels.pyz
scp dist/predict_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/predict.pyz
scp dist/evaluate_ffnn_latest.pyz <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/evaluate_ffnn.pyz
scp scripts/hpc/evaluateffnn.sh scripts/tools/cocci_eval_pipeline.py <user>@<cluster-host>:/home/<user>/pepseqpred_hpc/
```

### 3. Prepare on cluster and smoke-check artifacts

```bash
cd /home/<user>/pepseqpred_hpc
chmod +x *.sh
ls -lh *.pyz *.sh

# Run help checks for the app(s) you uploaded
python3 esm.pyz --help
```

Run SLURM jobs from this directory so relative `.pyz` filenames resolve correctly.

### 4. Update cycle after code changes

Any CLI code change requires rebuilding and redeploying corresponding `.pyz` files:
1. Re-run `python scripts/tools/buildpyz.py <app_name>`.
2. Re-transfer updated `dist/<app>_latest.pyz` to HPC as `<app>.pyz`.
3. Re-run jobs using the updated artifact.

## Pipeline Stages and Hardware Expectations

Run the main pipeline in this order:
1. Preprocess metadata/reactivity data.
2. Generate ESM-2 embeddings.
3. Generate residue-level labels.
4. Train FFNN model.
5. Predict epitopes on new FASTA input.
6. Evaluate FFNN outputs on labeled embeddings and/or Cocci reduced subsets.

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
| Evaluate FFNN | GPU recommended | `a100` (1 GPU), `8` CPU, `64G` RAM, `08:00:00` |

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

### Stage 7: Evaluate FFNN on Labeled Embeddings / Cocci Subsets

Purpose:
- Evaluate trained checkpoints or ensemble manifests on residue-level labels.
- Optionally run Cocci-specific reduced-dataset preparation and peptide-level 1-count comparison.

Required inputs (minimum residue-level eval):
- Trained checkpoint `.pt` or ensemble manifest `.json`.
- One or more embedding directories.
- One or more label shard `.pt` files.

Local CLI example:

```bash
pepseqpred-eval-ffnn \
  localdata/models/ffnn_smoke/run_001_split_11_train_101/fully_connected.pt \
  --embedding-dirs localdata/esm2_run/artifacts/pts/shard_000 \
  --label-shards localdata/labels/labels_shard_000.pt \
  --output-json localdata/eval/ffnn_eval_summary.json \
  --batch-size 64 \
  --num-workers 0
```

HPC script example (Cocci workflow):

```bash
sbatch --export=ALL,EVAL_MODE=nonreactive,SKIP_IF_EXISTS=1,RUN_PREP=1,RUN_EMBED=1,RUN_LABELS=1,RUN_PREDICT=1,RUN_EVAL=1,RUN_COMPARE=1 \
  scripts/hpc/evaluateffnn.sh \
  /scratch/$USER/models/phaseA/ffnn_ens_1.0_xxxxxxxx/ensemble_manifest.json \
  /scratch/$USER/evals/cocci_eval/nonreactive
```

Seeded sweep example (two flagship manifests, set indices 1..10):

```bash
HPC_DIR=/home/$USER/test \
SHARED=/scratch/$USER/evals/cocci_eval/combined \
OUT_BASE=/scratch/$USER/evals/cocci_eval/seeded_runs \
DATA_DIR=/scratch/$USER/data/CWP \
./evalffnnsweep.sh \
  /scratch/$USER/models/phaseB/flagship1/ensemble_manifest.json \
  /scratch/$USER/models/phaseB/flagship2/ensemble_manifest.json
```

Verify seeded run set-index alignment (`set_XX` directory must match evaluated ensemble set index):

```bash
python scripts/tools/verify_seeded_eval_set_indices.py \
  --base-dir localdata/evals/cocci_eval/seeded_runs \
  --models flagship1,flagship2 \
  --set-start 1 \
  --set-end 10
```

Expected outputs:
- `prepared/eval_metadata.tsv`, `prepared/eval_proteins.fasta`, `prepared/prepare_summary.json`.
- `embeddings/artifacts/eval_embedding_index.csv`.
- `labels/labels_eval.pt`.
- `prediction/predictions.fasta`.
- `evaluation/ffnn_eval_summary.json`.
- `peptide_compare/peptide_comparison.csv` and `peptide_compare/peptide_comparison_summary.json`.

### Stage Compatibility Notes

- Keep embedding key scheme consistent (`id` vs `id-family`) between embedding and label generation.
- Keep shard alignment explicit: embedding shard directories should map cleanly to label shard files.
- Use local smoke settings (`--subset`, low epochs, single shard) before submitting expensive HPC jobs.
- For `evaluateffnn.sh`, ensure `esm.pyz`, `labels.pyz`, `predict.pyz`, `evaluate_ffnn.pyz`, and `cocci_eval_pipeline.py` are present in the submission working directory.

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
- Plot generation produces no files when `--plot-dir` is set:
  Install `matplotlib` in the runtime environment (for HPC: ensure `envs/environment.hpc.yml` is applied with `matplotlib` present).
- `python: can't open file ... esm.pyz` (or `labels.pyz` / `predict.pyz`):
  Transfer missing `.pyz` files to the same working directory as the HPC shell script, or run from an installed package environment using CLI entrypoints.
- `labels_eval.pt` or `predictions.fasta` not found during evaluation:
  Upstream stage failed or was skipped; rerun with stage flags set (`RUN_EMBED=1`, `RUN_LABELS=1`, `RUN_PREDICT=1`) or disable dependent downstream stages.
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
