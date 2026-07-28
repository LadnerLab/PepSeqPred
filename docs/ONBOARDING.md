# PepSeqPred Onboarding

This guide is for future developers and researchers joining PepSeqPred work. It is a first-day orientation, not the full pipeline reference.

Use these docs together:

- [README.md](../README.md): full developer pipeline reference and artifact contracts.
- [README.pypi.md](../README.pypi.md): lightweight inference API quickstart.
- [CONTRIBUTING.md](../CONTRIBUTING.md): branch, issue, commit, and pull request rules.
- [PV1/CWP/BKP training note](pv1_cwp_bkp_merge_split_and_pos_weight.md): multi-source training, split, and positive-weight details.

## First Day Setup

PepSeqPred targets Python 3.12. For local CPU development, prefer the local Conda environment:

```bash
conda env create -f envs/environment.local.yml
conda activate pepseqpred
pip install -e .[dev]
```

For GPU or HPC development, use the HPC environment spec:

```bash
conda env create -f envs/environment.hpc.yml
conda activate pepseqpred
pip install -e .[dev]
```

A plain virtual environment also works for local package work:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

If you are changing the website under `web/`, install its dependencies separately:

```bash
cd web
npm ci
npm run lint
npm run build
```

## Repository Mental Model

The repository is organized around a staged residue-level epitope prediction workflow:

```text
prepare dataset inputs
generate ESM-2 embeddings
build residue labels
train model head
optionally tune with Optuna
predict residue masks
evaluate residue and peptide-level behavior
```

Important directories:

- `src/pepseqpred/apps/`: CLI entrypoints. Keep orchestration thin and preserve CLI names, flags, and defaults unless a change explicitly requires otherwise.
- `src/pepseqpred/core/`: reusable implementation for preprocessing, embeddings, labels, datasets, models, training, prediction, and I/O.
- `src/pepseqpred/api/`: stable Python inference API and bundled pretrained model registry.
- `scripts/hpc/`: SLURM wrappers and HPC/local helper scripts for production-style embedding, labeling, training, tuning, prediction, and evaluation.
- `scripts/tools/`: zipapp build tools and evaluation helpers.
- `tests/`: unit, integration, and end-to-end coverage.
- `envs/`: local CPU and HPC GPU Conda environment specs.
- `data/sample/`: small example input formats.
- `docs/`: deeper notes for specialized workflows.
- `web/`: Next.js results and reproducibility site.

## Development Workflow

Before editing, inspect the relevant CLI, core module, tests, and any HPC wrapper that exposes the behavior. The safest changes in this project are small, targeted, and aligned with existing pipeline boundaries.

Default workflow:

1. Identify the entrypoint that owns the behavior.
2. Trace from CLI argument parsing into the core implementation.
3. Check nearby tests for expected behavior and edge cases.
4. Make the smallest coherent change.
5. Run focused validation first.
6. Review the diff for unrelated churn.

Avoid broad refactors while fixing a narrow issue. Training and evaluation changes are especially sensitive because small semantic changes can silently invalidate experiment comparisons.

## Research Safety

PepSeqPred is research code, so reproducibility matters as much as passing tests.

Preserve these unless the task explicitly changes them:

- seed handling and deterministic run planning
- train, validation, and test split semantics
- `id` versus `id-family` leakage behavior
- uncertain-label masking
- metric calculations and threshold policies
- checkpoint, CSV, JSON, FASTA, and manifest schemas
- rank-scoped writes and distributed synchronization
- output paths that keep experiments traceable

When adding or changing outputs, prefer new explicit paths or names over overwriting previous experiment artifacts. When changing schemas, update both readers and writers and document the migration.

## HPC And Expensive Compute

Treat `scripts/hpc/` as first-class user interfaces. Researchers may depend on these wrappers for production runs.

Key guardrails:

- Do not assume `sbatch` or GPU access exists during local development.
- Do not run long training, Optuna, embedding, or end-to-end HPC jobs unless explicitly asked.
- Prefer smoke tests with tiny subsets and few epochs.
- Preserve SLURM resource behavior unless the task is about scheduler configuration.
- Be careful with DDP collectives, barriers, and rank-specific side effects.
- Ensure shared artifacts are written only by the intended rank.

For model-head training and Optuna, assume jobs may run on multiple GPUs through SLURM. A local single-rank pass is useful, but it does not prove distributed behavior is correct.

## Common Starting Points

Use these locations when beginning a change:

| Task | Start Here |
| --- | --- |
| CLI behavior or flags | `src/pepseqpred/apps/` and matching tests in `tests/unit/apps/` |
| Dataset preparation | `src/pepseqpred/core/preprocess/` and `tests/unit/core/preprocess/` |
| Embedding generation | `src/pepseqpred/core/embeddings/esm2.py` and `scripts/hpc/generateembeddings.sh` |
| Label generation | `src/pepseqpred/core/labels/builder.py` and `tests/unit/core/labels/` |
| Dataset loading/windowing | `src/pepseqpred/core/data/proteindataset.py` |
| Model definitions | `src/pepseqpred/core/models/` |
| Training, splits, thresholds, metrics, DDP | `src/pepseqpred/core/train/` |
| Prediction and evaluation | `src/pepseqpred/core/predict/` and prediction/eval CLIs |
| Public Python inference API | `src/pepseqpred/api/` and `tests/unit/api/` |
| HPC execution | `scripts/hpc/` |
| Website changes | `web/` |

When touching an HPC-facing behavior, check the Python CLI and its corresponding shell wrapper together.

## Validation Recipes

Start with the smallest relevant command, then broaden only when the change warrants it.

General Python checks:

```bash
ruff check .
pytest tests/unit
pytest tests/integration
pytest tests/e2e
```

Targeted examples:

```bash
pytest --no-cov tests/unit/core/train
pytest --no-cov tests/unit/apps/test_prediction_cli_config.py
pytest --no-cov tests/integration/test_train_clis_inprocess.py
```

These targeted commands disable coverage because the repository applies a global coverage threshold to every pytest invocation. Use the full validation command from [CONTRIBUTING.md](../CONTRIBUTING.md) when coverage enforcement is needed.

For web changes:

```bash
cd web
npm ci
npm run lint
npm run build
```

For documentation-only changes, code tests are usually unnecessary. Still verify links, paths, command names, and that the documentation does not conflict with the current README or contribution guide.

## Before Opening A Pull Request

Follow [CONTRIBUTING.md](../CONTRIBUTING.md). In short:

- work on a branch, not directly on `main`
- create or confirm an issue for the work, and link it from the PR
- use concise branch and commit names
- include a short PR summary
- include exact verification commands and outcomes
- mention any validation that was intentionally not run

For experiment-affecting changes, also explain the reproducibility impact: split behavior, seed behavior, artifact format, output paths, and any remaining distributed-training risk.
