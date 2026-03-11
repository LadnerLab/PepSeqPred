# AGENTS.md

## Project overview

PepSeqPred is a residue level epitope prediction pipeline for peptide and protein workflows.

This repository is organized around:
- `src/pepseqpred/apps/` for user facing CLIs
- `src/pepseqpred/core/` for reusable pipeline logic
- `scripts/hpc/` for SLURM batch execution on GPU clusters
- `tests/` for `unit`, `integration`, and `e2e` coverage
- `envs/`, `localdata/`, `notebooks/`, and `dist/` as supporting project directories

Primary goals when working in this repo:
- preserve scientific reproducibility
- keep training and evaluation behavior stable unless explicitly asked to change it
- prefer minimal, targeted edits
- avoid expensive or risky compute by default
- maintain compatibility with existing CLIs, scripts, and downstream outputs

## Repository structure

### Application entrypoints

The main CLIs are:
- `pepseqpred-esm`
- `pepseqpred-labels`
- `pepseqpred-predict`
- `pepseqpred-preprocess`
- `pepseqpred-train-ffnn`
- `pepseqpred-train-ffnn-optuna`

These map to files in `src/pepseqpred/apps/`.

### Core package layout

Important subpackages under `src/pepseqpred/core/`:
- `data/` for dataset loading
- `embeddings/` for ESM based embedding generation
- `io/` for logging and file writing helpers
- `labels/` for label generation logic
- `models/` for model definitions
- `predict/` for inference
- `preprocess/` for preprocessing workflows
- `train/` for DDP, splitting, metrics, thresholds, trainer logic, seeds, and class weighting

### HPC scripts

Batch scripts live in `scripts/hpc/`. These are part of the intended workflow, especially for:
- embedding generation
- label generation
- preprocessing
- prediction
- FFNN training
- FFNN Optuna tuning

Treat these scripts as first class project interfaces, not throwaway helpers.

## General working rules for any agents (Codex, Claude Code, etc.)

Before editing:
- inspect the relevant files first
- understand the existing CLI and core flow before proposing changes
- prefer the smallest possible diff
- do not rename modules, scripts, CLI flags, or output files unless the task requires it
- do not introduce dependencies unless clearly justified

While editing:
- follow the existing package structure
- preserve current naming conventions and CLI semantics
- preserve public script behavior unless the user explicitly asks for a behavior change
- keep functions explicit and readable
- add or update docstrings when behavior changes
- avoid unrelated refactors or cosmetic churn

After editing:
- run the smallest relevant validation first
- report exactly what changed
- note anything you could not validate

## Reproducibility and experiment safety

This is research code. Changes can silently invalidate experiments.

Always preserve:
- deterministic seed handling
- train, validation, and test split semantics
- masking behavior for uncertain labels
- metric calculation behavior
- checkpoint and result artifact formats, unless explicitly changing schema
- per run and per trial traceability

Do not:
- change default seeds casually
- change label meaning or preprocessing behavior without documenting it
- mix outputs from different experiments into ambiguous files
- overwrite prior results when a new output path is safer

If a change affects training or evaluation, explicitly check for:
- data leakage
- split leakage
- rank specific side effects
- output collisions across repeated runs or trials

## Distributed training and HPC guardrails

PepSeqPred training is designed around multi-GPU DistributedDataParallel and SLURM-based execution.

When touching training or Optuna code:
- assume jobs may run on at least 4 GPUs through SLURM
- be careful with `torch.distributed` collectives, barriers, and rank scoped logic
- ensure shared artifacts are only written by the correct rank
- avoid introducing deadlocks
- do not make changes that multiply compute cost unexpectedly
- preserve scheduler-friendly behavior

Prefer:
- local dry runs
- tiny subsets
- reduced epoch smoke tests
- single rank validation where possible before full scale recommendations

Do not assume:
- local laptop training is practical
- interactive GPU access exists
- paths outside repo root are portable unless already established by project scripts
- using `sbatch` locally will work, it will fail for local development

## Data and artifact handling

Never modify raw or source data in place.

Prefer:
- writing derived outputs to new paths
- append safe logs and result files
- explicit artifact names that encode experiment identity

Be careful with:
- checkpoint directories
- CSV summaries
- Optuna trial outputs
- per-rank logging
- temporary files on shared scratch storage

If a schema or file format must change:
- make the change explicit
- update readers and writers together
- document the migration clearly

## Validation expectations

Use the repo’s configured tooling where practical.

Default validation order:
1. `ruff check .`
2. targeted `pytest` invocation for affected tests
3. broader `pytest` if the change is cross-cutting
4. only then consider heavier runtime checks

Important:
- do not run long HPC style training jobs unless explicitly asked
- do not present expensive end-to-end training as routine validation
- for training code, prefer smoke tests over full experiments

If validation is incomplete:
- say what was not run
- say why
- identify the main remaining risks

## Commands

Common commands:
- install package: `pip install -e .`
- install dev tools: `pip install -e .[dev]`
- run tests: `pytest`
- lint: `ruff check .`
- format: `ruff format .`

Available CLIs:
- `pepseqpred-esm`
- `pepseqpred-labels`
- `pepseqpred-predict`
- `pepseqpred-preprocess`
- `pepseqpred-train-ffnn`
- `pepseqpred-train-ffnn-optuna`

## Testing guidance

The repo has:
- `tests/unit/`
- `tests/integration/`
- `tests/e2e/`

Prefer:
- unit tests for isolated logic changes
- integration tests for CLI to core interactions
- e2e only when a full pipeline boundary changed

Do not expand test scope unnecessarily if a small targeted test is enough.

## Documentation expectations

When behavior changes, update the relevant:
- docstrings
- CLI help text
- comments near tricky distributed logic
- any usage examples affected by the change

Note:
- the current root `README.md` is minimal, so do not assume broader user documentation already exists
- if you add a major new workflow, include enough inline guidance for future contributors

## What not to change without explicit approval

Do not, unless clearly requested:
- redesign package structure
- replace DDP or SLURM workflows
- alter default experiment semantics
- change model architecture defaults broadly
- change preprocessing formulas or label logic
- rewrite output schemas
- remove test categories
- introduce large framework migrations

## Preferred task workflow

For most tasks:
1. inspect relevant app, core, test, and script files
2. identify the smallest safe fix
3. implement minimally
4. run focused validation
5. summarize edits, validation, and remaining risks

## Directory specific notes

### `src/pepseqpred/apps/`
- preserve CLI compatibility
- do not break argument names or defaults without explicit instruction
- keep orchestration logic thin when possible

### `src/pepseqpred/core/train/`
- highest risk area
- be conservative with splits, seeds, metrics, thresholds, and DDP behavior
- verify rank aware writes and collective calls carefully

### `scripts/hpc/`
- preserve SLURM semantics
- avoid hard coding user specific assumptions unless already part of script conventions
- comment any scheduler related changes clearly

### `tests/`
- add targeted coverage for bug fixes
- do not rewrite unrelated fixtures or tests just for style