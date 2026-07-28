# Contributing to PepSeqPred

This document defines the required contribution workflow, naming conventions, verification expectations, and pull request standards for PepSeqPred.

## Core Rules

- Do not develop directly on `main`.
- All changes must be made on a separate branch and merged via pull request.
- Branch, issue, and commit names must follow the conventions below.
- Keep titles and descriptions short, clear, and specific.
- Prefer small, targeted changes that preserve pipeline reproducibility.

## Required Contribution Workflow

1. Create or confirm an issue for the work.
2. Create a branch from the latest `main` using the branch naming rules.
3. Set up a local development environment.
4. Implement the change and add/update relevant tests or documentation.
5. Run the smallest relevant local checks first, then broaden as needed.
6. Open a pull request into `main` with required summary and verification details.

## Development Setup

PepSeqPred targets Python 3.12.

Recommended editable install:

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

Conda environment options:

```bash
conda env create -f envs/environment.local.yml
conda activate pepseqpred
pip install -e .[dev]
```

For GPU/HPC-oriented development, use `envs/environment.hpc.yml` instead of `envs/environment.local.yml`.

Windows PowerShell virtual environment setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

## Branch Naming Conventions

Use lowercase and hyphen-separated descriptions.

Accepted patterns:
- `feat/short-description`
- `fix/short-description`
- `docs/short-description`
- `chore/short-description`
- `test/short-description`
- `refactor/short-description`

You can also be extra specific by adding the issue number associated with your code.

Examples:
- `feat/add-sharded-embedding-index-logging`
- `fix/issue-42-threshold-range-validation`
- `docs/update-readme-pipeline-section`

## Issue Naming and Content

Issue title format:
- `<type>: short description`

Use one of: `feat`, `fix`, `docs`, `chore`, `test`, or `refactor`.

Examples:
- `fix: label shard mismatch across embedding keys`
- `docs: add hpc setup troubleshooting`
- `chore: tighten local test gating in README`

Issue body requirements:
- `Summary`: a short statement of the problem or request.
- `Done when`: acceptance criteria, if applicable.

## Commit Message Conventions

Commit title format:
- `<type>: short description`

Use one of: `feat`, `fix`, `docs`, `chore`, `test`, or `refactor`.

Examples:
- `fix: id-family key validation in labels builder`
- `chore: remove unused import from prediction cli`
- `docs: add contributing workflow and naming rules`

Commit guidance:
- Keep the first line concise and specific.
- Keep one logical change per commit where possible.
- Do not mix unrelated experiment, code, data, and documentation changes in one commit.

## Pull Request Requirements

All pull requests to `main` must include:
- A concise summary of what changed.
- Linked issue(s), for example `Fixes #42`.
- A concise "How to verify" section with exact commands and outcomes.
- Any new or updated unit, integration, or e2e tests needed to verify behavior changes.
- Any intentionally skipped validation and why it was skipped.

PRs should not include changes unrelated to the issue unless they are minor and clearly explained.

For experiment-affecting changes, also describe impacts on:
- data split behavior
- seed/determinism behavior
- label masking or metric calculations
- checkpoint, CSV, JSON, FASTA, or manifest formats
- output paths and artifact traceability
- distributed training or rank-specific behavior

## Verification Expectations Before PR

Run the relevant checks locally before opening a PR.

Default Python checks:

```bash
ruff check .
pytest -m "unit or integration or e2e"
```

For focused development, start with targeted tests before running broader suites:

```bash
pytest --no-cov tests/unit
pytest --no-cov tests/integration/test_prediction_cli_smoke.py
```

Use `--no-cov` for narrow targeted runs when you only need a fast signal; the repository-level pytest configuration enforces a coverage threshold on normal pytest runs.

For documentation-only changes, code tests are usually unnecessary. Still review links, commands, file paths, and CLI names for accuracy.

For website changes under `web/`, validate separately:

```bash
cd web
npm ci
npm run lint
npm run build
```

Do not run long embedding, training, Optuna, prediction, evaluation, or SLURM/HPC jobs unless the change requires it and the PR explains the cost.

## Testing Scope Guidance

Choose the smallest test category that proves the change:

- `tests/unit/`: isolated logic and edge cases.
- `tests/integration/`: CLI-to-core interactions and smoke tests.
- `tests/e2e/`: full pipeline boundaries such as train-to-predict behavior.

When changing CLI behavior, check the matching file in `src/pepseqpred/apps/` and the relevant tests under `tests/unit/apps/` or `tests/integration/`.

When changing HPC-facing behavior, check both the Python CLI and the corresponding wrapper in `scripts/hpc/`.

## Reproducibility and HPC Guardrails

PepSeqPred is research software. Avoid changes that silently invalidate prior experiments.

Preserve unless the issue explicitly requires a change:
- deterministic seed handling
- train/validation/test split semantics
- `id` versus `id-family` leakage behavior
- uncertain-label masking
- metric and threshold policy behavior
- checkpoint and result artifact schemas
- rank-scoped writes and distributed synchronization

For training, evaluation, Optuna, and SLURM-related changes:
- prefer tiny smoke tests over full experiments
- avoid output collisions across repeated runs or trials
- write shared artifacts only from the intended rank
- document any schema or experiment-semantics migration in the PR

## PR Checklist

- [ ] Branch name follows convention.
- [ ] Issue title/body follow convention (`Summary` and `Done when` included when applicable).
- [ ] Commit messages follow `<type>: short description`.
- [ ] No development occurred directly on `main`.
- [ ] PR includes concise summary and reproducible verification steps.
- [ ] Relevant unit/integration/e2e tests were added or updated.
- [ ] Documentation was updated when behavior, commands, outputs, or workflows changed.
- [ ] Expensive or skipped validation is explicitly noted.

## Release Notes

Package publishing is maintainer-controlled. Tags matching `v*` trigger the release workflow, which runs tests, builds distributions, checks the artifacts, publishes to TestPyPI, and then publishes to PyPI through the configured environments.

Do not create release tags from feature branches or unreviewed changes.

## Maintainer Support and Escalation

- Use GitHub issues for normal development questions, bug reports, and feature requests.
- Use email for private or sensitive matters that should not be posted publicly.
- Maintainer contact: [Jeffrey Hoelzel](mailto:jmh2338@nau.edu) or [Jason Ladner](mailto:jason.ladner@nau.edu).
