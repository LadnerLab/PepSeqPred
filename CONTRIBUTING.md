# Contributing to PepSeqPred

This document defines required contribution workflow, naming conventions, and pull request expectations for this repository.

## Core Rules

- Do not develop directly on `main`.
- All changes must be made on a different branch and merged via pull request.
- Branch, issue, and commit names must follow the conventions below.
- Keep titles and descriptions short, clear, and specific.

## Required Contribution Workflow

1. Create or confirm an issue for the work.
2. Create a branch from the latest `main` using the branch naming rules.
3. Implement the change and add/update relevant tests.
4. Run required local checks.
5. Open a pull request into `main` with required summary and verification details.

## Branch Naming Conventions

Use lowercase and hyphen-separated descriptions.

Accepted patterns:
- `feat/short-description`
- `fix/short-description`
- `docs/short-description`
- `chore/short-description`
- `test/short-description`
- `refactor/short-description`

You can also be extra specific by adding the issue number associated with your code as seen below.

Examples:
- `feat/add-sharded-embedding-index-logging`
- `fix/issue-42-threshold-range-validation`
- `docs/update-readme-pipeline-section`

## Issue Naming and Content

Issue title format:
- `<type>: short description`

Examples:
- `bug: label shard mismatch across embedding keys`
- `docs: add hpc setup troubleshooting`
- `chore: tighten local test gating in README`

Issue body requirements:
- `Summary`: a short statement of the problem or request.
- `Done when`: acceptance criteria, if applicable.

## Commit Message Conventions

Commit title format:
- `<type>: short description`

Examples:
- `bug: fix id-family key validation in labels builder`
- `chore: remove unused import from prediction cli`
- `docs: add contributing workflow and naming rules`

Commit guidance:
- Keep the first line concise and specific.
- Keep one logical change per commit where possible.

## Pull Request Requirements

All pull requests to `main` must include:
- A concise summary of what changed.
- Linked issue(s) (for example, `Fixes #42`).
- A concise "How to verify" section with exact commands.
- Any new or updated unit, integration, or e2e tests needed to verify behavior changes.

PRs should not include changed unrelated to the issue unless it's minor, please use your own discretion.

## Verification Expectations Before PR

Run these checks locally before opening a PR:

```bash
ruff check .
pytest -m "unit or integration or e2e"
```

If behavior changed, include targeted test commands in the PR verification section, along with expected outcomes.

## PR Checklist

- [ ] Branch name follows convention.
- [ ] Issue title/body follow convention (`Summary` and `Done when` included when applicable).
- [ ] Commit messages follow `<type>: short description`.
- [ ] No development occurred directly on `main`.
- [ ] PR includes concise summary and reproducible verification steps.
- [ ] Relevant unit/integration/e2e tests were added or updated.

## Maintainer Support and Escalation

- Use GitHub issues for normal development questions, bug reports, and feature requests.
- Use email for private or sensitive matters that should not be posted publicly.
- Maintainer contact: [Jeffrey Hoelzel](mailto:jmh2338@nau.edu) or [Jason Ladner](mailto:jason.ladner@nau.edu).
