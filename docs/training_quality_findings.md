# PepSeqPred Training Quality Findings

This document summarizes the read-only exploratory review of PepSeqPred training, evaluation, label, and HPC code paths for unexpectedly poor multi-pathogen dataset results.

No code edits were made during the investigation. The review focused on likely training-quality failure modes rather than style or cosmetic issues.

## Scope Reviewed

Primary files inspected:

- `src/pepseqpred/apps/train_ffnn_cli.py`
- `src/pepseqpred/apps/train_ffnn_optuna_cli.py`
- `src/pepseqpred/apps/evaluate_ffnn_cli.py`
- `src/pepseqpred/core/models/ffnn.py`
- `src/pepseqpred/core/train/trainer.py`
- `src/pepseqpred/core/train/threshold.py`
- `src/pepseqpred/core/train/split.py`
- `src/pepseqpred/core/train/metrics.py`
- `src/pepseqpred/core/train/weights.py`
- `src/pepseqpred/core/data/proteindataset.py`
- `src/pepseqpred/core/labels/builder.py`
- `src/pepseqpred/core/preprocess/preparedataset.py`
- `scripts/hpc/trainffnn.sh`
- `scripts/hpc/trainffnnoptuna.sh`
- `scripts/hpc/evaluateffnn.sh`
- related tests under `tests/unit`, `tests/integration`, and `tests/e2e`

## Highest-Risk Findings

### 1. Label/objective mismatch for residue-level prediction

Current label generation expands peptide labels across every residue in a peptide alignment window. If a peptide is reactive, every residue in that peptide window becomes a positive residue.

Evidence:

- `src/pepseqpred/core/labels/builder.py`
  - `_build_labels_for_protein` marks `def_mask[start:stop] = True` for definite epitope peptides.
  - The FFNN then trains residue-level BCE on those expanded residue labels.

Why this can hurt:

- A reactive peptide means "the peptide contains an epitope signal", not necessarily "every residue in this peptide is epitope".
- This creates dense false-positive residue labels within positive peptides.
- The model is optimized for residue-wise correctness under noisy residue labels, while downstream use may care about peptide/protein regions or sparse true epitopes.
- The issue can compound in multi-pathogen data if peptide lengths, overlap density, or labeling criteria differ by source.

Planning direction:

- Add diagnostics comparing peptide-level labels to residue-level expansion density.
- Consider a multiple-instance learning objective, peptide-window objective, or soft/weak residue labels.
- If residue labels remain, consider down-weighting positive residues within reactive peptide spans or using boundary-aware smoothing.

### 2. Sparse or zero-valid windows still participate in training

The dataset yields all windows, including windows with no valid labeled residues. The trainer handles zero-valid masks by producing zero loss, but the optimizer still performs a step.

Evidence:

- `src/pepseqpred/core/data/proteindataset.py`
  - `ProteinDataset.__iter__` yields every window from `_iter_windows`.
  - The final mask may be all zero when a window contains only uncertain or padded residues.
- `src/pepseqpred/core/train/trainer.py`
  - `_batch_step` creates zero loss when `mask.sum() == 0`.
  - Training still calls `zero_grad`, `loss.backward()`, and `optimizer.step()`.

Why this can hurt:

- Adam step counters still advance on no-information batches.
- Learning-rate dynamics and optimizer moments can be affected without a meaningful gradient.
- If some pathogens or proteins have many uncertain/unlabeled regions, they can consume training steps without learning signal.

Planning direction:

- Add logging for zero-valid windows and zero-valid batches by split, fold, rank, and source.
- Skip optimizer steps when a batch has zero valid residues.
- Optionally filter zero-valid windows in `ProteinDataset` for training.

### 3. DDP and batch loss weighting are likely biased by local valid-residue counts

Each rank computes a local masked mean BCE loss, then DDP averages gradients across ranks. This is not equivalent to a global valid-residue-weighted loss when ranks or batches have different valid residue counts.

Evidence:

- `src/pepseqpred/core/train/trainer.py`
  - Loss is `(loss_raw * mask).sum() / mask.sum()` per local batch.
  - DDP then averages gradients across ranks equally.
- `src/pepseqpred/apps/train_ffnn_cli.py`
  - IDs are partitioned across ranks by estimated embedding file size, not by valid positive/negative residue count.

Why this can hurt:

- A rank with few valid residues can contribute the same gradient weight as a rank with many valid residues.
- Long proteins, label sparsity, and source/pathogen-specific label density can make this much worse.
- Multi-pathogen data is especially vulnerable if some groups have sparse labels or many uncertain residues.

Planning direction:

- Log valid residue count per batch and rank.
- Consider globally normalized loss using summed numerator and denominator across ranks.
- Alternatively ensure per-rank partitioning balances valid residues and positive residues, not just embedding file size.

### 4. Training uses overlapping windows while evaluation uses full proteins

Training defaults to windowed proteins with overlap, while evaluation defaults to full proteins.

Evidence:

- `src/pepseqpred/core/data/proteindataset.py`
  - `_iter_windows` supports `window_size` and `stride`.
- `scripts/hpc/trainffnn.sh`
  - `WINDOW_SIZE=1000`
  - `STRIDE=900`
- `src/pepseqpred/apps/evaluate_ffnn_cli.py`
  - Evaluation constructs datasets with `window_size=None` and `pad_last_window=False`.

Why this can hurt:

- Overlapped residues are duplicated during training and validation.
- Validation threshold selection is based on windowed validation arrays, but final evaluation is on full proteins.
- Overlap duplicates can overweight boundary regions or long proteins.
- If there are very long multi-pathogen proteins, length and overlap can distort training and metrics.

Planning direction:

- Run an ablation with `--window-size 0` where feasible.
- Run an ablation with non-overlapping windows, for example `stride == window_size`.
- Add metrics that count unique proteins/residues separately from yielded training residues.

### 5. Positive class weight is stale or fold-inappropriate

HPC scripts hard-code a positive class weight. When auto-computed, the weight is computed over all label shards, not the current training split.

Evidence:

- `scripts/hpc/trainffnn.sh`
  - `POS_WEIGHT="${POS_WEIGHT:-13.18999647945325}"`
  - The script always passes `--pos-weight "$POS_WEIGHT"`.
- `scripts/hpc/trainffnnoptuna.sh`
  - Same hard-coded default positive weight.
- `src/pepseqpred/apps/train_ffnn_cli.py`
  - If `--pos-weight` is absent, `pos_weight_from_label_shards(label_shards)` uses all provided label shard totals.
- `docs/pv1_cwp_bkp_merge_split_and_pos_weight.md`
  - Already notes that automatic train-time `pos_weight` uses shard-level totals, not train-only IDs.

Why this can hurt:

- Multi-pathogen data can have very different positive rates than the dataset used to derive the hard-coded value.
- K-fold training may have materially different positive rates per fold.
- Using validation labels to compute the training class weight is also a mild leakage/selection mismatch, even if it is not target leakage in the usual model-fitting sense.

Planning direction:

- Compute `pos_weight` from the current run's training IDs only.
- Log per-run and per-fold positive/negative residue counts before training.
- Stop hard-coding old positive weights in HPC defaults unless explicitly requested.

### 6. Threshold policy is hard-coded and can be overly conservative

Training chooses thresholds by maximizing recall subject to minimum precision 0.25. If that precision is unreachable, it falls back to the best precision threshold. Ensemble prediction then uses majority vote on member-thresholded binary masks.

Evidence:

- `src/pepseqpred/core/train/trainer.py`
  - Calls `find_threshold_max_recall_min_precision(y_true, y_prob, min_precision=0.25)`.
- `src/pepseqpred/core/train/threshold.py`
  - Fallback favors best precision when minimum precision is unreachable.
- `src/pepseqpred/core/predict/inference.py`
  - Ensemble prediction thresholds each member independently and uses majority vote.
  - Ties are effectively negative because `votes_needed = n_members // 2 + 1`.

Why this can hurt:

- A hard minimum precision can push thresholds high and crush recall on difficult or shifted pathogen groups.
- Majority vote over conservative masks further reduces positive calls.
- The reported validation threshold may not transfer to external multi-pathogen evaluation.

Planning direction:

- Make threshold selection configurable.
- Evaluate fixed thresholds, best-F1 thresholds, max-MCC thresholds, and recall-target thresholds.
- For ensembles, compare majority vote with mean-probability thresholding.
- Report threshold status, selected threshold, predicted positive fraction, and recall by pathogen/family.

### 7. Grouped splits prevent family leakage but are not label-stratified

The default split type keeps families/groups intact, which is good for leakage control. However, folds are assigned primarily by group size, not by positive/negative support or source balance.

Evidence:

- `src/pepseqpred/core/train/split.py`
  - `split_ids_grouped` and `build_grouped_kfold_splits` keep groups intact.
  - Grouped k-fold assigns larger groups first to balance fold size.

Why this can hurt:

- Folds can have very different positive rates.
- Some validation folds may contain source/pathogen groups not meaningfully represented in training.
- Metrics can look much worse than expected if the split is actually a difficult cross-family generalization test.

Planning direction:

- Add split reports with per-fold:
  - protein count
  - valid residue count
  - positive residue count
  - negative residue count
  - positive rate
  - source/pathogen/family counts
- Consider grouped stratified splitting where possible.
- Preserve leakage safety, but balance label support more deliberately.

### 8. Model has no local sequence modeling beyond ESM embeddings

The FFNN flattens residues and predicts each residue independently after the ESM embedding. It does not model neighboring residue interactions during classification.

Evidence:

- `src/pepseqpred/core/models/ffnn.py`
  - `PepSeqFFNN.forward` flattens `(B, L, D)` into `(B * L, D)`.
  - Output is reshaped back to `(B, L)`.

Why this can hurt:

- Residue-level epitope signals are often regional.
- The ESM embedding contains context, but the classifier cannot enforce local smoothness or peptide-level consistency.
- Weak peptide-expanded labels may need a model/objective that understands windows rather than independent residues.

Planning direction:

- First fix diagnostics, objective, and weighting before changing architecture.
- Then consider light local context heads, such as 1D convolution, CRF-like smoothing, or pooling over peptide windows.

### 9. Raw protein sequence length is appended as an unnormalized feature

The embedding pipeline appends the original sequence length as an additional scalar feature to every residue embedding.

Evidence:

- `src/pepseqpred/core/embeddings/esm2.py`
  - `append_seq_len` appends `float(seq_len)` as a column.
  - Both embedding generation and prediction embedding paths use this behavior.

Why this can hurt:

- Raw length can be orders of magnitude larger than normalized embedding features.
- It can become a source/pathogen shortcut if sequence length correlates with dataset or label source.
- Multi-pathogen training is particularly susceptible to spurious source-specific features.

Planning direction:

- Run an ablation without the length feature.
- If length is retained, normalize it, bucket it, or pass it through a controlled transform.

## Concrete Bug Found

### Optuna best checkpoint copy path mismatch

Optuna trial training saves trial checkpoints using one filename, but the CLI later tries to copy a different filename.

Evidence:

- `src/pepseqpred/core/train/trainer.py`
  - `fit_optuna` saves `fully_connected_by_score.pt`.
- `src/pepseqpred/apps/train_ffnn_optuna_cli.py`
  - Later tries to copy `best_model_by_score.pt` from the best trial directory.

Impact:

- The root-level `best_model_by_score.pt` may never be written.
- Users may accidentally evaluate stale or missing artifacts after Optuna.

Planning direction:

- Align checkpoint filenames between trainer and Optuna CLI.
- Add a test asserting that the best Optuna checkpoint is copied to the expected root path.

## Configuration Surprises

### `--use-pos-weight` in Optuna appeared misleading

Resolved: `--use-pos-weight` has been removed from `train_ffnn_optuna_cli.py`. Positive weighting is automatic when `--pos-weight` is omitted, and `--pos-weight` remains the manual override.

Impact:

- Users may have thought positive weighting was optional in Optuna when it was effectively always active unless code was changed.

Planning direction:

- No further action needed for this flag.

### Threshold minimum precision is not configurable

The value `0.25` is embedded in trainer evaluation and not exposed as a CLI option.

Impact:

- Experiments cannot easily compare different operating points without code edits.

Planning direction:

- Expose threshold policy and minimum precision as CLI options.
- Save threshold policy metadata in checkpoints and manifests.

## Diagnostics To Add Before Major Changes

Recommended first-pass diagnostics:

1. Split/fold label-balance report.
   - Per split, fold, rank, source, and family.
   - Counts for proteins, windows, valid residues, positive residues, negative residues, positive rate.

2. Window-validity report.
   - Number and fraction of zero-valid windows.
   - Number and fraction of windows with no positives.
   - Same metrics by source/family.

3. Training-loss denominator report.
   - Per batch and per rank valid residue counts.
   - Detect ranks/batches contributing gradients from very different valid counts.

4. Threshold diagnostics.
   - Selected threshold.
   - Threshold status.
   - Predicted positive residue fraction.
   - Precision, recall, F1, MCC, PR AUC at multiple fixed thresholds.

5. Per-protein and per-family evaluation.
   - Avoid relying only on flattened residue-level metrics.
   - Include peptide-level "any positive" metrics when evaluating reactive/nonreactive peptide tasks.

6. Positive-weight provenance.
   - Record whether `pos_weight` came from CLI, all label shards, or train IDs.
   - Record numerator and denominator counts.

## Validation Notes

This document was produced from static code inspection. No tests or training jobs were run for this report.
