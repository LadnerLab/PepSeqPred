# PV1 + CWP + BKP Multi-Source Training/Validation and Positive Weight

This document explains how PV1, CWP, and BKP datasets are normalized separately and then used together for model training/validation across one or more DDP ranks. It also explains how the positive class weight is calculated at label build and train time.

## 1) Per-dataset normalization before multi-source training

Each source dataset is first normalized independently with `pepseqpred-prepare-dataset` into the same 4-file contract:

- `prepared_targets.fasta`
- `prepared_labels_metadata.tsv`
- `prepared_embedding_metadata.tsv`
- `prepare_summary.json`

### PV1 normalization

- Uses existing PV1 preprocessing (`preprocess_pv1`) to derive `Def epitope`, `Uncertain`, `Not epitope`.
- Parses family from PV1 fullname `OXX` (last comma-delimited token).
- Uses PV1 FASTA to validate protein IDs and alignment bounds.
- Group/family is the parsed PV1 family (numeric).

### CWP normalization

- Keeps only `CodeName` rows listed in `--reactive-codes` or `--nonreactive-codes`.
- Label mapping:
  - reactive code -> `Def epitope=1`, `Not epitope=0`, `Uncertain=0`
  - nonreactive code -> `Def epitope=0`, `Not epitope=1`, `Uncertain=0`
- Uses `SequenceAccession` as normalized `ProteinID`.
- Uses `Cluster50ID` as group token.
- Resolves align columns with fallbacks (`StartIndex/AlignStart/...`, `StopIndex/AlignStop/...`).
- Builds deterministic numeric group IDs from sorted unique `Cluster50ID` values with an offset.
  - default offset in CLI: `100000000`

### BKP normalization

- Same reactive/nonreactive mapping logic as CWP.
- Uses `SequenceAccession` as normalized `ProteinID`.
- Uses `reClusterID_70` as group token.
- Resolves align columns with BKP-priority fallbacks (`alignStart`, `alignStop`, then others).
- Deterministic numeric group IDs from sorted unique `reClusterID_70` values with an offset.
  - default offset in CLI: `200000000`

### Shared normalized fullname format

After normalization, all datasets are rewritten to PV1-style fullnames:

`ID=<ProteinID> AC=<ProteinID> OXX=0,0,0,<GroupID>`

So downstream tools can parse a single `ID + family` pattern uniformly.

## 2) How PV1/CWP/BKP are used together for training/validation

There is currently no dedicated "multi-source merge" CLI, and the datasets remain separate source datasets. The integration-covered way to use them together is:

1. Concatenate FASTA records from each dataset's `prepared_targets.fasta` into one combined FASTA.
2. Row-concatenate all `prepared_labels_metadata.tsv` files.
3. Row-concatenate all `prepared_embedding_metadata.tsv` files, then de-duplicate on `["Name", "Family"]`.

That is the behavior exercised in `tests/integration/test_prepare_dataset_multisource_pipeline.py` and is used to form shared inputs for embedding, label generation, and training. However, generating both embeddings and labels can be done separately, and training can be done by passing in all dataset files as outlined [here](../README.md#stage-4-train-ffnn).

## 3) How training/validation partitioning is done

### 3.1 Base protein universe

`train_ffnn_cli` builds a `ProteinDataset` from provided embedding dirs + label shards, then uses:

- `protein_ids = intersection(embedding_index IDs, label_index IDs)`

So split candidates are only proteins that exist in both embeddings and labels.

### 3.2 `split_type=id-family` (default)

- Family is parsed from embedding filename stem (`<protein_id>-<family>.pt`).
- If family is missing for an ID, it is treated as a singleton group:
  - `__missing_family__:<protein_id>`
- Global split then uses grouped split functions so a family/group cannot appear in both train and validation.

### Seeded mode (`--train-mode seeded`)

- Uses `split_ids_grouped(ids, val_frac, split_seed, family_groups)`.
- Target val size is `floor(len(ids) * val_frac)`, but exact fraction can differ because whole groups move together.
- A leakage check is run after splitting and raises if any family overlaps.

### Ensemble-kfold mode (`--train-mode ensemble-kfold`)

- `--val-frac` is ignored.
- Uses grouped k-fold (`build_grouped_kfold_splits`) when `split_type=id-family`.
- Groups/families are assigned to folds intact, then each fold is one validation set.
- Leakage check is run per fold.

### 3.3 `split_type=id`

- No family grouping; IDs are split directly.
- Seeded mode uses `split_ids`:
  - shuffle IDs with `split_seed`
  - validation = first `floor(N * val_frac)` IDs
  - training = remaining IDs
- Ensemble-kfold mode uses plain `build_kfold_splits`.

### 3.4 DDP rank partitioning after global train/val split

For multi-rank training, each run's already-determined global train/val ID lists are partitioned across ranks using `partition_ids_weighted`:

- greedy load balance by estimated embedding file size
- optional grouping by label-shard path for locality
- train partition enforces non-empty per rank

This is a rank-level sharding step, not a new train/val split.

## 4) Positive class weight calculation

There are two connected pieces:

1. Label build time (`pepseqpred-labels --calc-pos-weight`)
   - Each label shard writes:
     - `class_stats.pos_count`
     - `class_stats.neg_count`
     - `class_stats.pos_weight`
   - Formula:
     - `pos_weight = neg_count / max(1, pos_count)`
   - For 3-column labels `[Def epitope, Uncertain, Not epitope]`, counts only include residues where `Uncertain == 0`.

2. Train time (`pepseqpred-train-ffnn`)
   - If `--pos-weight` is provided, that value is used directly.
   - Otherwise it reads `class_stats` from all provided label shards and recomputes:
     - `total_neg / max(1, total_pos)`
   - That scalar is passed to `BCEWithLogitsLoss(pos_weight=...)`.

Note: automatic train-time `pos_weight` uses shard-level totals, not run-specific train-only IDs.
