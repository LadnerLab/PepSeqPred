# PepSeqPred Methods

## Overview

PepSeqPred was developed as a residue-level epitope prediction pipeline for
protein and peptide datasets. The pipeline separates sequence representation
from residue classification: full protein sequences are first embedded with an
ESM-2 protein language model, peptide-level reactivity annotations are converted
into residue-level labels, and a lightweight neural network head is trained to
predict epitope probability for each residue.

The implementation was organized as a staged workflow so that data preparation,
embedding generation, label construction, model training, prediction, and
evaluation could be reproduced independently. Each stage writes explicit
artifacts, and downstream stages consume those artifacts instead of relying on
implicit in-memory state. This design supports local smoke tests while keeping
the production workflow compatible with SLURM-based multi-GPU execution.

## Dataset Normalization

PepSeqPred supports PV1, CWP/Cocci, and BKP source datasets through a shared
normalization contract. Each dataset is first converted into four prepared
outputs: a target FASTA, label metadata TSV, embedding metadata TSV, and JSON
summary. The normalized FASTA headers use the same PV1-compatible form:

```text
ID=<ProteinID> AC=<ProteinID> OXX=0,0,0,<GroupID>
```

This shared header format was chosen so downstream embedding, label, and
training code can parse protein identity and grouping information uniformly
across datasets.

PV1 data are prepared from metadata, z-score tables, and protein FASTA records.
The default PV1 label rules classify peptides as definite epitopes when the
z-score is at least 20 in at least 4 subjects, and classify peptides as
non-epitopes when all subject z-scores are at most 10. Peptides not satisfying
either definite criterion remain uncertain. PV1 group identifiers are parsed
from the final token of the source `OXX` field.

CWP/Cocci and BKP data are prepared from metadata, protein FASTA records,
reactive code lists, and non-reactive code lists. Reactive codes are mapped to
definite epitope labels, non-reactive codes are mapped to non-epitope labels,
and these datasets do not introduce uncertain labels during normalization.
CWP/Cocci uses `Cluster50ID` as the grouping token, while BKP uses
`reClusterID_70`. These group tokens are converted to deterministic numeric
group identifiers using sorted token order with dataset-specific offsets, so
the groups can be used by the same downstream family-aware split logic as PV1.

Rows are dropped during preparation when required identifiers, grouping tokens,
protein sequences, peptide sequences, or alignment bounds cannot be resolved.
Ambiguous FASTA accessions and out-of-bounds alignments are also excluded. Drop
counts and examples are recorded in each dataset's preparation summary.

## Residue Label Construction

PepSeqPred converts peptide-level annotations into dense per-protein residue
labels. Label tensors are stored as `(L, 3)` arrays with columns:

```text
[Def epitope, Uncertain, Not epitope]
```

where `L` is the protein length. Each peptide annotation contributes labels to
the residue interval defined by its alignment start and stop coordinates.
Residues are initialized as uncertain. Definite epitope intervals mark residues
as positive. Non-epitope intervals mark residues as negative only where no
definite epitope label is present. This overlap policy makes definite epitope
evidence take precedence over non-epitope evidence, and non-epitope evidence
take precedence over uncertainty.

During training, three-column labels are collapsed to binary targets by using
the definite epitope column as the positive class. The uncertain column is used
to build a validity mask, so uncertain residues do not contribute to loss or
metric calculations. This allows the model to use proteins with partially
annotated residue labels without treating unknown residues as negatives.

Label shards also retain peptide-level metadata for traceability. When requested
at label-build time, shards record positive and negative residue counts and a
positive class weight computed as:

```text
negative residues / max(1, positive residues)
```

The same masking rule is used for these counts: uncertain residues are excluded.

## Sequence Embeddings

Protein sequences are embedded with ESM-2, using `esm2_t33_650M_UR50D` by
default. PepSeqPred extracts per-residue representations from layer 33 and
writes one `.pt` tensor per protein. Each tensor has shape `(L, D)` by default,
where `L` is the number of cleaned amino acid residues and `D` is the ESM
embedding dimension.

Input sequences are normalized to uppercase amino acid tokens and restricted to
the accepted ESM alphabet characters used by the implementation. Sequences are
sorted by length before embedding and packed into batches under a token budget
to reduce padding overhead. The default ESM residue token limit is 1022, and the
default embedding batch size is 8.

Sequences that fit within the ESM context window are embedded in a single model
forward pass. Longer sequences are embedded with overlapping windows. Window
representations are stitched back into a full-length residue matrix by averaging
overlap contributions at each residue. This preserves full-protein output shape
while allowing proteins longer than the model's single-pass context limit to be
represented.

Embedding filenames can be keyed by protein ID alone or by `ID-family`. The
`ID-family` mode is the default for the multi-source training workflow because
it keeps leakage-control grouping information available to downstream training.
An optional sequence-length feature can be appended to each residue embedding.
The default is no appended length feature; supported alternatives append either
the raw protein length or its inverse.

## Training Dataset

Training examples are streamed from per-protein embedding files and label shard
files. The protein universe for a run is the intersection of proteins present in
the embedding index and the label index. This avoids training or evaluating on
proteins missing either representation or labels.

The dataset yields residue windows rather than requiring all proteins to have a
fixed length. Training uses a default window size of 1000 residues and a stride
of 900 residues. The final short window is padded by default, and padding is
removed from loss and metric calculations through a mask. Validation uses full
proteins, avoiding artificial validation-window fragmentation.

The final training mask combines two exclusions: padded positions and uncertain
residues. Only real residues with definite positive or definite negative labels
contribute to the binary classification objective.

## Model Architecture

PepSeqPred trains a lightweight residue classifier on top of frozen ESM-2
embeddings. The default model head is a feed-forward neural network that applies
the same classifier independently to each residue embedding. The default hidden
sizes are `150,120,45`, with dropout `0.1,0.1,0.1`, one binary output logit per
residue, and no layer normalization or residual connections unless explicitly
enabled.

An optional `conv1d` model head adds local sequence context before dense
classification. This head applies a small Conv1d stack across the residue axis,
concatenates local convolutional features with the original ESM embedding at
each residue, and then applies the same feed-forward classifier pattern. The
default convolutional configuration uses 64 channels, 2 layers, kernel size 9,
and dropout 0.1. The convolutional head preserves the same input and output
contract as the default feed-forward head: `(batch, length, embedding_dim)`
inputs and `(batch, length)` logits.

This separation between pretrained sequence representation and trainable model
head was chosen to keep training tractable on large residue collections while
still allowing the classifier to adapt to PepSeqPred's epitope-labeling task.

## Training Procedure

Training uses PyTorch with the Adam optimizer. The default learning rate is
`1e-3`, default weight decay is `0.0`, default batch size is 64, and default
training length is 10 epochs. The loss is binary cross entropy with logits,
computed per residue with `reduction="none"` so uncertain and padded residues
can be masked before averaging.

Positive class weighting is resolved per run. If a manual positive weight is
provided, it is used directly. Otherwise, PepSeqPred counts positive and
negative residues from the current run's training loader after global split
planning and, when applicable, after distributed rank partitioning. The
automatic training weight is:

```text
training negative residues / max(1, training positive residues)
```

This run-specific weighting was chosen because residue-level epitope labels are
highly imbalanced and because the effective class distribution may vary across
folds, seeds, subsets, and distributed partitions.

Randomness is controlled through explicit split and training seeds. If separate
split and training seeds are not supplied, both default to the global seed. The
training CLI supports single-run holdout mode and K-fold ensemble mode through a
single `--n-folds` interface. Model checkpoints store optimizer state, training
configuration, model configuration, best validation loss, and validation metrics.

The training loop supports DistributedDataParallel execution. Protein IDs are
first assigned to global train and validation splits, then the already chosen
training and validation ID lists are partitioned across ranks. Rank partitioning
uses estimated embedding-file size for greedy load balancing and can preserve
label-shard locality. Shared artifacts such as checkpoints, CSV rows, split
reports, summaries, and ensemble manifests are written by rank 0.

## Split Design

PepSeqPred uses `id-family` splitting by default. Under this split type,
proteins sharing the same parsed family or group identifier are kept together in
the same train or validation partition. This was chosen to reduce leakage from
closely related proteins across validation boundaries. If no family is available
for a protein, that protein is treated as a singleton group.

For `--n-folds 1`, PepSeqPred performs a holdout split using the requested
validation fraction, which defaults to 0.2. For `--n-folds K` where `K > 1`, it
builds K train/validation members per seed pair and ignores the holdout
validation fraction. In K-fold mode, each fold serves once as validation while
the remaining folds are used for training.

The default split strategy is `size-balanced`, which assigns groups to balance
protein counts while preserving group integrity. An optional `label-stratified`
strategy also preserves groups but uses per-protein valid, positive, and
negative residue support to balance label support and positive rates across
splits or folds.

Each training run writes a split report JSON by default. The report records
train and validation protein counts, valid residue counts, positive and negative
residue counts, positive rates, family counts, and label-shard counts. These
reports are part of the reproducibility record for comparing seeded runs and
folded ensembles.

## Threshold Selection And Metrics

PepSeqPred models output residue-level probabilities. Binary residue predictions
are produced by applying a validation-selected threshold or an explicit fixed
threshold. The default threshold policy is `max-recall-min-precision`, with a
minimum precision of 0.25. Under this policy, the selected threshold maximizes
recall among candidates satisfying the precision constraint; if the constraint
is unreachable, the highest-precision candidate is reported with a diagnostic
status.

Additional supported policies select the best F1 score, best Matthews
correlation coefficient, maximum precision subject to a minimum recall
constraint, or a fixed threshold. Threshold selection records the selected
threshold, status, precision, recall, F1, MCC, confusion counts, class support,
and predicted-positive fraction.

Evaluation metrics are computed over valid labeled residues only. Reported
metrics include precision, recall, F1, Matthews correlation coefficient, ROC
AUC, PR AUC, AUC over the low false-positive-rate region up to 0.10, and
balanced residue accuracy. Validation can also write ROC and precision-recall
curve artifacts for the best folds or epochs.

## Prediction And Ensembles

Single-checkpoint prediction loads the trained classifier, generates or consumes
protein embeddings, computes residue probabilities, and applies the selected
threshold to produce a binary residue mask. The prediction payload records mask
length, number and fraction of predicted epitope residues, mean and maximum
predicted probability, and threshold.

K-fold models are represented by ensemble manifest JSON files. Each valid member
contains its checkpoint path, fold index, member index, status, and validation
threshold. During ensemble prediction, each member first thresholds its own
probabilities. The default ensemble rule is majority voting over member binary
masks, with ties resolved as negative. A mean-probability aggregation path is
also available for explicit use.

Bundled pretrained predictors are exposed through the public `pepseqpred.api`
registry. The default registry entry is `flagship2-v1`, with `flagship2` and
`default` aliases. Bundled registry entries record their expected ESM model,
number of members, and provenance metadata.

## Reproducibility And Artifact Design

PepSeqPred's development emphasized traceability of scientific decisions and
outputs. The main reproducibility choices are:

- dataset preparation writes normalized source artifacts plus drop summaries;
- embeddings are stored one protein per tensor with an index CSV;
- labels are stored as shards with per-protein tensors and peptide metadata;
- training checkpoints include model and trainer configuration;
- split reports record the effective train and validation residue support;
- run CSVs and summary JSON files preserve per-run metrics and seeds;
- K-fold ensembles are described by manifest JSON files rather than implicit
  directory conventions;
- HPC wrappers preserve the same Python CLI contracts used by local smoke tests.

This artifact-oriented workflow allows expensive stages, such as ESM-2
embedding generation and multi-GPU training, to be reproduced or audited without
rerunning the full pipeline unnecessarily.
