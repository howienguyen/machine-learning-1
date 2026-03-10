The plan mentioned here is no longer valid - so do not use this document

# Development Guidelines: Dataset Supplementation After Manifest Build for `baseline_logmel_cnn_v21.py`

Mon Mar  9 2026

## Abstract

This document revises the earlier oversampling proposal for `baseline_logmel_cnn_v21.py`.
The earlier idea duplicated or resampled rows inside the in-memory training dataframe
just before `train_ds` was built. That approach can work, but it is the wrong layer for
this codebase.

For MelCNN-MGR, a better design is to add a **new post-manifest supplementation script**
that runs **after** `MelCNN-MGR/preprocessing/build_manifest.py`. Instead of modifying an
already-loaded dataframe inside one training script, the new script should create a
persisted, auditable, reproducible supplemented training dataset derived from the manifest.

That means the solution moves from:

- ephemeral in-memory oversampling tied to one notebook/script,

to:

- a reusable dataset-preparation stage that produces explicit supplemented parquet files
  for downstream training.

This revised design is more consistent with the project architecture because MelCNN-MGR
already treats the manifest parquet files as the data contract between preprocessing and
training.

---

## 1. Review of the Previous Proposal

The previous document proposed oversampling by modifying `train_index_u` inside
`baseline_logmel_cnn_v21.py` before calling `make_dataset(...)`.

That proposal is not fundamentally wrong, but it has several weaknesses for this project.

### 1.1 It operates too late in the pipeline

The old approach intervenes only after:

- manifest loading,
- log-mel cache/index creation,
- label encoding,
- and training-dataset assembly.

That makes the oversampling logic a training-time concern, when it is really a
data-preparation concern.

### 1.2 It is tied to one training script

If oversampling lives inside `baseline_logmel_cnn_v21.py`, then:

- v21 gets the feature,
- but other scripts do not automatically benefit,
- and the supplemented dataset cannot be reused consistently across experiments.

### 1.3 It is not auditable enough

If rows are duplicated only in memory, then the exact supplemented dataset is not preserved
as a concrete artifact. That weakens reproducibility and makes later comparison harder.

### 1.4 It cannot naturally search for extra real samples first

Your revised requirement is important: before duplicating existing train rows, the system
should try to find **additional real tracks** from FMA metadata across the available subset
versions. That behavior belongs naturally in a manifest-stage script, not inside the final
training dataframe assembly.

### 1.5 It risks colliding with manifest semantics

The current manifest is a metadata truth table keyed by real `track_id`. Blindly duplicating
rows with the same `track_id` into the same raw manifest can break assumptions about row
identity and provenance. A supplementation stage needs a clearer schema and stronger
provenance fields.

---

## 2. Revised Recommendation

The better solution is to create a new script, conceptually like:

```bash
python MelCNN-MGR/preprocessing/build_supplemented_manifest.py --subset medium
```

This script should run **after**:

```bash
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
```

and should:

1. load the base manifest outputs,
2. inspect only the **training split**,
3. detect genres that qualify for supplementation,
4. try to source additional real training tracks from `FMA/fma_metadata/tracks.csv`,
5. if still short, create controlled duplicate references to existing training rows,
6. write new supplemented parquet artifacts for downstream training.

This is the right architectural layer because it preserves the manifest-driven workflow.

---

## 3. Core Policy

Oversampling is **train-only**.

The supplementation script must never modify the composition of:

- validation,
- test,
- or evaluation reports that are meant to reflect the official held-out data.

Only the training split may be supplemented.

That means:

- `val_{subset}.parquet` remains untouched,
- `test_{subset}.parquet` remains untouched,
- only training-oriented outputs gain supplemented rows.

---

## 4. Which Genres Should Be Supplemented

The decision rule should be driven by the **training split counts** after the base manifest
has already applied all filtering, including the exclusion of `genre_top == "Experimental"`.

Let:

$$
r_c = \frac{n_c}{n_{\max}}
$$

where:

- $n_c$ is the number of usable training samples for genre $c$
- $n_{\max}$ is the largest training-genre count in the target subset

For the first version of the supplementation policy:

- if $0.05 < r_c \le 0.2$, supplementation is strongly worth considering.

This range should be used as the **eligibility rule** for supplementation.

### Important interpretation

This policy intentionally targets **minority but still viable** classes.
It does **not** automatically target the extreme tail below $0.05$.

That is a defensible first choice because classes below $0.05$ are often so small that
heavy duplication can become mostly memorization. If later experiments show value in helping
the extreme tail too, that should be introduced as a separate policy revision rather than
silently folded into this one.

---

## 5. What the Supplementation Script Should Use as Input

The script should read:

- `metadata_manifest_{subset}.parquet`
- `train_{subset}.parquet`
- `val_{subset}.parquet`
- `test_{subset}.parquet`
- `FMA/fma_metadata/tracks.csv`

The base manifest remains the source of truth for:

- reason-code filtering,
- excluded labels,
- resolved file paths,
- and the initial official split assignment.

The supplementation script should not re-implement the full manifest builder from scratch.
It should extend the outputs of that stage.

---

## 6. Recommended Output Design

The raw manifest should remain immutable for auditability.

That means the supplementation script should **not overwrite** the original files produced by
`build_manifest.py`. Instead, it should write a second set of derived artifacts, for example:

- `metadata_manifest_supplemented_{subset}.parquet`
- `train_supplemented_{subset}.parquet`
- `val_supplemented_{subset}.parquet`
- `test_supplemented_{subset}.parquet`
- `supplementation_config_{subset}.json`
- `supplementation_report_{subset}.txt`

In practice:

- `val_supplemented_{subset}.parquet` can be identical to `val_{subset}.parquet`
- `test_supplemented_{subset}.parquet` can be identical to `test_{subset}.parquet`
- only `train_supplemented_{subset}.parquet` actually changes

This keeps the workflow explicit and reversible.

---

## 7. Why the Raw Manifest Should Not Be Mutated In Place

Although it is tempting to “update `metadata_manifest.parquet` directly,” that is not the
best design.

The base manifest should remain the unmodified product of `build_manifest.py` because it is:

- the audited record of real FMA metadata after filtering,
- the canonical view of `reason_code`,
- and the clean boundary between preprocessing and derived experiments.

If supplemented rows are mixed into the raw manifest file itself, then future debugging gets
harder because one file starts serving two roles:

- real-source metadata,
- and synthetic training expansion.

The better design is:

- keep the base manifest untouched,
- write a supplemented manifest as a second explicit artifact.

That still satisfies the functional requirement of “updating the manifest accordingly,” but
does so in a controlled way that preserves provenance.

---

## 8. Supplementation Strategy

For each eligible genre, the script should attempt supplementation in two stages.

### Stage A: add more real training tracks first

The script should search `tracks.csv` for additional tracks of that genre from the available
FMA subset versions.

However, several constraints must be respected.

#### Constraint 1: train split only

Only rows whose official FMA split is `training` may be imported into the supplemented train
set. Validation or test rows must never be pulled into training.

#### Constraint 2: excluded labels remain excluded

Rows with excluded labels, including `Experimental`, must remain excluded.

#### Constraint 3: source rows must pass the same usability checks

Any imported track should satisfy the same effective usability contract as manifest rows:

- valid genre label,
- target reason equivalent to usable,
- audio file exists,
- duration meets the minimum requirement,
- and any other constraints already enforced by the base manifest builder.

#### Constraint 4: avoid duplicates of already-selected real tracks

If a track already exists in the target training split, it should not be re-added as a new
real candidate.

### Stage B: duplicate references only if Stage A is insufficient

If cross-subset sourcing still does not bring the genre up to the target supplementation level,
the script may create duplicate references to existing training rows of that genre.

Those duplicates should:

- point to the same audio filepath,
- preserve the same `genre_top`,
- remain in the training split,
- and be explicitly marked as duplicated references.

This is not new data in the strict sense. It is a controlled increase in training exposure.

---

## 9. Practical Meaning of “Use Every Version: small, medium, large”

The revised idea should be interpreted as follows.

For a target subset such as `medium`, the supplementation script may look for additional
**training** rows of the same genre from the official FMA metadata across the exact subset
labels available in `tracks.csv`:

- `small`
- `medium`
- `large`

subject to:

- not already being present in the target train split,
- not being excluded,
- and being usable under the manifest rules.

This is a better strategy than immediately duplicating existing rows because it prefers real
new tracks before synthetic repetition.

---

## 10. Target Count Recommendation

The ratio rule above tells us **which genres qualify for supplementation**, but the script
still needs a target count.

For the first version, the most coherent target is:

$$
n_c^{\text{target}} = \left\lceil 0.2 \cdot n_{\max} \right\rceil
$$

for every genre that satisfies:

$$
0.05 < r_c \le 0.2
$$

This has two advantages:

- it matches the upper edge of the chosen eligibility band,
- and it avoids trying to equalize all minority genres to the largest class.

That makes the intervention moderate instead of aggressive.

---

## 11. Duplicate-Reference Policy

If Stage A cannot supply enough additional real tracks, then the remaining shortfall should be
filled by duplicating existing training rows.

But duplication should not be purely random in a naive way.

The requirement you proposed is correct: duplicated references should prioritize rows with the
**fewest existing copies**.

That means the script should maintain a per-source duplication counter and sample from the
least-copied rows first.

Conceptually:

1. initialize copy count = 0 for each original training row in the target genre,
2. repeatedly choose among rows with the current minimum copy count,
3. break ties randomly with a fixed seed,
4. increment the chosen row’s copy count,
5. stop when the target count is reached.

This produces a more even duplication pattern than unrestricted random replacement.

---

## 12. Required Provenance Fields

The supplemented outputs should add explicit provenance columns. At minimum:

- `is_supplemented` — boolean
- `supplement_kind` — one of `original`, `external_track`, `duplicate_reference`
- `source_track_id` — original real FMA track id
- `sample_id` — unique row identifier for the supplemented dataset
- `copy_number` — 0 for originals, 0 for imported real external tracks, positive for duplicates
- `source_subset` — original subset label from `tracks.csv`
- `source_manifest_subset` — target subset for which supplementation was built

This matters because once duplicate references exist, `track_id` alone is no longer enough to
represent row identity safely.

### Important schema note

The supplemented dataset should prefer a new unique row identifier such as `sample_id` rather
than relying on `track_id` uniqueness.

That avoids violating the base-manifest assumption that one `track_id` refers to one real
metadata row.

---

## 13. Recommended Processing Order

The supplementation script should work in this order.

### Step 1: load base artifacts

Load:

- `metadata_manifest_{subset}.parquet`
- `train_{subset}.parquet`
- `val_{subset}.parquet`
- `test_{subset}.parquet`
- `tracks.csv`

### Step 2: compute training counts

Count samples per `genre_top` using only `train_{subset}.parquet`.

Compute:

- `n_max`
- `r_c = n_c / n_max`

### Step 3: select eligible genres

Keep only genres satisfying:

$$
0.05 < r_c \le 0.2
$$

### Step 4: define target count

For each eligible genre, set:

$$
n_c^{\text{target}} = \left\lceil 0.2 \cdot n_{\max} \right\rceil
$$

### Step 5: source extra real training tracks

Search `tracks.csv` for additional usable training rows of that genre from the available exact
subset labels.

### Step 6: backfill with duplicate references

If the genre is still below target, duplicate existing train rows using the minimum-copy-first
policy.

### Step 7: assemble supplemented outputs

Create:

- supplemented train parquet,
- supplemented manifest parquet,
- config snapshot,
- report file.

### Step 8: leave validation and test unchanged

Validation and test remain structurally identical to the base outputs.

---

## 14. How This Should Relate to `baseline_logmel_cnn_v21.py`

Under the revised design, `baseline_logmel_cnn_v21.py` should not be responsible for deciding
how oversampling happens.

Instead, the training script should simply load whichever dataset version is requested:

- base manifest outputs,
- or supplemented manifest outputs.

That keeps the training script simpler and makes supplementation a reusable preprocessing step.

The role of v21 becomes:

- consume prepared split files,
- build caches if needed,
- train the model,
- evaluate normally.

That separation of concerns is cleaner.

---

## 15. What Should Stay Unchanged in Training

For the first version of this revised approach, the following should remain unchanged in the
training pipeline:

- log-mel extraction parameters,
- cache format and cache paths,
- normalization logic,
- model architecture,
- learning-rate schedule,
- Macro-F1 checkpointing,
- validation pipeline,
- test pipeline,
- evaluation metrics.

The main change is only the composition of the training rows that feed the training dataset.

---

## 16. Class Weights Under the Revised Design

The earlier document recommended disabling class weights for the first clean experiment.
That recommendation still stands.

If the supplementation script is used, the first experiment should be:

- supplemented training data enabled,
- class weights disabled,
- validation and test untouched,
- everything else unchanged.

That makes the impact of supplementation easier to interpret.

Later, a second experiment may test:

- supplemented training data,
- plus class weights.

But that should remain a separate experiment.

---

## 17. Reproducibility Requirements

The supplementation script must be deterministic under a fixed seed.

That means:

- candidate selection order should be stable,
- tie-breaking among equal-priority duplicate sources should use a seeded RNG,
- output files should be written with stable ordering,
- and the config file should record the full supplementation policy.

The config snapshot should include at least:

- target subset,
- eligibility rule,
- target ratio,
- random seed,
- whether external-track sourcing was enabled,
- whether duplicate-reference fallback was enabled,
- counts before and after supplementation by genre.

---

## 18. Reporting Requirements

The supplementation report should clearly distinguish:

- original training count per genre,
- target count per eligible genre,
- number of added real external tracks,
- number of added duplicate references,
- final supplemented training count per genre.

It should also list which genres were eligible under the rule:

$$
0.05 < r_c \le 0.2
$$

and which were not.

This keeps the transformation auditable.

---

## 19. Important Caveats

### 19.1 This is not true data augmentation

Duplicate references point to the same underlying audio. They increase exposure frequency but
do not create new musical content.

### 19.2 The extreme tail remains difficult

Genres below the lower bound $r_c \le 0.05$ are intentionally outside the first supplementation
policy. That is conservative, but it also means the most data-starved genres may remain weak.

### 19.3 Split leakage must be prevented strictly

The supplementation script must never import validation or test rows into training just because
they share a genre.

### 19.4 The base manifest contract should remain readable

The raw output of `build_manifest.py` should remain a clean representation of filtered FMA
metadata, not a mixture of real rows and synthetic training expansions.

---

## 20. Minimal Implementation Contract

The new script should do the following.

1. Run after `build_manifest.py`.
2. Read the base manifest outputs plus `tracks.csv`.
3. Compute train-only genre ratios.
4. Select eligible genres using:

   $$
   0.05 < r_c \le 0.2
   $$

5. Try to add real extra training tracks first.
6. If still below target, add duplicate references using minimum-copy-first balancing.
7. Write supplemented parquet outputs and a report.
8. Leave validation and test unchanged.

That is the correct shape for version 1.

---

## 21. Final Recommendation

For MelCNN-MGR, the revised idea is stronger than the previous one.

The old in-memory oversampling proposal was serviceable for a quick experiment, but it placed
the logic in the wrong layer and made provenance weaker. A post-manifest supplementation script
is the better engineering choice because it:

- matches the manifest-driven architecture,
- persists the exact supplemented dataset,
- can search for new real tracks before duplication,
- keeps train-only supplementation explicit,
- and preserves the raw manifest as an auditable base artifact.

In short: for this codebase, oversampling should be implemented as **dataset supplementation
after manifest build**, not as an invisible last-minute manipulation of an in-memory training
dataframe.