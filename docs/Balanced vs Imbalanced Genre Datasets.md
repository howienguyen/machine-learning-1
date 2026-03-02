## Guide: Balanced vs Imbalanced Genre Datasets (for Music Genre Classification)

### 1) First: what does “balanced” mean?

“Balanced” only makes sense after you choose the **unit** you’re balancing.

**A. Balanced by tracks (most common)**
Each genre has roughly the same number of tracks/clips.

* Let (n_g) be the number of tracks in genre (g).
* A simple imbalance score is:

[
\text{imbalance ratio} = \frac{\max_g n_g}{\min_g n_g}
]

Perfectly balanced means ratio = 1. In practice, people call it “reasonably balanced” if the ratio is small (often ≤ 2–5).

**B. Balanced by duration**
Each genre has roughly the same total audio time. Mostly matters when track lengths vary a lot.

**C. Balanced by artist (anti-leakage balance)**
Each genre has a similar number of distinct artists, and you avoid the same artist appearing in both train and test. This helps prevent the model from learning “artist fingerprints” instead of genre cues.

**For FMA-medium:** clips are fixed length, so “balance by duration” is less important; “balance by tracks” and “artist leakage control” matter more.

---

### 2) Why imbalance is a problem

Genre data in the wild is often **long-tailed**: a few genres have tons of tracks; many genres have very few.

Imbalance causes two classic issues:

**A. Training bias**
The model gets more updates from majority genres, so it becomes very good at predicting common genres and bad at rare ones.

**B. Misleading evaluation**
Overall accuracy can look high even if the model is terrible on minority classes, because the model can “cheat” by over-predicting common genres.

---

### 3) Is a balanced dataset “better” for training?

**Often yes — for fair learning and clean experiments — but it’s not always the right goal.**

Balanced training tends to give:

* **Fairer per-genre performance** (minor genres get enough exposure).
* **More stable gradients** during training (minor genres appear in minibatches more often).
* **Cleaner comparisons** between models and representations.

But here’s the important reality check: **“making” a dataset balanced can create trade-offs that fight your actual deployment goals.**

#### The trade-offs when forcing balance

If you aggressively equalize counts by downsampling/upsampling, you can pay several costs:

* **You may throw away a lot of data** if you downsample big genres (wasting signal you already have).
* **You can overfit** if you upsample tiny genres (you keep showing the model the same few examples).
* **You may train on an unrealistic distribution.** Real music catalogs are long-tailed; a perfectly balanced training set can produce models whose predicted probabilities are **miscalibrated** for real-world usage (they may over-predict rare genres because training made them “seem” common).
* **Full-taxonomy balancing is awkward.** With datasets like FMA where the tail is huge, “balance every genre” quickly becomes impractical because many genres have too few tracks to learn reliably.

#### A practical principle

* If your priority is **fairness / per-genre performance / scientific comparison**, then balancing (or approximating balance) is useful.
* If your priority is **realistic deployment behavior**, don’t force the dataset to be perfectly balanced. Instead, **keep the natural long-tail distribution** and handle imbalance during training with **sampling and/or class weights**.

That principle lets you preserve real-world realism while still preventing the model from ignoring minority genres.

---

### 4) Practical strategies (recommended order)

You usually don’t need to physically “make the dataset balanced” by discarding tons of tracks. Instead, keep the data and adjust training + evaluation.

**Strategy 1 — Use metrics that respect imbalance (must-do)**
Report at least one metric that treats each genre equally:

* **Macro-F1** (average F1 over genres)

Also useful:

* **Balanced accuracy** (average recall over genres)
* Per-class precision/recall/F1 table

Keep accuracy/micro-F1 too, but don’t rely on them alone.

**Strategy 2 — Class-weighted loss (easy win)**
Give more weight to minority genres in the loss, e.g.:

[
w_g \propto \frac{1}{n_g} \quad \text{or} \quad w_g \propto \frac{1}{\sqrt{n_g}}
]

This pushes the model to care about rare genres without duplicating data.

**Strategy 3 — Balanced sampling (very common in deep learning)**
During mini-batch construction:

* sample genres more evenly, or
* oversample rare genres (ideally with augmentation to reduce overfitting)

This improves learning signal for minority classes while keeping your full dataset.

**Strategy 4 — Top-K or minimum-count filtering (taxonomy-friendly)**
For large taxonomies (like FMA’s long tail), it’s often cleaner to:

* restrict to **Top-K genres**, or
* keep only genres with at least **N tracks**

This avoids “classes with 7 examples,” which is less “genre classification” and more “guessing with vibes.”

**Strategy 5 — True rebalancing (use carefully)**

* **Downsample** majority classes only if you have huge redundancy or need strict balance for an experiment.
* **Upsample** minority classes only if you also use augmentation or strong regularization.

---

### 5) A good default policy for your FMA-medium project

**Recommended baseline policy:**

1. Keep the official split and dataset as-is.
2. Choose your label granularity (coarse genres first is often easier).
3. Train with either:

   * class-weighted loss, or
   * balanced sampling.
4. Evaluate with:

   * macro-F1 (primary “fairness” metric),
   * plus accuracy/micro-F1 (overall performance).

This gives you:

* a fair model that doesn’t ignore minority genres,
* realistic exposure to the real long-tail distribution,
* and evaluation that doesn’t lie.

---

### 6) Quick “definition” you can put in your report

> “We call the dataset balanced when the number of tracks per genre in the training set is approximately uniform, and we monitor imbalance using the ratio (\max(n_g)/\min(n_g)). Because real music data is naturally long-tailed, we primarily keep the natural distribution and address imbalance through class-weighted loss or balanced sampling, and we report macro-F1 to reflect per-genre performance.”

---

### 7) One last trap: artist leakage

Even with perfect genre balance, you can get fake-good results if the same artist appears in both train and test and the model learns production/voice/instrument signatures.

So sanity-check:

* artist overlap across splits (if you have artist IDs),
* and track-level duplicates / near-duplicates.

---

Core mental model: **Balance is not just a dataset property — it’s a training + evaluation decision, and “perfect balance” isn’t always aligned with real deployment.**
