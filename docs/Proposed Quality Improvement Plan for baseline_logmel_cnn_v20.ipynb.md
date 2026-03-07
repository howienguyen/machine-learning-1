# Proposed Quality Improvement Plan for `baseline_logmel_cnn_v20.ipynb`

Friday, March 6, 2026 7:58:55 PM

## 1. Purpose and scope

This document presents a **proposal for improving** the existing notebook **`baseline_logmel_cnn_v20.ipynb`**. The current notebook is a solid baseline for music genre classification using **10-second fixed-length audio clips**, log-mel spectrograms, and a CNN. However, it is still primarily a **baseline-oriented experiment notebook** rather than a more **production-like** training and inference pipeline.

The purpose of this document is to describe a set of recommended improvements that can help the next version move in a more practical direction: better predictive quality, better robustness, and behavior that is closer to what one would want from a real deployed model. The recommendations here are not random feature glitter. Each one is tied to a problem it is intended to address.

The proposed improvements are:

* keep 10-second fixed-length training
* add class weights
* switch to milder time pooling
* add SpecAugment
* try AdamW
* optionally use label smoothing = 0.05
* at inference, use 3 deterministic crops and average probabilities

---

## 2. Current baseline context

The current `baseline_logmel_cnn_v20.ipynb` already does several things well:

* it uses a **fixed 10-second input policy**
* it applies a consistent preprocessing path
* it uses a valid **log-mel + CNN** baseline design
* it normalizes features in a train-safe way
* it has a reasonable training loop with Adam, early stopping, and LR reduction
* it evaluates the model using useful metrics such as macro-F1 and a confusion matrix

So the situation is not that the notebook is broken. The issue is more subtle:

> the notebook is a good baseline, but there are several predictable reasons why its predictive quality is likely below what a stronger, more production-like version could achieve.

That is the starting point for the proposals below.

---

## 3. Design direction for the upgraded version

The upgraded version should continue to preserve the core shape of the current approach:

* 10-second fixed-length training inputs
* spectrogram-based representation
* CNN-based classifier
* deterministic preprocessing
* reproducible training

This is important because the goal is **not** to throw away the baseline and replace it with a completely different system. The goal is to **evolve** the baseline into a stronger version while keeping the design understandable and methodologically defensible.

In other words, the system should become more robust and more accurate without mutating into an experimental zoo where no one remembers which knob caused what.

---

# 4. Recommended improvements

## 4.1. Keep 10-second fixed-length training

### Problem being addressed

The system needs a stable and consistent definition of what a training sample is. If input duration is allowed to vary, then feature shapes vary, model assumptions become more complicated, and comparison across experiments becomes less clean.

### Why this matters

The current project has already moved from a 30-second baseline toward a 10-second fixed-length regime. That move is sensible because:

* 10-second clips are cheaper to process
* 10-second inputs better match many realistic use cases
* a fixed-length design keeps the model architecture simple
* it makes caching, batching, and deployment easier

For an upgraded version that leans toward production-like behavior, this consistency is valuable.

### Proposed solution

Keep the existing **10-second fixed-length training policy**.

### What this is for

This is not an “improvement” in the sense of changing the method. It is a decision to **preserve a good constraint** that supports reproducibility, stable engineering, and easier deployment.

### How it should work

The upgraded version should continue using the current policy:

* if the source clip is longer than 10 seconds, derive an exact 10-second clip using the midpoint-centered rule
* if the source clip is exactly 10 seconds, use it as-is
* if the source clip is shorter than 10 seconds, pad with silent samples to reach exactly 10 seconds

### Recommendation

Keep this as a foundation. Do not re-open the variable-length rabbit hole at this stage.

---

## 4.2. Add class weights

### Problem being addressed

The genre classification dataset is imbalanced. Some genres have many more samples than others. When training uses plain categorical cross-entropy without class weighting, the model tends to optimize more strongly for majority classes.

### Why this matters

Without imbalance-aware training, the model may:

* achieve acceptable overall accuracy
* but perform poorly on minority genres
* under-represent rare classes
* produce weaker macro-F1
* become biased toward the dominant genre distribution

This is especially important in genre classification because a model that mostly learns the “big genres” can look decent numerically while still being weak in practice.

### Proposed solution

Compute **class weights** from the **training split only**, then pass them to `model.fit()`.

### What this is for

Class weighting makes the training objective pay more attention to minority classes. It tells the optimizer that mistakes on under-represented classes should matter more.

### How it should work

A standard implementation would:

1. compute integer labels for the training split
2. calculate balanced class weights based on training frequencies
3. pass the resulting dictionary to Keras through `class_weight=...`

Conceptually:

* rare class → larger weight
* common class → smaller weight

### Expected effect

This change is expected to improve:

* minority-class recall
* macro-F1
* class balance in predictions

It may or may not improve plain accuracy, but it is highly likely to improve the model in a more meaningful multiclass sense.

### Recommendation priority

**Very high**

---

## 4.3. Switch to milder time pooling

### Problem being addressed

The current CNN appears to downsample the **time axis** too aggressively through pooling. This can destroy temporal detail too early in the network.

### Why this matters

Music genre recognition is not only about static frequency content. It also depends on temporal structure such as:

* rhythm
* texture evolution
* repeated motifs
* short-term patterns across time

If the time dimension is compressed too aggressively in early layers, the network may lose useful information before it has a chance to learn from it.

This issue becomes more important in a **10-second regime**, where the total available time context is already shorter than before.

### Proposed solution

Use **milder temporal pooling** than the current architecture.

### What this is for

The goal is to preserve more time information deeper into the network so that the CNN can learn richer temporal patterns instead of crushing them too early.

### How it should work

Instead of strong pooling such as repeated `(2, 4)` over frequency × time, use gentler patterns such as:

* `(2, 2)`
* `(2, 2)`
* `(2, 2)`
* `(2, 2)`

or another similarly moderate design.

The exact configuration can still be tuned, but the principle is clear:

> keep some temporal resolution alive longer.

### Expected effect

This change is expected to improve:

* temporal feature retention
* rhythm-sensitive discrimination
* overall classification quality for genres where timing patterns matter

### Recommendation priority

**High**

---

## 4.4. Add SpecAugment

### Problem being addressed

The current training data pipeline is relatively clean and static. Each training example is a fixed spectrogram derived from a fixed 10-second clip, with little or no augmentation.

### Why this matters

Without augmentation, the model may become too dependent on narrow local patterns in the training data. That can hurt generalization. In real usage, audio conditions and local feature arrangements vary. A more production-like system should be more robust to such variation.

### Proposed solution

Add **SpecAugment-style masking** during training.

### What this is for

SpecAugment is a regularization and robustness technique for spectrogram models. It usually involves randomly masking:

* time regions
* frequency regions

This forces the model to rely less on a few brittle cues and to learn more distributed patterns.

### How it should work

During training only:

* randomly zero or mask small time spans
* randomly zero or mask small frequency bands

Validation and test data should remain unaugmented.

This can be implemented inside the TensorFlow data pipeline or as a model-side augmentation layer.

### Expected effect

This change is expected to improve:

* generalization
* robustness to local missing features
* resistance to overfitting
* predictive quality on unseen examples

### Recommendation priority

**High**

---

## 4.5. Try AdamW

### Problem being addressed

The current optimizer choice, plain Adam, is already a good baseline, but it may not provide the best generalization. In some cases, Adam can fit training data well while regularizing less cleanly than desired.

### Why this matters

If the goal is to move toward a more production-like training setup, it is reasonable to test an optimizer variant that often behaves better in practice when combined with modern regularization.

### Proposed solution

Try **AdamW** instead of plain Adam.

### What this is for

AdamW introduces **decoupled weight decay**, which is generally a cleaner and more principled way to regularize model weights than relying only on Adam plus dropout.

### How it should work

Replace:

* `Adam(learning_rate=1e-3)`

with something like:

* `AdamW(learning_rate=1e-3, weight_decay=1e-4)`

The exact weight decay may require tuning, but `1e-4` is a reasonable starting point.

### Expected effect

Possible benefits include:

* slightly better generalization
* cleaner regularization behavior
* improved validation performance

### Important note

This is a useful improvement, but it is **not** expected to be the single biggest lever. It is more of a refinement than a fundamental correction.

### Recommendation priority

**Medium**

---

## 4.6. Optionally use label smoothing = 0.05

### Problem being addressed

The current training uses hard one-hot labels. That means the loss assumes absolute certainty about the correct class and absolute zero for all others.

### Why this matters

Genre labels are often somewhat fuzzy. A track labeled as one genre may also contain qualities of neighboring genres. Hard one-hot supervision can encourage the model to become too confident and too sharp in its output probabilities.

That can lead to:

* overconfidence
* weaker calibration
* unnecessarily brittle decision boundaries

### Proposed solution

Optionally use **categorical cross-entropy with label smoothing = 0.05**.

### What this is for

Label smoothing softens the training target slightly. Instead of saying:

* correct class = 1.0
* all others = 0.0

it says something more modest, such as:

* correct class = mostly correct
* other classes = small non-zero mass

This can reduce overconfidence and improve generalization in some multiclass tasks.

### How it should work

Use a loss such as:

* `CategoricalCrossentropy(label_smoothing=0.05)`

A smoothing value of `0.05` is a good conservative starting point. It is gentler than `0.1` and more suitable for a baseline-sized model.

### Expected effect

Potential benefits include:

* better calibration
* reduced output overconfidence
* modest regularization
* possible generalization gains

### Important note

This should be treated as **optional** because its impact is often smaller than class weighting or augmentation. It is a secondary lever, not the first hammer to reach for.

### Recommendation priority

**Medium**

---

## 4.7. At inference, use 3 deterministic crops and average probabilities

### Problem being addressed

The current prediction path appears to use a **single 10-second derived clip** for inference. That means the final prediction depends on only one temporal view of the source audio.

### Why this matters

A single 10-second segment may miss useful genre cues that appear elsewhere in the source clip. Music is not uniformly informative across time. One section may emphasize rhythm, another instrumentation, another texture.

Using only one crop makes the prediction more fragile.

### Proposed solution

At inference time, derive **3 deterministic 10-second crops** and average their predicted probabilities.

### What this is for

This acts like a lightweight test-time ensemble across temporal regions of the same clip. It improves robustness without changing the core model.

### How it should work

For clips longer than 10 seconds, choose three deterministic views, for example:

* early-centered crop
* middle-centered crop
* late-centered crop

or another fixed, documented three-crop policy.

Then:

1. preprocess each crop independently
2. run the model on each crop
3. average the resulting probability vectors
4. take the final class from the averaged probabilities

For short clips that are padded to 10 seconds, the same normalized clip may simply be reused.

### Expected effect

This change is expected to improve:

* inference robustness
* prediction stability
* final classification quality

It is especially helpful when genre cues are distributed across different song regions.

### Recommendation priority

**High for production-like inference**
**Medium for strict baseline purity**

---

# 5. Why these improvements belong together

Each recommendation addresses a different weakness in the current notebook. They are complementary rather than redundant.

* **keep 10s fixed length** addresses stability and operational simplicity
* **class weights** address imbalance in the label distribution
* **milder time pooling** addresses premature loss of temporal information
* **SpecAugment** addresses generalization and robustness
* **AdamW** addresses optimizer-side regularization quality
* **label smoothing** addresses overconfidence and soft-label realism
* **3-crop inference** addresses fragility from relying on one temporal view

So the upgraded version is not just “a stronger model.” It is a more balanced system in which:

* training is more fair across classes
* the architecture is less destructive
* the model is more robust during learning
* the optimizer regularizes more cleanly
* predictions are more stable at inference time

That is exactly the kind of shift one would expect when moving from notebook baseline toward something more production-like.

---

# 6. Suggested implementation order

To keep development controlled, the upgrades should not all be introduced chaotically at once. A staged plan is better.

## Stage 1 — highest-value structural improvements

Start with:

* keep 10s fixed-length training
* add class weights
* switch to milder time pooling
* add SpecAugment

These are the most important quality-oriented improvements.

## Stage 2 — training refinement

Then test:

* AdamW
* optionally label smoothing = 0.05

These are useful refinements once the larger issues are addressed.

## Stage 3 — inference improvement

Finally add:

* 3 deterministic crops at inference and probability averaging

This improves real prediction behavior and makes the upgraded system more production-like.

---

# 7. Proposed upgraded-version intent

The next version should be framed as:

> an upgraded development version derived from `baseline_logmel_cnn_v20.ipynb`, designed to improve predictive quality and robustness while remaining reproducible, understandable, and closer to production-like behavior.

That wording matters. It signals that the upgraded notebook is not abandoning the baseline spirit. It is extending it in a disciplined way.

---

# 8. Final recommendation summary

Yes — that should be stated explicitly.

A clean way to revise **Section 8** is this:

---

# 8. Final recommendation summary

Based on the existing **`baseline_logmel_cnn_v20.ipynb`**, the proposed upgraded version should:

* continue using **10-second fixed-length training**
* incorporate **class weights** to handle imbalance
* adopt **milder temporal pooling** to preserve time information
* add **SpecAugment** for stronger training-time robustness
* test **AdamW** as a regularized optimizer upgrade
* optionally use **label smoothing = 0.05**
* use **3 deterministic crops at inference** and average their probabilities

At the same time, this proposal is **not intended to replace the current model architecture with a fundamentally different one**. The overall architectural approach should remain the same: a **log-mel spectrogram input pipeline feeding a CNN-based classifier**. In other words, the proposal aims to **preserve the core model architecture and baseline design philosophy**, while improving training behavior, regularization, temporal information retention, and inference robustness.

So the intent of this proposal is to **upgrade the existing baseline rather than redesign it from scratch**. The model should remain recognizably the same family of solution, which helps preserve comparability with `baseline_logmel_cnn_v20.ipynb`, keeps development controlled, and supports a more production-like evolution path.

Together, these changes target the main weaknesses of the current notebook while keeping the solution grounded in the same core architecture. The result should be a stronger and more robust version of the baseline, not a completely different modeling approach dressed up in new shoes.

In shoret, this proposal does not aim to introduce a fundamentally new architecture. The core solution should remain a fixed-length log-mel + CNN model + existing layers, so that the upgraded version stays comparable to `baseline_logmel_cnn_v20.ipynb` while improving robustness, generalization, and prediction quality.



---

## Appendix A. Clarification on architectural changes and model-family preservation

This appendix clarifies an important point in the proposal: some of the recommended improvements may introduce **limited architectural changes**, but these changes are **not intended to replace the current model with a fundamentally different model family**.

That distinction matters because the phrase “architecture change” can sound larger and more disruptive than what is actually being proposed.

### A.1. What remains unchanged at the model-family level

The upgraded version is intended to remain within the same overall modeling approach as `baseline_logmel_cnn_v20.ipynb`. In that sense, the **model family is preserved**.

The following core characteristics remain the same:

* the input remains a **fixed-length 10-second audio clip**
* the waveform is converted into a **log-mel spectrogram representation**
* the classifier remains a **2D convolutional neural network (CNN)**
* the model still follows the same broad processing pattern:

  * spectrogram input
  * convolutional feature extraction
  * pooling-based feature compression
  * global aggregation
  * softmax classification

So, at the high level, the proposed upgraded system remains a **fixed-length log-mel + CNN classifier**.

This means the proposal is **not** moving to a different family such as:

* a raw-waveform model
* a recurrent model such as a CRNN
* a transformer-based model
* a ResNet-style redesign
* a variable-length architecture

In other words, the upgraded version is intended to remain recognizably the same type of solution.

---

### A.2. What may change at the internal architectural level

Although the model family remains the same, some recommended improvements may still change **internal architectural details**.

This means the exact layer-level implementation may be adjusted, even though the overall design philosophy remains unchanged.

The clearest example is the recommendation to use **milder temporal pooling**.

In the existing model, pooling is relatively aggressive along the time axis. If that pooling strategy is changed, then the exact model definition changes as well. For example:

* pooling window sizes may be reduced
* temporal resolution may be preserved longer
* intermediate feature-map shapes may differ

This is an architectural refinement because pooling layers are part of the model structure. However, it is still a refinement **within the same CNN family**, not a transition to a new architecture family.

So the proposal allows for **internal architectural tuning**, but not a full architectural replacement.

---

### A.3. Why this distinction matters

Without this clarification, one might incorrectly interpret the proposal in one of two extreme ways.

The first incorrect interpretation would be:

> the upgraded version is a completely new model

That is not true. The proposal does not call for abandoning the current log-mel + CNN approach.

The second incorrect interpretation would be:

> the model architecture remains exactly identical layer by layer

That is also not necessarily true. Some changes, especially to pooling behavior, may alter the exact internal layer configuration.

The accurate position is between those two extremes:

> the upgraded version preserves the same overall model family, while allowing limited internal architectural refinements.

That is the intended meaning.

---

### A.4. Which proposed improvements do and do not affect architecture

The recommended improvements do not all operate at the same level. Some affect training, some affect inference, and some may affect internal architecture.

#### Improvements that do **not** inherently change the model architecture

The following recommendations mainly affect preprocessing, training, loss behavior, or inference procedure:

* keep 10-second fixed-length training
* add class weights
* try AdamW
* optionally use label smoothing = 0.05
* use 3 deterministic crops at inference and average probabilities

These do not require changing the core CNN layer stack itself.

#### Improvements that **may** affect internal architectural details

The following recommendations may affect the internal structure or computation path:

* switch to milder time pooling
* add SpecAugment, depending on implementation

For **milder time pooling**, the change is clearly architectural because pooling is part of the layer design.

For **SpecAugment**, the answer depends on implementation:

* if it is applied in the data pipeline before the model, it is not a model-architecture change
* if it is implemented as augmentation layers inside the model graph, then it does affect the model structure

So not every proposal item touches architecture, but some do.

---

### A.5. Practical interpretation for this proposal

For this document, the intended interpretation should be:

> The proposal preserves the current fixed-length log-mel + CNN modeling approach, while allowing limited internal refinements where needed to improve robustness and predictive quality.

This means the upgraded version should still be comparable to `baseline_logmel_cnn_v20.ipynb` in a meaningful way. It remains the same overall kind of system, but it is allowed to become slightly more refined internally.

That is consistent with the broader goal of this proposal: to evolve the baseline toward a more production-like version without replacing it with a fundamentally different solution.

---

### A.6. Suggested wording to reference in the main document

A concise statement that can be cited or reused in the main body of the document is:

> This proposal may introduce limited internal architectural refinements, but it does not change the overall model family. The solution remains a fixed-length log-mel + CNN model derived from the existing `baseline_logmel_cnn_v20.ipynb` baseline.

A slightly more formal variant is:

> The proposed upgraded version is intended to preserve the existing fixed-length log-mel + CNN architectural approach, while allowing selected internal refinements such as milder pooling behavior and other training or inference improvements.

These phrasings make the distinction explicit and help prevent misunderstandings.

---

### A.7. Final clarification

The simplest way to understand the proposal is this:

* **same model family**
* **possibly refined internal architecture**
* **same baseline lineage**
* **stronger training and inference behavior**
* **The current proposal does not inherently require adding more layers or change layer types.**

So yes, some limited architectural changes may occur, but only at a level that refines the existing model rather than changing what kind of model it is. That is the key point this appendix is intended to make.
