First, tiny terminology cleanup: **“fault vs normal” is not multi-class**. It is **binary classification** made by collapsing the original **12 classes** of the Infrared Solar Modules dataset into **2 super-classes**:

* **Normal** = `No-Anomaly`
* **Fault** = everything else

In that dataset, the anomaly classes include **Cell, Cell-Multi, Cracking, Diode, Diode-Multi, Hot-spot, Hot-spot-Multi, Shadowing, Soiling, Vegetation, and Offline-Module**, while `No-Anomaly` is the healthy class. 

## What “fault” means here

For this dataset, a **fault** means:

> a solar module image shows an **undesired abnormal condition** in its thermal pattern, compared with a healthy panel.

That abnormality may come from:

* **electrical issues** like hot cells, diode problems, offline modules
* **physical damage** like cracking
* **operational/environmental problems** like shadowing, soiling, vegetation blockage

So in this dataset, “fault” is really closer to:

> **anomaly or problematic condition that can reduce performance or indicate risk**

That distinction matters. Some classes, like **shadowing** or **vegetation**, are not internal hardware failures in the strict engineering sense, but they are still labeled as problematic states for inspection and diagnosis, so they belong in the **fault** side for a binary setup. 

## When is an image labeled as fault?

In the binary version of the dataset:

* label it as **normal** only when the original class is **No-Anomaly**
* label it as **fault** whenever the original class is **any one of the 11 anomaly classes**

So the decision rule is very simple:

[
y =
\begin{cases}
0, & \text{if No-Anomaly} \
1, & \text{if any anomaly/fault class}
\end{cases}
]

A nice subtle point: the binary **fault** class is **heterogeneous**. It contains many different visual patterns, not one single pattern. That makes the problem easier than 12-class classification, but still not trivial, because “fault” includes many kinds of weirdness.

Also, in the dataset counts reported in the paper, `No-Anomaly` has about **10,000** images and the combined anomaly classes total about **10,006**, so the **binary version is almost perfectly balanced already**. That is great news and means you usually do **not** need aggressive class weighting for the 2-class setup. 

---

# Recommended CNN architecture

You asked for a model that balances **accuracy** and **computing cost**. For this binary task, I would **not** use a huge network. That would be bringing a bulldozer to rearrange a teacup.

The original images are very small, around **24×40 grayscale**, and the paper shows that compact models can already work very well on this data. 

## Recommendation: small residual CNN with light SE attention

Use a **small custom CNN** on the native or near-native resolution.

### Input

* `40 x 24 x 1` grayscale
  or
* `48 x 32 x 1` if you want a tiny bit more spatial room

I would start with **40×24×1** to preserve speed.

### Architecture

```text
Input (40, 24, 1)

Block 1
- Conv 3x3, 32 filters
- BatchNorm
- ReLU
- Conv 3x3, 32 filters
- BatchNorm
- ReLU
- MaxPool 2x2

Block 2
- Conv 3x3, 64 filters
- BatchNorm
- ReLU
- Conv 3x3, 64 filters
- BatchNorm
- ReLU
- Squeeze-and-Excitation (light)
- Residual/shortcut if shapes match
- MaxPool 2x2

Block 3
- Conv 3x3, 96 filters
- BatchNorm
- ReLU
- Conv 3x3, 96 filters
- BatchNorm
- ReLU
- Squeeze-and-Excitation (light)
- Residual/shortcut if shapes match
- MaxPool 2x2

Head
- GlobalAveragePooling
- Dense 64
- ReLU
- Dropout 0.25
- Dense 1 (logit)
```

## Why this is a good balance

### Why not something bigger?

Because this is only **binary classification** on **tiny thermal images**. You do not need ResNet50 to detect that a panel looks thermally cursed.

### Why residual connections?

They help optimization and make the network train more stably.

### Why SE blocks?

They add a small amount of channel attention, which is useful because faults often appear as **specific thermal cues** rather than big semantic objects.

### Why Global Average Pooling?

It greatly reduces parameter count and overfitting risk compared with flattening.

---

# Training setup I recommend

## 1. Labels

Map labels like this:

* `No-Anomaly` → `0`
* all other classes → `1`

## 2. Data split

Use a **stratified** split:

* **70% train**
* **15% validation**
* **15% test**

or

* **80% train**
* **10% validation**
* **10% test**

Stratified matters so both classes stay balanced across splits.

Do **not** augment before splitting.
Split first, then augment **train only**. Otherwise you risk leakage gremlins.

## 3. Preprocessing

* resize to chosen input size
* normalize pixels to `[0, 1]`
* keep as **1-channel grayscale**

## 4. Data augmentation

Use **mild** augmentation only:

* horizontal flip
* vertical flip
* small translation
* small rotation, around `±10°`
* slight brightness jitter
* maybe tiny Gaussian noise

Do not go wild. Thermal images are not Instagram.

## 5. Loss

Use:

* **Binary Cross Entropy with logits**

  * PyTorch: `BCEWithLogitsLoss`
  * Keras: `BinaryCrossentropy(from_logits=True)`

Because the binary dataset is already close to balanced, plain BCE is a good first choice.

## 6. Optimizer

Use:

* **AdamW**
* learning rate: `1e-3`
* weight decay: `1e-4`

That is a strong default.

## 7. Batch size

Start with:

* `64` if memory allows
* otherwise `32`

## 8. Epochs

Train for:

* `30–50 epochs`

with early stopping.

## 9. Early stopping and checkpointing

This part matters more than people expect.

Monitor:

* **validation F1 for the fault class**
* or **balanced accuracy**
* or **AUROC**

Do **not** use plain validation accuracy as the main checkpoint metric. Accuracy can flatter a model too easily.

Recommended:

* early stopping patience: `6–8`
* save best model by **val_fault_F1** or **val_balanced_accuracy**

## 10. Decision threshold

Default threshold is `0.5`, but for deployment I would tune it on validation data.

Why? Because in practice, **missing a fault** is usually worse than a false alarm.

So after training:

* scan thresholds like `0.3, 0.35, 0.4, ...`
* choose the one that gives the best tradeoff for your use case

For safety-oriented inspection, you often prefer a threshold that improves **fault recall**.

---

# Metrics you should care about

For this task, do **not** judge the model by accuracy alone.

Use:

* **Recall for fault**
* **Precision for fault**
* **F1-score for fault**
* **Balanced accuracy**
* **ROC-AUC**
* **Confusion matrix**

Why? Because the real question is not just “how often was I right?”
It is:

> “How often did I successfully catch faulty modules without drowning in false alarms?”

That is the practical engineering question.

---

# A very solid baseline setup

Here is the compact version I would actually start with:

## Model

* input: `40×24×1`
* 3 CNN blocks: `32 → 64 → 96`
* BN + ReLU
* light SE on blocks 2 and 3
* GAP
* Dense 64
* Dropout 0.25
* Dense 1 logit

## Training

* optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
* loss: `BCEWithLogitsLoss`
* batch size: `64`
* epochs: `40`
* early stopping on `val_fault_f1`
* mild augmentation on train only
* threshold tuning after training

That setup should give a very reasonable balance between:

* **good accuracy**
* **low overfitting risk**
* **fast training**
* **low inference cost**

---

# If you want an even cheaper model

Use this:

```text
Input
Conv(32) -> BN -> ReLU -> MaxPool
Conv(64) -> BN -> ReLU -> MaxPool
Conv(96) -> BN -> ReLU -> MaxPool
GAP
Dense(32) -> ReLU
Dropout(0.2)
Dense(1)
```

This is cheaper, simpler, and a good first baseline.
Then add residual + SE only if needed.

---

# If you want a bit more accuracy without exploding compute

Use a **MobileNetV3-Small** style backbone adapted for grayscale input.

That is a strong option, but for this dataset size and image size, a **small custom CNN** is often cleaner and more compute-efficient.

---

# My recommendation in one sentence

For **Infrared Solar Modules fault-vs-normal**, define **fault** as **any non–No-Anomaly abnormal condition**, then use a **small residual CNN with light SE attention, GAP, AdamW, BCEWithLogitsLoss, mild train-only augmentation, and checkpoint on validation fault F1**.

That is the sweet spot. Lean, practical, and not wearing a tuxedo to fix a roof.

I can turn this into a **ready-to-run TensorFlow or PyTorch model script** next.
