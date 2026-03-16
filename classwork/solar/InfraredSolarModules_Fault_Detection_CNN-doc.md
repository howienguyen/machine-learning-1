# Explanation and reasoning behind the solution

## 1. This is still binary classification, but encoded as 2-class softmax
The original dataset contains multiple fault categories plus one healthy class.

For this notebook, the labels are collapsed into:

- **normal** = `No-Anomaly`
- **fault** = all other classes

That means the *task* is still binary in meaning, but the *model output* is now **categorical with 2 classes**.

Why do this?

Because a 2-class softmax directly returns both probabilities:

- `P(normal | image)`
- `P(fault | image)`

So you can get an output like:

- `fault = 0.80`
- `normal = 0.20`

This is exactly what you asked for.

---

## 2. Why switch from one logit to softmax
A single-logit binary head is perfectly valid, but it naturally gives you one main probability, usually the probability of the positive class.

A **2-class softmax** is often easier to explain in reports and demos because it explicitly shows the model's belief over both categories.

So this notebook uses:

```text
Dense(2) -> Softmax
```

and trains with:

- **SparseCategoricalCrossentropy**

instead of:

- binary cross-entropy with logits

---

## 3. Why the notebook now reads `module_metadata.json`
The Infrared Solar Modules dataset is provided through a metadata file where each module record contains fields such as:

- `image_filepath`
- `anomaly_class`

That means the dataset is not naturally organized as one folder per class.  
So the notebook now loads the images by following the metadata file and then derives the 2-class target from `anomaly_class`.

This is the correct alignment for the dataset format.

---

## 4. Why stratify by original anomaly class
Even though the final task is only `normal` vs `fault`, the source dataset still contains many different anomaly subclasses such as cracking, diode faults, shadowing, soiling, and so on.

If we stratify only on the final binary label, some rare original fault types could be distributed badly across splits.

So the split is stratified by the **original class first**, which preserves the source distribution more faithfully.

---

## 5. Why use the same XPU runtime selection style as `logmel_cnn_v2_2.py`
The notebook now mirrors the runtime-selection logic used in `model_training/logmel_cnn_v2_2.py`:

1. try CUDA GPU first
2. if not available, try Intel XPU through `intel_extension_for_tensorflow`
3. otherwise fall back to CPU

It also performs a small **smoke-test matrix multiplication** on the selected device.

This is useful because it makes the notebook behave more like your existing training stack and gives you a practical path to Intel XPU acceleration when the environment supports it.

---

## 6. Why keep the CNN compact
The thermal images in this dataset are small, so a giant backbone would often add computation faster than it adds useful signal.

This compact CNN is a good balance between:

- accuracy
- speed
- memory usage
- deployment practicality

Architecture:

```text
Conv(32) -> BN -> ReLU -> MaxPool
Conv(64) -> BN -> ReLU -> MaxPool
Conv(96) -> BN -> ReLU -> MaxPool
GAP
Dense(32) -> ReLU
Dropout(0.2)
Dense(2) -> Softmax
```

It stays light, but still has enough capacity to separate normal thermal patterns from abnormal ones.

---

## 7. Why GAP is still a good choice
**Global Average Pooling (GAP)** replaces each final feature map with one average value.

This gives you:

- fewer parameters
- lower overfitting risk
- lower compute cost
- cleaner deployment behavior

In plain language, GAP asks each learned feature map:

> “Across the whole image, how strongly did this pattern appear?”

That is a very sensible question for fault-vs-normal thermal classification.

---

## 8. Why only light augmentation
Thermal imagery is more delicate than natural-image photography.  
Aggressive augmentation can distort the thermal meaning of the sample.

So this notebook keeps augmentation modest:

- flips
- slight brightness jitter

That gives extra variety without turning the data into nonsense soup.

---

## 9. Which metrics matter most
Even though the model is trained with softmax, the practical evaluation question is still:

> how well does it detect faulty modules?

So the key metrics remain:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Confusion matrix**

And because the model outputs two probabilities, you can also inspect uncertain examples more easily.

---

## 10. Bottom line
This notebook now does three aligned things:

1. loads the dataset the way it is actually provided through `module_metadata.json`
2. predicts with a **2-class softmax** so you get explicit `fault` and `normal` probabilities
3. uses the same **XPU/GPU/CPU runtime selection pattern** as your `logmel_cnn_v2_2.py`

So the result is cleaner, more consistent with your training stack, and more demo-friendly.
