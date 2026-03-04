Mon Mar  2 14:09:18 UTC 2026

For **data understanding + EDA** on **FMA**, the best starting point is actually already in the official repo:

### 1) **`analysis.ipynb` (official, EDA-focused)**

This is the “true EDA notebook” of the project: it explores **metadata, splits, and the provided features**, and it’s the notebook used to generate figures for the original paper. If your goal is *understand what’s inside FMA and how it’s structured*, this is the one. 

What you’ll learn from it (typical flow):

* what the metadata tables contain (tracks/artists/albums/genres)
* how the **train/validation/test splits** are defined
* label structure (genre hierarchy, imbalance, etc.)
* basic distribution sanity checks + plots
* how the provided feature matrices look and behave

### 2) **`usage.ipynb` (official, “how to load and not suffer”)**

Not pure EDA, but it’s the cleanest “getting oriented” notebook: loading the dataset, using the helper utilities, and making sure you can actually access tracks/metadata/features correctly. I treat it as the “dataset boot sequence.” ([GitHub][1])

### 3) Optional but useful: **`baselines.ipynb` (official, sanity-check modeling)**

Not EDA, but great for “does this dataset behave like reality?” It runs baseline classifiers on provided features and some deep learning setups; useful as a **reference point** after EDA so you know what performance is “normal.” ([GitHub][2])

---

## If you want a more “EDA + audio intuition” notebook (community)

If you want EDA that includes **raw waveform / spectrogram poking**, one example repo explicitly calls out an `analysis.ipynb` used for EDA on raw audio + mel spectrograms. This can complement the official metadata-focused EDA nicely. ([GitHub][3])

## If you’re doing “biggish data” EDA (PySpark-style)

There’s also a PySpark tutorial notebook built around FMA, useful if your EDA starts hitting memory/scale pain (especially with medium/large). ([GitHub][4])

---

### My practical recommendation (so you don’t wander the dataset maze forever)

* **Start with `usage.ipynb`** to confirm paths + loaders work. ([GitHub][1])
* Then do **`analysis.ipynb`** as your core “dataset understanding + EDA.” ([GitHub][1])
* If you’re working with **FMA-medium/large** and feel your laptop sweating, borrow ideas from the **PySpark notebook** for scalable EDA patterns. ([GitHub][4])

The meta-lesson: with FMA, **most confusion is metadata/schema confusion**, not model confusion—so front-load that EDA and your whole project becomes less cursed.

[1]: https://github.com/mdeff/fma?utm_source=chatgpt.com "mdeff/fma - A Dataset For Music Analysis"
[2]: https://github.com/mdeff/fma/blob/master/baselines.ipynb?utm_source=chatgpt.com "fma/baselines.ipynb at master"
[3]: https://github.com/saiteki-kai/music-classification?utm_source=chatgpt.com "saiteki-kai/music-classification"
[4]: https://github.com/juanmanuel-tirado/pyspark-tutorial/blob/main/pyspark_fma.ipynb?utm_source=chatgpt.com "pyspark-tutorial/pyspark_fma.ipynb at main"
