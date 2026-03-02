### Introduction to **FMA-medium** (Free Music Archive – medium subset)

**FMA-medium** is a curated subset of the **Free Music Archive (FMA)** dataset designed for **music information retrieval (MIR)** and ML tasks like **genre recognition**. The broader FMA project was created to provide a large, openly usable collection of **Creative Commons–licensed** music with **standard splits**, **precomputed features**, and rich metadata—so researchers can train and compare models without reinventing the dataset plumbing every time. ([arXiv][1])

#### Core characteristics / properties

* **Audio content:** **25,000 tracks**, each **30 seconds** long, distributed as MP3 audio (this fixed clip length is great for consistent feature extraction and batching). ([Kaggle][2])
* **Genre labeling:** **16 “unbalanced” genres** in the medium subset (i.e., class counts are not equal—realistic, but it affects training/evaluation choices). ([Kaggle][2])
* **Scale & storage:** roughly **~22 GiB** for the medium audio archive (practical for local development compared to the large/full releases). ([Kaggle][2])
* **Metadata + features:** the official `fma_metadata.zip` includes structured tables (e.g., track/artist/album info, genre taxonomy, and a recommended split), and the dataset release is designed to support reproducible experiments. ([GitHub][3])
* **Intended use:** benchmarking and experimentation across MIR tasks, especially **genre recognition**, with baselines provided in the project materials/paper. ([arXiv][1])

#### Why FMA-medium is a sweet spot for ML projects

FMA-medium is big enough to train meaningful neural baselines (CNNs on log-mel / MFCC, etc.) while still being small enough to iterate quickly on a single machine. And because the overall FMA ecosystem includes a genre hierarchy + official splits + metadata, you can scale your project up later (to FMA-large/full) without changing the whole pipeline—just swapping the subset. ([arXiv][1])

[1]: https://arxiv.org/abs/1612.01840?utm_source=chatgpt.com "FMA: A Dataset For Music Analysis"
[2]: https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium/data?utm_source=chatgpt.com "FMA - Free Music Archive - Small & Medium"
[3]: https://github.com/mdeff/fma?utm_source=chatgpt.com "mdeff/fma - A Dataset For Music Analysis"
