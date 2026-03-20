## Log-Mel Pipeline: Core Mathematical Expression

The essence of the script’s log-Mel feature extraction can be summarized by:

[
L[r,m] = \log\left(1 + \sum_k H_r[k]\left|\sum_n \tilde{y}[mH+n],w[n],e^{-j2\pi kn/L}\right|^2\right)
]

Where:

* (L[r,m]): final **log-Mel spectrogram** value at Mel band (r) and time frame (m)
* (\tilde{y}[mH+n]): fixed-length audio waveform, sampled at the current frame position
* (H): **hop length**, controlling how far the window moves each step
* (w[n]): analysis **window function** applied to the frame
* (\sum_n \tilde{y}[mH+n] w[n] e^{-j2\pi kn/L}): **STFT / FFT-based frequency analysis** of one short audio frame
* (\left| \cdot \right|^2): converts the complex spectrum into **power / energy**
* (H_r[k]): **Mel filterbank** weights that map FFT bins (k) into Mel band (r)
* (\sum_k H_r[k](\cdot)): computes the **Mel-band energy**
* (\log(1+\cdot)): **log compression**, reducing dynamic range and improving numerical stability

In short, the pipeline takes a short window of waveform, converts it into frequency-domain energy using FFT, groups that energy into perceptual Mel bands, and then applies logarithmic compression to produce the final log-Mel representation.
