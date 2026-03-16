# Infrared Solar Modules dataset 

The **Infrared Solar Modules dataset** is a computer-vision dataset used to detect faults in solar photovoltaic (PV) panels using **thermal (infrared) images**. It is commonly used to train CNN models for **automatic solar panel inspection**.

---

## 1. What the dataset contains

The dataset consists of **infrared images of solar panels** captured during inspections of solar farms.

Key characteristics:

* **~20,000 thermal images**
* Image resolution: **24 × 40 pixels**
* Image type: **grayscale infrared (thermal)**
* Each image represents **one solar module**

Thermal cameras detect **temperature differences**, which helps reveal defects that are invisible in normal RGB images.

Faulty areas usually appear **hotter than surrounding cells**.

---

## 2. Classification labels

Each image is labeled into **12 categories** representing normal panels and different fault types.

Typical classes include:

| Class          | Meaning                       |
| -------------- | ----------------------------- |
| Cell           | Single-cell hotspot           |
| Cell-Multi     | Multiple hot cells            |
| Cracking       | Physical cracks in the panel  |
| Diode          | Bypass diode failure          |
| Diode-Multi    | Multiple diode faults         |
| Hot-spot       | Isolated hotspot              |
| Hot-spot-Multi | Multiple hotspots             |
| No-Anomaly     | Normal panel                  |
| Offline-Module | Entire module malfunction     |
| Shadowing      | Shade blocking sunlight       |
| Soiling        | Dust or debris covering panel |
| Vegetation     | Plants blocking panel         |

So the ML task is **multi-class image classification**.

---

## 3. Where the data comes from

The dataset was originally created by **Raptor Maps**, a company specializing in solar farm inspection.

Images were collected using:

* **drone-mounted infrared cameras**
* **handheld thermal cameras**

These inspections were performed on real solar installations, meaning the dataset reflects **real-world operating conditions**.

---

## 4. Why infrared images are used

Thermal imaging is ideal for solar fault detection because electrical faults produce heat patterns.

Examples:

* damaged cells → local hotspots
* diode failure → vertical heat stripes
* shading → irregular temperature distribution

A CNN can learn to recognize these patterns automatically.

---

## 5. Typical machine learning task

The dataset is commonly used for:

**Image classification**

Pipeline:

```
Infrared image
      ↓
CNN feature extraction
      ↓
Fully connected layer
      ↓
Softmax
      ↓
fault type prediction
```

Some research also uses it for:

* anomaly detection
* segmentation
* object detection in PV systems

---

## 6. Why the dataset is interesting

This dataset is popular because:

* real industrial inspection data
* small images → good for lightweight models
* suitable for **edge AI deployment**
* useful for **drone-based solar monitoring**

Many papers (including the **TinyTripleNet** paper you uploaded earlier) use this dataset.

---

## 7. A useful ML insight

Although these are **thermal images**, mathematically they are still just **2D arrays of numbers**.

That means the same CNN techniques used for:

* object detection
* medical imaging
* satellite analysis

can also be applied to **solar panel fault detection**.

In fact, the same CNN you use for **log-mel spectrograms in music classification** works for this dataset too — because spectrograms are also images.


---

# TinyTripleNet: Lightweight Deep Learning Architecture for Solar Panel Fault Detection (Rewritten)

## Abstract

The rapid expansion of solar energy systems creates a strong need for reliable and efficient fault detection methods in photovoltaic (PV) modules. Faults such as hot spots, cracks, and shading can reduce energy production and increase maintenance costs.

This work proposes **TinyTripleNet**, a lightweight convolutional neural network designed for **real-time solar panel fault classification on edge devices**. The architecture combines convolution layers with **Residual connections and Squeeze-and-Excitation (SE) attention blocks** to improve feature extraction while keeping the model compact.

TinyTripleNet contains **less than 0.9 million parameters**, making it suitable for embedded deployment. The model is trained using **binary cross-entropy loss and the Adam optimizer** on a thermographic PV dataset.

The model achieved the following classification accuracies:

* **96.71%** for 2-class classification
* **92.16%** for 8-class classification
* **90.25%** for 11-class classification
* **93.07%** for 12-class classification

Compared with larger deep learning models such as VGG16, ResNet50, and MobileNet, TinyTripleNet reduces **inference time by about 75%** and **memory usage by more than 80%**. On a Coral Edge TPU device, the model runs with an inference latency of **4.8 ms per image** and uses only **1.7 MB of runtime memory**.

These results demonstrate that TinyTripleNet provides an effective balance between accuracy, computational efficiency, and deployment practicality for **drone-based solar panel inspection systems**. 

---

# 1. Introduction

Solar energy is one of the most widely adopted renewable energy sources because it is environmentally friendly, scalable, and accessible in many regions. However, photovoltaic (PV) systems often suffer from various problems such as:

* partial shading
* dust accumulation
* aging of components
* environmental damage
* electrical faults such as short circuits or arc faults

These issues reduce system efficiency and increase maintenance costs.

Traditionally, PV fault detection relied on manual inspection or classical monitoring techniques. However, with the increasing size of solar farms, these approaches have become inefficient and time-consuming.

Deep learning methods have recently shown strong performance in image-based fault detection. Convolutional Neural Networks (CNNs) can automatically learn patterns from infrared images of solar panels and identify anomalies such as hot spots or cracks.

Despite their effectiveness, many deep learning models require powerful GPUs and large amounts of memory, which limits their use in field environments.

To address this limitation, this study introduces **TinyTripleNet**, a lightweight neural network designed specifically for **edge computing devices such as drones or embedded processors**. 

---

# 2. Dataset and Data Preparation

## Dataset Description

The experiments use the **Infrared Solar Module Dataset**, which contains approximately **20,000 infrared thermal images** of solar panels.

Each image has a resolution of **24 × 40 pixels** and belongs to one of **12 categories**, including normal panels and several fault types.

Examples of faults include:

* single-cell hotspots
* multi-cell hotspots
* cracks
* diode failures
* shadowing
* vegetation obstruction
* dust or soiling
* complete module failure

The dataset therefore includes **11 anomaly classes plus one normal class**.

After augmentation, the dataset contained about **80,000 images**, divided as follows:

* **64,000 training samples**
* **11,200 validation samples**
* **4,800 test samples**

These images were captured using infrared cameras mounted on drones or handheld devices, providing realistic inspection conditions. 

---

# 3. Data Preprocessing and Augmentation

Because PV datasets are often imbalanced and relatively small, several preprocessing techniques were applied to improve model performance.

### Image preprocessing

Images were processed using the following steps:

1. **Noise reduction**
   A Gaussian filter was applied to reduce noise in infrared images.

2. **Resizing and normalization**
   Images were resized to **224×224 during training** and normalized so pixel values fall between **0 and 1**.

3. **Dataset splitting**
   The dataset was divided into:

   * 70% training
   * 15% validation
   * 15% testing

### Data augmentation

To increase diversity and balance classes, several augmentation techniques were applied:

* horizontal and vertical flipping
* brightness adjustments
* translation of image pixels
* rotations up to 15 degrees

Pixel values were clipped to remain within the valid range after augmentation. These techniques help the model generalize better to different environmental conditions. 

---

# 4. TinyTripleNet Architecture

The proposed architecture takes **grayscale images with shape 40 × 24 × 1** as input.

The network consists of **three parallel processing paths**, each designed to extract features from the solar panel images.

The key architectural components include:

### Initial convolution layer

Each path begins with a **3×3 convolution with 64 filters**, followed by:

* batch normalization
* ReLU activation

### Residual blocks

Residual connections allow the network to learn deeper features without suffering from vanishing gradients.

The residual block combines:

* convolution layers
* shortcut connections
* activation functions

### Squeeze-and-Excitation (SE) blocks

SE blocks provide **channel-wise attention** by learning which feature maps are most important.

They work in two steps:

1. **Squeeze:** global average pooling summarizes spatial information.
2. **Excitation:** fully connected layers compute weights that scale each feature channel.

This mechanism allows the network to emphasize important features while suppressing irrelevant ones.

### Pooling layers

Max pooling layers reduce spatial dimensions and computation cost while preserving essential information.

### Global average pooling

Instead of flattening large feature maps, global average pooling converts each channel into a single value, reducing parameters and improving efficiency.

### Fully connected layer

The extracted features are merged and passed through a dense layer with **256 neurons and ReLU activation**. Dropout with a rate of **0.3** is applied to reduce overfitting.

### Output layer

The final layer uses **Softmax activation** to predict probabilities for the **12 fault classes**. 

---

# 5. Training and Experimental Setup

The model was trained using:

* **Adam optimizer**
* **Batch size:** 64
* **Epochs:** 50
* **Categorical cross-entropy loss**
* **Label smoothing**

Adam was chosen because it provides adaptive learning rates and faster convergence compared to SGD or RMSProp.

Training curves showed stable convergence, and validation accuracy closely matched training accuracy, indicating good generalization. 

---

# 6. Experimental Results

TinyTripleNet achieved strong performance across multiple classification tasks.

| Task     | Accuracy |
| -------- | -------- |
| 2-class  | 96.71%   |
| 8-class  | 92.16%   |
| 11-class | 90.25%   |
| 12-class | 93.07%   |

The model maintained high precision and recall across different types of faults.

Compared with traditional CNN models, TinyTripleNet provides a better trade-off between **accuracy and model size**.

The architecture contains **only 0.9 million parameters**, while models such as VGG16 require tens or hundreds of millions. 

---

# 7. Edge Deployment Performance

TinyTripleNet was deployed on the **Coral Dev Board Edge TPU**, which provides specialized hardware acceleration for neural networks.

Key deployment results:

* **Inference time:** 4.8 ms per image
* **Memory usage:** 1.7 MB
* **Power consumption:** ~2 W

These results demonstrate that the model can perform **real-time solar panel inspection directly on edge devices**, without needing cloud processing.

Such deployment enables practical systems where drones capture thermal images of solar farms and automatically detect faults in real time. 

---

# 8. Conclusion

TinyTripleNet provides an efficient solution for automated solar panel fault detection. By combining residual connections and SE attention blocks in a lightweight architecture, the model achieves high accuracy while maintaining extremely low computational cost.

The architecture demonstrates strong performance across multiple classification tasks and can operate efficiently on edge devices such as the Coral Edge TPU.

Because of its small size, low power requirements, and real-time inference capability, TinyTripleNet is well suited for **drone-based solar farm monitoring systems**.

Future work may focus on:

* improving robustness under difficult environmental conditions
* detecting previously unseen fault types
* integrating temporal analysis for monitoring fault progression over time. 
