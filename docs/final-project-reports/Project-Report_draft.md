# Machine Learning 1 (MLE501.22), MSA30DN, FSB | Nguyen Sy Hung | Mar 2026

# Deep Learning for Music Genre Prediction Using Log-Mel Spectrograms and CNNs

**Nguyễn Sỹ Hùng (25MSA33055)**

**Machine Learning 1 (MLE501.22), FSB – Lecturer: PhD. Nguyen Thi Kim Truc**

## Abstract

This report presents my final project for the Machine Learning 1 course.

The project established a comparation between the solutions:  
1) MFCC + CNN baseline vs  
2) LogMel + CNN baseline – which actually compares 2 types of input data representation.

From there, I successfully delivered a production-like prototype solution: an end-to-end, reproducible of collecting data and analysis of the dataset from raw to model-training-ready consumed data, to mode training, to evaluation and any demo application that consume the model (inference) and applied to a Music Genres Classification application

**Keywords.**

## Introduction

## Project Overview

Music genre classification is difficult because a “genre” is not defined in a fully exact or scientific way. Music does have measurable properties such as rhythm, tempo, beat, pitch, and frequency patterns, so it can be analyzed by computers. However, genre labels are also shaped by human perception, culture, place, time, and the fact that many songs mix multiple styles. Because of that, one song may fit more than one genre, and genre labels should be treated as approximate categories rather than absolute truth.

This project treats music genre classification as a practical supervised learning problem. The goal is to train a deep learning model to predict the dominant genre of a 15-second music segment. To do this, the audio is converted into log-Mel spectrograms, which are compact time-frequency representations of sound. These spectrograms are then used as input to a Convolutional Neural Network (CNN), which learns patterns related to genre from labeled training data.

This project is mainly for learning and practical exploration. It is not intended to produce a publishable research result or a fully commercial system. Instead, it focuses on applying course knowledge, instructor guidance, and self-study to a realistic problem, while improving understanding of data preparation, model design, training, evaluation, and result analysis.

## Problem Statement

The main problem in this project is to classify a n-second music segment into a genre using deep learning. More specifically, the system should predict the dominant genre label from the audio.

This is hard because music is a complex signal, and genre labels themselves are often unclear or overlapping. Different genres may share similar sound patterns, and short audio clips may not always contain enough information for perfect classification.

To handle this, the project converts audio into log-Mel spectrograms and uses CNN models to learn genre-related patterns from them. The model does not try to define the true meaning of music genre. Instead, it tries to learn from the labeled dataset and make the best possible prediction of the dominant genre.

A key question behind this project is whether it is possible to train a compact music genre classification model, using personal computing resources, that still gives useful results. This question is important because it affects choices about model size, architecture, training setup, and the balance between accuracy, time, and computing cost.

## Goals and Objectives

The main goals of this project are:

- Prepare the dataset, preprocessing pipeline, and exploratory data analysis (EDA)
- Re-engineer the provided MFCC + CNN baseline for a better speed performance (the original training script for the MFCC + CNN baseline runs quite slow)
- Build a Log-Mel + CNN baseline with a similar setup for fair comparison
- Build a more practical, production-like Log-Mel + CNN version
- Compare and benchmark the different model versions
- Evaluate the results and analyze strengths and weaknesses
- Develop a demo application
- Explore whether a compact model can still be useful in practice

Overall, the project moves step by step from a basic instructional baseline toward a stronger proof-of-concept model.

## III. Scope and Constraints

This project is limited by time, effort, hardware, and computing resources. It is a final course assignment, so its purpose is to show how the learned concepts can be applied in practice, not to build a full industrial product.

The project assumes a home-lab setup with a normal consumer GPU, not a large cloud or multi-GPU system. Training time should stay within a few hours, or at most around 24 hours, not multiple days. Because of this, the model must stay reasonably small and efficient.

Also, this is not a safety-critical system. The model is expected to classify at least eight common music genres, and occasional mistakes are acceptable because they do not cause serious harm. In practice, the model is meant to be a supportive tool for tasks such as tagging, browsing, or lightweight recommendation support, rather than a perfect decision-making system.

## Methodology and Experimental Setup

This project follows a step-by-step experimental approach. It begins with dataset preparation, preprocessing, and exploratory data analysis (EDA). Audio clips are converted into fixed n-second segments and represented as log-Mel spectrograms for CNN-based training. For comparison, the project first uses the provided MFCC + CNN baseline trained on the small FMA dataset. Based on this reference, a new Log-Mel + CNN baseline is defined to remain as close as possible to the original model, with only minor adjustments required by the different input representation. This enables a more controlled comparison of how MFCC and log-Mel features affect model performance.

Model development then proceeds incrementally. After re-engineering and understanding the baseline, the project extends the work by building a custom dataset from several open datasets together with manually collected and processed audio clips. The model is then further improved into a more production-like version through practical refinements in preprocessing, architecture, and training design. This version is finally demonstrated through a proof-of-concept application. The models are trained, benchmarked, and evaluated under realistic personal computing constraints, and the results are analyzed to assess both their practical usefulness and their limitations.

## Collecting Data, Building a Custom Dataset, and EDA

(todo...)

## Baselines: MFCC vs Log-Mel

### Model Architectures & Training Setup

- MFCC version (provided via the FMA dataset)

(todo...)

- Log-Mel CNN (my own defined)

(todo...)

- Training Setup
(todo...)

### Benchmarking and Evaluation - Comparation Result

(todo...)

## Production-like Neural Network Model & Music Genre Prediction Application

(todo: a brief of paragraph to introduce...)

### Model Architecture

(todo...)

### Results and Evaluation

(todo...)

### Demo Applications (Consumer Layer)

(todo: a brief of description and some screenshots)

## Discussion and Reflection

(todo...)

## Conclusion

(todo...)

## References

(todo...)