
# 🔍 Classifier-Guided Anomaly Detection with CAE (MVTec AD)

This repository implements a **two-stage anomaly detection pipeline** that combines image classification with feature-level reconstruction using **Convolutional Autoencoders (CAE)**.

The system first identifies the object category, then applies a **category-specific anomaly detector** trained only on normal samples. This design enables scalable and accurate multi-class industrial anomaly detection.

---

## 🚀 Pipeline Overview

```
Input Image
     ↓
ResNet-18 Classifier
     ↓
Predicted Category
     ↓
ResNet-50 Feature Extractor
     ↓
Category-Specific CAE
     ↓
Reconstruction Error Map
     ↓
Top-K Scoring + Threshold
     ↓
Normal / Anomalous
```

---

## 🧠 Core Components

### 1️⃣ Category Classifier (ResNet-18)

* Pretrained on ImageNet and fine-tuned on MVTec classes
* Predicts one of 15 object categories
* Used only for routing the image to the correct anomaly model

Supported categories:

```
bottle, cable, capsule, carpet, grid, hazelnut,
leather, metal_nut, pill, screw, tile,
toothbrush, transistor, wood, zipper
```

---

### 2️⃣ Feature Extraction (ResNet-50)

Instead of raw pixels, anomaly detection operates on **deep feature maps**.

Features are extracted from:

* `layer2` (512 channels)
* `layer3` (1024 channels)

Processing steps:

* 3×3 average pooling
* Spatial alignment via bilinear interpolation
* Channel-wise concatenation

Final feature tensor:

```
1536 channels = 512 + 1024
```

These feature maps preserve semantic and structural information useful for anomaly detection.

---

### 3️⃣ Category-Specific Convolutional Autoencoders (CAE)

Each object category has its **own CAE**, trained only on *normal samples*.

#### CAE design:

* Uses **1×1 convolutions only**
* Preserves spatial resolution
* Learns channel-wise dependencies

Architecture:

```
Encoder: 1536 → hidden → latent (100)
Decoder: latent → hidden → 1536
```

Each model is saved as:

```
cae_<category>.pth
```

During inference, anomalies cause poor reconstruction, producing high error values.

---

## 🔥 Anomaly Scoring

### Reconstruction Error

For feature map `F` and reconstruction `R`:

```
E = mean_c (F - R)²
```

This produces a spatial anomaly heatmap.

### Top-K Scoring

Instead of averaging all pixels, the system focuses on the most suspicious regions:

```
score = mean(top-10(E))
```

This improves sensitivity to small localized defects.

---

## 🎯 Threshold-Based Decision

Each category has its own threshold stored in `thresholds.json`.

Threshold computation:

```
T = μ + 3σ
```

Where μ and σ are computed from **normal training samples**.

Decision rule:

```
score ≥ T → anomaly
score < T → normal
```

---

## 🖼️ Visualization & Interface (Gradio)

The Gradio app provides:

* Image upload
* Predicted object category
* Anomaly score + threshold
* Binary decision (Normal / Defective)
* Heatmap visualization
* Overlayed anomaly regions

Required files:

```
best_classifier.pth
cae_<category>.pth
thresholds.json
```

---

## ✅ Key Features

* Modular classifier → detector pipeline
* Category-aware anomaly detection
* Deep feature–based reconstruction
* Lightweight CAE with fast inference
* Localized anomaly heatmaps
* Easily extensible to new categories
* Clean separation of training and inference

---

## 📌 Summary

This project combines **supervised classification** with **unsupervised anomaly detection** to build a scalable, category-aware inspection system. By routing inputs through specialized autoencoders and operating in feature space, it achieves robust detection and localization of defects.
