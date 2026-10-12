# Technical Model Explanation & Architecture Guide
**Project: Multi-Modal Plant Disease Classification (v3)**  
*Document prepared for Project Evaluation & Viva Defense*

---

## 1. System Overview & Problem Formulation

Agricultural plant disease diagnosis traditionally relies solely on leaf imagery. However, leaf symptoms often overlap across different diseases or are conditioned by specific environmental factors (temperature, humidity, rainfall). 

This project implements a **Multi-Modal Deep Learning Framework** that fuses:
1. **Visual Features ($I$)**: RGB leaf images ($300 \times 300 \times 3$) processed via a Convolutional Neural Network backbone (**EfficientNetV2-S**).
2. **Meteorological Features ($W$)**: 7-day historical weather sequences ($7 \times 11$) processed via a 1D Convolutional + Multi-Head Self-Attention Network (**WeatherEncoder**).

The system classifies plant conditions across **4 major crops** (Wheat, Rice, Sugarcane, Cotton) into **33 unified disease and healthy classes**.

```
                         ┌───────────────────────────┐
  Leaf Image [3x300x300] │ EfficientNetV2-S Backbone │ ──> [512] ──┐
                         └───────────────────────────┘             │
                                                                   ├──> Concatenate [640] ──> Classifier FC Layers ──> Softmax [33 Classes]
                         ┌───────────────────────────┐             │
  Weather Data [7x11]    │  1D CNN + Self-Attention  │ ──> [128] ──┘
                         └───────────────────────────┘
```

---

## 2. Data Pipeline & Weather Feature Engineering

### 2.1 Weather Data Collection Strategy (`data_collection.py`)
To prevent artificial memorization and ensure high generalization across geographic regions, **v3** uses a **Stratified Profile Pool Strategy**:
* **Pool Generation**: 91 validated historical weather profiles generated per class ($91 \times 33 = 3,003$ total weather files).
* **Geographic & Temporal Diversity**: Each profile pool is forced to span $\ge 3$ distinct historical years (from $[2020, 2021, 2022, 2023, 2024]$) and $\ge 3$ distinct agricultural states/districts in India.
* **Spatial Centroids**: District latitude/longitude coordinates extracted from ICRISAT agricultural dataset & shapefiles (`district_centroids.json`).
* **Open-Meteo Historical API**: Daily aggregation of hourly historical weather observations for specific district coordinates and target date ranges.

### 2.2 Epidemiological Windowing & Favorable Condition Validation (`utils.py`)
Diseases do not occur randomly; pathogens require specific epidemiological thresholds for spore germination and lesion development.

* **7-Day Weather Window**: Captures acute pre-symptomatic weather conditions leading to disease onset.
* **Agronomic Rule-Based Validation**: Weather profiles fetched from Open-Meteo must satisfy biological criteria for at least **$\ge 3$ days out of 7** (`MIN_FAVORABLE_DAYS = 3`).
  * *Example (Wheat Yellow Rust)*: Temperature $8^\circ\text{C} \le T_{\text{mean}} \le 18^\circ\text{C}$ and Relative Humidity $\text{RH}_{\text{mean}} > 75\%$.
  * *Example (Rice Leaf Blast)*: Temperature $22^\circ\text{C} \le T_{\text{mean}} \le 30^\circ\text{C}$ and Relative Humidity $\text{RH}_{\text{mean}} > 85\%$.
  * *Example (Sugarcane Red Rot)*: Temperature $24^\circ\text{C} \le T_{\text{mean}} \le 35^\circ\text{C}$ and Relative Humidity $\text{RH}_{\text{mean}} > 75\%$.

### 2.3 11 Meteorological Features
Each day in the 7-day sequence contains **11 aggregated parameters**:
1. `temp_max`: Daily maximum 2m air temperature ($^\circ\text{C}$)
2. `temp_min`: Daily minimum 2m air temperature ($^\circ\text{C}$)
3. `temp_mean`: Daily average 2m air temperature ($^\circ\text{C}$)
4. `rh_max`: Daily maximum relative humidity ($\%$)
5. `rh_mean`: Daily average relative humidity ($\%$)
6. `precip_sum`: Cumulative daily precipitation ($\text{mm}$)
7. `dewpoint_mean`: Daily average dew point temperature ($^\circ\text{C}$)
8. `wind_max`: Daily maximum wind speed at 10m ($\text{m/s}$)
9. `soil_temp`: Daily average surface soil temperature ($^\circ\text{C}$)
10. `month_sin`: $\sin\left(\frac{2\pi \cdot \text{month}}{12}\right)$ (Cyclic seasonal sine encoding)
11. `month_cos`: $\cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$ (Cyclic seasonal cosine encoding)

> **Why Cyclic Month Encoding?**  
> Representing months as raw numbers ($1, 2, \dots, 12$) introduces an artificial discontinuity between December ($12$) and January ($1$). Sine-Cosine transform embeds time on a continuous unit circle so the model understands Dec-Jan proximity.

### 2.4 Anti-Leakage Profile Assignment
* **District-Level Split**: 70% Train, 15% Validation, 15% Test.
* **Max Reuse Cap**: $\text{Cap} = \lceil \frac{N_{\text{class\_images}}}{N_{\text{pool}}} \times 1.5 \rceil \approx 3\%$ maximum assignment per profile per class.
* **Randomized Round-Robin**: Prevents identical profiles from clustering within sequential image batches.

---

## 3. Deep Learning Architecture (`train_multimodal.py`)

### 3.1 Image Encoder: EfficientNetV2-S
* **Pretrained Weights**: ImageNet-1k (`tf_efficientnetv2_s.in1k`).
* **Architecture**: Fused-MBConv (Mobile Inverted Bottleneck Convolution) blocks utilizing Squeeze-and-Excitation (SE) attention units and progressive learning scaling.
* **Dimensionality Reduction**: Extracts a 1280-dimensional feature vector, projected via a Dense layer to **512 dimensions** followed by ReLU activation and 40% Dropout.

$$\mathbf{f}_{\text{img}} = \text{Dropout}_{0.4}\left(\text{ReLU}\left(\mathbf{W}_{\text{img}} \cdot \text{EfficientNetV2}(I) + \mathbf{b}_{\text{img}}\right)\right) \in \mathbb{R}^{512}$$

### 3.2 Weather Encoder: 1D CNN + Multi-Head Self-Attention
Sequential weather data requires modeling local multi-variable correlations (via Conv1D) and long-range temporal trends (via Self-Attention).

1. **Temporal 1D Convolutional Blocks**:
   $$\text{Conv1D}(11 \rightarrow 64, k=3, p=1) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Conv1D}(64 \rightarrow 128, k=3, p=1) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU}$$
   * Output shape: $[B, 128, 7]$ (128 channels across 7 timesteps).

2. **Multi-Head Self-Attention (MHSA)**:
   * Uses $h = 4$ attention heads over $d_{\text{model}} = 128$ dimensions.
   * Transposed to $[B, 7, 128]$ so attention is computed **across timesteps**:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

3. **Residual Connection & Layer Normalization**:

$$\mathbf{X}_{\text{attn}} = \text{LayerNorm}\left(\mathbf{X} + \text{MHSA}(\mathbf{X})\right)$$

4. **Global Pooling & Projection**:
   * Adaptive Average Pooling over timesteps $\rightarrow [B, 128]$.
   * Fully Connected layer projects to **128 dimensions**.

$$\mathbf{f}_{\text{wx}} = \mathbf{W}_{\text{wx}} \cdot \text{Pool}(\mathbf{X}_{\text{attn}}) + \mathbf{b}_{\text{wx}} \in \mathbb{R}^{128}$$

### 3.3 Fusion & Classification Head
* **Late Fusion (Concatenation)**: Combines visual and weather vectors:

$$\mathbf{f}_{\text{fused}} = [\mathbf{f}_{\text{img}} \,||\, \mathbf{f}_{\text{wx}}] \in \mathbb{R}^{640}$$

* **Classification Head**:
  $$\text{Dense}(640 \rightarrow 256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.5) \rightarrow \text{Dense}(256 \rightarrow 33)$$

---

## 4. Advanced Training Techniques & Optimization

### 4.1 Two-Phase Training Strategy
1. **Phase 1 — Head Warmup (5 Epochs)**:
   * Image encoder backbone is frozen (`requires_grad = False`).
   * Optimizer: AdamW ($lr = 3 \times 10^{-4}$, weight decay $= 10^{-3}$).
   * LR Scheduler: `ReduceLROnPlateau` (factor $= 0.5$, patience $= 3$).
   * *Purpose*: Prevents gradient corruption of pretrained image weights while training random weights in fusion classifier and weather encoder.
2. **Phase 2 — Backbone Fine-Tuning (50 Epochs)**:
   * Top 40% of EfficientNetV2 convolutional blocks unfrozen (`unfreeze_backbone(model, fraction=0.40)`).
   * Optimizer: AdamW ($lr = 2 \times 10^{-4}$, weight decay $= 10^{-3}$).
   * LR Scheduler: `CosineAnnealingWarmRestarts` ($T_0 = 10, T_{\text{mult}} = 2, \eta_{\min} = 10^{-6}$).

### 4.2 Data Augmentations & Regularization
* **Visual Augmentations**: `RandomResizedCrop(300, scale=(0.75, 1.0))`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)`.
* **Weather Jittering (Um et al., 2017)**: Additive Gaussian noise ($\sigma = 0.03$) added to normalized weather tensors during training:
  $$\mathbf{W}_{\text{train}} = \mathbf{W}_{\text{norm}} + \mathcal{N}(0, \sigma^2 \mathbf{I})$$
  *Prevents overfitting to exact synthetic floating-point values while preserving seasonal trends.*
* **Mixup Regularization ($\alpha = 0.2$)**: Blends pairs of training examples and their target labels using a Beta distribution:
  $$\lambda \sim \text{Beta}(\alpha, \alpha)$$
  $$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1 - \lambda) \mathbf{x}_j, \quad \tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1 - \lambda) \mathbf{y}_j$$
* **Inverse Class Frequency Weighting**: Rebalances loss to prevent majority class dominance:
  $$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
* **Label Smoothing ($\epsilon = 0.1$)**: Softens target labels to prevent overconfident logit outputs:
  $$y^{\text{smooth}}_c = (1 - \epsilon) y_c + \frac{\epsilon}{K}$$
* **Early Stopping**: Patience of 8 epochs tracking validation accuracy.
* **Mixed Precision Training**: PyTorch `torch.amp.autocast` + `GradScaler` for faster FP16 forward pass and GPU memory optimization.

---

## 5. Explainable AI (XAI) & Interpretability (`xai_analysis.py`)

1. **Visual Interpretability (Grad-CAM)**:
   * Computes gradients of the target class logit with respect to the feature maps of EfficientNetV2's final convolutional layer (`conv_head`).
   * Generates spatial heatmaps showing leaf spots, rust pustules, or lesions driving predictions.

2. **Weather Interpretability (Attention Weight Extraction)**:
   * Extracts $7 \times 7$ attention weight matrix $\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)$ from `WeatherEncoder.attn`.
   * Maps which of the 7 daily timesteps exhibited the highest cross-attention, providing agronomists with insights into peak weather trigger days.

---

## 6. Model Ablation Study (Multimodal vs. Single-Modal)

The training script supports 3 variants (`--variant multimodal|image-only|weather-only`):
* **Multimodal (Full)**: Leaf Image + 7-day Weather Fusion ($\mathbf{f}_{\text{img}} || \mathbf{f}_{\text{wx}}$).
* **Image-Only Baseline**: EfficientNetV2-S operating on leaf images alone.
* **Weather-Only Baseline**: WeatherEncoder operating on 7-day weather sequences alone.

*Key Takeaway for Viva*: Multi-modal fusion resolves visually ambiguous classes (e.g., Early vs. Late Blight, or Healthy vs. Mildew under suboptimal lighting) by using meteorological cues as a strong secondary prior.
