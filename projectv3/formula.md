# Mathematical Formulas & Evaluation Metrics Guide
**Project: Multi-Modal Plant Disease Classification (v2 & v3)**  
*Comprehensive Mathematical Reference for Performance Evaluation & Viva Defense*

---

## 1. Classification Performance Metrics

All evaluation metrics in `projectv2` and `projectv3` are computed using `sklearn.metrics.classification_report`, `confusion_matrix`, and `f1_score`.

### 1.1 Confusion Matrix Building Blocks
For a multi-class problem with $K = 33$ classes, we evaluate each class $c \in \{1, 2, \dots, K\}$ using a One-vs-Rest binary decomposition:

* **True Positives ($TP_c$)**: Number of samples belonging to class $c$ correctly predicted as class $c$.
* **False Positives ($FP_c$)**: Number of samples NOT belonging to class $c$ incorrectly predicted as class $c$.
* **False Negatives ($FN_c$)**: Number of samples belonging to class $c$ incorrectly predicted as another class.
* **True Negatives ($TN_c$)**: Number of samples NOT belonging to class $c$ correctly predicted as NOT class $c$.

---

### 1.2 Per-Class Metrics

#### Precision ($P_c$)
Measures the proportion of positive identifications that were actually correct.

$$P_c = \frac{TP_c}{TP_c + FP_c}$$

#### Recall / Sensitivity / True Positive Rate ($R_c$)
Measures the proportion of actual positives that were correctly identified.

$$R_c = \frac{TP_c}{TP_c + FN_c}$$

#### F1-Score ($F1_c$)
Harmonic mean of Precision and Recall. Gives a balanced score even under class imbalance.

$$F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c} = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$$

*Note: If $TP_c + FP_c = 0$ or $TP_c + FN_c = 0$, $F1_c$ uses `zero_division=0` setting $F1_c = 0$.*

---

### 1.3 Aggregate Global Metrics

#### Overall Accuracy ($\text{Acc}$)
Proportion of total correct predictions across all classes $N_{\text{total}} = \sum_{c=1}^K (TP_c + FN_c)$.

$$\text{Accuracy} = \frac{\sum_{c=1}^K TP_c}{N_{\text{total}}} = \frac{\text{Total Correct Predictions}}{\text{Total Test Samples}}$$

#### Macro Average Metric ($\text{Macro } Metric$)
Unweighted mean of metric values across all $K$ classes. Treats all classes equally regardless of sample size.

$$\text{Macro Precision} = \frac{1}{K} \sum_{c=1}^K P_c$$

$$\text{Macro Recall} = \frac{1}{K} \sum_{c=1}^K R_c$$

$$\text{Macro F1} = \frac{1}{K} \sum_{c=1}^K F1_c$$

#### Weighted Average Metric ($\text{Weighted } Metric$)
Weighted mean of metric values across all $K$ classes, weighted by class support $N_c = TP_c + FN_c$. Accounts for class imbalance.

$$\text{Weighted Precision} = \sum_{c=1}^K \left( \frac{N_c}{N_{\text{total}}} \cdot P_c \right)$$

$$\text{Weighted Recall} = \sum_{c=1}^K \left( \frac{N_c}{N_{\text{total}}} \cdot R_c \right)$$

$$\text{Weighted F1} = \sum_{c=1}^K \left( \frac{N_c}{N_{\text{total}}} \cdot F1_c \right)$$

#### Micro Average Metric ($\text{Micro } Metric$)
Aggregates total $TP$, $FP$, and $FN$ globally across all classes prior to computing metrics.

$$\text{Micro Precision} = \frac{\sum_{c=1}^K TP_c}{\sum_{c=1}^K (TP_c + FP_c)}$$

$$\text{Micro Recall} = \frac{\sum_{c=1}^K TP_c}{\sum_{c=1}^K (TP_c + FN_c)}$$

$$\text{Micro F1} = 2 \cdot \frac{\text{Micro Precision} \cdot \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}$$

*(Note: For single-label multi-class classification where every sample is assigned exactly one prediction, $\text{Micro Precision} = \text{Micro Recall} = \text{Micro F1} = \text{Overall Accuracy}$.)*

---

## 2. Loss Functions & Training Formulations

### 2.1 Categorical Cross-Entropy Loss with Inverse Class Weighting
For sample $i$ with true class label $y_i \in \{1, \dots, K\}$ and predicted probability vector $\hat{\mathbf{y}}_i = \text{Softmax}(\mathbf{z}_i)$:

$$L_{\text{CE}}(i) = - w_{y_i} \cdot \log(\hat{y}_{i, y_i})$$

Where class weights $w_c$ are calculated to balance under-represented classes:

$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$

---

### 2.2 Label Smoothing Regularization ($\epsilon = 0.1$)
To prevent the model from becoming overconfident in its logit predictions, hard target vectors $\mathbf{y} = [0, \dots, 1, \dots, 0]^T$ are converted to smooth target vectors $\mathbf{y}^{\text{smooth}}$:

$$y_c^{\text{smooth}} = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K}, & \text{if } c = \text{true class} \\[6pt] \frac{\epsilon}{K}, & \text{if } c \neq \text{true class} \end{cases}$$

The smooth cross-entropy loss becomes:

$$L_{\text{smooth}}(i) = - \sum_{c=1}^K y_{i,c}^{\text{smooth}} \cdot \log(\hat{y}_{i,c})$$

---

### 2.3 Mixup Regularization Loss (`mixup_data` & `mixup_criterion`)
Given two training samples $(\mathbf{x}_a, \mathbf{y}_a)$ and $(\mathbf{x}_b, \mathbf{y}_b)$, a synthetic training sample $(\tilde{\mathbf{x}}, \tilde{\mathbf{y}})$ is generated via convex combination:

$$\lambda \sim \text{Beta}(\alpha, \alpha) \quad \text{with } \alpha = 0.2$$

$$\tilde{\mathbf{x}}_{\text{img}} = \lambda \mathbf{x}_{a,\text{img}} + (1 - \lambda) \mathbf{x}_{b,\text{img}}$$

$$\tilde{\mathbf{x}}_{\text{wx}} = \lambda \mathbf{x}_{a,\text{wx}} + (1 - \lambda) \mathbf{x}_{b,\text{wx}}$$

The total loss for the mixed batch is computed as:

$$L_{\text{mixup}}(\hat{\mathbf{y}}, \mathbf{y}_a, \mathbf{y}_b, \lambda) = \lambda \cdot L_{\text{criterion}}(\hat{\mathbf{y}}, \mathbf{y}_a) + (1 - \lambda) \cdot L_{\text{criterion}}(\hat{\mathbf{y}}, \mathbf{y}_b)$$

---

## 3. Data Preprocessing & Feature Engineering Formulas

### 3.1 Z-Score Standardization (`compute_stats.py`)
Weather feature values $x_{i, t, j}$ for timestep $t \in \{1, \dots, T\}$ and feature channel $j \in \{1, \dots, 11\}$ are standardized using parameters computed strictly over the training split:

$$\mu_j = \frac{1}{N_{\text{train}} \cdot T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^T x_{i, t, j}$$

$$\sigma_j = \sqrt{\frac{1}{N_{\text{train}} \cdot T} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^T (x_{i, t, j} - \mu_j)^2}$$

Standardized input:

$$z_{i, t, j} = \frac{x_{i, t, j} - \mu_j}{\sigma_j + 10^{-8}}$$

---

### 3.2 Cyclic Month Encoding (`utils.py`)
Encodes calendar month $m \in \{1, 2, \dots, 12\}$ into a continuous 2D coordinate system:

$$\text{month}_{\sin} = \sin\left( \frac{2 \pi \cdot m}{12} \right)$$

$$\text{month}_{\cos} = \cos\left( \frac{2 \pi \cdot m}{12} \right)$$

Satisfies the trigonometric identity $\text{month}_{\sin}^2 + \text{month}_{\cos}^2 = 1$.

---

### 3.3 Weather Gaussian Jitter Augmentation (`train_multimodal.py`)
Applied only during training (`split == 'train'`):

$$\mathbf{W}_{\text{augmented}} = \mathbf{W}_{\text{normalized}} + \boldsymbol{\eta}, \quad \boldsymbol{\eta} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}) \quad \text{with } \sigma = 0.03$$

---

## 4. Multi-Head Self-Attention Formulation (`WeatherEncoder`)

Given input tensor sequence $\mathbf{X} \in \mathbb{R}^{7 \times 128}$ from the 1D CNN:

Query, Key, and Value projections for head $h \in \{1, 2, 3, 4\}$:

$$\mathbf{Q}_h = \mathbf{X} \mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{X} \mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{X} \mathbf{W}_h^V \quad \left(\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V \in \mathbb{R}^{128 \times 32}\right)$$

Attention weights for head $h$:

$$\mathbf{A}_h = \text{Softmax}\left( \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{7 \times 7} \quad \left(d_k = 32\right)$$

Head output:

$$\mathbf{head}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{7 \times 32}$$

Multi-head concatenation & final projection:

$$\text{MHSA}(\mathbf{X}) = \left[ \mathbf{head}_1 \,||\, \mathbf{head}_2 \,||\, \mathbf{head}_3 \,||\, \mathbf{head}_4 \right] \mathbf{W}^O \in \mathbb{R}^{7 \times 128}$$

---

## 5. Comparison Summary: Project v2 vs. Project v3

| Parameter / Concept | Project v2 Formula / Specification | Project v3 Formula / Specification | Technical Motivation |
|---|---|---|---|
| **Weather Window ($T$)** | $T = 30$ days ($30 \times 11$) | $T = 7$ days ($7 \times 11$) | 7-day window isolates immediate pre-symptomatic infection period and reduces noise. |
| **Favorable Validation Threshold** | $\ge 7$ favorable days out of 30 days ($\sim 23.3\%$) | $\ge 3$ favorable days out of 7 days ($\sim 42.8\%$) | Proportionally stricter validation ensures higher biological realism for short windows. |
| **Weather Augmentation** | None (Raw normalized weather) | Gaussian Jitter ($\sigma = 0.03$) | Prevents overfitting to exact synthetic weather values while preserving temporal shapes. |
| **Profile Pool Size** | Single profiles / direct assignments | 91 profiles / class ($91 \times 33 = 3003$ total) | Ensures vast weather diversity spanning $\ge 3$ years and $\ge 3$ states. |
| **Max Profile Reuse Cap** | Uncapped profile reuse | $\text{Cap} = \lceil \frac{N_{\text{class}}}{N_{\text{pool}}} \times 1.5 \rceil \approx 3\%$ cap | Anti-leakage constraint ensuring no single profile dominates a disease class. |
| **XAI Attention Extraction** | Standard forward pass | `return_attention=True` returning $7 \times 7$ matrix | Enables explicit temporal heatmaps showing peak trigger days for disease onset. |
