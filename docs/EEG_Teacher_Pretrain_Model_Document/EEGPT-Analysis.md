# EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals

**Paper:** NeurIPS 2024 (38th Conference on Neural Information Processing Systems)
**Authors:** Guagnyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, Haifeng Li (Harbin Institute of Technology)
**Code:** https://github.com/BINE022/EEGPT

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Dual Self-Supervised Pretraining](#3-dual-self-supervised-pretraining)
4. [Model Variants and Scaling](#4-model-variants-and-scaling)
5. [Linear-Probing Downstream Method](#5-linear-probing-downstream-method)
6. [Key Hyperparameters](#6-key-hyperparameters)
7. [Experimental Results](#7-experimental-results)
8. [Sleep Staging Results (Sleep-EDFx)](#8-sleep-staging-results-sleep-edfx)
9. [Ablation Studies](#9-ablation-studies)
10. [Application to Our Project](#10-application-to-our-project)
11. [EEGPT vs SleepTransformer](#11-eegpt-vs-sleeptransformer)
12. [Strengths and Limitations](#12-strengths-and-limitations)

---

## 1. Overview

EEGPT is a 10-million-parameter (in the base variant; up to 101M in the large variant) pretrained transformer model designed for universal EEG feature extraction. The key problems it addresses are:

- **Low signal-to-noise ratio (SNR)** in EEG recordings, making raw-signal self-supervised learning unreliable
- **High inter-subject variability** across individuals
- **Channel mismatch** between different EEG acquisition devices (different montages, different channel counts)
- **Task-dependent variations** in EEG patterns across paradigms

EEGPT's core innovation is a **dual self-supervised learning** (dual SSL) strategy that combines:
1. **Spatio-temporal representation alignment** - aligns the encoder's predictions with a momentum encoder's output using high-SNR EEG representations (not raw signals)
2. **Mask-based reconstruction** - the standard masked autoencoder approach, reconstructing masked patches from context

By training on these two objectives jointly, EEGPT learns features that are both semantically rich (from alignment) and locally precise (from reconstruction). The model is pretrained on a mixed multi-task EEG dataset spanning motor imagery, SSVEP, emotion recognition, and identification tasks, then applied to downstream tasks via a lightweight linear-probing method.

---

## 2. Architecture

### 2.1 High-Level Pipeline

The EEGPT processing pipeline consists of the following stages:

```
Raw EEG [M channels x T timepoints]
    |
    v
Patching: divide into patches p_{i,j} of size d along time axis per channel
    |
    v
Embedding: linear projection + channel embedding (from Codex book)
    |
    v
Masking: 50% time, 80% channel patches masked
    |
    v
[Masked part M]          [Unmasked part M_bar]
    |                           |
    v                           v
Encoder (Transformer)     Momentum Encoder (Transformer)
    |                           |
    v                           v
enc_j (spatial features)   menc_j (target features)
    |                           |
    v                           |
Predictor (Transformer)         |
    |                           |
    v                           v
pred_j  <--- Alignment Loss (MSE) ---> LN(menc_j)
    |
    v
Reconstructor (Transformer)
    |
    v
rec_{u,t} <--- Reconstruction Loss (MSE) ---> LN(p_{i,j})
```

### 2.2 Patching

The input EEG signal x is in R^{M x T}, where M is the number of channels and T is the number of time points. The signal is divided into non-overlapping patches in the spatio-temporal dimensions:

- Each patch `p_{i,j} = x_{i, (j-1)*d : j*d}` corresponds to channel i, time segment j
- Patch length d = 64 time points at 256 Hz sampling rate, giving **250ms time windows per patch**
- Total time patches per channel: N = T/d (for T=1024 at 256 Hz with 4s windows, N = 16)
- Total patches: M x N (e.g., 58 channels x 16 time patches = 928 patches)

### 2.3 Local Spatio-Temporal Embedding

This is a key design choice that separates spatial and temporal processing:

**Patch embedding:** Each patch is linearly projected into the embedding space:
```
Embed(p_{i,j}) = W_p^T * p_{i,j} + b_p
```
where W_p is in R^{d x d_e} and b_p is in R^{d_e}.

**Channel embedding (Codex book):** EEGPT constructs a **Codex book** - a lookup table of learnable channel embedding vectors {c_i} in R^{d_e} for each of M_total possible channels. A mapping function R maps channel names from the input data to the corresponding Codex book entries:

```
R: {c_i}_{i=1}^{M_input} -> {c_i}_{i=1}^{M_input}
```

This is critical for channel adaptation. The Codex book contains embeddings for all standard 10-20 system electrode positions. When a new dataset uses a subset of channels (or different channel names), the mapping R selects the appropriate embeddings. This means the model does not need retraining for different channel configurations.

**Token construction:** The final embedded token combines both:
```
token_{i,j} = Embed(p_{i,j}) + c_i
```

### 2.4 Masking Strategy

EEGPT uses an aggressive masking ratio to force the model to learn robust representations:

- **50% of time patches** are masked
- **80% of channel patches** are masked
- The masked part M is fed to the encoder
- The unmasked part M_bar is fed to the momentum encoder

This is intentionally asymmetric: the encoder must work from heavily incomplete data, while the momentum encoder sees a fuller picture to produce stable alignment targets.

### 2.5 Encoder

The encoder is a standard Vision Transformer (ViT) architecture adapted for EEG:

- Processes all masked tokens `{token_{i,j}}` at time step j
- Produces spatial features `enc_j` for each time segment
- Uses S learnable **summary tokens** (similar to CLS tokens) to aggregate information within each time segment
- The encoder processes spatial information (across channels) for each time step, which decouples spatial from temporal processing

The hierarchical structure works as follows:
1. **Within each time step:** The transformer attends over all channel tokens at that time step, producing a spatial representation
2. **Across time steps:** The predictor then attends over the temporal sequence of encoded features

This factorization reduces computational complexity from O((M*N)^2) to O(M^2 * N + N^2), since spatial and temporal attention are computed separately rather than jointly.

### 2.6 Predictor

The predictor takes the encoder's output and predicts the full (unmasked) representation:

```
{pred_t} = PRED({enc_j + pos_j})
```

Key details:
- Adds **rotary position embeddings** (RoPE) to encode relative temporal positions
- Uses a learnable query vector for generating predictions of unmasked positions
- The predictor is also a transformer that operates across the temporal dimension
- Its output is aligned with the momentum encoder's output via the alignment loss

### 2.7 Momentum Encoder

Structure is identical to the encoder, but:
- Receives ALL tokens (both masked and unmasked)
- Parameters are updated via exponential moving average (EMA) of the encoder with factor tau = 0.01
- Produces stable target representations `menc_j` for the alignment loss

```
menc_j = MENC({token_{i,j}}_{(i,j) in M union M_bar})
```

### 2.8 Reconstructor

Takes features from both the encoder (for masked parts) and predictor (for unmasked parts), plus positional information, to reconstruct the original raw patches:

```
{rec_{u,t}} = REC({enc_j + pos_j} union {pred_j + pos_j})
```

A skip connection from encoder features to the reconstructor helps maintain feature quality and accelerates convergence.

---

## 3. Dual Self-Supervised Pretraining

### 3.1 Why Dual SSL?

Standard masked autoencoders for EEG face a fundamental problem: EEG signals have very low SNR, meaning that reconstructing raw signals from masked inputs often forces the model to learn noise patterns rather than meaningful brain activity features. Previous methods like BENDR and EEG2VEC rely solely on reconstruction objectives and suffer from this limitation.

EEGPT addresses this with two complementary objectives:

### 3.2 Spatio-Temporal Representation Alignment (L_A)

The alignment loss matches the predictor's output with the momentum encoder's output:

```
L_A = -(1/N) * sum_{j=1}^{N} ||pred_j, LN(menc_j)||_2^2
```

where LN is layer normalization. Key properties:
- Operates on **representations** (high-level features), not raw signals, so alignment targets have higher SNR
- The momentum encoder provides stable, slowly-evolving targets (similar to BYOL/MoCo in vision)
- Layer normalization prevents representation collapse and mitigates extreme values
- Encourages the encoder to learn features with consistent spatio-temporal semantics

### 3.3 Mask-Based Reconstruction (L_R)

The reconstruction loss aligns reconstructed patches with the original raw patches:

```
L_R = -(1/|M_bar|) * sum_{(i,j) in M_bar} ||rec_{i,j}, LN(p_{i,j})||_2^2
```

Key properties:
- Uses MSE loss on layer-normalized raw patches
- Focuses on the unmasked part M_bar (the reconstructor attempts to recover what was not seen by the encoder)
- Leverages the spatial and temporal consistency inherent in EEG signals
- Provides complementary gradient signal to the alignment loss

### 3.4 Total Pretraining Loss

```
L = L_A + L_R
```

Both losses are weighted equally (no tunable coefficient between them).

### 3.5 Why Both Objectives Matter

From the ablation study (Table 5 in the paper):
- Without L_A: 6-9% performance degradation on downstream tasks - this is the most impactful loss
- Without layer normalization on reconstruction targets: 1-7% degradation due to extreme values and covariate shift
- Without the skip connection in the reconstructor: 1-3% degradation
- With all components (D: with all): best performance across the board

The alignment loss is the primary driver of representation quality, while the reconstruction loss provides complementary local signal structure learning.

---

## 4. Model Variants and Scaling

EEGPT defines 8 variants with different embedding dimensions, depths, and summary token counts:

| Variant | d_e (embed dim) | Layers (enc/pred/rec) | S (summary tokens) | Parameters | L_A | L_R | BCIC-2A BAC (%) |
|---------|----------|------------------------|-----|-----------|------|------|----------|
| tiny1   | 64       | 2/2/4                  | 1   | 0.4M      | 0.32 | 0.60 | 49.19    |
| tiny2   | 64       | 2/2/4                  | 4   | 0.5M      | 0.36 | 0.60 | 50.03    |
| tiny3   | 64       | 8/8/8                  | 4   | 1.6M      | 0.17 | 0.59 | 51.58    |
| little  | 128      | 8/8/8                  | 4   | 6.4M      | 0.18 | 0.57 | 54.18    |
| base1   | 256      | 6/6/6                  | 1   | 19M       | 0.24 | 0.56 | 54.53    |
| base2   | 256      | 8/8/8                  | 4   | 25M       | 0.33 | 0.56 | 56.48    |
| base3   | 512      | 6/6/6                  | 1   | 76M       | 0.14 | 0.58 | 54.47    |
| **large** | **512** | **8/8/8**              | **4** | **101M** | **0.24** | **0.56** | **58.46** |

Key observations from scaling:
- **Larger models consistently achieve lower alignment loss and better downstream accuracy**
- Scaling law for accuracy: ACC = (33.6 * N)^0.029 where N is parameter count
- Scaling law for reconstruction loss: L_R = (0.72 * N)^{-0.014}
- The large model (512 embed dim, 8 layers, 4 summary tokens, 101M params) achieves the best performance
- Summary tokens S=4 consistently outperforms S=1 at the same depth
- Deeper models (8 layers) outperform shallower ones (2 or 6 layers) at the same embedding dimension

The paper uses the **large** model (101M parameters) for all downstream evaluation results.

### Comparison with Other Pretrained EEG Models

| Model | Parameters |
|-------|-----------|
| SPaRCNet | 0.79M |
| ContraWR | 1.6M |
| CNN-T | 3.2M |
| BIOT | 3.2M |
| ST-T | 3.5M |
| EEGPT-Tiny | 4.7M |
| LaBraM | 5.8M |
| **EEGPT (large)** | **25M (base2) / 101M (large)** |

Note: The paper reports EEGPT as "25M" in Tables 2-4 (using the base2 variant for fair model-size comparison on some tasks) and the full large model (101M) for the primary downstream evaluations.

---

## 5. Linear-Probing Downstream Method

For downstream tasks, EEGPT uses a lightweight linear-probing approach rather than full fine-tuning:

### 5.1 Architecture

```
Raw EEG input
    |
    v
Adaptive Spatial Filter (1x1 Conv) -- trainable
    |
    v
Pretrained Encoder (FROZEN)
    |
    v
Summary Token Features
    |
    v
Linear Classification Layer -- trainable
    |
    v
Logits (per class)
```

### 5.2 Key Components

**Adaptive Spatial Filter:** A 1x1 convolution layer that maps the input EEG channels to the model's expected channel configuration. This handles channel mismatch between the dataset and the pretrained model. For example, if the downstream dataset has 22 channels but EEGPT was pretrained with 58, the spatial filter learns to map the 22-channel input to the appropriate representation.

**Frozen Encoder:** The pretrained encoder weights are completely frozen. No gradients flow through the encoder during downstream training. This:
- Prevents overfitting on small downstream datasets
- Significantly reduces training time and memory
- Tests the quality of the pretrained representations directly

**Linear Layer:** A simple linear projection from the summary token features to the number of output classes. Only this layer and the adaptive spatial filter are trained.

### 5.3 Sleep Stage Detection Specific Configuration

For the sleep staging task (Sleep-EDFx), the paper uses a special configuration:
- A **4-layer transformer encoder** is used as the classifier (not just a linear layer)
- This classifier integrates the encoder output every 0.25 seconds
- Purpose: sleep staging requires processing **30-second epochs** (a long task), so the classifier aggregates features over the full epoch
- The pretrained EEGPT encoder still processes 4-second windows; the 4-layer classifier handles temporal aggregation over the 30s window

### 5.4 How This Relates to the FewShotKDVigilance Implementation

In the FewShotKDVigilance codebase (stage1_EEGPT_fewshot_train.py), the downstream pipeline works differently from the paper's linear-probing. The practical implementation for vigilance detection:

1. **EEGPT encoder is loaded from checkpoint** (`eegpt_mcae_58chs_4s_large4E.ckpt`) and set to eval mode
2. A **channel convolution** (`Conv1dWithConstraint`) maps 26 input channels to the encoder's expected channel count
3. The encoder processes 30-second segments (256 Hz * 30s = 7680 time points), producing features of shape [B, 120, 2048]
4. Sinusoidal position embeddings are added to the encoded features
5. Features are reshaped into 5 temporal segments and concatenated with a learnable CLS token
6. An **MLP mapper** (98304 -> 1024 -> 512) reduces dimensionality
7. A **classifier MLP** (512*5 -> 256 -> 2) produces binary vigilance predictions

The trainable parameters are: channel conv, CLS token, mapper MLP, and classifier MLP. The EEGPT encoder itself remains frozen (set to eval mode in forward pass).

---

## 6. Key Hyperparameters

### 6.1 Pretraining Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input sampling rate | f_s = 256 Hz |
| Input signal length | T = 1024 time points (4 seconds) |
| Patch length | d = 64 time points (250 ms) |
| Number of channels used | M = 58 (standard 10-20 extended) |
| Time patches per channel | N = T/d = 16 |
| Time masking ratio | 50% |
| Channel masking ratio | 80% |
| Momentum encoder EMA factor | tau = 0.01 |
| Optimizer | AdamW |
| Learning rate schedule | OneCycle (initial 2.5e-4, max 5e-4, min 3.13e-5) |
| Epochs | 200 |
| Batch size | 64 |
| Precision | 16-bit mixed precision |
| Hardware | 8x Nvidia 3090 GPUs |
| Validation split | 10% of training data |

### 6.2 Downstream Hyperparameters (from FewShotKDVigilance Implementation)

| Parameter | Stage 1 (Teacher Fine-tuning) | Stage 2 (KD) |
|-----------|------|------|
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-5 (max, OneCycle) | 7e-4 (max, OneCycle) |
| Weight decay | 0.01 | 0.01 |
| Batch size | 32 | 32 |
| Epochs | 10 | 30 |
| Precision | 16-bit | 16-bit |
| LR schedule | OneCycle, pct_start=0.2 | OneCycle, pct_start=0.2 |
| KD temperature | - | 2.0 |

### 6.3 Downstream Hyperparameters (from Paper, Various Tasks)

| Task | LR (start) | LR (max) | LR (min) | Epochs | Batch Size |
|------|-----------|----------|----------|--------|------------|
| BCIC-2A | 5e-4 | 1e-3 | 4.22e-7 | 100 | 72 |
| BCIC-2B | 1.6e-5 | 4e-4 | 1.51e-7 | 100 | 64 |
| KaggleERN | 1.6e-5 | 4e-4 | 1.51e-7 | 100 | 64 |
| PhysioP300 | 3.2e-5 | 8e-4 | 3.02e-7 | 100 | 64 |
| Sleep-EDFx | 10-fold CV, ratio 6:2:2 | - | - | - | - |

---

## 7. Experimental Results

### 7.1 Pretraining Datasets

EEGPT is pretrained on a mixed multi-task dataset combining five public EEG datasets:

| Dataset | Paradigm | Subjects | Classes |
|---------|----------|----------|---------|
| PhysioMI | Motor Imagery & Execution | 109 | 5 |
| HGD | Motor Imagery | 14 | 4 |
| TSU | SSVEP | 35 | 40 |
| SEED | Emotion Recognition | 15 | 3 |
| M3CV | Multi-paradigm | 106 | - |

All datasets are preprocessed uniformly: 4-second crops, average re-referencing, channel selection, scaling to mV, resampling to 256 Hz. Motor imagery datasets additionally apply 0-38 Hz bandpass filtering.

### 7.2 Downstream Performance Summary

**TUAB (Abnormal EEG Detection - Binary):**

| Method | Model Size | Balanced Accuracy | AUROC |
|--------|-----------|-------------------|-------|
| SPaRCNet | 0.79M | 0.7896 +/- 0.0018 | 0.8676 +/- 0.0012 |
| BIOT | 3.2M | 0.7959 +/- 0.0057 | 0.8815 +/- 0.0043 |
| EEGPT-Tiny | 4.7M | 0.7959 +/- 0.0021 | 0.8716 +/- 0.0041 |
| **EEGPT** | **25M** | **0.7983 +/- 0.0030** | **0.8718 +/- 0.0050** |

**TUEV (Event Type Classification - 6 Classes):**

| Method | Model Size | Balanced Accuracy | Weighted F1 | Cohen's Kappa |
|--------|-----------|-------------------|-------------|---------------|
| BIOT | 3.2M | 0.5281 +/- 0.0225 | 0.7492 +/- 0.0082 | 0.5273 +/- 0.0249 |
| EEGPT-Tiny | 4.7M | 0.5670 +/- 0.0066 | 0.7535 +/- 0.0097 | 0.5085 +/- 0.0173 |
| **EEGPT** | **25M** | **0.6232 +/- 0.0114** | **0.8187 +/- 0.0063** | **0.6351 +/- 0.0134** |

On TUEV, EEGPT improves over BIOT by 9.5% in balanced accuracy and 6.9% in weighted F1 - a significant improvement.

**Cross-Dataset Comparison (Table 4 in Paper):**

| Dataset | Method | Balanced Accuracy | Cohen's Kappa | Weighted F1 / AUROC |
|---------|--------|-------------------|---------------|---------------------|
| BCIC-2A | BENDR | 0.4899 +/- 0.0070 | 0.3199 +/- 0.0094 | 0.4836 +/- 0.0076 |
| BCIC-2A | BIOT | 0.4590 +/- 0.0196 | 0.2787 +/- 0.0261 | 0.4282 +/- 0.0289 |
| BCIC-2A | LaBraM | 0.5613 +/- 0.0052 | 0.4151 +/- 0.0069 | 0.5520 +/- 0.0052 |
| **BCIC-2A** | **EEGPT** | **0.5846 +/- 0.0070** | **0.4462 +/- 0.0094** | **0.5715 +/- 0.0051** |
| BCIC-2B | EEGPT | **0.7212 +/- 0.0019** | **0.4426 +/- 0.0037** | **0.8059 +/- 0.0032** |
| Sleep-EDFx | EEGPT | **0.6917 +/- 0.0069** | **0.6857 +/- 0.0019** | **0.7654 +/- 0.0023** |
| KaggleERN | EEGPT | **0.5837 +/- 0.0064** | **0.1882 +/- 0.0110** | **0.6621 +/- 0.0096** |
| PhysioP300 | EEGPT | **0.6502 +/- 0.0063** | **0.2999 +/- 0.0139** | **0.7168 +/- 0.0051** |

EEGPT achieves the best balanced accuracy on all five downstream tasks compared to BENDR, BIOT, and LaBraM.

---

## 8. Sleep Staging Results (Sleep-EDFx)

This section is particularly relevant to our project since we are building a sleep/vigilance classification system.

### 8.1 Dataset Details

- **Sleep-EDFx:** 197 recordings from 78 healthy subjects, all-night sleep recordings
- **Channels:** EEG, EOG, chin EMG, event markers
- **Preprocessing:** Convert to mV, 30 Hz low-pass filter, 30-second non-overlapping epochs, channel-wise z-score normalization
- **Classes:** 5 sleep stages (Wake, N1, N2, N3, REM)
- **Evaluation:** 10-fold cross-validation with ratio 6:2:2 (training:validation:test)

### 8.2 Results

| Method | Balanced Accuracy | Cohen's Kappa | Weighted F1 / AUROC |
|--------|-------------------|---------------|---------------------|
| BENDR | 0.6655 +/- 0.0043 | 0.6659 +/- 0.0043 | 0.7507 +/- 0.0029 |
| BIOT | 0.6622 +/- 0.0013 | 0.6461 +/- 0.0017 | 0.7415 +/- 0.0010 |
| LaBraM | 0.6771 +/- 0.0022 | 0.6710 +/- 0.0006 | 0.7592 +/- 0.0005 |
| **EEGPT** | **0.6917 +/- 0.0069** | **0.6857 +/- 0.0019** | **0.7654 +/- 0.0023** |

EEGPT improvements over baselines on Sleep-EDFx:
- vs BENDR: +2.6% balanced accuracy, +2.0% kappa, +1.5% weighted F1
- vs BIOT: +3.0% balanced accuracy, +4.0% kappa, +2.4% weighted F1
- vs LaBraM: +1.5% balanced accuracy, +1.5% kappa, +0.6% weighted F1

### 8.3 Significance for Our Project

These results demonstrate that EEGPT can effectively classify sleep states from EEG, which is directly relevant to our vigilance classification task. Key observations:

1. **5-class performance translates to simpler tasks:** If EEGPT achieves ~69% balanced accuracy on a 5-class sleep staging task, it should perform significantly better on our simpler 3-class vigilance task (alert/intermediate/drowsy) or binary (wake/sleep) task.

2. **Special classifier needed for long windows:** The sleep staging task uses a 4-layer transformer classifier to aggregate EEGPT features over 30-second epochs. Our 1-minute intervals will similarly require a temporal aggregation strategy.

3. **Linear probing is sufficient:** Even without fine-tuning the encoder, EEGPT outperforms fully fine-tuned BENDR on sleep staging.

---

## 9. Ablation Studies

### 9.1 Pretraining Method Ablations (Table 5/7)

| Variant | L_A | L_R | BCIC-2A-BAC | BCIC-2B-AUROC | KaggleERN-AUROC |
|---------|------|------|-------------|---------------|-----------------|
| A: w/o L_A | 37.13 | 0.57 | 0.5287 +/- 0.0086 | 0.7264 +/- 0.0381 | 0.5752 +/- 0.0164 |
| B: w/o LN | 0.15 | 0.002 | 0.5567 +/- 0.0088 | 0.7920 +/- 0.0012 | 0.5891 +/- 0.0227 |
| C: w/o skip | 0.12 | 0.56 | 0.5796 +/- 0.0011 | 0.7702 +/- 0.0122 | 0.6356 +/- 0.0296 |
| **D: with all** | **0.24** | **0.56** | **0.5846 +/- 0.0070** | **0.8059 +/- 0.0032** | **0.6621 +/- 0.0096** |

Key findings:
- **Removing alignment loss (A):** Most severe degradation, 6-9% drop. The alignment loss is the most important component.
- **Removing layer normalization (B):** L_R drops to near-zero (0.002) but downstream performance degrades 3-7%. The model overfits to extreme values in raw patches.
- **Removing skip connection (C):** 1-3% drop. The skip connection from encoder to reconstructor helps maintain feature quality.
- **All components (D):** Best performance across all datasets.

### 9.2 Predictor Ablation (Appendix A.1)

Removing the predictor causes the reconstruction loss L_R to not decrease during training. Without the predictor, directly aligning encoder and momentum encoder outputs leads to **representation collapse** - the model cannot learn meaningful representations. The predictor acts as a bottleneck that prevents this collapse.

### 9.3 Fine-Tuning Method Ablation (Table 8)

| Variant | Adaptive Spatial Filter | Linear Probing | BCIC-2A-BAC | BCIC-2B-AUROC | KaggleERN-AUROC |
|---------|--------|------|-------------|---------------|-----------------|
| A (full FT, no ASF) | No | No | 0.5774 | 0.7871 | 0.6078 |
| B (full FT, with ASF) | Yes | No | 0.5183 | 0.7541 | 0.6110 |
| C (LP, no ASF) | No | Yes | 0.5586 | 0.7974 | 0.6463 |
| **D (LP, with ASF)** | **Yes** | **Yes** | **0.5846** | **0.8059** | **0.6621** |

Linear probing with adaptive spatial filter (D) outperforms full fine-tuning (A, B) in most cases. This is a remarkable finding: freezing the encoder and only training a spatial filter + linear layer is actually better than updating all parameters.

### 9.4 Self-Supervised Pretraining Value (Table 11)

| Method | Model Size | Balanced Accuracy | AUROC |
|--------|-----------|-------------------|-------|
| BIOT | 3.2M | 0.7959 +/- 0.0057 | 0.8815 +/- 0.0043 |
| EEGPT (no pretrained) | 25M | 0.7553 +/- 0.0014 | 0.8260 +/- 0.0018 |
| **EEGPT (pretrained)** | **25M** | **0.7983 +/- 0.0030** | **0.8718 +/- 0.0050** |

Without pretraining, EEGPT performs worse than the much smaller BIOT model. Pretraining adds ~4% balanced accuracy and ~5% AUROC. This confirms that the pretrained representations are critical, not just the architecture.

---

## 10. Application to Our Project

### 10.1 Role as EEG Teacher in Knowledge Distillation Pipeline

In our project, EEGPT serves as the **teacher model** in a knowledge distillation (KD) pipeline:

```
NatView EEG data (simultaneous EEG-fMRI)
    |
    v
EEGPT Teacher (pretrained + few-shot fine-tuned)
    |
    v
Soft probability distributions over vigilance classes
    |
    v
KD Loss: alpha * CE(true_label, student) + (1-alpha) * KL(teacher_soft, student_soft)
    |
    v
fMRI GNN Student Model
```

The teacher produces soft labels (probability distributions) that contain richer information than hard labels (one-hot). The temperature-scaled softmax reveals inter-class similarities that the student can learn from.

### 10.2 Channel Adaptation for NatView EEG

**The challenge:** NatView uses ~64-channel EEG (likely the 10-20 extended system), but the standard EEGPT model expects M=58 channels from specific positions.

**The solution - Codex Book:** EEGPT's Codex book mechanism handles this naturally:
1. Identify which of NatView's ~64 channels overlap with EEGPT's 58-channel configuration
2. The mapping function R selects the appropriate channel embeddings from the Codex book
3. For channels in NatView that are not in the Codex book, we have two options:
   - Drop those channels (simplest)
   - Use the adaptive spatial filter (1x1 convolution) to learn a mapping from NatView's channels to the model's expected channels

**How the FewShotKDVigilance repo handles this:** The Vanderbilt lab's implementation uses only 26 channels from their EEG-fMRI dataset and maps them via a `Conv1dWithConstraint` layer (1x1 convolution from 26 to the EEGPT encoder's expected channel count). The channel names they use are:

```python
use_channels_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                       'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                       'FPZ', 'FZ', 'CZ', 'PZ', 'POZ', 'OZ', 'F7', 'F8',
                       'T7', 'T8']
```

Note: The last 4 channels are duplicates of F7, F8, T7, T8 - this appears to be a padding strategy to map 22 unique channels to 26 inputs.

For our NatView dataset, we should:
1. Check which channels are available in NatView EEG
2. Select the subset that overlaps with EEGPT's 58-channel configuration
3. Use `prepare_chan_ids()` to get the corresponding Codex book indices
4. Add a Conv1d spatial filter if needed to adapt dimensions

### 10.3 Window Size Considerations

**EEGPT pretraining:** 4-second windows at 256 Hz (T=1024 time points)

**FewShotKDVigilance usage:** 30-second segments (256 Hz * 30 = 7680 time points). They handle this by:
1. Temporally interpolating the input to 7680 time points
2. The EEGPT encoder processes this as one long sequence
3. Features are reshaped into 5 sub-segments of 6 seconds each
4. A CLS token is concatenated per sub-segment
5. MLP maps each sub-segment's features (98304 -> 512)
6. All 5 sub-segment features are concatenated (512*5 = 2560) and classified

**Our 1-minute intervals:** Each interval is 60 seconds. Options:

1. **Follow the FewShotKDVigilance approach:** Divide each 1-minute interval into multiple sub-windows (e.g., 10 x 6-second or 2 x 30-second segments), run EEGPT on each, aggregate features
2. **Use sliding 4-second windows:** Extract EEGPT features every 4 seconds, getting 15 feature vectors per minute, then aggregate with a small temporal model
3. **Use 30-second windows:** Split each minute into two 30-second halves, process each as in the FewShotKDVigilance code, then average the resulting logits

Recommendation: **Option 1 or 3**, following the FewShotKDVigilance approach. The temporal interpolation from their sampling rate to 256*30 = 7680 is already validated. For a 1-minute interval, we could either:
- Process two 30-second segments and average logits/features
- Process the full 60 seconds with appropriate reshaping (12 sub-segments of 5 seconds each)

### 10.4 Which Model Variant to Use

**Recommendation: Use the large checkpoint (512 embed dim, 8/8/8 layers, 4 summary tokens)**

Justification:
- The FewShotKDVigilance repo uses `eegpt_mcae_58chs_4s_large4E.ckpt` - the large variant
- The large model achieves the best downstream accuracy across all tasks
- Our dataset is small (~220 samples), but since we are using linear probing (frozen encoder), the risk of overfitting the large model is minimal - only the small adapter layers are trained
- The large model checkpoint is available from the EEGPT GitHub repository
- Compute cost is manageable because the encoder is frozen during fine-tuning

Configuration for the large model:
```python
target_encoder = EEGTransformer(
    img_size=[num_channels, 256*30],  # channels x time points
    patch_size=32*2,                   # patch size = 64
    embed_num=4,                       # 4 summary tokens
    embed_dim=512,                     # 512 embedding dimension
    depth=8,                           # 8 transformer layers
    num_heads=8,                       # 8 attention heads
    mlp_ratio=4.0,                     # MLP hidden dim = 4 * embed_dim = 2048
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    init_std=0.02,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)
```

### 10.5 Extracting Soft Probability Distributions for KD

The pipeline for generating teacher soft labels:

```python
# Stage 1: Fine-tune EEGPT teacher on NatView EEG for vigilance classification
# (Following FewShotKDVigilance stage1_EEGPT_fewshot_train.py pattern)

# 1. Load pretrained EEGPT encoder
encoder = EEGTransformer(...)
encoder.load_state_dict(pretrained_weights)
encoder.eval()  # Freeze encoder

# 2. Add trainable adapter layers
chan_conv = Conv1dWithConstraint(input_channels, encoder_channels, 1)
mapper = nn.Sequential(nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2))
classifier = nn.Sequential(nn.Linear(512*num_segments, 256), nn.ReLU(),
                            nn.Dropout(0.2), nn.Linear(256, num_classes))

# 3. Fine-tune only adapter layers on NatView EEG vigilance labels
# Use AdamW, OneCycle LR, ~10 epochs, lr ~3e-5

# 4. Extract soft labels for KD
with torch.no_grad():
    eeg_features, logits = teacher_forward(eeg_input)
    # Temperature-scaled soft labels
    T = 2.0  # or tune in range 2-10
    soft_labels = F.softmax(logits / T, dim=-1)
```

In the KD loss (Stage 2):
```python
# From FewShotKDVigilance stage2 code
T = 2.0
student_soft = F.log_softmax(fmri_logits / T, dim=-1)
teacher_soft = F.softmax(eeg_logits / T, dim=-1)
kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
```

The T^2 scaling factor compensates for the reduced gradient magnitude from temperature scaling, keeping the KD gradient on the same scale as the cross-entropy gradient.

### 10.6 Multi-Teacher Ensemble (Advanced)

The FewShotKDVigilance stage 2 uses **three** EEG teacher models simultaneously:
1. **EEGPT** - the primary teacher
2. **LaBraM** - a large brain model (5.8M params)
3. **BIOT** - biosignal transformer (3.2M params)

Their logits are combined:
```python
# For KD logit loss:
eeg_logits_combined = (eeg_logits + eeg_logits_labram + eeg_logits_biot) / 3

# For pseudo-labels:
eeg_logits_combined = 0.5 * eeg_logits + 0.5 * eeg_logits_labram
pseudo_labels = torch.argmax(eeg_logits_combined.detach(), dim=-1)
```

For our project, we could start with EEGPT alone and optionally add LaBraM or BIOT as additional teachers if time permits.

### 10.7 Two-Stage KD Training (from FewShotKDVigilance)

The stage 2 training uses a two-phase approach within the same training run:

**Phase 1 (epochs 0-19): Heavy KD, light supervision**
```python
loss_kd_logit: 0.4
loss_kd_feat: 0.4
loss_fmri_cls: 0.2   # Uses pseudo-labels from teacher
```

**Phase 2 (epochs 20-30): Light KD, heavy supervision**
```python
loss_fmri_cls: 0.8   # Uses real ground truth labels
loss_kd_logit: 0.1
loss_kd_feat: 0.1
```

This curriculum-style training first teaches the student to mimic the teacher, then gradually shifts to optimizing on real labels. The pseudo-labeling in Phase 1 is particularly interesting: the student is trained on teacher-generated labels rather than ground truth, which can be useful when ground truth is noisy or limited.

---

## 11. EEGPT vs SleepTransformer

For our project, the choice between EEGPT and SleepTransformer as the EEG teacher model is an important architectural decision.

### 11.1 Key Differences

| Aspect | EEGPT | SleepTransformer |
|--------|-------|------------------|
| **Pretraining** | Self-supervised on mixed multi-task EEG data | No pretraining; trained end-to-end on sleep data |
| **Architecture** | Pretrained encoder + linear probing | Epoch-level + sequence-level transformers |
| **Input** | Raw EEG patches (any paradigm) | STFT spectrograms (sleep-specific) |
| **Channel handling** | Codex book adapts to any montage | Fixed channel configuration |
| **Transfer learning** | Designed for it (pretrained universal features) | Must train from scratch on new data |
| **Venue** | NeurIPS 2024 | IEEE TNNLS 2022 |
| **Parameters** | 25M-101M (encoder only) | Task-specific, smaller |
| **Sleep staging** | 69.2% BAC on Sleep-EDFx (5-class) | 84.3% accuracy on Sleep-EDFx (not directly comparable due to different metrics/splits) |
| **Data efficiency** | High (pretrained features need few labeled samples) | Lower (needs substantial labeled sleep data) |

### 11.2 Why EEGPT is the Better Choice for Our Project

1. **Small dataset:** NatView has only ~22 subjects with ~220 labeled intervals. EEGPT's pretrained features + linear probing are specifically designed for low-data regimes. SleepTransformer would need to be trained from scratch on this small dataset, risking severe overfitting.

2. **Channel flexibility:** NatView uses a simultaneous EEG-fMRI setup with a specific channel configuration that may not match standard sleep EEG montages. EEGPT's Codex book handles this gracefully; SleepTransformer's STFT-based input would need custom adaptation.

3. **Proven KD pipeline:** The FewShotKDVigilance repo already demonstrates EEGPT working in a KD pipeline for EEG-to-fMRI knowledge distillation - exactly our use case. There is no equivalent reference implementation for SleepTransformer in a KD context.

4. **Reference implementation:** We can directly adapt the FewShotKDVigilance codebase, which uses EEGPT as the teacher. This significantly reduces implementation effort and risk.

5. **Universal representations:** EEGPT extracts general EEG features (not sleep-specific), which may generalize better to our vigilance classification task that includes awake states. SleepTransformer is optimized specifically for sleep staging.

6. **State-of-the-art:** EEGPT is a NeurIPS 2024 paper with demonstrated superiority over multiple baselines including BENDR, BIOT, and LaBraM on multiple tasks.

### 11.3 When SleepTransformer Might Be Preferred

- If we had a large labeled sleep dataset (not our case)
- If the task required fine-grained N1/N2/N3/REM discrimination (not our case - we have 3 vigilance levels)
- If the STFT-based input was specifically desired for interpretability

---

## 12. Strengths and Limitations

### 12.1 Strengths

1. **Universal representations:** Pretrained on diverse EEG paradigms, demonstrated to work across motor imagery, sleep staging, emotion recognition, ERP detection, and abnormal EEG detection.

2. **Channel adaptation (Codex book):** The most practical innovation for real-world use. Different EEG devices have different channel configurations; EEGPT handles this without retraining.

3. **Data efficiency:** Linear probing with frozen encoder prevents overfitting on small datasets. Critical for our ~220-sample NatView dataset.

4. **Dual SSL quality:** The spatio-temporal alignment produces significantly better features than reconstruction alone (6-9% improvement in ablation).

5. **Hierarchical processing:** Factoring spatial and temporal processing reduces computational complexity and improves flexibility.

6. **Scaling properties:** Clear positive scaling laws for both model size and pretraining data size, suggesting the architecture can benefit from larger training efforts.

7. **Reproducibility:** Code is publicly available, and the FewShotKDVigilance repo provides a complete working example of EEGPT in a KD pipeline.

### 12.2 Limitations

1. **Computational cost:** The large model has 101M parameters. While the encoder is frozen during downstream training, inference requires running this full model for every input. For our 220-sample dataset this is manageable, but it rules out real-time applications.

2. **Pretraining data bias:** Pretrained primarily on motor imagery and BCI data, not sleep data. While it shows good transfer to sleep staging, a model pretrained on sleep-specific data might perform better on that task specifically.

3. **Fixed 256 Hz assumption:** The model expects input at 256 Hz. Data at other sampling rates must be resampled, which can introduce artifacts. NatView EEG may need resampling.

4. **4-second native window:** The encoder is pretrained on 4-second windows. Longer inputs (like our 1-minute intervals) require temporal interpolation or windowed processing, which is not a native capability.

5. **Limited sleep-specific evaluation:** The paper only evaluates on Sleep-EDFx with 10-fold CV. No results on simultaneous EEG-fMRI sleep/vigilance data, which is our exact use case.

6. **No attention to subcortical markers:** EEGPT processes scalp EEG and knows nothing about subcortical structures. This means the teacher's knowledge is purely cortical, while our GNN student additionally has subcortical node features from fMRI.

7. **Binary limitation in FewShotKDVigilance:** The existing Vanderbilt implementation uses binary classification (2 classes). We need to adapt for 3-class vigilance (alert/intermediate/drowsy), which requires changing the classifier output dimension and potentially adjusting the training procedure.

### 12.3 Risk Mitigation for Our Project

| Risk | Mitigation |
|------|-----------|
| EEGPT teacher performs poorly on NatView EEG | Start with the FewShotKDVigilance pipeline on their data first; adapt to NatView only after confirming the pipeline works |
| Channel mismatch causes poor adaptation | Use the Conv1dWithConstraint approach from FewShotKDVigilance; experiment with different channel subsets |
| 3-class task is harder than binary | Start with binary (alert vs drowsy), add intermediate class later |
| Compute constraints prevent using large model | Fall back to base2 (25M) or even tiny3 (1.6M) - still competitive with baselines |
| Teacher gives poor soft labels | Add ground-truth cross-entropy loss with higher weight (increase alpha in KD loss) |

---

## Appendix: Quick Reference Code Patterns

### Loading EEGPT Pretrained Weights

```python
from functools import partial
import torch.nn as nn
from Modules.models.EEGPT_mcae import EEGTransformer

# Initialize encoder with large model config
encoder = EEGTransformer(
    img_size=[num_channels, 256*30],
    patch_size=64,
    embed_num=4,
    embed_dim=512,
    depth=8,
    num_heads=8,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    init_std=0.02,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)

# Load from pretrained checkpoint
ckpt = torch.load("eegpt_mcae_58chs_4s_large4E.ckpt")
encoder_state = {k[15:]: v for k, v in ckpt['state_dict'].items()
                 if k.startswith("target_encoder.")}
encoder.load_state_dict(encoder_state)

# Prepare channel IDs for your dataset
chan_ids = encoder.prepare_chan_ids(your_channel_names)
```

### Temporal Interpolation Utility

```python
def temporal_interpolation(x, desired_length, mode='nearest', use_avg=True):
    """Resample EEG temporal dimension to desired length.
    Args:
        x: [B, C, T] or [C, T] tensor
        desired_length: target number of time points
        mode: interpolation mode ('nearest' or 'linear')
        use_avg: if True, subtract temporal mean before interpolation
    """
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return F.interpolate(x.unsqueeze(0), desired_length, mode=mode).squeeze(0)
    elif len(x.shape) == 3:
        return F.interpolate(x, desired_length, mode=mode)
```

### KD Loss Computation

```python
T = 2.0  # temperature
student_soft = F.log_softmax(student_logits / T, dim=-1)
teacher_soft = F.softmax(teacher_logits / T, dim=-1)
kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

# Combined loss
total_loss = alpha * ce_loss + (1 - alpha) * kd_loss
```
