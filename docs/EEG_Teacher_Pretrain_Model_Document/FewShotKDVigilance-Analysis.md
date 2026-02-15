# FewShotKDVigilance Repository Analysis

**Repository:** [neurdylab/FewShotKDVigilance](https://github.com/neurdylab/FewShotKDVigilance)
**Authors:** Vanderbilt University, neurdylab (Neural Dynamics Lab)
**Paper:** Submitted to SPIE 2026
**Related publication:** "Few-Shot Knowledge Distillation for EEG-to-fMRI Vigilance Estimation" (Imaging Neuroscience, MIT Press)

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Overall Pipeline Architecture](#2-overall-pipeline-architecture)
3. [Stage 1: EEG Teacher Training (EEGPT)](#3-stage-1-eeg-teacher-training-eegpt)
4. [Stage 1: EEG Teacher Training (LaBraM)](#4-stage-1-eeg-teacher-training-labram)
5. [Stage 1: EEG Teacher Training (BIOT)](#5-stage-1-eeg-teacher-training-biot)
6. [Stage 2: Knowledge Distillation to fMRI Student](#6-stage-2-knowledge-distillation-to-fmri-student)
7. [Dataset Structure and Vigilance Labels](#7-dataset-structure-and-vigilance-labels)
8. [fMRI Preprocessing](#8-fmri-preprocessing)
9. [Model Architecture Details](#9-model-architecture-details)
10. [Key Differences from Our Project](#10-key-differences-from-our-project)
11. [Reusable Components](#11-reusable-components)
12. [Recommended Adaptations](#12-recommended-adaptations)
13. [HuggingFace Checkpoints and Pretrained Weights](#13-huggingface-checkpoints-and-pretrained-weights)

---

## 1. Repository Overview

FewShotKDVigilance implements a two-stage knowledge distillation pipeline that transfers vigilance classification ability from EEG-based teacher models to an fMRI-based student model. The core idea is that EEG provides high-temporal-resolution vigilance signals but is noisy and hard to collect at scale, while fMRI is more robust but lacks direct access to the rapid neural dynamics that distinguish alert from drowsy states. By distilling EEG knowledge into an fMRI model, the fMRI student can classify vigilance without requiring simultaneous EEG at inference time.

### Key facts

- **Task:** Binary vigilance classification (alert vs. drowsy)
- **Modalities:** Simultaneous EEG-fMRI recordings
- **Datasets:** Vanderbilt EEGfMRI-VU dataset (primary), NIH-ECR dataset (zero-shot evaluation)
- **Teachers:** Three pretrained EEG foundation models - EEGPT, LaBraM, BIOT
- **Student:** 2-layer Transformer encoder operating on fMRI ROI timeseries
- **Framework:** PyTorch + PyTorch Lightning
- **Python version:** 3.8
- **Key dependencies:** PyTorch 2.4.1, CUDA 12.4, pytorch-lightning 2.4.0, timm 1.0.19, nilearn 0.10.4, mne 1.6.1

### Repository file structure

```
FewShotKDVigilance/
  stage1_EEGPT_fewshot_train.py          # Train EEGPT teacher
  stage1_EEGPT_fewshot_test.py           # Test EEGPT teacher
  stage1_labram_fewshot_train.py         # Train LaBraM teacher
  stage1_labram_fewshot_test.py          # Test LaBraM teacher
  stage1_BIOT_fewshot_train.py           # Train BIOT teacher
  stage1_BIOT_fewshot_test.py            # Test BIOT teacher
  stage2_labramBIOTEEGPT_kd_transformer_train.py   # KD training
  stage2_labramBIOTEEGPT_kd_transformer_test.py    # KD testing
  stage2_..._test_visualizations.py      # Feature extraction for viz
  stage2_..._test_visualizations_vpat_gt.py  # Patient GT visualization
  utils_EEGPT.py                         # Temporal interpolation utility
  models/
    transformer.py                       # TransformerEncoder/Decoder (from DETR)
    helpers.py                           # MLP, weight init, norms, activations
  vigilance_datasets/
    __init__.py                          # Dataset registry + builders
    eegfmri_vu_alphatheta_smallinterval_1024.py   # Primary VU dataset
    eegfmri_vu_pat_alphatheta_smallinterval_1024.py  # Patient fMRI-only
    eegfmri_vu_pat_alphatheta_smallinterval_1024_gt.py  # Patient fMRI+GT
    NIHECR_alphatheta_smallinterval_1024.py  # NIH-ECR dataset
  data_preprocessing/
    eeg_filtering/    # MATLAB scripts for EEG channel removal
    fmri_atlas/
      EEGfMRI_VU_fmri_fit_atlas_batch_1024.py   # VU fMRI parcellation
      NIH_fmri_fit_atlas_batch_1024.py           # NIH fMRI parcellation
  environment.yaml                       # Conda environment spec
```

---

## 2. Overall Pipeline Architecture

The pipeline has two sequential stages:

### Stage 1: EEG Teacher Fine-Tuning

Three pretrained EEG foundation models are independently fine-tuned on the vigilance classification task using only EEG data from simultaneous EEG-fMRI recordings:

1. **EEGPT** - A masked channel autoencoder transformer pretrained on 58 EEG channels
2. **LaBraM** - A ViT-style model pretrained on EEG with patch embeddings
3. **BIOT** - A biosignal transformer using STFT-based input representations

Each teacher learns to classify binary vigilance (alert=1, drowsy=0) from EEG segments. The pretrained encoder weights are frozen during fine-tuning; only the classification head (mapper + classifier) is trained.

### Stage 2: Multi-Teacher Knowledge Distillation

All three frozen teacher models produce soft logits and intermediate features for each EEG segment. An fMRI-based Transformer student model is trained using a 4-component loss function that combines:
- Logit-level KD (KL divergence from teacher ensemble)
- Feature-level KD (MSE between student and teacher features)
- Classification loss (cross-entropy, using either pseudo-labels or ground truth)
- Contrastive loss on fMRI features

The student model only needs fMRI data at inference time.

### Data flow diagram

```
                    Stage 1 (x3 teachers)
EEG data -----> [Pretrained EEG Encoder] -----> mapper -----> classifier -----> binary label
                    (frozen)                  (trained)       (trained)

                    Stage 2
EEG data -----> [Frozen Teacher 1: EEGPT]  -----> teacher logits + features
            |-> [Frozen Teacher 2: LaBraM] -----> teacher logits + features  ---> averaged
            |-> [Frozen Teacher 3: BIOT]   -----> teacher logits + features

fMRI data ----> [Trainable fMRI Transformer] ---> mapper ---> classifier ---> student logits

Loss = w1 * KL_div(student, teacher_avg) + w2 * MSE(student_feat, teacher_feats)
     + w3 * CE(student, pseudo_labels_or_gt) + w4 * contrastive(student_feats)
```

---

## 3. Stage 1: EEG Teacher Training (EEGPT)

**File:** `stage1_EEGPT_fewshot_train.py`

### 3.1 Pretrained Encoder Loading

The EEGPT encoder is an `EEGTransformer` (from `Modules.models.EEGPT_mcae`) loaded from a pretrained checkpoint at `../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`. The checkpoint contains a `target_encoder` state dict that is extracted by stripping the `target_encoder.` prefix:

```python
pretrain_ckpt = torch.load(load_path)
target_encoder_stat = {}
for k,v in pretrain_ckpt['state_dict'].items():
    if k.startswith("target_encoder."):
        target_encoder_stat[k[15:]]=v
self.eeg_encoder.load_state_dict(target_encoder_stat)
```

The encoder is set to `eval()` mode during forward passes, ensuring batch norm and dropout layers behave as inference-only. Its parameters are NOT included in the optimizer, so only the classification head is trained.

### 3.2 EEGTransformer Configuration

```python
EEGTransformer(
    img_size=[26, 256*30],    # 26 channels x 7680 samples
    patch_size=32*2,           # 64-sample patches
    embed_num=4,
    embed_dim=512,
    depth=8,                   # 8 transformer layers
    num_heads=8,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    init_std=0.02,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)
```

### 3.3 Channel Selection

The model uses 26 EEG channels from the international 10-20 system:

```python
use_channels_names_original = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FPZ', 'FZ', 'CZ', 'PZ',
    'POZ', 'OZ', 'FT9', 'FT10', 'TP9', 'TP10']
```

Note: The `use_channels_names` list (used for the model) replaces the last 4 channels (FT9, FT10, TP9, TP10) with duplicates of F7, F8, T7, T8. This is likely because EEGPT's channel ID system does not have embeddings for FT9/FT10/TP9/TP10, so they map these to the nearest known channels.

A `Conv1dWithConstraint` layer (`chan_conv`) with kernel size 1 maps from the 26 raw channels to the 26 model channels (26 -> 26), applying a max-norm constraint of 1.0.

### 3.4 Temporal Interpolation

The raw EEG sampling rate is 250 Hz, and each fMRI TR is 2.1 seconds, yielding 525 EEG samples per TR. The model processes 5 TRs at a time (2625 samples), but EEGPT expects 7680 samples (256 Hz x 30 seconds). The `temporal_interpolation` function handles this:

```python
def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
```

This performs nearest-neighbor interpolation from 2625 to 7680 samples, effectively upsampling the signal by a factor of ~2.93. Before interpolation, the channel-wise mean is subtracted for zero-centering.

### 3.5 Classification Head Architecture

After the frozen EEGPT encoder processes the interpolated input, the output goes through several steps:

1. **Encoder output:** `z` with shape `[B, 120, 2048]` (120 patch tokens, 2048-dim embeddings after flattening)
2. **Sinusoidal positional encoding:** Added to the patch embeddings
3. **Reshape into 5 temporal segments:** `[B, 5, 49152]` (each segment = 24 patches x 2048 dim)
4. **CLS token concatenation:** A learnable CLS token of dim 49152 is concatenated to each segment
5. **Feature mapping:** `eeg_mapper` reduces each segment: 98304 -> 1024 -> 512 (with ReLU, Dropout 0.2)
6. **Flatten all 5 segments:** 512 x 5 = 2560 features
7. **Classification:** `eeg_classfier` maps 2560 -> 256 -> 2 (binary output)

### 3.6 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Max epochs | 10 |
| Learning rate | 3e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| LR scheduler | OneCycleLR (pct_start=0.2) |
| Precision | FP16 |
| Loss | CrossEntropyLoss (weight=1.0) |
| Metric | Macro F1 score |
| Checkpoint selection | Best val_mf1 |
| Random seed | 11 |
| Weight initialization | Xavier uniform (for new layers) |

Only these parameters are optimized (encoder is frozen):
- `eeg_mapper` parameters
- `chan_conv` parameters
- `cls_token`
- `eeg_classfier` parameters

---

## 4. Stage 1: EEG Teacher Training (LaBraM)

**File:** `stage1_labram_fewshot_train.py`

### 4.1 Model Loading

LaBraM is a ViT-style model loaded via `timm.models.create_model("labram_base_patch200_200")`. The pretrained weights come from `Modules/LaBraM/labram-base.pth`, with state dict keys prefixed by `student.`:

```python
checkpoint = torch.load("Modules/LaBraM/labram-base.pth")
new_checkpoint = {}
for k,v in checkpoint['model'].items():
    if k.startswith('student.'):
        new_checkpoint[k[len('student.'):]] = v
model.load_state_dict(new_checkpoint, strict=False)
```

### 4.2 LaBraM-specific Input Processing

LaBraM expects EEG input in a specific patch format. The 2625-sample EEG segment is interpolated to 3000 samples (200 Hz x 15 seconds), then reshaped into patches:

```python
x_labram = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 200*15)  # [B, 26, 3000]
x_labram = x_labram.reshape((B, C, 15, 200))  # [B, 26, 15, 200] = 15 patches of 200 samples
```

The forward pass manually calls the encoder's internal layers:
```python
feats = self.eeg_encoder.patch_embed(x_labram)
feats = self.eeg_encoder.pos_drop(feats)
for block in self.eeg_encoder.blocks:
    feats = block(feats)
feats = self.eeg_encoder.norm(feats)
feats = self.eeg_encoder.fc_norm(feats)  # [B, 390, 200]
```

### 4.3 Classification Head

The LaBraM output is reshaped to `[B, 5, 15600]` (5 temporal segments, each 78 x 200), then:
- `eeg_mapper_labram`: 15600 -> 1024 -> 512 (ReLU, Dropout 0.2)
- Flatten: 512 x 5 = 2560
- `eeg_classfier_labram`: 2560 -> 256 -> 2

### 4.4 Training Hyperparameters

Same as EEGPT: batch_size=32, epochs=10, lr=3e-5, AdamW, OneCycleLR.

---

## 5. Stage 1: EEG Teacher Training (BIOT)

**File:** `stage1_BIOT_fewshot_train.py`

### 5.1 Model Loading

BIOT (Biosignal Transformer) uses a different architecture based on STFT input representations. It is loaded from pretrained checkpoints:

```python
pretrain_models = ["Modules/BIOT/EEG-PREST-16-channels.ckpt",
                   "Modules/BIOT/EEG-SHHS+PREST-18-channels.ckpt",
                   "Modules/BIOT/EEG-six-datasets-18-channels.ckpt"]
```

The default choice is index 2 (18-channel, six-dataset pretraining). A `Conv1dWithConstraint` maps from 26 channels to 18 channels since the pretrained BIOT expects 18 inputs.

```python
BIOTClassifier(
    n_classes=2,
    n_channels=18,
    n_fft=200,
    hop_length=100,
)
```

### 5.2 BIOT-specific Input Processing

BIOT also interpolates to 3000 samples, applies channel convolution, then uses the internal BIOT encoder:

```python
x_biot = temporal_interpolation(eeg[:, :, i*525*5:(i+1)*525*5], 200*15)  # [B, 26, 3000]
x_biot = self.chan_conv(x_biot)  # [B, 18, 3000]
h = self.eeg_encoder.biot(x_biot)  # [B, 256]
```

### 5.3 Classification Head

BIOT produces a 256-dim feature vector (much smaller than EEGPT or LaBraM):
- `eeg_mapper_biot`: 256 -> 1024 -> 2560 (ReLU, Dropout 0.2)
- `eeg_classfier_biot`: 2560 -> 256 -> 2

Note: The mapper here upscales rather than downscales, expanding from 256 to 2560 so the classifier head has the same input dimension as the other teachers.

### 5.4 Training Hyperparameters

Same as EEGPT and LaBraM: batch_size=32, epochs=10, lr=3e-5, AdamW, OneCycleLR.

---

## 6. Stage 2: Knowledge Distillation to fMRI Student

**File:** `stage2_labramBIOTEEGPT_kd_transformer_train.py`

This is the core of the pipeline. All three frozen EEG teachers and the trainable fMRI student are combined into a single `LitEEGPTCausal` PyTorch Lightning module.

### 6.1 Teacher Loading

All three teachers are loaded from their respective Stage 1 checkpoints:
- EEGPT: `EEGvigilance/stage0_EEGPT_fewshot/best.ckpt`
- LaBraM: `EEGvigilance/stage0_labram_fewshot/best.ckpt`
- BIOT: `EEGvigilance/stage0_BIOT_fewshot/best.ckpt`

Each teacher's full pipeline (encoder + mapper + classifier) is loaded and frozen. The CLS token for EEGPT is set to `requires_grad=False`.

### 6.2 fMRI Student Architecture

The student is a 2-layer Transformer encoder that processes fMRI ROI timeseries:

```python
self.fmri_encoder = TransformerEncoder(
    TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=512),
    num_layers=2
)
```

**Input processing:**
- fMRI data shape: `[B, 5, 1024]` (5 TRs x 1024 DiFuMo ROIs)
- Permuted to `[B, 1024, 5]`, then segments of 5 TRs are fed as `[B, 5, 1024]` to the encoder

The Transformer encoder uses:
- Pre-norm (LayerNorm before attention and FFN)
- `d_model=1024` (matching the 1024 DiFuMo ROI dimensions)
- `nhead=4` attention heads
- `dim_feedforward=512`
- Dropout 0.1 on attention and FFN

After encoding:
- `fmri_mapper`: 1024 -> 1024 -> 512 (ReLU, Dropout 0.2)
- Flatten 5 segments: 512 x 5 = 2560
- `fmri_classifier`: 2560 -> 256 -> 2

### 6.3 The 4-Component Loss Function

The `SetCriterion` class in Stage 2 implements five loss functions (four are active during training):

#### 6.3.1 Logit KD Loss (`loss_kd_logit`)

Uses KL divergence with temperature scaling. Teacher logits are averaged across all three teachers:

```python
T = 2.0  # temperature
eeg_logits_combined = (eeg_logits + eeg_logits_labram + eeg_logits_biot) / 3
student_soft = F.log_softmax(fmri_logits / T, dim=-1)
teacher_soft = F.softmax(eeg_logits_combined / T, dim=-1)
kd_logit_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
```

The `T^2` scaling factor normalizes the gradient magnitude, which is standard practice in knowledge distillation (Hinton et al., 2015).

#### 6.3.2 Feature KD Loss (`loss_kd_feat`)

MSE loss between student and each teacher's intermediate features, averaged:

```python
loss_a = F.mse_loss(fmri_all_feats, eeg_all_feats)       # vs EEGPT
loss_b = F.mse_loss(fmri_all_feats, eeg_all_feats_labram) # vs LaBraM
loss_c = F.mse_loss(                                       # vs BIOT (normalized)
    F.normalize(fmri_all_feats, dim=-1),
    F.normalize(eeg_all_feats_biot, dim=-1)
)
total_loss = (loss_a + loss_b + loss_c) / 3
```

Note that BIOT features are L2-normalized before comparison, likely because BIOT's feature space has a different scale than the other teachers.

#### 6.3.3 Classification Loss (`loss_fmri_cls`)

Has two modes controlled by `self.use_pseudo_labels`:

**Pseudo-label mode (Stage 1 of training):**
```python
eeg_logits_combined = 0.5 * eeg_logits + 0.5 * eeg_logits_labram
pseudo_labels = torch.argmax(eeg_logits_combined.detach(), dim=-1)
loss = F.cross_entropy(fmri_logits, pseudo_labels)
```

**Ground-truth mode (Stage 2 of training):**
```python
loss = ce_loss_fn(fmri_logits, vigilance_seg)
```

#### 6.3.4 Contrastive Loss (`loss_fmri_contrastive`)

A supervised contrastive loss on fMRI features. For each sample in the batch, positive pairs are samples with the same vigilance label, and negative pairs have different labels. Uses cosine similarity with temperature 0.1.

### 6.4 Two-Phase Training Strategy

Training is split into two phases based on epoch number:

#### Phase 1: KD-Heavy (Epochs 0-19)

```python
loss_weight_dict = {
    "loss_kd_logit": 0.4,
    "loss_kd_feat": 0.4,
    "loss_fmri_cls": 0.2,   # uses pseudo-labels from teacher ensemble
}
```

In this phase, 80% of the loss comes from KD (logit + feature alignment), and 20% from classification using teacher-generated pseudo-labels. The student focuses on mimicking teacher behavior.

#### Phase 2: Supervised-Heavy (Epochs 20-30)

```python
loss_weight_dict = {
    "loss_fmri_cls": 0.8,    # uses ground-truth labels
    "loss_kd_logit": 0.1,
    "loss_kd_feat": 0.1,
}
```

In this phase, 80% of the loss comes from supervised classification with ground-truth labels, and only 20% from KD. The student refines its predictions with real labels.

The contrastive loss weight is fixed at 0.1 throughout both phases (set in `build_criterion()`'s initial weights).

### 6.5 Training Hyperparameters (Stage 2)

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Max epochs | 30 |
| Learning rate | 7e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| LR scheduler | OneCycleLR (pct_start=0.2) |
| Precision | FP16 |
| KD temperature | 2.0 |
| Contrastive temperature | 0.1 |
| Checkpoint selection | Best val_fmri_mf1 |

Only the fMRI student parameters are optimized:
- `fmri_encoder` parameters
- `fmri_mapper` parameters
- `fmri_classifier` parameters

### 6.6 Optimizer Configuration

The optimizer for Stage 2 has a significantly higher learning rate (7e-4) compared to Stage 1 (3e-5). This makes sense because Stage 2 trains the fMRI student from scratch (randomly initialized with Xavier uniform), whereas Stage 1 only fine-tunes classification heads on top of frozen pretrained encoders.

---

## 7. Dataset Structure and Vigilance Labels

### 7.1 Primary Dataset: EEGfMRI-VU

**File:** `vigilance_datasets/eegfmri_vu_alphatheta_smallinterval_1024.py`

The `EEGfMRIVuAlphaThetaSmallInterval1024Dataset` class handles the Vanderbilt simultaneous EEG-fMRI dataset.

**Data sources per scan:**
- **EEG:** `.set` files (EEGLAB format), 26 channels at 250 Hz
- **fMRI:** CSV files with 1024 DiFuMo ROI values per TR
- **Physio:** MATLAB `.mat` files with 5 physiological regressors (from REGS field)
- **Vigilance index:** MATLAB `.mat` files with `VIG_SIG` structure containing multiple vigilance metrics

**Key data dimensions after loading:**
- fMRI: `[num_scans, num_TRs, 1024]` (1024 DiFuMo ROIs)
- EEG: `[num_scans, num_samples, 26]` (26 channels)
- EEG-to-fMRI ratio: 525 EEG samples per fMRI TR (250 Hz x 2.1s)

### 7.2 Vigilance Label Extraction

The repository computes vigilance labels from EEG using the alpha/theta power ratio:

1. **Channel selection for alertness:** P3, P4, Pz, O1, O2, Oz (posterior channels)
2. **Signal averaging:** Mean across selected channels
3. **Band power computation per TR window (2.1s = 525 samples):**
   - Alpha band: 8-12 Hz, using Welch's method PSD
   - Theta band: 3-7 Hz, using Welch's method PSD
   - RMS amplitude = sqrt(band_power)
4. **Alpha/theta ratio:** `alpha_power / (theta_power + 1e-10)`
5. **Smoothing:** Moving average with window_size=5, edge padding

The actual binary labels come from a pre-computed `VIG_SIG` MATLAB structure:
- `VIG_SIG[0][0][1]` = linear raw vigilance index
- `VIG_SIG[0][0][3]` = binary vigilance labels (pre-thresholded)

### 7.3 Windowing Strategy

The dataset creates windowed segments for training:

```python
step_size = 5         # stride in TRs
step_seg_length = 5   # window length in TRs
vigilance_window_size = 5   # TRs per vigilance label
vigilance_threshold = -1    # threshold for binary label
```

Each window contains:
- fMRI: `[5, 1024]` (5 TRs x 1024 ROIs)
- EEG: `[2625, 26]` (5 TRs x 525 samples x 26 channels)
- Vigilance label: 1 value per window (binary: alert or drowsy)

The vigilance label per window is computed by summing the binary vigilance scores within the window and thresholding. With `vigilance_threshold=-1`, any sum > -1 (i.e., sum >= 0) means alert. Since binary scores are 0 or 1, this effectively means the window is labeled as alert if at least one of the 5 TRs has a positive binary label.

### 7.4 Hemodynamic Response Compensation

The fMRI data is shifted forward by 2 TRs to account for the hemodynamic response delay:

```python
slide_cnt = (fmri_data_00.shape[0] - 2) // 5
fmri_data = fmri_data_00.iloc[2:slide_cnt*5+2, :1024]
```

This means fMRI data starting from TR 3 is aligned with EEG data starting from TR 1.

### 7.5 Train/Test Splitting

The dataset supports multiple split modes:
- `train` / `test`: Predefined subject-level splits via `FMRI_PROC_DIRS_TRAIN` / `FMRI_PROC_DIRS_TEST`
- `fewshot_train` / `fewshot_test`: Smaller splits for few-shot learning scenarios
- `zero_shot`: All scans (for cross-dataset evaluation)

The split is subject-level (not scan-level), preventing data leakage between related scans from the same person.

### 7.6 fMRI Data Preprocessing in Dataset

Before windowing, the fMRI ROI timeseries are standardized per scan:
```python
scaler = StandardScaler()
temp_fmri_data = scaler.fit_transform(temp_fmri_data_raw)
```

Physiological confounds are imputed via linear interpolation for any NaN values.

### 7.7 NIH-ECR Dataset

**File:** `vigilance_datasets/NIHECR_alphatheta_smallinterval_1024.py`

The NIH-ECR dataset follows the same structure but has different preprocessing requirements:
- Extra channels that need to be dropped: FC1, FC2, CP1, CP2, FC5, FC6, CP5, CP6
- Missing channels zero-padded: FPz, POz, FT9, FT10
- Channel renaming: TP9 -> TP9', TP10 -> TP10'
- Different event naming conventions
- Polynomial drift confound removal (4th order) for fMRI

### 7.8 Dataset `__getitem__` Return Format

Each sample returns an 8-tuple:
```python
(fmri, eeg, physio, eeg_index_linear_raw, eeg_index_linear_smoothed,
 eeg_index_binary, alpha_theta_ratio, vigilance_seg)
```

Where:
- `fmri`: `[5, 1024]` - 5 TRs x 1024 ROIs
- `eeg`: `[2625, 26]` - 2625 samples x 26 channels
- `physio`: `[5, 5]` - 5 TRs x 5 regressors
- `eeg_index_linear_raw`: `[5]` - raw vigilance index per TR
- `eeg_index_linear_smoothed`: `[5]` - smoothed vigilance index
- `eeg_index_binary`: `[5]` - binary vigilance per TR
- `alpha_theta_ratio`: `[5]` - alpha/theta ratio per TR
- `vigilance_seg`: `[1]` - binary label for the window

---

## 8. fMRI Preprocessing

### 8.1 Atlas: DiFuMo 1024

**Files:** `data_preprocessing/fmri_atlas/EEGfMRI_VU_fmri_fit_atlas_batch_1024.py`, `NIH_fmri_fit_atlas_batch_1024.py`

The fMRI data is parcellated using the DiFuMo 1024 atlas:

```python
atlas = datasets.fetch_atlas_difumo(dimension=1024, resolution_mm=2)
maps_img = atlas.maps
maps_masker = input_data.NiftiMapsMasker(
    maps_img=maps_img,
    verbose=1,
    detrend=True,
    standardize=True,
    standardize_confounds=True,
    high_variance_confounds=False,
)
```

DiFuMo (Dictionary of Functional Modes) provides soft probabilistic parcellations rather than hard boundaries, which can capture more nuanced spatial patterns. The 1024-dimensional atlas provides fine-grained brain region coverage.

### 8.2 Confound Handling

**VU dataset:** Motion parameters from `.volreg_par` files (6 parameters: 3 rotation + 3 translation) are used as confounds during `NiftiMapsMasker.fit_transform()`.

**NIH dataset:** Motion parameters plus 4th-order polynomial drift regressors:
```python
poly_confound = poly_drift(4, time_seq)
confounds = pd.concat([motion_confound, poly_df], axis=1)
```

### 8.3 Global Signal Extraction

Both preprocessing scripts also extract global signal (with and without confound removal) and append it to the ROI timeseries:

```python
df_fmri['global signal clean'] = ts_global_signal_clean
df_fmri['global signal raw'] = ts_global_signal
```

### 8.4 EEG Preprocessing

EEG preprocessing is done in MATLAB (files in `data_preprocessing/eeg_filtering/`):
- `EEGfMRI_VU_convertEEG_to_set_batch.m` - Convert raw EEG to EEGLAB `.set` format
- `EEGfMRI_VU_EEG_removechannels_batch.m` - Remove non-standard channels, keep 26

---

## 9. Model Architecture Details

### 9.1 Transformer Encoder (`models/transformer.py`)

The TransformerEncoder is a modified version from DETR (Facebook Research). Key features:

- **Pre-norm architecture:** LayerNorm is applied before attention and FFN (not after)
- **Positional encoding support:** Positions are added to query and key (not value)
- **Masking support:** Optional attention masks (used in MaskedTransformerEncoder variant)
- **Returns 3 values:** `(xyz, output, xyz_inds)` where only `output` matters for fMRI use

The TransformerEncoderLayer:
```python
TransformerEncoderLayer(
    d_model=1024,        # matching DiFuMo ROI count
    nhead=4,             # 4 attention heads
    dim_feedforward=512, # FFN hidden dim (half of d_model)
    dropout=0.1,
    normalize_before=True,  # pre-norm
    norm_name="ln",
    use_ffn=True,
)
```

### 9.2 Helper Modules (`models/helpers.py`)

Provides:
- `NORM_DICT`: BatchNorm, LayerNorm, Identity
- `ACTIVATION_DICT`: ReLU, GELU, LeakyReLU
- `WEIGHT_INIT_DICT`: Xavier uniform
- `GenericMLP`: Flexible MLP with configurable layers, norms, dropout
- `SequentialCNN2D`: 2D CNN for spectrogram features (not used in main pipeline)
- `get_clones`: Deep copies a module N times for TransformerEncoder layers

### 9.3 Temporal Interpolation (`utils_EEGPT.py`)

A simple utility for resampling EEG signals:

```python
def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
```

Uses nearest-neighbor interpolation by default and subtracts the channel-wise mean before resampling.

---

## 10. Key Differences from Our Project

### 10.1 Atlas and Parcellation

| Aspect | FewShotKDVigilance | Our Project |
|--------|-------------------|-------------|
| Atlas | DiFuMo 1024 (soft probabilistic maps) | Schaefer 200 (hard parcels, 7-network) |
| ROI count | 1024 | ~210 (200 cortical + ~10 subcortical) |
| Subcortical | Included implicitly via DiFuMo's brain-wide coverage | Explicitly added from FreeSurfer aparc+aseg |
| fMRI features | Raw ROI timeseries (1024-dim per TR) | Correlation matrices between ROIs |

**Implications:** Their fMRI input is a `[5, 1024]` timeseries tensor per window. Our fMRI input will be a graph with ~210 nodes where each node has a connectivity profile + gene expression vector. Their `d_model=1024` is set to match DiFuMo dimensions; we need to set our GNN input dimension to match our concatenated feature size.

### 10.2 Student Model Architecture

| Aspect | FewShotKDVigilance | Our Project |
|--------|-------------------|-------------|
| Model type | 2-layer Transformer encoder | GNN (GCN/GAT from PyTorch Geometric) |
| Input | fMRI ROI timeseries [5, 1024] | Brain graph (~210 nodes, top-k edges) |
| Spatial structure | None (treats ROIs as sequence tokens) | Explicit graph connectivity |
| Pooling | Flatten 5 segments x 512 dims | Global mean/max/attention pooling |

**Implications:** Their Transformer treats each TR as a token and attends over the temporal dimension. Our GNN treats each brain region as a node and performs message passing over the connectivity graph. The graph structure provides explicit spatial priors about brain organization.

### 10.3 Classification Task

| Aspect | FewShotKDVigilance | Our Project |
|--------|-------------------|-------------|
| Classes | 2 (alert vs drowsy) | 3 (alert, intermediate, drowsy) |
| Output dim | 2 | 3 |
| Label source | Pre-computed VIG_SIG binary + alpha/theta | EEG alpha/theta ratio with 3 thresholds |
| Window size | 5 TRs (~10.5 seconds) | 1 minute (per Darrin/Maggie) |

**Implications:** We need to adjust all classifier output dimensions from 2 to 3. The KD loss functions (KL divergence, cross-entropy) work the same way with 3 classes - just different output dimensions. Our larger window sizes mean fewer samples per subject but potentially more stable labels.

### 10.4 Node Features

| Aspect | FewShotKDVigilance | Our Project |
|--------|-------------------|-------------|
| Node features | Raw fMRI activation timeseries | fMRI connectivity profile + gene expression |
| Genomics | None | ~294 sleep/circadian genes per ROI |
| Feature source | DiFuMo atlas extraction | Schaefer parcellation + Allen Brain Atlas |

**Implications:** Our genomics-informed features are a unique contribution. Each node will have a concatenated feature vector of (connectivity_dim + 294) dimensions, providing biologically meaningful priors about sleep-related gene expression patterns.

### 10.5 Teacher Ensemble

| Aspect | FewShotKDVigilance | Our Project |
|--------|-------------------|-------------|
| Teachers | 3 (EEGPT + LaBraM + BIOT) | 1 (EEGPT only, as decided at this stage) |
| Logit averaging | Mean of 3 teacher logits | Single teacher logits |
| Feature KD | MSE against each teacher, averaged | MSE against single teacher |

**Implications:** With a single teacher, our KD pipeline is simpler. The logit KD loss does not need averaging, and the feature KD loss targets one feature space. If time permits, we could consider adding LaBraM or BIOT as additional teachers, reusing their Stage 1 fine-tuning code directly from this repo.

### 10.6 fMRI Preprocessing

| Aspect | FewShotKDVigilance | Our Project |
|--------|-------------------|-------------|
| Tool | NiftiMapsMasker (nilearn) | Schaefer parcellation (nilearn/NatView pre-extracted) |
| Confounds | Motion (6 params) + optional polynomial drift | TBD (Maggie's preprocessing) |
| Normalization | StandardScaler per scan | TBD |
| Hemodynamic delay | 2 TR forward shift | 5 second forward shift |

---

## 11. Reusable Components

### 11.1 Directly Reusable

1. **`utils_EEGPT.py` - `temporal_interpolation()`:** Essential for feeding any EEG data into EEGPT. Can be used as-is.

2. **`models/transformer.py` - `TransformerEncoder` and `TransformerEncoderLayer`:** If we want a Transformer-based baseline for comparison, these are clean implementations. Modified from DETR with pre-norm, positional encoding support.

3. **`models/helpers.py`:** Useful utilities: `WEIGHT_INIT_DICT`, `NORM_DICT`, `ACTIVATION_DICT`, `GenericMLP`, `get_clones`.

4. **Stage 1 EEGPT training pattern:** The entire `LitEEGPTCausal` class from `stage1_EEGPT_fewshot_train.py` shows exactly how to load, freeze, and fine-tune EEGPT for vigilance classification. This is directly adaptable for our teacher.

5. **Loss function architecture (`SetCriterion`):** The weighted multi-loss framework with dynamic weight updating is well-structured. The KD loss implementations (KL divergence with temperature, feature MSE) can be adapted.

### 11.2 Adaptable with Modifications

1. **Stage 2 KD training loop:** The two-phase training strategy (KD-heavy then supervised-heavy) is a proven approach. We adapt it for our GNN student by:
   - Replacing `fmri_encoder` (Transformer) with our GNN
   - Changing `d_model` to match our feature dimensions
   - Adjusting output from 2 to 3 classes

2. **Channel selection and mapping:** The 26-channel selection and `use_channels_names` pattern are reusable for any 10-20 system EEG data. The channel duplication trick for missing EEGPT channel IDs is important to note.

3. **Contrastive loss implementation:** The supervised contrastive loss with cosine similarity is generic and can work with any feature representation.

### 11.3 Not Directly Applicable

1. **Dataset classes:** Our data format (NatView with Schaefer parcellation) is completely different from VU/NIH with DiFuMo 1024. We need custom dataset classes.

2. **fMRI student architecture:** The Transformer student is designed for 1024-dim timeseries input. Our GNN student operates on graph-structured data. However, the mapper + classifier MLP pattern (feature_dim -> 1024 -> 512 -> 256 -> num_classes) is a useful reference.

---

## 12. Recommended Adaptations

### 12.1 EEG Teacher (Priority)

Reuse the EEGPT fine-tuning pattern from `stage1_EEGPT_fewshot_train.py`:

1. Load EEGPT encoder from the same pretrained checkpoint
2. Use the same 26 channels and channel mapping
3. Adapt the temporal interpolation for our NatView EEG sampling rate
4. Change the classifier output from 2 to 3 classes
5. Keep the same training hyperparameters as a starting point (lr=3e-5, epochs=10)

If David chooses a different teacher model (SleepTransformer), the distillation framework still applies - we just replace the encoder.

### 12.2 KD Loss Function

Adapt the `SetCriterion` from Stage 2:

```python
# Simplified for single teacher, 3 classes
loss_weight_dict = {
    "loss_fmri_cls": 0.2,     # CE with pseudo or GT labels
    "loss_kd_logit": 0.4,     # KL div from teacher
    "loss_kd_feat": 0.3,      # MSE on features
    "loss_contrastive": 0.1,  # supervised contrastive
}
```

Key adaptations:
- Remove multi-teacher averaging (single EEGPT teacher)
- Change output dim from 2 to 3 in all losses
- Keep T=2.0 as starting temperature
- Keep two-phase training strategy

### 12.3 GNN Student

Replace the Transformer student with our GNN:

```python
# Their pattern:
fMRI [B, 5, 1024] -> TransformerEncoder -> mapper [1024->512] -> classifier [2560->2]

# Our pattern:
Graph [~210 nodes, edges, features] -> GCN/GAT layers -> global_pool -> mapper -> classifier [feat_dim->3]
```

The mapper + classifier MLP structure is transferable. Match the intermediate feature dimension (512) if we want to try feature KD from their pretrained teachers.

### 12.4 Two-Phase Training

Adopt the same two-phase approach:
- Phase 1 (first 2/3 of epochs): High KD weight, pseudo-labels from teacher
- Phase 2 (last 1/3): High supervised weight, ground-truth labels

### 12.5 Evaluation

Their evaluation pattern:
- Macro F1 score as primary metric (handles class imbalance)
- Per-class F1 breakdown (F1_drowsy, F1_alert)
- Few-shot (within-dataset) and zero-shot (cross-dataset) testing
- Feature visualization via saved features (`harvest_feats` function)

We should adopt macro F1 as primary metric and add per-class F1 for all 3 classes.

---

## 13. HuggingFace Checkpoints and Pretrained Weights

### 13.1 EEGPT

The EEGPT pretrained checkpoint is `eegpt_mcae_58chs_4s_large4E.ckpt`:
- Architecture: Masked Channel Autoencoder with EEGTransformer backbone
- Pretraining: Self-supervised on 58-channel EEG data, 4-second windows
- Variant: "large4E" (large model with 4 embed tokens)
- Parameters: embed_dim=512, depth=8, num_heads=8
- Source: [BINE022/EEGPT GitHub](https://github.com/BINE022/EEGPT)

The checkpoint contains the full autoencoder; only the `target_encoder` portion is used for downstream tasks.

### 13.2 LaBraM

Pretrained checkpoint: `Modules/LaBraM/labram-base.pth`
- Architecture: ViT-style transformer with patch embeddings for EEG
- Model name registered as `labram_base_patch200_200` in timm
- Patch size: 200 samples per patch
- State dict uses `student.` prefix (from knowledge distillation pretraining)
- Loaded with `strict=False` to handle missing classification head weights
- Source: [935963004/LaBraM](https://github.com/935963004/LaBraM)

### 13.3 BIOT

Three pretrained variants available:
1. `EEG-PREST-16-channels.ckpt` (16 channels)
2. `EEG-SHHS+PREST-18-channels.ckpt` (18 channels)
3. `EEG-six-datasets-18-channels.ckpt` (18 channels, default choice)

The weights are loaded into `BIOTClassifier.biot` (the encoder portion):
```python
model.biot.load_state_dict(torch.load(pretrain_models[2]))
```
- Architecture: Transformer on STFT representations (n_fft=200, hop_length=100)
- Source: [ycq091044/BIOT](https://github.com/ycq091044/BIOT)

### 13.4 Fine-tuned Checkpoints

After Stage 1 training, checkpoints are saved at:
- `EEGvigilance/stage1_EEGPT_fewshot/best-test-mf1-{epoch}-{val_mf1}.ckpt`
- `EEGvigilance/stage1_labram_fewshot/best-test-mf1-{epoch}-{val_mf1}.ckpt`
- `EEGvigilance/stage1_BIOT_fewshot/best-test-mf1-{epoch}-{val_mf1}.ckpt`

After Stage 2 KD training:
- `EEGvigilance/stage2_labramBIOTEEGPT_kd_fewshot_transformer/best-test-mf1-{epoch}-{val_fmri_mf1}.ckpt`

These checkpoints contain the complete `state_dict` of the PyTorch Lightning module, including all teachers and the student. They are not published on HuggingFace but could theoretically be loaded for transfer learning.

---

## Summary

FewShotKDVigilance provides a complete, working implementation of multi-teacher knowledge distillation from EEG to fMRI for vigilance classification. The key insights for our project are:

1. **Freeze pretrained encoders, train only classification heads** in Stage 1
2. **Multi-component loss with two-phase training** is more effective than simple KD
3. **Temporal interpolation** is critical for matching different encoder input expectations
4. **Feature-level KD** (MSE on intermediate representations) complements logit-level KD
5. **Pseudo-labels from teachers** provide useful supervision when ground truth is limited
6. **The fMRI student architecture is modular** - we can swap their Transformer for our GNN while keeping the same loss framework and training strategy

The main architectural differences between their project and ours (Transformer vs GNN student, DiFuMo 1024 vs Schaefer 210, binary vs 3-class, no genomics vs genomics-informed) do not prevent us from leveraging their distillation framework. The loss functions, training strategy, and teacher fine-tuning code are all adaptable to our setting.
