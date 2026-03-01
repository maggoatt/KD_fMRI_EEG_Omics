# EEGPT-KD Pipeline Status and Design Decisions

## Overview

Knowledge distillation pipeline: EEGPT (frozen pretrained EEG transformer) as teacher,
ConnectivityGCN (EdgeConv-based GNN) as student, trained on fMRI brain graphs to classify
vigilance states (alert/drowsy). Uses LOSO cross-validation on NatView EEG-fMRI data.

## Architecture

```
                    STAGE 1: Teacher Training
                    -------------------------
Raw EEG (64ch, 5000Hz)
  -> Drop to 58 EEGPT channels
  -> Resample to 256 Hz
  -> Cut into 30-second windows (7680 samples each)
  -> temporal_interpolation to exactly 7680 if needed
  -> Frozen EEGPT encoder -> 120 patches x 2048-dim features
  -> Add sinusoidal positional encoding
  -> Mean pool across patches -> 2048-dim feature vector
  -> Trainable linear head -> 2-class logits
  -> Supervised with binary vigilance labels (from Darrin's alpha/theta pipeline)
  -> Cache: teacher_logits (N, 2) + teacher_features (N, 2048)
  NOTE: "4s" in checkpoint name = pretraining window. Downstream uses 30s windows.
        Patch embeddings (64 samples each) are length-agnostic.

                    STAGE 2: Student KD Training (per LOSO fold)
                    --------------------------------------------
fMRI correlation matrices (210 x 210 per interval)
  -> Top-k=10 graph construction
  -> Node features: mean connectivity (210, 1) [+ gene expression (210, 294)]
  -> ConnectivityGCN (3x EdgeConv, BatchNorm, global_mean_pool)
  -> KD Loss: alpha * CE(student, true) + (1-alpha) * KL(teacher_soft, student_soft) * T^2
  -> Two-phase schedule: epochs 0-19 alpha=0.2 (KD-heavy), epochs 20-29 alpha=0.8 (CE-heavy)
```

## Data Sources and Status

### 1. EEG Vigilance Labels (DONE)

Source: Darrin's Google Drive upload
Location: `eeg_data/eeg_vig_TR=0.25s_Patch=120/`
Converted to: `EEGPT-KD/data/labels/sub-*_labels.npy`

Format of source TSVs:
- `*_frames.tsv`: Per-0.25s frame. Columns: vigilance_score (alpha/theta ratio), label_ternary (-1/0/+1)
- `*_patches.tsv`: Per-30s window. Columns: window_sum, label_binary (0=drowsy, 1=alert)
- `*_epochs.tsv`: Consecutive same-label runs

Pipeline that produced these (Darrin's eeg_prep.py):
1. Compute alpha (8-12 Hz) / theta (4-8 Hz) power ratio per 0.25s frame
2. Smooth with 5-frame moving average
3. Percentile threshold: top 33% = alert(+1), bottom 33% = drowsy(-1), middle = intermediate(0)
4. Apply HRF delay shift (~5.5s backward) to align with fMRI BOLD lag
5. 120-frame (30s) sliding window: sum ternary scores, threshold at -1
   - sum >= -1 -> Alert (1)
   - sum < -1 -> Drowsy (0)

Dataset summary:
- 10 subjects (sub-01 through sub-10; sub-11 incomplete)
- 16 sessions total (some subjects have 2 sessions)
- 308 total patches (30s each)
- 165 alert (53.6%) / 143 drowsy (46.4%) - roughly balanced
- ~20 patches per session, ~31 per subject (subjects with 2 sessions have more)

### 2. Raw EEG for EEGPT Teacher (NOT YET AVAILABLE)

Needed for: Stage 1 teacher training (feeding raw signal through EEGPT encoder)
Format expected: Raw continuous EEG at 5000 Hz, 64 channels (.set or .edf files)
Source: NatView dataset, simultaneous EEG-fMRI

Preprocessing required before EEGPT:
- MR gradient artifact removal (if not already done)
- Select 58 channels matching EEGPT_CHANNELS (drop 6 non-standard channels)
- Resample from 5000 Hz to 256 Hz
- Cut into 4-second windows (1024 samples = img_size[1] that EEGPT expects)
- Per 30s interval: 7-8 windows of 4s each, average EEGPT features across windows

EEGPT input constraint: Fixed positional embeddings, MUST be exactly (B, 58, 1024).
Cannot handle variable length - the checkpoint was trained on this exact shape.

### 3. fMRI Correlation Matrices (NOT YET AVAILABLE)

Needed for: Stage 2 student GNN training
Format expected: sub-*_interval_corr.npy (N_intervals, 210, 210) per subject
Source: Maggie's fMRI preprocessing pipeline

Key question: What is Maggie's interval duration?
- If 60s intervals: ~10 per session, need to aggregate 2 EEG patches per fMRI interval
- If 30s intervals: 1:1 mapping with EEG patches (ideal)
- The label-to-graph alignment MUST match exactly

### 4. Gene Expression (DONE)

Location: `KD_fMRI_EEG_Omics/sample_data/` or `preprocessing/omics_preprocessing/`
Format: (210, 294) matrix - 210 ROIs x 294 sleep/circadian genes
Source: Allen Human Brain Atlas, preprocessed by Gautham
Status: Static features, same for all subjects/intervals

## Interval Alignment Problem

CRITICAL: The EEG labels use 30-second patches, but the original CLAUDE.md says fMRI uses
60-second (1-minute) intervals. These MUST be aligned:

Option A (if fMRI = 30s intervals): Direct 1:1 mapping. Each fMRI graph gets one binary label.
Option B (if fMRI = 60s intervals): Aggregate 2 EEG patches per fMRI interval:
  - Both alert -> alert
  - Both drowsy -> drowsy
  - Mixed -> majority vote, or use the more recent patch

Need to confirm with Maggie what her interval duration is.

## File Structure

```
EEGPT-KD/
  setup.sh                    # Clone EEGPT repo, checkpoint download instructions
  .gitignore                  # vendor/, checkpoints/, __pycache__, *.npy, *.ckpt
  requirements.txt            # torch, torch-geometric, numpy, pandas, sklearn, etc.
  teacher/
    __init__.py               # exports EEGPTTeacher
    eegpt_adapter.py          # Frozen EEGPT encoder + linear head, feature caching
  student/
    __init__.py               # exports ConnectivityGCN, VigilanceGraphDataset
    gnn_model.py              # 3-layer EdgeConv GNN, returns {logits, features}
    graph_dataset.py          # Loads fMRI graphs + optional gene expression
  distillation/
    __init__.py               # exports KDLoss, LOSOTrainer
    kd_loss.py                # alpha*CE + (1-alpha)*KL*T^2, optional feature MSE
    trainer.py                # LOSO CV with two-phase alpha schedule
  evaluation/
    __init__.py               # exports compute_metrics, print_results_table
    metrics.py                # balanced_accuracy, f1, auroc, confusion matrix
  scripts/
    convert_eeg_labels.py     # Converts Darrin's TSVs to .npy label files
  data/
    labels/                   # Converted .npy label files (sub-*_labels.npy)
  notebooks/
    kaggle_pipeline.ipynb     # Self-contained Kaggle notebook with synthetic fallback
  docs/
    pipeline-status.md        # This file
```

## Design Decisions

### Why 2-class (binary) instead of 3-class?

The original plan discussed 3-class (alert/intermediate/drowsy), but Darrin's labeling
pipeline produces binary labels. The Vanderbilt reference paper also uses binary.
Rationale: intermediate is ambiguous and adds noise; binary is cleaner for KD.

### Why EdgeConv instead of standard GCN?

Teammate's original architecture (model/gnn_pipeline.ipynb) uses EdgeConv with max
aggregation. EdgeConv is better for brain graphs because:
- It operates on edge features (correlation weights), not just adjacency
- Max aggregation captures strongest connection patterns
- More expressive than standard message passing for correlation-based graphs

### Why two-phase alpha schedule?

- Phase 1 (epochs 0-19, alpha=0.2): Student learns mostly from teacher soft labels.
  The teacher has rich 2048-dim feature knowledge the student needs early.
- Phase 2 (epochs 20-29, alpha=0.8): Student fine-tunes on hard labels.
  Prevents the student from being locked into teacher's mistakes.

### Why pre-cache teacher outputs?

EEGPT is large (512 embed_dim, 8 layers, 8 heads). Running it every epoch during
student training would be wasteful. Instead:
1. Run teacher once on all EEG data -> cache logits + features
2. Attach cached outputs to PyG Data objects
3. DataLoader shuffling works correctly since cache travels with each sample

### Why not use fMRI TR (2.1s) as input to EEGPT?

EEGPT expects raw 256 Hz EEG in 4-second windows (1024 samples). It was pretrained
on this input format with fixed positional embeddings. Feeding TR-level data (one
value per 2.1s) would:
- Not match the expected temporal resolution
- Produce meaningless patch embeddings (model expects continuous waveforms)
- Require interpolation that distorts the signal beyond recognition

## What's Blocking Progress

1. **Raw EEG files** - Need continuous EEG for EEGPT teacher. Darrin has the preprocessed
   labels but we need the actual signal data (.set or .edf files).

2. **fMRI correlation matrices** - Need Maggie's sub-*_interval_corr.npy files for the
   GNN student. Without these, we can only test with synthetic data.

3. **Interval alignment confirmation** - Need to confirm fMRI interval duration (30s vs 60s)
   to ensure label-graph alignment.

4. **EEGPT checkpoint** - Need to download from Figshare (~large file). run setup.sh.

## Next Steps (when data arrives)

1. Download EEGPT checkpoint (run setup.sh)
2. Preprocess raw EEG: 64ch/5000Hz -> 58ch/256Hz -> 4s windows
3. Train teacher head (linear probe on frozen EEGPT features)
4. Cache teacher outputs for all training samples
5. Load fMRI correlation matrices into VigilanceGraphDataset
6. Align labels with fMRI intervals
7. Run LOSO baseline (hard labels only) and KD experiments
8. Ablation: connectivity-only vs connectivity+genomics node features
