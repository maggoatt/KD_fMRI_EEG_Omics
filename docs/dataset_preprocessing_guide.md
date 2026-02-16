# Dataset Preprocessing Guide

**Last updated**: February 16, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Neuroscience Glossary](#2-neuroscience-glossary)
3. [Dataset: NatView EEG-fMRI](#3-dataset-natview-eeg-fmri)
4. [EEG Preprocessing (eeg_prep.py)](#4-eeg-preprocessing-eeg_preppy)
5. [Brain Atlases Used](#5-brain-atlases-used)
6. [What the fMRI Preprocessing Notebook Does](#6-what-the-fmri-preprocessing-notebook-does)
7. [How the Current Graphs and Visualizations Are Constructed](#7-how-the-current-graphs-and-visualizations-are-constructed)
8. [Interval / Epoch Segmentation (Implemented)](#8-interval--epoch-segmentation-implemented)
9. [Remaining Graph Construction Steps](#9-remaining-graph-construction-steps)
10. [Data Shapes and What They Mean](#10-data-shapes-and-what-they-mean)
11. [How This Feeds Into the Model](#11-how-this-feeds-into-the-model)

---

## 1. Project Overview

### The Big Picture

We are building a system that can detect whether a person is **alert or drowsy** using only their brain scan (fMRI), without needing an EEG cap on their head. The trick: we first train a "teacher" model on EEG data (which is very good at detecting alertness), then transfer that knowledge to a "student" model that only needs fMRI data. This transfer process is called **knowledge distillation**.

### Why This Matters

- **EEG** (electroencephalography) measures electrical brain activity directly. It's excellent at detecting alertness in real-time, but requires wearing a cap with electrodes — impractical in many clinical settings.
- **fMRI** (functional magnetic resonance imaging) measures blood flow changes in the brain. It doesn't require any wearable equipment (just the MRI scanner), but is much worse at detecting alertness on its own.
- By distilling knowledge from EEG into an fMRI model, we get the best of both worlds: fMRI-only alertness detection with EEG-level accuracy.

### The Three Components

| Component | Role | Data Source |
|-----------|------|-------------|
| **EEG Teacher Model** | Learns to classify alertness from brainwave patterns | EEG recordings |
| **fMRI Student Model** | Learns to classify alertness from brain connectivity graphs | fMRI brain scans |
| **Omics Attention** | Tells the fMRI model which brain regions are genetically relevant to alertness | Allen Human Brain Atlas (gene expression data) |

This document covers **EEG and fMRI preprocessing** for the NatView dataset: how we turn raw EEG recordings into vigilance labels (teacher side) and raw brain scans into structured FC graphs (student side).

---

## 2. Neuroscience Glossary

### Brain Imaging Terms

| Term | Plain-English Definition |
|------|------------------------|
| **fMRI** | Functional Magnetic Resonance Imaging. A brain scanning technique that detects which brain areas are active by measuring changes in blood oxygen levels. When neurons fire, they need more oxygen, so nearby blood vessels deliver more oxygenated blood. fMRI detects this change. |
| **BOLD signal** | Blood-Oxygen-Level-Dependent signal. The actual measurement that fMRI captures — the ratio of oxygenated to deoxygenated blood in each tiny cube (voxel) of the brain, measured over time. When a brain region is active, its BOLD signal goes up. |
| **EEG** | Electroencephalography. Measures electrical activity of the brain using electrodes placed on the scalp. Much faster temporal resolution than fMRI (milliseconds vs. seconds), but much worse spatial resolution (can't pinpoint exactly where in the brain the signal comes from). |
| **Resting state** | A scan condition where the participant lies in the scanner doing nothing — no task, just resting with eyes open or closed. Brain regions still show spontaneous, organized activity patterns even at rest, and these patterns are informative about brain health and cognitive state. |
| **TR (Repetition Time)** | The time between consecutive fMRI measurements. In our dataset, TR = 2.1 seconds. This means we get one complete brain snapshot every 2.1 seconds. A 600-second (10-minute) scan produces ~288 snapshots (timepoints). |
| **Voxel** | A 3D pixel — the smallest unit of an fMRI image. Our data uses 3mm voxels, meaning each cube is 3mm × 3mm × 3mm. A typical brain contains ~50,000–100,000 voxels. |

### Brain Anatomy Terms

| Term | Plain-English Definition |
|------|------------------------|
| **Cortex / Cortical** | The outer layer of the brain (the wrinkly surface). Responsible for higher-level functions: thinking, perception, language, decision-making. Divided into left and right **hemispheres**. |
| **Subcortical** | Structures deeper inside the brain, below the cortex. Includes regions like the thalamus, hippocampus, and amygdala. Often involved in more automatic/primitive functions: arousal, memory, emotion. |
| **Thalamus** | A subcortical relay station that routes sensory information to the cortex. Critically involved in consciousness and arousal — it's the "gatekeeper" that determines whether you stay awake or fall asleep. The single most important subcortical region for our vigilance classification task. |
| **Hippocampus** | A subcortical structure essential for memory formation. Shows connectivity changes during drowsiness transitions. |
| **Amygdala** | A subcortical structure involved in emotional processing and arousal regulation. |
| **Caudate / Putamen** | Parts of the basal ganglia (subcortical motor/reward circuits). Involved in maintaining wakefulness and arousal. |
| **ROI (Region of Interest)** | A defined brain area selected for analysis. Instead of analyzing all ~50,000 voxels individually, we group them into ~200 ROIs. Each ROI is a cluster of voxels with similar function. |
| **Parcellation / Parcel** | The process of dividing the brain into non-overlapping ROIs using an atlas. A "parcel" is one such region. Think of it like a map dividing a country into states. |

### Connectivity and Graph Terms

| Term | Plain-English Definition |
|------|------------------------|
| **Functional Connectivity (FC)** | A statistical measure of how synchronized two brain regions are. If two ROIs have similar BOLD signal patterns over time (i.e., they "activate together"), they are functionally connected. Measured using Pearson correlation. |
| **Correlation Matrix** | A square matrix (N_ROIs × N_ROIs) where each cell contains the correlation between two ROIs' timeseries. Values range from -1 (perfectly anti-correlated) to +1 (perfectly correlated). The diagonal is always 1 (a region correlates perfectly with itself). |
| **FC Graph** | A graph data structure representing the brain as a network. Each ROI becomes a **node**, and the strongest correlations between ROIs become **edges**. This is the input to the GNN student model. |
| **Node** | A vertex in the graph. Each node = one brain ROI. We have 210 nodes (200 cortical + 10 subcortical). |
| **Edge** | A connection between two nodes, weighted by their correlation. We only keep the top-K strongest edges per node to create a sparse graph. |
| **Top-K edges** | A graph sparsification method: for each node, keep only the K strongest connections (by absolute correlation) and discard the rest. This removes weak/noisy connections and makes the graph manageable. |
| **GNN (Graph Neural Network)** | A type of neural network designed to operate on graph-structured data. It learns by aggregating information from a node's neighbors, then its neighbors' neighbors, and so on. Perfect for brain networks where the spatial arrangement matters. |

### Preprocessing Terms

| Term | Plain-English Definition |
|------|------------------------|
| **Atlas** | A predefined map of the brain that assigns every voxel to a labeled region. Like a political map that assigns every point on Earth to a country. We use two atlases: Schaefer 2018 (cortical) and Harvard-Oxford (subcortical). |
| **MNI space (MNI152)** | A standard brain coordinate system. Everyone's brain is slightly different in shape/size, so to compare across people, all brains are warped ("registered") to match a standard template brain called MNI152. All our data is in MNI space. |
| **GSR (Global Signal Regression)** | A preprocessing step that removes the average signal across the entire brain from every region's timeseries. Controversial: it removes noise but may also remove real signal. We use the **non-GSR** variant for our pipeline. |
| **Timeseries** | A sequence of measurements over time. For each ROI, the timeseries is the average BOLD signal value at each timepoint (each TR). Shape: (288 timepoints,) per ROI. |
| **Fisher z-transformation** | A mathematical transform (`z = arctanh(r)`) applied to correlation values to make them approximately normally distributed. Useful for cross-subject statistical comparisons and task-based analyses, but **not needed** for our resting-state within-subject FC pipeline (per advisor guidance). |

### Vigilance / Alertness Terms

| Term | Plain-English Definition |
|------|------------------------|
| **Vigilance** | The level of alertness/wakefulness. Ranges from fully alert to drowsy to asleep. |
| **Alpha waves (8-12 Hz)** | EEG oscillations associated with relaxed wakefulness. Prominent when eyes are closed but still awake. |
| **Theta waves (3-7 Hz)** | Slower EEG oscillations associated with drowsiness and early sleep. |
| **Alpha/Theta ratio** | A simple vigilance metric: high alpha relative to theta = alert; low alpha relative to theta = drowsy. Used to generate ground truth labels for our classification task. |
| **HRF delay** | Hemodynamic Response Function delay (~5-6 seconds). When neurons fire, the BOLD signal doesn't change instantly — it takes about 5-6 seconds for the blood flow change to peak. This means the fMRI signal at time T actually reflects neural activity from ~5-6 seconds earlier. We account for this when aligning EEG and fMRI data. |
| **Patch / Epoch** | A short time window extracted from the full scan. In our pipeline, each patch is **28 TRs (58.8 seconds, ~1 minute)** — a non-overlapping window matching the EEG vigilance label granularity. Each patch gets a single binary vigilance label (alert/drowsy) and a corresponding FC graph. |

---

## 3. Dataset: NatView EEG-fMRI

### Overview

We use the **EEG/fMRI Naturalistic Viewing (NatView) Dataset**, collected at the Nathan Kline Institute (NKI). This is a publicly available dataset of simultaneous EEG and fMRI recordings.

| Property | Value |
|----------|-------|
| **Participants** | 22 healthy adults (ages 23-51) |
| **Condition used** | Resting state (eyes open, no task) |
| **fMRI TR** | 2.1 seconds |
| **Scan duration** | 600 seconds (10 minutes) |
| **Timepoints per scan** | ~288 |
| **fMRI resolution** | 3mm isotropic voxels |
| **EEG channels** | 64 total (61 cortical + 2 EOG + 1 ECG) |
| **EEG sampling rate** | 250 Hz |
| **Coordinate space** | MNI152 (standard brain template) |
| **Data format** | BIDS (Brain Imaging Data Structure) |
| **Access** | Public, via Amazon S3 |

### Why This Dataset?

1. **Simultaneous EEG-fMRI** — Both modalities recorded at the same time in the same session. This is essential for knowledge distillation: the teacher (EEG) and student (fMRI) models see the exact same brain states.
2. **Resting state** — Vigilance naturally fluctuates during rest. About one-third of participants fall asleep within 3 minutes of a resting-state scan, providing natural variation in alertness levels.
3. **Pre-computed derivatives** — NKI provides preprocessed versions of the data, including pre-extracted timeseries for the Schaefer 2018 atlas. This saves significant preprocessing work.

### Dataset File Structure

```
natview/
├── rawdata/                          # Raw, unprocessed data
│   ├── sub-01/
│   │   └── ses-01/
│   │       ├── anat/                 # Structural (anatomical) MRI
│   │       ├── eeg/                  # Raw EEG recordings
│   │       └── func/                 # Raw fMRI data
│   │           └── sub-01_ses-01_task-rest_bold.nii.gz   # 4D fMRI file
│   ├── sub-02/
│   └── ... (22 subjects total)
│
└── derivatives/                      # Preprocessed data
    └── natview_nki_release/
        ├── sub-01/
        │   └── ses-01/
        │       └── func/
        │           └── sub-01_ses-01_task-rest_bold/
        │               ├── func_atlas/       # Pre-extracted timeseries (what we use)
        │               │   ├── ..._Schaefer2018_dens-200parcels7networks_desc-sm0_bold.tsv
        │               │   └── ..._Schaefer2018_dens-200parcels7networks_desc-sm0gsr_bold.tsv
        │               ├── func_preproc/     # Preprocessed BOLD volumes (for subcortical extraction)
        │               │   ├── func_pp_nofilt_sm0.mni152.3mm.nii.gz      # No GSR
        │               │   └── func_pp_nofilt_gsr_sm0.mni152.3mm.nii.gz  # With GSR
        │               └── func_seg/         # Brain segmentations (native space)
        └── ... (22 subjects total)
```

### Key Files We Use

| File | What It Contains | Shape |
|------|-----------------|-------|
| `desc-sm0_bold.tsv` | Pre-extracted Schaefer 200-parcel timeseries (no GSR). Each row = 1 parcel, each column = 1 timepoint. | (200 parcels, 288 timepoints) |
| `desc-sm0gsr_bold.tsv` | Same but with Global Signal Regression applied. | (200 parcels, 288 timepoints) |
| `func_pp_nofilt_sm0.mni152.3mm.nii.gz` | Full preprocessed 4D fMRI volume in MNI space. Used for subcortical timeseries extraction. | (3D brain × 288 timepoints) |

**Note on TSV orientation**: The TSV files store data as *parcels × timepoints* (rows are parcels, columns are timepoints). We transpose after loading so that rows = timepoints and columns = parcels, which is the standard orientation for timeseries analysis.

---

## 4. EEG Preprocessing (eeg_prep.py)

The EEG preprocessing pipeline (`preprocessing/eeg_preprocessing/eeg_prep.py`) produces **vigilance labels** used as ground truth for the knowledge distillation task. It turns preprocessed EEG into per-TR frame-wise scores and per-patch binary labels (alert/drowsy) that are temporally aligned with fMRI via an HRF delay. The pipeline follows the approach used in the FewShotKDVigilance reference.

### 4.1 Purpose

- **Frame-wise**: One vigilance score and one ternary label (-1/0/+1 = drowsy/intermediate/alert) per fMRI TR (~286 TRs in a 600 s scan).
- **HRF alignment**: Labels are shifted by ~2–3 TRs so that the label at fMRI TR *t* reflects the EEG state that *drove* the BOLD signal at that TR.
- **Patch-wise**: Non-overlapping 28-TR windows (~1 minute) get a single **binary training label** (1 = alert, 0 = drowsy). These patches are what the fMRI pipeline uses for interval-level FC and GNN training.

### 4.2 Input

- **Preprocessed EEG** in EEGLAB format: `*_task-rest_eeg.set` (e.g. `sub-01_ses-01_task-rest_eeg.set`).
- The script expects data that has already been preprocessed (e.g. filtered, artifact-corrected) so that alpha and theta power are interpretable.

### 4.3 Pipeline Steps (in order)

#### Step 1: Load and pick vigilance channels

- Load the `.set` file with MNE (`mne.io.read_raw_eeglab`), preload into memory.
- Keep only EEG channels (`pick_types(eeg=True)`).
- Pick the subset of channels used for **alpha** (posterior) and **theta** (frontal) power:
  - **Alpha (posterior)**: O1, O2, Oz, Pz, P1–P8, POz, PO3, PO4, PO7, PO8, CP1–CP6, CPz.
  - **Theta (frontal)**: Fp1, Fp2, Fpz, Fz, F1–F8, AF3, AF4, AF7, AF8, FC1–FC6.
- Only channels that exist in the file are kept; if none of the vigilance channels are present, the script raises an error.

#### Step 2: Compute vigilance ratio per TR

- For each **fMRI TR** (duration = 2.1 s), take the corresponding EEG segment (same clock/time).
- Number of TRs = `total_duration_s / TR_S` (e.g. 600 / 2.1 ≈ 286).
- For each TR:
  - Extract the segment: `data[:, start_samp:end_samp]` where the window length is `TR_S * sfreq` samples.
  - Compute **power spectral density (PSD)** via Welch method (`psd_array_welch`), 1–30 Hz, `n_fft` up to 2048.
  - **Alpha power**: mean of PSD in **8–12 Hz** over **alpha channels only**.
  - **Theta power**: mean of PSD in **4–8 Hz** over **theta channels only**.
  - **Ratio** = `alpha_power / (theta_power + eps)` to avoid division by zero.
- Result: one ratio per TR (length ≈ 286 for a 600 s recording). High ratio → more alpha than theta → more alert; low ratio → more theta → more drowsy.

#### Step 3: Smooth the ratio

- Apply a **moving average** over the ratio time series: `np.convolve(ratio, np.ones(window)/window, mode="same")`.
- **Window** = 5 TRs (`SMOOTH_WINDOW_TR = 5`) to reduce TR-to-TR noise while keeping temporal structure.

#### Step 4: Frame-wise ternary labels (-1, 0, +1)

- **Percentile-based** thresholds on the smoothed ratio:
  - 33rd and 67th percentiles of the full smoothed series.
  - Smoothed ≥ 67th percentile → **1 (alert)**.
  - Smoothed ≥ 33rd and < 67th → **0 (intermediate)**.
  - Smoothed < 33rd → **-1 (drowsy)**.
- Yields one integer label per TR: `label_ternary` in {-1, 0, 1}.

#### Step 5: Apply HRF delay (align to fMRI)

- **Hemodynamic delay** = 5.5 s (`HRF_DELAY_S = 5.5`). In TRs: `delay_tr = round(5.5 / 2.1)` ≈ 2–3 TRs.
- **Shift** both the ternary labels and the smoothed vigilance scores **forward in time** by `delay_tr` (e.g. `np.roll(..., delay_tr)`), so that the label at fMRI TR index *t* corresponds to the EEG state that caused the BOLD at *t*.
- Output: `label_ternary_fmri`, `vig_fmri` — same length as number of TRs, now aligned to fMRI time.

#### Step 6: Patch-wise binary labels (for GNN training)

- **Patch** = non-overlapping window of **28 TRs** (~58.8 s, ~1 minute). `PATCH_WINDOW_TR = 28`, `PATCH_STRIDE_TR = 28`.
- For each patch:
  - Take the 28 frame-wise ternary scores (-1, 0, or +1) in that window.
  - **Sum** them.
  - **Binary label**: `1 (alert)` if sum ≥ -1, else `0 (drowsy)`. (`PATCH_SUM_THRESHOLD = -1`.)
- These patch labels are written to `*_vigilance_patches.tsv` and are the labels used by the fMRI pipeline for each 28-TR interval (one label per patch, one FC matrix per patch).

#### Step 7: Epoch boundaries (reference only)

- **Epochs** = consecutive runs of the same frame-wise ternary label (e.g. a block of all 1s, then a block of 0s). Used for visualization/reference only; not used for training. Written to `*_vigilance_epochs.tsv`.

### 4.4 Key constants (from eeg_prep.py)

| Constant | Value | Meaning |
|----------|-------|--------|
| `TR_S` | 2.1 | fMRI repetition time (seconds). One vigilance value per TR. |
| `HRF_DELAY_S` | 5.5 | Hemodynamic delay (seconds). Labels shifted by ~2–3 TRs to align with BOLD. |
| `SMOOTH_WINDOW_TR` | 5 | Moving-average window (TRs) for ratio before ternary thresholding. |
| `PATCH_WINDOW_TR` | 28 | Patch length in TRs (~1 min). Same as fMRI interval length. |
| `PATCH_STRIDE_TR` | 28 | Stride between patches (non-overlapping). |
| `PATCH_SUM_THRESHOLD` | -1 | Patch binary label: sum of 28 frame scores ≥ -1 → alert (1), else drowsy (0). |
| `ALPHA_HZ` | (8, 12) | Alpha band for posterior channels (Hz). |
| `THETA_HZ` | (4, 8) | Theta band for frontal channels (Hz). |

### 4.5 Output files (per subject)

All written to a configurable output directory (default: same as input file, or `vigilance_outputs/` when processing a directory).

| File | Description |
|------|-------------|
| `{stem}_vigilance_frames.tsv` | One row per fMRI TR: `fmri_tr_index`, `t_start_s`, `t_end_s`, `vigilance_score`, `label_ternary` (-1/0/1). |
| `{stem}_vigilance_patches.tsv` | One row per 28-TR patch: `patch_index`, `start_tr`, `end_tr`, `t_start_s`, `t_end_s`, `window_sum`, `label_binary` (1=alert, 0=drowsy). **This file is used by the fMRI pipeline** to define intervals and assign labels. |
| `{stem}_vigilance_epochs.tsv` | Epoch boundaries (consecutive same ternary label): `start_tr`, `end_tr`, `start_s`, `end_s`, `label_ternary`. |

Stem is derived from the `.set` filename (e.g. `sub-01_ses-01_task-rest`).

### 4.6 How this connects to fMRI preprocessing

- The **fMRI interval/patch pipeline** reads `*_vigilance_patches.tsv` to get `start_tr`, `end_tr`, and `label_binary` for each 28-TR window.
- No extra HRF shift is applied on the fMRI side — the EEG pipeline has already aligned labels to fMRI time.
- Each 28-TR fMRI patch gets one 210×210 correlation matrix and one binary vigilance label (alert/drowsy), which together form one sample for the GNN student model.

### 4.7 Running the pipeline

- **Single file**: `python eeg_prep.py path/to/sub-01_ses-01_task-rest_eeg.set` — outputs next to the file (or pass a second argument for `out_dir`).
- **Directory**: `python eeg_prep.py path/to/natview/rawdata` — finds all `**/*task-rest_eeg.set`, processes each, and writes all outputs to `vigilance_outputs/` (or a path given as second argument).

---

## 5. Brain Atlases Used

### Schaefer 2018 Atlas (Cortical)

The **Schaefer 2018** atlas divides the cortex into parcels using a data-driven approach based on functional connectivity patterns across many subjects.

| Property | Value |
|----------|-------|
| **Number of parcels** | 200 (our configuration) |
| **Coverage** | Cortex only (no subcortical regions) |
| **Network assignment** | Each parcel belongs to one of 7 Yeo resting-state networks |
| **Resolution** | 2mm in MNI152 space |
| **Source** | `nilearn.datasets.fetch_atlas_schaefer_2018()` |

**The 7 Yeo Networks**:

| Network | Abbreviation | Function | Example Regions |
|---------|-------------|----------|----------------|
| Visual | `Vis` | Processing visual information | Occipital cortex |
| Somatomotor | `SomMot` | Body sensation and movement | Pre/post-central gyrus |
| Dorsal Attention | `DorsAttn` | Voluntary, focused attention | Intraparietal sulcus, frontal eye fields |
| Salience / Ventral Attention | `SalVentAttn` | Detecting important stimuli, reorienting attention | Anterior insula, anterior cingulate |
| Limbic | `Limbic` | Emotion, motivation | Orbitofrontal cortex, temporal poles |
| Control / Frontoparietal | `Cont` | Executive control, working memory | Lateral prefrontal, posterior parietal |
| Default Mode | `Default` | Mind-wandering, self-referential thought; active at rest | Medial prefrontal, posterior cingulate, precuneus |

Each parcel label encodes its network: `7Networks_LH_Vis_1` = Left Hemisphere, Visual network, region 1.

### Harvard-Oxford Atlas (Subcortical)

Since the Schaefer atlas only covers the cortex, we supplement it with subcortical regions from the **Harvard-Oxford** subcortical atlas, which is also in MNI152 space (so the coordinates align).

| Property | Value |
|----------|-------|
| **Total regions** | 21 (we use 10) |
| **Our selected regions** | Thalamus, Caudate, Putamen, Hippocampus, Amygdala (all bilateral: L + R = 10) |
| **Resolution** | 2mm in MNI152 space |
| **Source** | `nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')` |

### Combined Atlas: 210 ROIs

By combining both atlases, we get:
- **200 cortical ROIs** from Schaefer (pre-extracted as TSVs)
- **10 subcortical ROIs** from Harvard-Oxford (extracted from preprocessed BOLD using `NiftiLabelsMasker`)
- **Total: 210 ROIs = 210 graph nodes**

---

## 6. What the fMRI Preprocessing Notebook Does

The notebook (`correlation_matrix_extraction.ipynb`) performs the following steps:

### Step 1: Load Cortical Timeseries

Loads the pre-extracted Schaefer 200-parcel timeseries TSV for each subject. The NatView derivatives already computed these — each file contains the average BOLD signal per parcel per timepoint.

- **Input**: `desc-sm0_bold.tsv` (tab-separated, no header)
- **Processing**: Load with `pd.read_csv(header=None)`, transpose to (timepoints × parcels)
- **Output**: DataFrame of shape **(288, 200)** — 288 timepoints, 200 cortical ROIs

### Step 2: Compute Cortical Correlation Matrix

Computes pairwise Pearson correlation between all 200 cortical ROI timeseries.

- **Input**: (288, 200) timeseries
- **Output**: (200, 200) correlation matrix
- **What it means**: Cell (i, j) = how synchronized ROI i and ROI j are. High positive values = they activate together. Values near zero = unrelated.

### Step 3: Visualize Cortical FC

Plots the correlation matrix as a heatmap and the connectome (brain network) on a glass brain.

### Step 4: Extract Subcortical Timeseries

Uses the Harvard-Oxford subcortical atlas to extract timeseries for 10 subcortical ROIs from the preprocessed 4D BOLD NIfTI.

- **Input**: `func_pp_nofilt_sm0.mni152.3mm.nii.gz` + filtered Harvard-Oxford atlas
- **Processing**: `NiftiLabelsMasker` averages all voxels within each subcortical region, producing one timeseries per region
- **Output**: Array of shape **(288, 10)** — 288 timepoints, 10 subcortical ROIs

### Step 5: Combine and Compute Full Correlation Matrix

Horizontally stacks the cortical (288 × 200) and subcortical (288 × 10) timeseries into a combined (288 × 210) matrix. Computes the full correlation matrix.

- **Output**: **(210, 210)** correlation matrix — the complete functional connectivity matrix for 200 cortical + 10 subcortical ROIs

### Step 6: Visualize Combined FC

Plots the full 210 × 210 correlation matrix and the combined cortical+subcortical connectome.

### Step 7: Save Correlation Matrices for All Subjects

Iterates over all 22 subjects, extracts cortical and subcortical timeseries, combines them, computes the full-scan (210 × 210) correlation matrix, and saves each as a `.npy` file. Also saves the combined ROI label list (`roi_labels.npy`) for downstream use.

- **Output directory**: `sample_data/corr_matrices/`
- **Per-subject file**: `sub-XX_combined_corr.npy` — shape **(210, 210)**
- **Label file**: `roi_labels.npy` — 210 ROI names (200 Schaefer + 10 Harvard-Oxford)

### Step 8: Compute Interval-Level Correlation Matrices

Uses the EEG-derived vigilance patch labels to compute a **per-patch FC matrix** for each subject. The EEG team has produced VIGALL-based vigilance labels aligned to fMRI time (HRF delay already applied).

- **Input**: Combined (288, 210) timeseries + `*_ses-01_task-rest_vigilance_patches.tsv` (from `sample_data/eeg_28TR_interval_labels/`)
- **Patch definition**: 28-TR non-overlapping windows (58.8 seconds each), with binary labels (1 = alert, 0 = drowsy)
- **Processing**: For each of the ~10 patches per subject, slice the corresponding 28 TRs from the 210-ROI timeseries, compute a 210 × 210 Pearson correlation matrix
- **Output directory**: `sample_data/28TR_interval_corr_matrices/`
- **Per-subject files**:
  - `sub-XX_interval_corr.npy` — shape **(N_patches, 210, 210)** — all patch correlation matrices
  - `sub-XX_labels.npy` — shape **(N_patches,)** — binary vigilance labels

**Actual results** (ses-01 only, 22 subjects):

| Subject | Patches | Alert | Drowsy | Subject | Patches | Alert | Drowsy |
|---------|---------|-------|--------|---------|---------|-------|--------|
| sub-01 | 10 | 5 | 5 | sub-12 | 10 | 5 | 5 |
| sub-02 | 11 | 7 | 4 | sub-13 | 10 | 6 | 4 |
| sub-03 | 11 | 8 | 3 | sub-14 | 10 | 5 | 5 |
| sub-04 | 11 | 5 | 6 | sub-15 | 11 | 7 | 4 |
| sub-05 | 11 | 5 | 6 | sub-16 | 10 | 6 | 4 |
| sub-06 | 10 | 6 | 4 | sub-17 | 10 | 6 | 4 |
| sub-07 | 16 | 8 | 8 | sub-18 | 10 | 7 | 3 |
| sub-08 | 11 | 9 | 2 | sub-19 | 11 | 7 | 4 |
| sub-09 | 10 | 4 | 6 | sub-20 | 11 | 6 | 5 |
| sub-10 | 11 | 7 | 4 | sub-21 | 10 | 5 | 5 |
| sub-11 | 10 | 6 | 4 | sub-22 | 10 | 5 | 5 |

**Notes**: Most subjects have 10–11 patches (288 TRs / 28 ≈ 10.3). Sub-07 has 16 due to a longer scan. The alert/drowsy split is approximately 57/43, slightly more balanced than the previous 5-TR version. **Total: 235 labeled FC graphs.**

---

## 7. How the Current Graphs and Visualizations Are Constructed

This section documents exactly how the correlation matrices and visualizations in the notebook are built as of this writing, so the team understands the baseline before interval-level and graph-level additions are made.

### 7.1 Correlation Matrix Computation

The notebook computes functional connectivity (FC) using **Pearson correlation** across the full scan duration.

| Stage | Method | Input | Output |
|-------|--------|-------|--------|
| Cortical-only FC | `ts.corr()` (Pandas Pearson) | (288, 200) timeseries DataFrame | (200, 200) correlation matrix |
| Combined FC | `np.corrcoef(combined_ts, rowvar=False)` (NumPy Pearson) | (288, 210) stacked array | (210, 210) correlation matrix |

**Key characteristics of the current approach**:

- **Full-scan AND interval-level correlations**: The pipeline produces two types of correlation matrices:
  - **Full-scan** (1 per subject): Summarizes FC across the entire ~10-minute scan (all 288 TRs). Saved in `sample_data/corr_matrices/`.
  - **Interval-level** (~10 per subject): Each captures FC within a single 28-TR (58.8s) non-overlapping patch, matched to EEG-derived vigilance labels. Saved in `sample_data/28TR_interval_corr_matrices/`.
- **Raw Pearson values**: Correlation values are kept as-is (range [-1, +1]). Per advisor guidance, **Fisher z-transformation is not needed** for resting-state within-subject FC — raw Pearson is appropriate for our pipeline.
- **No GSR variant**: The non-GSR timeseries (`desc-sm0_bold.tsv` and `func_pp_nofilt_sm0.mni152.3mm.nii.gz`) are used, preserving the global signal.
- **No edge filtering in the saved matrices**: The saved `.npy` files are dense 210 × 210 matrices — every ROI pair has a correlation value. Edge pruning / top-K selection is only applied at the visualization stage, not stored.
- **Vigilance labels included**: Each interval-level file has a corresponding `sub-XX_labels.npy` with binary labels (1 = alert, 0 = drowsy).

### 7.2 Visualizations Produced

The notebook generates four types of plots for a single demonstration subject (sub-01):

#### A. Cortical-Only FC Heatmap

```python
plotting.plot_matrix(
    corr_matrix.values,
    colorbar=True, vmax=1, vmin=-1,
    title="Schaefer 200 - FC Matrix Pre-Edge Pruning (sub-01)"
)
```

- Displays the **200 × 200** cortical correlation matrix as a color-coded heatmap.
- Color scale ranges from **-1 (blue/anti-correlated)** to **+1 (red/correlated)**.
- Titled "Pre-Edge Pruning" to indicate this is the full dense matrix before any sparsification.
- No ROI labels are shown (omitted for readability at 200 × 200 scale).

#### B. Cortical-Only Connectome

```python
plotting.plot_connectome(
    corr_matrix.values,
    coords,                    # Schaefer 200 ROI center coordinates
    edge_threshold="90%",      # show only top 10% strongest edges
    colorbar=True,
    title="Schaefer 200 - Connectome (sub-01)"
)
```

- Renders the cortical FC as a **glass-brain network** — each ROI is a node plotted at its MNI coordinate, with lines (edges) between connected regions.
- `edge_threshold="90%"` means only the **top 10% of edges by absolute correlation strength** are drawn. This is a **visualization-only** threshold — the underlying data is not modified.
- ROI coordinates are derived via `find_parcellation_cut_coords(labels_img=atlas_img)`, which computes the center of mass of each Schaefer parcel.

#### C. Combined FC Heatmap (Cortical + Subcortical)

```python
plotting.plot_matrix(
    combined_corr,
    colorbar=True, vmax=1, vmin=-1,
    title="Schaefer 200 + Subcortical - FC Matrix (sub-01)"
)
```

- Displays the full **210 × 210** combined correlation matrix.
- The bottom-right 10 × 10 block captures subcortical-to-subcortical correlations; the off-diagonal rectangles capture cortical-to-subcortical relationships.

#### D. Combined Connectome (Cortical + Subcortical)

```python
plotting.plot_connectome(
    combined_corr,
    combined_coords,           # 200 cortical + 10 subcortical coordinates
    edge_threshold="90%",
    colorbar=True,
    title="Schaefer 200 + Subcortical - Connectome (sub-01)"
)
```

- Same glass-brain visualization as the cortical-only connectome, but now includes 210 nodes.
- `combined_coords` is built by vertically stacking cortical coordinates (from the Schaefer atlas) and subcortical coordinates (from the filtered Harvard-Oxford atlas).
- Top 10% edge threshold is applied identically.

### 7.3 Implementation Status

#### Completed

- Full-scan (288-TR) correlation matrix extraction and saving for all 22 subjects
- Interval/epoch segmentation of the timeseries into 28-TR non-overlapping patches
- Per-interval (28-TR) correlation matrices computed and saved for all 22 subjects
- Vigilance label assignment per interval (binary: alert/drowsy) from EEG-derived VIGALL labels
- HRF-aligned temporal coupling with EEG patches (HRF delay already applied by EEG preprocessing pipeline)
- Visualization and saving of all full-scan and interval-level FC heatmaps

#### Remaining (Graph Construction Phase)

- Top-K edge selection per node (K=10) as a data processing step
- Sparse graph construction as PyG `Data` objects
- Packaging as PyG `InMemoryDataset` for GNN training

---

## 8. Interval / Epoch Segmentation (Implemented)

### 8.1 Why Intervals?

The pipeline produces two levels of FC matrices:

1. **Full-scan** (1 per subject): Summarizes the entire 10-minute scan. Useful as a reference but **not suitable for vigilance classification** because alertness fluctuates throughout the scan.
2. **Interval-level** (~10 per subject): Each captures FC within a ~1-minute window matched to a binary vigilance label. **This is the input for the GNN student model.**

Interval-level graphs are critical because:
- The EEG teacher model classifies vigilance on a per-patch basis.
- The fMRI student model (GNN) needs per-patch FC graphs to learn from.
- Knowledge distillation requires matched teacher/student inputs at the same temporal granularity.

### 8.2 Patch Parameters (Implemented)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Patch duration** | 28 TRs = 58.8 seconds (~1 min) | Provides enough timepoints for a stable correlation estimate (rank 27 for 210 ROIs). Matches ~1-minute vigilance scoring windows. |
| **TRs per patch** | 28 | Set in `eeg_prep.py` (`PATCH_WINDOW_TR = 28`). |
| **Patches per subject** | ~10–11 (varies, see table in Section 6 Step 8) | 288 TRs / 28 ≈ 10.3 → 10 complete patches for most subjects. Sub-07 has 16 (longer scan). |
| **Total labeled patches** | **235** across 22 subjects | Each patch has a binary label: 1 (alert) or 0 (drowsy). |
| **Overlap** | None | Non-overlapping windows (stride = 28 TRs) — every TR belongs to exactly one patch. |
| **Label source** | `*_ses-01_task-rest_vigilance_patches.tsv` | From the EEG preprocessing pipeline (`sample_data/eeg_28TR_interval_labels/`). |

### 8.3 Segmentation Strategy (Implemented)

Patch boundaries are defined by the EEG preprocessing pipeline's output (the `start_tr` and `end_tr` columns in each subject's `*_vigilance_patches.tsv` file), not recomputed on the fMRI side. This ensures exact alignment.

```
Full timeseries:  TR_0 ... TR_27 | TR_28 ... TR_55 | TR_56 ... TR_83 | ... | TR_252 ... TR_279 | (TR_280-287 remainder)
                  ←── Patch 0 ──→  ←── Patch 1 ───→  ←── Patch 2 ───→       ←── Patch 9 ────→   (discarded)
                    label: 1          label: 1           label: 0
```

- Each patch slice is a **(28, 210)** sub-matrix of the combined cortical+subcortical timeseries.
- Remainder TRs at the end (if the total is not divisible by 28) are discarded.

### 8.4 Per-Interval Correlation Matrices (Implemented)

For each 28-TR patch, a **210 × 210 Pearson correlation matrix** is computed:

```
Per patch (e.g., Patch 0):
    combined_ts[0:28, :]              →  (28, 210) timeseries slice
    np.corrcoef(patch_ts, rowvar=False)  →  (210, 210) correlation matrix for Patch 0
```

**Rank note**: A 28 × 210 matrix has rank 27 (at most). While still rank-deficient for 210 variables, this is a **major improvement** over the previous 5-TR approach (rank 4). With 27 degrees of freedom:
- Correlation estimates are much more stable and meaningful.
- The matrix is still not invertible (ruling out partial correlation), but the top-K graph construction only needs to rank individual edges, not invert the full matrix.
- Far fewer spurious ±1 correlations compared to 5-TR patches, resulting in more realistic FC heatmaps.

### 8.5 Alignment with EEG Patches via HRF Delay

The EEG preprocessing pipeline has **already applied the HRF delay** (~2–3 TRs) when generating the `*_vigilance_patches.tsv` files. The `start_tr` and `end_tr` columns in those files refer to fMRI TRs that have been shifted to account for the hemodynamic lag. This means:

- We do **not** need to apply any additional HRF correction on the fMRI side.
- The `start_tr` / `end_tr` values can be used directly to index into the fMRI timeseries.
- The binary label (`label_binary`) for each patch already corresponds to the neural activity occurring ~5–6 seconds *before* the fMRI measurement window.

### 8.6 Label Distribution

Across all 22 subjects (ses-01 only):

| Metric | Value |
|--------|-------|
| **Total patches** | 235 |
| **Alert (label=1)** | 135 (57.4%) |
| **Drowsy (label=0)** | 100 (42.6%) |
| **Alert:Drowsy ratio** | ~1.35:1 |

The class balance (57/43) is nearly even and unlikely to require explicit balancing, though class weighting remains an option during GNN training if needed.

### 8.7 Output Files

All interval-level data is saved to `sample_data/28TR_interval_corr_matrices/`:

| File | Shape | Description |
|------|-------|-------------|
| `sub-XX_interval_corr.npy` | (N_patches, 210, 210) | Stacked Pearson correlation matrices for all patches (N_patches ≈ 10) |
| `sub-XX_labels.npy` | (N_patches,) | Binary vigilance labels (1=alert, 0=drowsy) |
| `images/sub-XX_patchNN_alert.png` | — | Saved FC heatmap for each patch (for visual inspection) |

**Note**: The full-scan matrices remain available in `sample_data/corr_matrices/` as a reference baseline. Previous 5-TR interval matrices are archived in `sample_data/5TR_interval_corr_matrices/`.

---

## 9. Remaining Graph Construction Steps

The FC correlation matrices (both full-scan and interval-level) have been computed and saved. The following steps convert those matrices into the sparse, labeled graph objects that the GNN student model will consume.

### 9.1 Fisher Z-Transformation — Skipped

Per advisor guidance: for resting-state, within-subject functional connectivity fed into a GNN, **raw Pearson correlation is appropriate**. Fisher z-transformation is primarily needed for cross-subject statistical comparisons and task-based statistical thresholding — neither of which applies to our pipeline. The GNN learns its own feature representations and is robust to the bounded [-1, +1] range. Additionally, since top-K selection only cares about relative edge ranking (which is preserved under the monotonic Fisher z transform), the same edges would be selected either way.

### 9.2 Top-K Edge Selection (Per Node) — Next Step

**What**: For each of the 210 nodes, retain only the **K=10 strongest edges** (by absolute Pearson correlation) and discard the rest.

**Current state**: The saved `.npy` interval correlation matrices are **fully dense** (210 × 210). Edge filtering is only applied during visualization (`edge_threshold="90%"`).

**Implementation plan**: For each node:
1. Rank all 209 connections by `|r_ij|` (absolute Pearson correlation).
2. Keep the top K=10 connections.
3. Zero out all other entries for that node.

**Result**: A sparse 210 × 210 matrix per patch with at most 210 × 10 = 2,100 directed edges.

**Why K=10**: Balances informativeness with sparsity. Consistent with the FewShotKDVigilance reference paper. May be tuned.

### 9.3 Graph Data Structure (PyG Integration)

**What**: Convert each patch's sparse matrix into a PyTorch Geometric (PyG) `Data` object.

**Planned structure per patch graph**:

| PyG Field | Shape | Description |
|-----------|-------|-------------|
| `x` | (210, F) | Node feature matrix. F = feature dimensionality TBD (e.g., mean BOLD, variance within the patch). |
| `edge_index` | (2, E) | COO-format edge list. E ≤ 2,100 after top-K filtering. |
| `edge_attr` | (E, 1) | Edge weights (raw Pearson correlation values for retained edges). |
| `y` | (1,) | Binary vigilance label: **1** = alert, **0** = drowsy. |

**Storage**: PyG `InMemoryDataset`. This is preferred over zeroed-out TSVs because:
- Sparse COO format is memory-efficient.
- Directly compatible with the GNN training loop.
- PyG handles batching, shuffling, and data splitting natively.

### 9.4 Per-Interval Visualizations

| Visualization | Status | Description |
|---------------|--------|-------------|
| **Per-patch FC heatmaps** | **DONE** | 210 × 210 correlation matrix heatmap for each 28-TR patch, saved as PNG. Located in `sample_data/28TR_interval_corr_matrices/images/`. |
| **Full-scan FC heatmaps** | **DONE** | Per-subject full-scan heatmaps saved in `sample_data/corr_matrices/images/`. |
| **Full-scan connectomes** | **DONE** | Glass-brain connectome plots for demonstration subjects. |
| **Per-patch connectomes** | Planned | Glass-brain plots per patch with top-K filtering. |
| **Patch-to-patch FC variability** | Planned | Summary of how FC fluctuates across patches within a subject. |
| **Alert vs. drowsy FC comparison** | Planned | Average FC matrix for alert patches vs. drowsy patches to visualize differences. |
| **Top-K sparsity visualization** | Planned | Adjacency matrix after top-K filtering, showing which edges survived. |

### 9.5 Summary: What's Done vs. What's Remaining

| Aspect | Status | Details |
|--------|--------|---------|
| **Full-scan FC extraction** | **DONE** | 22 subjects × 1 matrix each → `sample_data/corr_matrices/` |
| **Interval-level FC extraction** | **DONE** | 235 total labeled patches (28-TR) → `sample_data/28TR_interval_corr_matrices/` |
| **EEG vigilance labels** | **DONE** | Binary (alert/drowsy), HRF delay pre-applied by EEG pipeline |
| **Visualization (heatmaps)** | **DONE** | Full-scan + interval-level heatmaps saved as PNG |
| **Fisher z-transformation** | **Skipped** | Not needed for resting-state within-subject FC (per advisor) |
| **Top-K edge selection** | Remaining | K=10 edges per node from raw Pearson matrices |
| **PyG graph construction** | Remaining | Convert sparse matrices + labels into PyG `Data` objects |
| **PyG InMemoryDataset packaging** | Remaining | Bundle all 235 graphs for GNN training |
| **Node feature definition** | Remaining | Decide what goes into `x` (node features) beyond connectivity |

---

## 10. Data Shapes and What They Mean

A quick reference for the key data structures in the notebook:

### Current Full-Scan Shapes

| Variable | Shape | Rows | Columns | Description |
|----------|-------|------|---------|-------------|
| `ts` | (288, 200) | Timepoints (TRs) | Schaefer cortical parcels | Cortical BOLD timeseries |
| `subcortical_ts` | (288, 10) | Timepoints (TRs) | Subcortical ROIs | Subcortical BOLD timeseries |
| `combined_ts` | (288, 210) | Timepoints (TRs) | All ROIs (cortical + subcortical) | Full brain timeseries |
| `corr_matrix` | (200, 200) | Cortical ROIs | Cortical ROIs | Cortical-only FC matrix |
| `combined_corr` | (210, 210) | All ROIs | All ROIs | Full FC matrix |

### Interval-Level Shapes (Implemented)

| Variable | Shape | Description |
|----------|-------|-------------|
| `patch_ts` | (28, 210) | Timeseries slice for one 28-TR patch (58.8 seconds) |
| `patch_corr` | (210, 210) | Pearson correlation matrix for one patch (rank ≤ 27) |
| `sub-XX_interval_corr.npy` | (N_patches, 210, 210) | All patch correlation matrices for one subject (N_patches ≈ 10) |
| `sub-XX_labels.npy` | (N_patches,) | Binary vigilance labels for one subject (1=alert, 0=drowsy) |

### Planned Graph Shapes (Next Phase)

| Variable | Shape | Description |
|----------|-------|-------------|
| `patch_adj_topk` | (210, 210) | Sparse adjacency after top-K=10 filtering |
| `edge_index` | (2, E) | COO edge list for PyG (E ≤ 2,100) |
| `edge_attr` | (E, 1) | Raw Pearson correlation weights for retained edges |

**Reading the correlation matrix**: Row i, Column j = Pearson correlation between ROI i's timeseries and ROI j's timeseries. The matrix is symmetric (cell i,j = cell j,i) and the diagonal is 1.0.

---

## 11. How This Feeds Into the Model

The preprocessing pipeline produces **~10 labeled FC matrices per subject** (235 total across 22 subjects), each paired with a binary vigilance label. After the remaining graph construction steps, these become the input to the GNN student model.

### Completed Pipeline

```
fMRI scan (10 min, per subject)
    → Load 200 cortical ROI timeseries (Schaefer 2018 pre-extracted TSV)
    → Extract 10 subcortical ROI timeseries (Harvard-Oxford via NiftiLabelsMasker)
    → Combine into (288, 210) timeseries
    → Compute full-scan 210 × 210 Pearson correlation matrix → save as .npy     ✓
    → Load EEG-derived vigilance patch labels (28-TR, non-overlapping, HRF pre-applied)
    → Per 28-TR patch:
        → Slice (28, 210) timeseries segment
        → Compute 210 × 210 Pearson correlation matrix (rank ≤ 27)
        → Pair with binary vigilance label (1=alert, 0=drowsy)
    → Save per-subject: interval_corr.npy + labels.npy                          ✓
    → Visualize and save all FC heatmaps                                         ✓
```

### Remaining Pipeline (Graph Construction Phase)

```
Per patch (starting from saved interval_corr.npy):
    → Select top-K=10 edges per node (by absolute Pearson correlation)
    → Build sparse PyG Data graph (210 nodes, ≤2,100 edges, binary label)
→ Package all 235 graphs as PyG InMemoryDataset
→ Train GNN student via knowledge distillation from EEG teacher
```

### Dataset Summary

| Metric | Value |
|--------|-------|
| **Subjects** | 22 (ses-01 only) |
| **Total labeled FC graphs** | 235 |
| **Alert graphs** | 135 (57.4%) |
| **Drowsy graphs** | 100 (42.6%) |
| **Graph nodes** | 210 (200 cortical + 10 subcortical) |
| **Planned edges per graph** | ≤ 2,100 (top-K=10 per node) |
| **Patch duration** | 28 TRs = 58.8 seconds (~1 min) |
| **Correlation matrix rank** | Up to 27 (vs. rank 4 with 5-TR patches) |
| **Label type** | Binary (1=alert, 0=drowsy) |
