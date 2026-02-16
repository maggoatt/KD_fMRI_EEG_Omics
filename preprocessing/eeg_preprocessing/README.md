# EEG Preprocessing: Vigilance Labeling Pipeline

EEG-derived vigilance labels for resting-state data (NatView dataset). Produces per-TR frame-wise scores and per-patch binary labels (alert/drowsy) for knowledge distillation training. Labels are temporally aligned to fMRI via HRF delay.

**For detailed documentation**, see [`../../docs/dataset_preprocessing_guide.md`](../../docs/dataset_preprocessing_guide.md#4-eeg-preprocessing-eeg_preppy).

---

## Quick Start

### Single file
```bash
python eeg_prep.py path/to/sub-01_ses-01_task-rest_eeg.set
```
Outputs are written next to the input file (or specify `out_dir` as second argument).

### Batch processing (directory)
```bash
python eeg_prep.py path/to/natview/rawdata [output_directory]
```
Processes all `**/*task-rest_eeg.set` files found under the directory. All outputs go to `vigilance_outputs/` by default (or the specified output directory).

---

## Pipeline Overview

The pipeline (`eeg_prep.py`) follows the FewShotKDVigilance approach:

1. **Load EEG**: Pick vigilance channels (posterior alpha, frontal theta)
2. **Compute ratio per TR**: Alpha (8–12 Hz) / Theta (4–8 Hz) power ratio for each 2.1s fMRI TR
3. **Smooth**: 5-TR moving average to reduce noise
4. **Ternary labels**: Percentile-based thresholds → -1 (drowsy), 0 (intermediate), 1 (alert)
5. **HRF alignment**: Shift labels by ~2–3 TRs (5.5s) to align with fMRI BOLD signal
6. **Patch labels**: 28-TR non-overlapping windows → sum ternary scores → binary (alert/drowsy)

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TR_S` | 2.1 s | fMRI repetition time |
| `HRF_DELAY_S` | 5.5 s | Hemodynamic delay (~2–3 TRs) |
| `SMOOTH_WINDOW_TR` | 5 | Moving-average window for ratio smoothing |
| `PATCH_WINDOW_TR` | 28 | Patch length in TRs (~58.8 s, ~1 min) |
| `PATCH_STRIDE_TR` | 28 | Stride between patches (non-overlapping) |
| `PATCH_SUM_THRESHOLD` | -1 | Binary label threshold: sum ≥ -1 → alert (1), else drowsy (0) |
| `ALPHA_HZ` | (8, 12) | Alpha frequency band (Hz) |
| `THETA_HZ` | (4, 8) | Theta frequency band (Hz) |

### Channel Selection

- **Alpha (posterior)**: O1, O2, Oz, Pz, P1–P8, POz, PO3, PO4, PO7, PO8, CP1–CP6, CPz
- **Theta (frontal)**: Fp1, Fp2, Fpz, Fz, F1–F8, AF3, AF4, AF7, AF8, FC1–FC6

Only channels present in the input file are used.

---

## Output Files

One set of three TSV files per subject/session (e.g. `sub-01_ses-01_task-rest_*`).

### **Frames** (`*_vigilance_frames.tsv`)

**One row per fMRI volume (TR).** Frame-level vigilance state.

| Column | Meaning |
|--------|--------|
| `fmri_tr_index` | fMRI volume index (0, 1, 2, …) |
| `t_start_s`, `t_end_s` | Time span of this TR (seconds) |
| `vigilance_score` | Smoothed alpha/theta ratio (posterior alpha, frontal theta) |
| `label_ternary` | **-1** drowsy, **0** intermediate, **1** alert (percentile-based) |

**Use for**: Time-aligned vigilance traces, plots, or to recompute patch labels with different window/stride.

---

### **Patches** (`*_vigilance_patches.tsv`)

**One row per 28-TR non-overlapping window.** Binary labels for training (e.g. GNN).

| Column | Meaning |
|--------|--------|
| `patch_index` | Window index |
| `start_tr`, `end_tr` | TR range [start_tr, end_tr) — 28 consecutive TRs |
| `t_start_s`, `t_end_s` | Same window in seconds (each window ≈ 58.8 s) |
| `window_sum` | Sum of the 28 frame-wise `label_ternary` values (-28 to +28) |
| `label_binary` | **1** alert, **0** drowsy. Rule: `window_sum >= -1` → 1, else 0 |

**Use for**: Training/evaluation. Each row = one labeled sample (e.g. one FC graph from that 28-TR window). **This file is used by the fMRI preprocessing pipeline** to define intervals and assign labels.

---

### **Epochs** (`*_vigilance_epochs.tsv`)

**One row per contiguous run of the same frame-wise ternary label.** No overlap; boundaries where the state changes.

| Column | Meaning |
|--------|--------|
| `start_tr`, `end_tr` | TR range of this run |
| `start_s`, `end_s` | Same in seconds |
| `label_ternary` | -1, 0, or 1 for the whole run |

**Use for**: Summarizing alert/drowsy/intermediate segment lengths, visualization of state transitions. Reference only; not used for training.

---

## Pipeline Details

### Step-by-Step Process

1. **Load and pick vigilance channels**
   - Load `.set` file with MNE (`mne.io.read_raw_eeglab`)
   - Keep only EEG channels (`pick_types(eeg=True)`)
   - Select alpha (posterior) and theta (frontal) channels

2. **Compute vigilance ratio per TR**
   - For each fMRI TR (2.1s), extract corresponding EEG segment
   - Compute PSD via Welch method (1–30 Hz, `n_fft` up to 2048)
   - Alpha power: mean PSD in 8–12 Hz over alpha channels only
   - Theta power: mean PSD in 4–8 Hz over theta channels only
   - Ratio = `alpha_power / (theta_power + eps)`

3. **Smooth the ratio**
   - Apply 5-TR moving average (`np.convolve`, mode="same")
   - Reduces TR-to-TR noise while preserving temporal structure

4. **Frame-wise ternary labels**
   - Compute 33rd and 67th percentiles of smoothed ratio
   - Smoothed ≥ 67th percentile → **1 (alert)**
   - Smoothed ≥ 33rd and < 67th → **0 (intermediate)**
   - Smoothed < 33rd → **-1 (drowsy)**

5. **Apply HRF delay (align to fMRI)**
   - Shift labels forward by `delay_tr = round(5.5 / 2.1)` ≈ 2–3 TRs
   - Label at fMRI TR *t* now corresponds to EEG state that caused BOLD at *t*

6. **Patch-wise binary labels**
   - Non-overlapping 28-TR windows (~58.8 s, ~1 min)
   - For each patch: sum the 28 frame-wise ternary scores
   - Binary label: `1 (alert)` if sum ≥ -1, else `0 (drowsy)`

7. **Epoch boundaries** (reference only)
   - Consecutive runs of the same ternary label
   - Used for visualization/reference; not used for training

---

## Integration with fMRI Preprocessing

- The **fMRI interval/patch pipeline** reads `*_vigilance_patches.tsv` to get `start_tr`, `end_tr`, and `label_binary` for each 28-TR window.
- **No extra HRF shift** is applied on the fMRI side — the EEG pipeline has already aligned labels to fMRI time.
- Each 28-TR fMRI patch gets one 210×210 correlation matrix and one binary vigilance label (alert/drowsy), which together form one sample for the GNN student model.

---

## Requirements

- Python 3.x
- `mne` (MNE-Python)
- `numpy`
- `pandas`
- `python-dotenv` (for environment variables)

---

## Notes

- **Input format**: Preprocessed EEG in EEGLAB `.set` format (e.g. `sub-01_ses-01_task-rest_eeg.set`)
- **Expected preprocessing**: Data should be filtered and artifact-corrected so that alpha and theta power are interpretable
- **Patch duration**: 28 TRs = 58.8 seconds (~1 minute) — provides enough timepoints for stable correlation estimates in downstream fMRI FC analysis
- **HRF delay**: Already applied in output files — fMRI pipeline uses `start_tr`/`end_tr` directly without additional shifting

---

## References

- FewShotKDVigilance: [EEG-to-fMRI knowledge distillation empowers few-shot resting-state fMRI vigilance detection](https://spie.org/medical-imaging/presentation/EEG-to-fMRI-knowledge-distillation-empowers-few-shot-resting-state/13925-1) ([GitHub Repository](https://github.com/neurdylab/FewShotKDVigilance/tree/main))
