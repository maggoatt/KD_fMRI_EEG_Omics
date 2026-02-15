# Vigilance labeling outputs

EEG-derived vigilance labels for resting-state data (NatView). One set of three files per subject/session (e.g. `sub-01_ses-01_task-rest_*`). TR = 2.1 s. Labels are aligned to fMRI time (HRF delay applied).

---

## File types

### **Frames** (`*_vigilance_frames.tsv`)

**One row per fMRI volume (TR).** Frame-level vigilance state.

| Column | Meaning |
|--------|--------|
| `fmri_tr_index` | fMRI volume index (0, 1, 2, …) |
| `t_start_s`, `t_end_s` | Time span of this TR (seconds) |
| `vigilance_score` | Smoothed alpha/theta ratio (posterior alpha, frontal theta) |
| `label_ternary` | **-1** drowsy, **0** intermediate, **1** alert (percentile-based) |

Use for: time-aligned vigilance traces, plots, or to recompute patch labels with a different window/stride.

---

### **Patches** (`*_vigilance_patches.tsv`)

**One row per 5-TR non-overlapping window.** Binary labels for training (e.g. GNN).

| Column | Meaning |
|--------|--------|
| `patch_index` | Window index |
| `start_tr`, `end_tr` | TR range [start_tr, end_tr) — 5 consecutive TRs |
| `t_start_s`, `t_end_s` | Same window in seconds (each window = 10.5 s) |
| `window_sum` | Sum of the 5 frame-wise `label_ternary` values (-5 to +5) |
| `label_binary` | **1** alert, **0** drowsy. Rule: `window_sum >= -1` → 1, else 0 |

Use for: training/evaluation. Each row = one labeled sample (e.g. one FC graph from that 5-TR window).

---

### **Epochs** (`*_vigilance_epochs.tsv`)

**One row per contiguous run of the same frame-wise ternary label.** No overlap; boundaries where the state changes.

| Column | Meaning |
|--------|--------|
| `start_tr`, `end_tr` | TR range of this run |
| `start_s`, `end_s` | Same in seconds |
| `label_ternary` | -1, 0, or 1 for the whole run |

Use for: summarizing alert/drowsy/intermediate segment lengths, visualization of state transitions.

---

## Pipeline summary

- **EEG:** Posterior alpha (8–12 Hz) / frontal theta (3–7 Hz) per TR → smoothed → ternary -1/0/1.
- **HRF:** Labels shifted by ~2–3 TRs so they align with fMRI.
- **Patches:** 5-frame windows, stride 5 (non-overlapping), sum then threshold at -1 → binary.
