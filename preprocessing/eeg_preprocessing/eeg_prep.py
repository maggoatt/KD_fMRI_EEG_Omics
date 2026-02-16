"""
EEG vigilance labeling for EEG-to-fMRI knowledge distillation (NatView task-rest).

Pipeline (aligned with FewShotKDVigilance / lead researcher):
1. Frame-wise: alpha/theta ratio per TR → smooth → ternary -1/0/+1 (drowsy/intermediate/alert).
2. HRF delay: shift labels by ~2–3 TRs so they align with fMRI frames.
3. Patch-wise: 5-fMRI-frame sliding window; sum the 5 frame scores; threshold at -1
   → binary training label: sum >= -1 → Alert (1), sum < -1 → Drowsy (0).

Outputs: per-TR frame-wise scores (-1/0/+1) and per-patch binary labels for GNN training.
"""
import mne
from mne.time_frequency import psd_array_welch
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Config (NatView: TR=2.1s, ~600s; duration taken from each recording)
# -----------------------------------------------------------------------------
# TR_S: Repetition time (s). One vigilance label per fMRI volume (e.g. 2.1 s → ~286 TRs in 600 s).
TR_S = 2.1

# HRF_DELAY_S: Hemodynamic response delay in seconds. BOLD signal in fMRI lags
#              neural activity (and thus EEG) by ~5–6 s. When aligning
#              EEG-derived vigilance to fMRI timepoints, we shift labels back
#              by this amount so the label at fMRI TR t reflects the EEG state
#              that drove the BOLD at that TR.
HRF_DELAY_S = 5.5

# Frame-level smoothing before ternary thresholding
SMOOTH_WINDOW_TR = 5

# Patch-level (training) labels: 5-frame window, sum scores, threshold at -1 (confirmed pipeline)
PATCH_WINDOW_TR = 5
PATCH_SUM_THRESHOLD = -1   # sum >= -1 → Alert (1), sum < -1 → Drowsy (0)
# FewShotKDVigilance repo uses step_size=5, step_seg_length=5 → non-overlapping windows
PATCH_STRIDE_TR = 5       # 5 = non-overlapping (~57 patches), matches paper; 1 = overlapping (~282)
# All posterior (alpha) and frontal (theta) regions from 59-channel layout
ALPHA_CHANNELS = [
    "O1", "O2", "Oz", "Pz",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO7", "PO8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
]
THETA_CHANNELS = [
    "Fp1", "Fp2", "Fpz",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "AF3", "AF4", "AF7", "AF8",
    "FC1", "FC2", "FC3", "FC5", "FC6",
]
ALPHA_HZ = (8.0, 12.0)
THETA_HZ = (4.0, 8.0)
EPS = 1e-10  # avoid division by zero

# -----------------------------------------------------------------------------
# 1. Load preprocessed EEG and pick channels needed for alpha and theta
# -----------------------------------------------------------------------------
def load_and_pick_vigilance_channels(set_path: str):
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    raw.pick_types(eeg=True)
    needed = list(dict.fromkeys(ALPHA_CHANNELS + THETA_CHANNELS))  # order preserved, no dupes
    available = [ch for ch in needed if ch in raw.ch_names]
    if not available:
        raise ValueError(
            f"None of the vigilance channels {needed} found in {raw.ch_names}"
        )
    raw = raw.pick_channels(available, ordered=False)
    return raw


# -----------------------------------------------------------------------------
# 2. Compute posterior alpha power and frontal theta power per TR, then ratio
# -----------------------------------------------------------------------------
def compute_vigilance_ratio_per_tr(raw: mne.io.Raw, total_duration_s: float) -> np.ndarray:
    data, times = raw.get_data(return_times=True)
    ch_names = raw.ch_names
    alpha_inds = [i for i, ch in enumerate(ch_names) if ch in ALPHA_CHANNELS]
    theta_inds = [i for i, ch in enumerate(ch_names) if ch in THETA_CHANNELS]
    if not alpha_inds:
        raise ValueError(f"No alpha channels {ALPHA_CHANNELS} in picked data: {ch_names}")
    if not theta_inds:
        raise ValueError(f"No theta channels {THETA_CHANNELS} in picked data: {ch_names}")

    sfreq = raw.info["sfreq"]
    n_tr = int(total_duration_s / TR_S)
    n_samp_per_tr = int(TR_S * sfreq)

    ratios = []
    for i in range(n_tr):
        start_samp = i * n_samp_per_tr
        end_samp = start_samp + n_samp_per_tr
        if end_samp > data.shape[1]:
            break
        segment = data[:, start_samp:end_samp]  # (n_chans, n_samp)

        psds, freqs = psd_array_welch(
            segment, sfreq=sfreq, fmin=1.0, fmax=30.0, n_fft=min(2048, segment.shape[1])
        )
        alpha_freq_mask = (freqs >= ALPHA_HZ[0]) & (freqs <= ALPHA_HZ[1])
        theta_freq_mask = (freqs >= THETA_HZ[0]) & (freqs <= THETA_HZ[1])
        # Posterior alpha: mean power in 8–12 Hz over alpha channels only
        alpha_power = np.mean(psds[np.ix_(alpha_inds, alpha_freq_mask)])
        # Frontal theta: mean power in 3–7 Hz over theta channels only
        theta_power = np.mean(psds[np.ix_(theta_inds, theta_freq_mask)])
        ratio = alpha_power / (theta_power + EPS)
        ratios.append(ratio)

    return np.array(ratios)


# -----------------------------------------------------------------------------
# 3. Smooth with moving average (window=5)
# -----------------------------------------------------------------------------
def smooth_ratio(ratio: np.ndarray, window: int = SMOOTH_WINDOW_TR) -> np.ndarray:
    return np.convolve(ratio, np.ones(window) / window, mode="same")


# -----------------------------------------------------------------------------
# 4. Frame-wise ternary labels (-1, 0, 1)
# -----------------------------------------------------------------------------
def threshold_frame_labels(smoothed: np.ndarray) -> np.ndarray:
    """Ternary per TR: -1 = drowsy, 0 = intermediate, 1 = alert (percentiles)."""
    p33, p67 = np.percentile(smoothed, [33, 67])
    return np.where(smoothed >= p67, 1, np.where(smoothed >= p33, 0, -1)).astype(int)


# -----------------------------------------------------------------------------
# 5. Apply HRF delay: align EEG-derived labels to fMRI frames
# -----------------------------------------------------------------------------
def apply_hrf_delay(
    label_ternary: np.ndarray,
    vigilance_smoothed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift frame-wise labels and scores by HRF delay (≈2–3 TRs for NatView)."""
    delay_tr = int(round(HRF_DELAY_S / TR_S))
    label_ternary_fmri = np.roll(label_ternary, delay_tr)
    vig_fmri = np.roll(vigilance_smoothed, delay_tr)
    return label_ternary_fmri, vig_fmri


# -----------------------------------------------------------------------------
# 6. Patch-wise binary labels (5-frame sliding window, sum, threshold at -1)
# -----------------------------------------------------------------------------
def compute_patch_labels(
    label_ternary_fmri: np.ndarray,
    tr_s: float,
    window_tr: int = PATCH_WINDOW_TR,
    stride_tr: int = PATCH_STRIDE_TR,
    threshold: int = PATCH_SUM_THRESHOLD,
) -> list[dict]:
    """
    For each 5-fMRI-frame window: sum the 5 frame-wise scores (-1/0/+1),
    then threshold at -1 → binary training label (1=alert, 0=drowsy).
    """
    n_tr = len(label_ternary_fmri)
    patches = []
    for start_tr in range(0, n_tr - window_tr + 1, stride_tr):
        end_tr = start_tr + window_tr
        window_scores = label_ternary_fmri[start_tr:end_tr]
        window_sum = int(np.sum(window_scores))
        label_binary = 1 if window_sum >= threshold else 0
        patches.append({
            "patch_index": len(patches),
            "start_tr": start_tr,
            "end_tr": end_tr,
            "t_start_s": start_tr * tr_s,
            "t_end_s": end_tr * tr_s,
            "window_sum": window_sum,
            "label_binary": label_binary,
        })
    return patches


# -----------------------------------------------------------------------------
# 7. Epoch boundaries (consecutive same frame-wise ternary label)
# -----------------------------------------------------------------------------
def build_epoch_boundaries(
    label_ternary: np.ndarray, tr_s: float
) -> list[dict]:
    """Consecutive same frame-wise label runs (for reference)."""
    boundaries = []
    n_tr = len(label_ternary)
    i = 0
    while i < n_tr:
        lab = label_ternary[i]
        j = i
        while j < n_tr and label_ternary[j] == lab:
            j += 1
        boundaries.append({
            "start_tr": i,
            "end_tr": j,
            "start_s": i * tr_s,
            "end_s": j * tr_s,
            "label_ternary": lab,
        })
        i = j
    return boundaries


def run_pipeline(set_path: str, out_dir: str | None = None) -> pd.DataFrame:
    if out_dir is None:
        out_dir = Path(set_path).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_and_pick_vigilance_channels(set_path)
    # Use actual recording duration (handles 600, 601, 602 s etc across subjects)
    total_duration_s = float(raw.times[-1]) + (1.0 / raw.info["sfreq"])
    n_tr = int(total_duration_s / TR_S)

    ratio_per_tr = compute_vigilance_ratio_per_tr(raw, total_duration_s)
    n_tr = min(n_tr, len(ratio_per_tr))
    ratio_per_tr = ratio_per_tr[:n_tr]

    smoothed = smooth_ratio(ratio_per_tr)
    label_ternary = threshold_frame_labels(smoothed)
    label_ternary_fmri, vig_fmri = apply_hrf_delay(label_ternary, smoothed)

    # Frame-wise table (fMRI-aligned): one row per TR with -1/0/+1 and vigilance score
    frame_rows = []
    for i in range(n_tr):
        frame_rows.append({
            "fmri_tr_index": i,
            "t_start_s": i * TR_S,
            "t_end_s": (i + 1) * TR_S,
            "vigilance_score": float(vig_fmri[i]),
            "label_ternary": int(label_ternary_fmri[i]),  # -1=drowsy, 0=intermediate, 1=alert
        })
    frame_table = pd.DataFrame(frame_rows)

    # Patch-wise table: 5-frame window sum, threshold at -1 → binary training label (1=alert, 0=drowsy)
    patches = compute_patch_labels(
        label_ternary_fmri[:n_tr], TR_S,
        window_tr=PATCH_WINDOW_TR, stride_tr=PATCH_STRIDE_TR, threshold=PATCH_SUM_THRESHOLD,
    )
    patch_table = pd.DataFrame(patches)

    # Epoch boundaries (consecutive same frame-wise ternary; for reference)
    epochs = build_epoch_boundaries(label_ternary_fmri[:n_tr], TR_S)
    epochs_df = pd.DataFrame(epochs)

    stem = Path(set_path).stem.replace("_eeg", "")  # e.g. sub-01_ses-01_task-rest
    frame_path = out_dir / f"{stem}_vigilance_frames.tsv"
    frame_table.to_csv(frame_path, sep="\t", index=False)
    patch_path = out_dir / f"{stem}_vigilance_patches.tsv"
    patch_table.to_csv(patch_path, sep="\t", index=False)
    epochs_path = out_dir / f"{stem}_vigilance_epochs.tsv"
    epochs_df.to_csv(epochs_path, sep="\t", index=False)
    print(f"Saved frame-wise (per TR): {frame_path}")
    print(f"Saved patch-wise (training labels): {patch_path}  n_patches={len(patches)}")
    print(f"Saved epoch boundaries: {epochs_path}")
    return frame_table


# -----------------------------------------------------------------------------
# Iterate over all resting-state .set files in a data directory
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "vigilance_outputs"  # batch results go here when processing a directory


def run_pipeline_on_directory(
    data_root: str,
    pattern: str = "*task-rest_eeg.set",
    out_dir: str | None = None,
) -> list[str]:
    """
    Find all .set files under data_root matching pattern, sort by subject/session,
    and run the vigilance pipeline on each. All outputs are written to out_dir
    (default: cwd/vigilance_outputs), so results stay in one folder.
    Returns list of processed file paths.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise NotADirectoryError(str(data_root))
    set_files = sorted(data_root.glob(pattern))
    if not set_files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_root}")
    if out_dir is None:
        out_dir = Path.cwd() / DEFAULT_OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")
    processed = []
    for i, set_path in enumerate(set_files, 1):
        print(f"[{i}/{len(set_files)}] {set_path.name}")
        run_pipeline(str(set_path), out_dir=str(out_dir))
        processed.append(str(set_path))
    return processed


# -----------------------------------------------------------------------------
# Run: single file, list of files, or data directory
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if Path(arg).is_dir():
            # Directory → process all *task-rest_eeg.set; save to vigilance_outputs/ (or 2nd arg)
            out = sys.argv[2] if len(sys.argv) > 2 else None
            run_pipeline_on_directory(arg, out_dir=out)
        else:
            # One or more .set files (saved next to each file unless out_dir given)
            for set_path in sys.argv[1:]:
                run_pipeline(set_path)
    else:
        set_path = "sub-01_ses-01_task-rest_eeg.set"
        frame_df = run_pipeline(set_path)
        print("Frame-wise (first 10 TRs):")
        print(frame_df.head(10))
        # Patch-wise labels are in {stem}_vigilance_patches.tsv (use for GNN training)
