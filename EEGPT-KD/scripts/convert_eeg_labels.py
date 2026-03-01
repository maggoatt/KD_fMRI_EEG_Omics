"""Convert Darrin's EEG vigilance patch TSVs to .npy label files.

Reads *_patches.tsv files from eeg_data/ and writes sub-*_labels.npy
files that match what VigilanceGraphDataset expects.

NOTE on interval alignment:
  - EEG labels: 30s patches (120 frames at 0.25s TR)
  - fMRI graphs: ~60s intervals (Maggie's 1-min intervals)

  If fMRI uses 60s intervals, we need to aggregate 2 EEG patches per
  fMRI interval. Current script saves per-patch labels (30s each).
  Adjust FMRI_INTERVAL_S to match Maggie's actual interval duration.
"""

import os
import glob

import numpy as np
import pandas as pd


# Source directory with Darrin's TSVs
EEG_LABEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "eeg_data", "eeg_vig_TR=0.25s_Patch=120",
)

# Output directory for .npy label files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "labels")


def convert_patches_to_npy(eeg_label_dir, output_dir):
    """Convert all patch TSVs to .npy label arrays.

    Groups sessions by subject. For subjects with multiple sessions,
    concatenates labels across sessions.

    Output: one {subject_id}_labels.npy per subject with binary labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    patch_files = sorted(glob.glob(
        os.path.join(eeg_label_dir, "*_patches.tsv")
    ))

    if not patch_files:
        print(f"No patch files found in {eeg_label_dir}")
        return

    # Group by subject (e.g. sub-04 has ses-01 and ses-02)
    subject_labels = {}
    for path in patch_files:
        fname = os.path.basename(path)
        # e.g. sub-04_ses-02_task-rest_vigilance_patches.tsv
        parts = fname.split("_")
        subject_id = parts[0]  # sub-04

        df = pd.read_csv(path, sep="\t")
        labels = df["label_binary"].values.astype(np.int64)

        if subject_id not in subject_labels:
            subject_labels[subject_id] = []
        subject_labels[subject_id].append(labels)

    # Save concatenated labels per subject
    total_samples = 0
    for subject_id, label_arrays in sorted(subject_labels.items()):
        all_labels = np.concatenate(label_arrays)
        out_path = os.path.join(output_dir, f"{subject_id}_labels.npy")
        np.save(out_path, all_labels)
        n_alert = (all_labels == 1).sum()
        n_drowsy = (all_labels == 0).sum()
        total_samples += len(all_labels)
        print(
            f"  {subject_id}: {len(all_labels)} labels "
            f"({n_alert} alert, {n_drowsy} drowsy) -> {out_path}"
        )

    print(f"\nTotal: {total_samples} labels across {len(subject_labels)} subjects")
    print(f"Output: {output_dir}")


def print_summary(eeg_label_dir):
    """Print dataset summary from patch files."""
    patch_files = sorted(glob.glob(
        os.path.join(eeg_label_dir, "*_patches.tsv")
    ))
    total_alert = 0
    total_drowsy = 0
    subjects = set()

    for path in patch_files:
        fname = os.path.basename(path)
        subject_id = fname.split("_")[0]
        subjects.add(subject_id)
        df = pd.read_csv(path, sep="\t")
        total_alert += (df["label_binary"] == 1).sum()
        total_drowsy += (df["label_binary"] == 0).sum()

    total = total_alert + total_drowsy
    print("EEG Vigilance Label Summary")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Sessions: {len(patch_files)}")
    print(f"  Total patches: {total}")
    print(f"  Alert: {total_alert} ({100*total_alert/total:.1f}%)")
    print(f"  Drowsy: {total_drowsy} ({100*total_drowsy/total:.1f}%)")
    print(f"  Patch duration: 30s (120 frames at 0.25s)")


if __name__ == "__main__":
    eeg_dir = os.path.abspath(EEG_LABEL_DIR)
    out_dir = os.path.abspath(OUTPUT_DIR)

    print(f"Source: {eeg_dir}\n")
    print_summary(eeg_dir)
    print(f"\nConverting to .npy...")
    convert_patches_to_npy(eeg_dir, out_dir)
