"""Graph dataset for fMRI brain vigilance classification.

Adapted from BrainDataset (preprocessing/omics_preprocessing/create_omics_dataset.py).
Loads fMRI correlation matrices, constructs PyG graphs with top-k edges,
and optionally incorporates gene expression node features.
"""

import os
import glob

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class VigilanceGraphDataset(Dataset):
    """PyG dataset of brain graphs for vigilance classification.

    Each sample is a graph representing one time interval of one subject.
    Nodes are brain ROIs, edges connect the k most correlated ROI pairs.

    Args:
        fmri_path: directory with sub-*_ses-*_interval_corr.npy and sub-*_ses-*_labels.npy
        gene_expression_path: optional path to gene expression .npy (num_nodes, num_genes)
        k: number of top connections per node (default 10)
    """

    def __init__(self, fmri_path, gene_expression_path=None, k=10):
        super().__init__()
        self.k = k
        self.samples = []
        self._subjects = []

        # Load gene expression if provided
        self.gene_expression = None
        if gene_expression_path is not None and os.path.exists(gene_expression_path):
            self.gene_expression = torch.FloatTensor(np.load(gene_expression_path))

        # Load fMRI correlation matrices and labels
        # Supports both sub-*_ses-*_interval_corr.npy (new) and sub-*_interval_corr.npy (legacy)
        corr_files = sorted(glob.glob(os.path.join(fmri_path, "sub-*_ses-*_interval_corr.npy")))
        if not corr_files:
            corr_files = sorted(glob.glob(os.path.join(fmri_path, "sub-*_interval_corr.npy")))

        for corr_file in corr_files:
            filename = os.path.basename(corr_file)
            parts = filename.split("_")
            subject_id = parts[0]
            # Detect session: sub-01_ses-01_interval_corr.npy vs sub-01_interval_corr.npy
            if parts[1].startswith("ses-"):
                ses_id = parts[1]
                label_file = os.path.join(fmri_path, f"{subject_id}_{ses_id}_labels.npy")
            else:
                ses_id = "ses-01"
                label_file = os.path.join(fmri_path, f"{subject_id}_labels.npy")

            corr_intervals = np.load(corr_file)

            if not os.path.exists(label_file):
                continue
            labels = np.load(label_file)

            for i in range(len(corr_intervals)):
                if i >= len(labels):
                    break
                self.samples.append({
                    "subject": subject_id,
                    "session": ses_id,
                    "interval": i,
                    "corr_matrix": corr_intervals[i],
                    "label": int(labels[i]),
                })
                if subject_id not in self._subjects:
                    self._subjects.append(subject_id)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample = self.samples[idx]
        corr = sample["corr_matrix"]

        edge_index, edge_attr = self._corr_to_graph(corr, self.k)

        # Node features: mean connectivity
        mean_conn = torch.FloatTensor(corr.mean(axis=1)).unsqueeze(1)  # (N, 1)

        # Optionally concatenate gene expression
        if self.gene_expression is not None:
            num_nodes = mean_conn.shape[0]
            gene_feat = self.gene_expression[:num_nodes]
            x = torch.cat([mean_conn, gene_feat], dim=1)  # (N, 1 + num_genes)
        else:
            x = mean_conn

        y = torch.tensor(sample["label"], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.subject = sample["subject"]
        data.session = sample["session"]
        data.interval = sample["interval"]
        return data

    def _corr_to_graph(self, corr_matrix, k):
        """Convert correlation matrix to top-k graph edges per node."""
        num_nodes = corr_matrix.shape[0]
        edges = []
        weights = []

        for node in range(num_nodes):
            corr_values = np.abs(corr_matrix[node, :].copy())
            corr_values[node] = 0.0  # no self-loops
            top_k_idx = np.argsort(corr_values)[-k:]
            for neighbor in top_k_idx:
                edges.append((node, neighbor))
                weights.append(corr_matrix[node, neighbor])

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_attr

    def get_subjects(self):
        """Return list of unique subject IDs."""
        return list(self._subjects)

    def get_subject_indices(self, subject_id):
        """Return sample indices belonging to a specific subject."""
        return [i for i, s in enumerate(self.samples) if s["subject"] == subject_id]

    @property
    def num_node_features(self):
        if len(self.samples) == 0:
            return 1
        sample = self.get(0)
        return sample.x.shape[1]

    @property
    def num_classes(self):
        return 2


def create_synthetic_dataset(
    num_subjects=5,
    intervals_per_subject=10,
    num_nodes=210,
    num_genes=0,
    k=10,
    save_dir=None,
):
    """Create a synthetic dataset for testing without real data.

    Generates random correlation matrices and binary labels,
    optionally with random gene expression features.

    Args:
        num_subjects: number of synthetic subjects
        intervals_per_subject: time intervals per subject
        num_nodes: brain ROIs per graph
        num_genes: gene features per node (0 to skip)
        k: top-k edges per node
        save_dir: if provided, save .npy files here; otherwise use a temp directory

    Returns:
        VigilanceGraphDataset loaded from the generated files
    """
    import tempfile

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="synthetic_brain_")

    os.makedirs(save_dir, exist_ok=True)

    gene_path = None
    if num_genes > 0:
        gene_expression = np.random.randn(num_nodes, num_genes).astype(np.float32)
        gene_path = os.path.join(save_dir, "gene_expression.npy")
        np.save(gene_path, gene_expression)

    for subj_idx in range(num_subjects):
        subject_id = f"sub-{subj_idx + 1:02d}"

        # Random symmetric correlation matrices
        corr_matrices = []
        for _ in range(intervals_per_subject):
            rand = np.random.randn(num_nodes, num_nodes).astype(np.float32)
            corr = (rand + rand.T) / 2
            np.fill_diagonal(corr, 1.0)
            corr = np.clip(corr, -1, 1)
            corr_matrices.append(corr)
        corr_matrices = np.stack(corr_matrices)

        # Random binary labels
        labels = np.random.randint(0, 2, size=intervals_per_subject)

        np.save(os.path.join(save_dir, f"{subject_id}_ses-01_interval_corr.npy"), corr_matrices)
        np.save(os.path.join(save_dir, f"{subject_id}_ses-01_labels.npy"), labels)

    return VigilanceGraphDataset(save_dir, gene_expression_path=gene_path, k=k)
