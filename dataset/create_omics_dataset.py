import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import os
import glob

# currently implemented in accordance to PyTorch Dataset object
class BrainOmicsDataset(Dataset):
    samples = []
    def __init__(self, fmri_path, expression_path, k=10):
        self.k = k
        self.expression_data = torch.FloatTensor(np.load(expression_path))
        
        corr_files = sorted(glob.glob(f"{fmri_path}/sub-*_ses-*_interval_corr.npy"))

        for corr_file in corr_files:
            filename = os.path.basename(corr_file)
            parts = filename.split('_')
            subject_id = parts[0]  # 'sub-01'
            ses_id = parts[1]      # 'ses-01'
            corr_intervals = np.load(corr_file)
            
            label_file = os.path.join(fmri_path, f"{subject_id}_{ses_id}_labels.npy")
            if not os.path.exists(label_file):
                print(f"Warning: Labels not found for {subject_id}/{ses_id}, skipping.")
                continue
            labels = np.load(label_file)

            # create one sample per interval
            for i in range(len(corr_intervals)):
                self.samples.append({
                    'subject': subject_id,
                    'session': ses_id,
                    'interval': i,
                    'corr_matrix': corr_intervals[i],
                    'label': labels[i]
                })
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # convert correlation matrix to graph
        graph = self.corr_to_graph(sample['corr_matrix'], self.k)
        node_activity = torch.FloatTensor(sample['corr_matrix'].mean(axis=1))  # Example node feature: mean connectivity
        expression = self.expression_data  # Shape: (210, num_genes)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return graph, node_activity, expression, label

    def corr_to_graph(self, corr_matrix, k):
        num_nodes = corr_matrix.shape[0]
        edges =[]
        weights = []

        for node in range(num_nodes):
            corr_values = np.abs(corr_matrix[node, :])
            corr_values[node] = 0.0

            top_k_indices = np.argsort(corr_values)[-k:]

            # add edges based on top-k correlations
            for neighbor in top_k_indices:
                edges.append((node, neighbor))
                weights.append(corr_matrix[node, neighbor])

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weights

# currently implemented with PyG Dataset Object
class BrainDataset(Dataset):
    # for benchmarking + fallback purposes.

    def __init__(self, fmri_path, k=10):
        super().__init__()
        self.k = k
        self.samples = []
        corr_files = sorted(glob.glob(f"{fmri_path}/sub-*_ses-*_interval_corr.npy"))

        for corr_file in corr_files:
            filename = os.path.basename(corr_file)
            parts = filename.split('_')
            subject_id = parts[0]  # 'sub-01'
            ses_id = parts[1]      # 'ses-01'
            corr_intervals = np.load(corr_file)

            label_file = os.path.join(fmri_path, f"{subject_id}_{ses_id}_labels.npy")
            if not os.path.exists(label_file):
                print(f"Warning: Labels not found for {subject_id}/{ses_id}, skipping.")
                continue
            labels = np.load(label_file)

            for i, _ in enumerate(corr_intervals):
                self.samples.append({
                    'subject': subject_id,
                    'session': ses_id,
                    'interval': i,
                    'corr_matrix': corr_intervals[i],
                    'label': labels[i]
                })

    def len(self):  # should equal num of interval graphs
        return len(self.samples)

    def get(self, idx):  # modified to a single PyG Data for PyG DataLoader and PyG models
        sample = self.samples[idx]

        edge_index, edge_weights = self._corr_to_graph(sample['corr_matrix'], self.k)

        node_activity = torch.FloatTensor(sample['corr_matrix'].mean(axis=1))  # mean connectivity

        x = node_activity.unsqueeze(1)  # (num_nodes, 1)

        y = torch.tensor(sample['label'], dtype=torch.long).unsqueeze(0)  # this is label

        data_object = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)  # Data object
        data_object.subject = sample['subject']  # adding on some metadata for analysis
        data_object.session = sample['session']
        data_object.interval = sample['interval']

        return data_object

    def _corr_to_graph(self, corr_matrix, k):
        num_nodes = corr_matrix.shape[0]
        edges = []
        weights = []

        for node in range(num_nodes):
            corr_values = np.abs(corr_matrix[node, :].copy())
            corr_values[node] = 0.0

            top_k_indices = np.argsort(corr_values)[-k:]

            for neighbor in top_k_indices:
                edges.append((node, neighbor))
                weights.append(corr_matrix[node, neighbor])

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float32)

        return edge_index, edge_weights

### EXAMPLE USAGE ###
# dataset = BrainOmicsDataset(
#     fmri_path='/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/60s_interval_corr_matrices/',
#     expression_path='/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/gene_expression_schaefer210.npy',
#     k=10
# )

# dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=0
# )

# for i, batch in enumerate(dataloader):
#     graph, node_activity, expression, label = batch
#     print(f"Batch {i}:")
#     print("Graph edge index shape:", graph[0].shape)
#     print("Graph edge weights shape:", graph[1].shape)
#     print("Node activity shape:", node_activity.shape)
#     print("Expression data shape:", expression.shape)
#     print("Label shape:", label.shape)
#     if i == 0:
#         break


class BrainOmicsDataset5State(Dataset):
    """
    Dataset that loads 5-state sleep probabilities
    """
    def __init__(self, fmri_path, expression_path=None, k=10, 
                 use_expression=True, use_soft_labels=True):
        self.samples = []
        self.k = k
        self.use_expression = use_expression
        self.use_soft_labels = use_soft_labels
        
        # Load expression if needed
        if use_expression and expression_path:
            expression_raw = np.load(expression_path)
            self.expression_data = torch.FloatTensor(expression_raw)
            self.expression_data = (self.expression_data - self.expression_data.mean(dim=0)) / \
                                   (self.expression_data.std(dim=0) + 1e-8)
        else:
            self.expression_data = None
        
        # Load fMRI data and 5-state labels
        corr_files = sorted(glob.glob(f"{fmri_path}/sub-*_interval_corr.npy"))
        
        all_features = []
        
        for corr_file in corr_files:
            filename = os.path.basename(corr_file)
            subject_id = filename.split('_')[0]
            corr_intervals = np.load(corr_file)
            
            # Load 5-state probabilities
            prob_file = os.path.join(fmri_path, f"{subject_id}_5state_probs.npy")
            if not os.path.exists(prob_file):
                print(f"Warning: 5-state probs not found for {subject_id}, skipping")
                continue
            
            state_probs = np.load(prob_file)  # (n_intervals, 5)
            
            # Also load binary labels for supervision
            label_file = os.path.join(fmri_path, f"{subject_id}_ses-01_labels.npy")
            binary_labels = np.load(label_file)
            
            n_corr = len(corr_intervals)
            n_probs = len(state_probs)
            n_labels = len(binary_labels)
            
            if not (n_corr == n_probs == n_labels):
                print(f"Warning: {subject_id} - Length mismatch: "
                      f"corr={n_corr}, probs={n_probs}, labels={n_labels}")
                # Use minimum length
                min_len = min(n_corr, n_probs, n_labels)
                corr_intervals = corr_intervals[:min_len]
                state_probs = state_probs[:min_len]
                binary_labels = binary_labels[:min_len]
                print(f"  Trimmed to {min_len} intervals")

            for i in range(len(corr_intervals)):
                corr_mat = corr_intervals[i]
                if np.isnan(corr_mat).any() or np.isinf(corr_mat).any():
                    continue
                
                # Extract features for normalization
                corr_clip = np.clip(corr_mat, -0.999, 0.999)
                fisher_z = np.arctanh(corr_clip)
                
                node_features = np.column_stack([
                    fisher_z.mean(axis=1),
                    fisher_z.std(axis=1),
                    (fisher_z > 0).mean(axis=1),
                    np.percentile(fisher_z, 75, axis=1),
                    np.percentile(fisher_z, 25, axis=1)
                ])
                
                all_features.append(node_features)
                
                self.samples.append({
                    'subject': subject_id,
                    'interval': i,
                    'corr_matrix': corr_mat,
                    'label': binary_labels[i],  # Binary label for final classification
                    '5state_probs': state_probs[i],  # 5-state soft labels
                    '5state_label': state_probs[i].argmax()  # 5-state hard label
                })
        
        # Global normalization
        all_features = np.concatenate(all_features, axis=0)
        self.feature_mean = all_features.mean(axis=0, keepdims=True)
        self.feature_std = all_features.std(axis=0, keepdims=True) + 1e-8
        
        print(f"✓ Loaded {len(self.samples)} samples with 5-state distributions")
        
        # Statistics
        awake = sum(1 for s in self.samples if s['label'] == 0)
        drowsy = sum(1 for s in self.samples if s['label'] == 1)
        print(f"  Binary: Awake={awake}, Drowsy={drowsy}")
        
        state_counts = [0] * 5
        for s in self.samples:
            state_counts[s['5state_label']] += 1
        state_names = ['Alert', 'N1', 'N2', 'N3', 'REM']
        print(f"  5-state distribution:")
        for i, (name, count) in enumerate(zip(state_names, state_counts)):
            print(f"    {name}: {count} ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        graph = self.corr_to_graph(sample['corr_matrix'], self.k)
        corr_matrix = sample['corr_matrix']
        
        corr_clip = np.clip(corr_matrix, -0.999, 0.999)
        fisher_z = np.arctanh(corr_clip)
        
        node_features = np.column_stack([
            fisher_z.mean(axis=1),
            fisher_z.std(axis=1),
            (fisher_z > 0).mean(axis=1),
            np.percentile(fisher_z, 75, axis=1),
            np.percentile(fisher_z, 25, axis=1)
        ])
        
        # Global normalization
        node_features = (node_features - self.feature_mean) / self.feature_std
        
        node_activity = torch.FloatTensor(node_features)
        label = torch.tensor(sample['label'], dtype=torch.long)  # Binary label
        
        # Return 5-state probabilities for knowledge distillation
        if self.use_soft_labels:
            state_probs = torch.FloatTensor(sample['5state_probs'])  # (5,)
            # Convert to binary soft labels (Alert vs Sleep stages)
            binary_soft = torch.FloatTensor([
                state_probs[0],  # P(Awake) = P(Alert)
                state_probs[1:].sum()  # P(Drowsy) = P(N1+N2+N3+REM)
            ])
            
            return {
                'graph': graph,
                'fmri_features': node_activity,
                'label': label,
                'soft_label': binary_soft,  # For KD
                '5state_probs': state_probs  # Full distribution
            }
        else:
            return {
                'graph': graph,
                'fmri_features': node_activity,
                'label': label
            }
    
    def corr_to_graph(self, corr_matrix, k):
        num_nodes = corr_matrix.shape[0]
        edges = []
        weights = []
        
        for node in range(num_nodes):
            corr_values = np.abs(corr_matrix[node, :])
            corr_values[node] = 0.0
            top_k_indices = np.argsort(corr_values)[-k:]
            
            for neighbor in top_k_indices:
                edges.append((node, neighbor))
                weights.append(np.clip(corr_matrix[node, neighbor], -1.0, 1.0))
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weights