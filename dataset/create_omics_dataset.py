import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import os
import glob

class BrainOmicsDataset(Dataset):
    samples = []
    def __init__(self, fmri_path, expression_path, k=10, augment=False, use_expression=True):
        self.samples = []
        self.k = k
        self.augment = augment
        self.use_expression = use_expression

        expression_raw = np.load(expression_path)
        
        # Clip extreme values and normalize
        expression_raw = np.clip(expression_raw, np.percentile(expression_raw, 1), 
                                                 np.percentile(expression_raw, 99))


        self.expression_data = torch.FloatTensor(np.load(expression_path))
        self.expression_data = (self.expression_data - self.expression_data.mean(dim=0)) / \
                               (self.expression_data.std(dim=0) + 1e-8)

        corr_files = sorted(glob.glob(f"{fmri_path}/sub-*_interval_corr.npy"))

        for corr_file in corr_files:
            filename = os.path.basename(corr_file)
            subject_id = filename.split('_')[0]  # 'sub-01'
            corr_intervals = np.load(corr_file)
            
            # TODO: change to real labels
            label_file = os.path.join(fmri_path, f"{subject_id}_synthetic_labels.npy")
            if not os.path.exists(label_file):
                print(f"Warning: Labels not found for {subject_id}, skipping.")
                continue
            labels = np.load(label_file)

            # create one sample per interval
            for i in range(len(corr_intervals)):
                corr_mat = corr_intervals[i]
                if np.isnan(corr_mat).any() or np.isinf(corr_mat).any():
                    print(f"Warning: NaN/Inf in {subject_id} interval {i}, skipping")
                    continue
                self.samples.append({
                    'subject': subject_id,
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
        corr_matrix = sample['corr_matrix']

        if self.augment:
            noise = np.random.normal(0, 0.02, corr_matrix.shape)
            corr_matrix += noise
        
        # corr_matrix = np.clip(corr_matrix, -0.999, 0.999)  # Ensure values remain in valid range
        # corr_matrix = sample['corr_matrix']
        corr_matrix_clip = np.clip(corr_matrix, -0.999, 0.999)  # Avoid log(0)
        fisher_z = np.arctanh(corr_matrix_clip)
        node_features = np.column_stack([
            fisher_z.mean(axis=1),              # Mean connectivity (Fisher Z)
            fisher_z.std(axis=1),               # Connectivity variability
            (fisher_z > 0).mean(axis=1),        # Proportion of positive connections (NOT count)
            np.percentile(fisher_z, 75, axis=1), # 75th percentile connectivity
            np.percentile(fisher_z, 25, axis=1)  # 25th percentile connectivity
        ])
    
        # Standardize features
        #node_features = (node_features - node_features.mean(axis=0)) / (node_features.std(axis=0) + 1e-8)


        node_activity = torch.FloatTensor(node_features)  # Shape: (210, 3)
        #node_activity = torch.FloatTensor(sample['corr_matrix'].mean(axis=1))  # Example node feature: mean connectivity
        expression = self.expression_data  # Shape: (210, num_genes)
        label = torch.tensor(sample['label'], dtype=torch.long)

        if not self.use_expression:
            return graph, node_activity, label
        
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
                # weights.append(corr_matrix[node, neighbor])
                weight = np.clip(corr_matrix[node, neighbor], -1.0, 1.0)
                weights.append(weight)

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

