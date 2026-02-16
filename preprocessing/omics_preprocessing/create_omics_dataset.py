import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob

class BrainOmicsDataset(Dataset):
    samples = []
    def __init__(self, fmri_path, expression_path, k=10):
        self.k = k
        self.expression_data = torch.FloatTensor(np.load(expression_path))
        
        corr_files = sorted(glob.glob(f"{fmri_path}/sub-*_interval_corr.npy"))

        for corr_file in corr_files:
            filename = os.path.basename(corr_file)
            subject_id = filename.split('_')[0]  # 'sub-01'
            corr_intervals = np.load(corr_file)
            
            label_file = os.path.join(fmri_path, f"{subject_id}_labels.npy")
            if not os.path.exists(label_file):
                print(f"Warning: Labels not found for {subject_id}, skipping.")
                continue
            labels = np.load(label_file)

            # create one sample per interval
            for i in range(len(corr_intervals)):
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

