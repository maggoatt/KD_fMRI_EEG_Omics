import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.create_omics_dataset import BrainOmicsDataset


class ROIConnectivityDataset(Dataset):
    """
    Creates training samples: (gene_expr, state) → connectivity_pattern
    
    For each fMRI sample and each ROI:
    - Input: Gene expression for that ROI + brain state
    - Target: That ROI's connectivity to all other 209 ROIs
    """
    def __init__(self, brain_dataset, expression_data):
        self.samples = []
        
        
        for sample_idx, sample in enumerate(brain_dataset.samples):
            corr_matrix = sample['corr_matrix']
            state = sample['label']  # 0=awake, 1=drowsy
            
            # Skip bad data
            if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
                continue
            
            # Fisher Z transform
            corr_clip = np.clip(corr_matrix, -0.999, 0.999)
            fisher_z = np.arctanh(corr_clip)
            
            # Create one training sample per ROI
            for roi in range(210):
                # Input: Gene expression for this ROI
                gene_expr = expression_data[roi, :]
                
                # Target: This ROI's connectivity to all others (exclude self-connection)
                connectivity = np.concatenate([
                    fisher_z[roi, :roi], 
                    fisher_z[roi, roi+1:]
                ])  # Shape: (209,)
                
                self.samples.append({
                    'gene_expr': torch.FloatTensor(gene_expr),
                    'state': torch.FloatTensor([state]),
                    'connectivity': torch.FloatTensor(connectivity),
                    'roi': roi,
                    'sample_idx': sample_idx
                })
        
        print(f"✓ Created {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]