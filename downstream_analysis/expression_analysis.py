import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx
from sklearn.decomposition import PCA
import pandas as pd
from dataset.create_omics_dataset import BrainOmicsDataset


class SleepROIDiscovery:
    def __init__(self, dataset, expression_data, gene_names_path=None):
        """
        Args:
            dataset: Your BrainOmicsDataset instance
            expression_data: Gene expression tensor (210, n_genes)
        """
        self.dataset = dataset
        self.expression_data = expression_data.numpy() if torch.is_tensor(expression_data) else expression_data
        self.n_rois = 200
        
        if gene_names_path:
            try:
                self.gene_names = np.load(gene_names_path, allow_pickle=True).tolist()
            except Exception as e:
                self.gene_names = [f"Gene_{i}" for i in range(self.expression_data.shape[1])]
        else:
            self.gene_names = [f"Gene_{i}" for i in range(self.expression_data.shape[1])]
 

    def compute_roi_importance(self):
        awake_samples = [s for s in self.dataset.samples if s['label'] == 0]
        sleep_samples = [s for s in self.dataset.samples if s['label'] == 1]


        roi_importance = np.zeros(self.n_rois)
        roi_connectivity_awake = np.zeros((self.n_rois, len(awake_samples)))
        roi_connectivity_drowsy = np.zeros((self.n_rois, len(sleep_samples)))
        roi_pvalues = np.zeros(self.n_rois)

        for roi in range(self.n_rois):
            # Collect connectivity for this ROI across samples
            awake_conn = []
            drowsy_conn = []
            
            for i, sample in enumerate(awake_samples):
                corr_matrix = sample['corr_matrix']
                fisher_z = np.arctanh(np.clip(corr_matrix, -0.999, 0.999))
                # Mean connectivity of this ROI to all others
                awake_conn.append(fisher_z[roi, :].mean())
                roi_connectivity_awake[roi, i] = fisher_z[roi, :].mean()
            
            for i, sample in enumerate(sleep_samples):
                corr_matrix = sample['corr_matrix']
                fisher_z = np.arctanh(np.clip(corr_matrix, -0.999, 0.999))
                drowsy_conn.append(fisher_z[roi, :].mean())
                roi_connectivity_drowsy[roi, i] = fisher_z[roi, :].mean()
            
            awake_conn = np.array(awake_conn)
            drowsy_conn = np.array(drowsy_conn)
            
            # Statistical test
            t_stat, p_val = stats.ttest_ind(awake_conn, drowsy_conn)
            roi_pvalues[roi] = p_val
            
            # Effect size (Cohen's d)
            mean_diff = awake_conn.mean() - drowsy_conn.mean()
            pooled_std = np.sqrt((awake_conn.var() + drowsy_conn.var()) / 2)
            cohens_d = abs(mean_diff) / (pooled_std + 1e-8)
            
            roi_importance[roi] = cohens_d
        
        # Find significant ROIs
        significant_rois = np.where(roi_pvalues < 0.05)[0]
        
        
        # Top 10 most changing ROIs
        top_10 = np.argsort(roi_importance)[-10:][::-1]
        
        self.roi_importance = roi_importance
        self.roi_pvalues = roi_pvalues
        self.roi_connectivity_awake = roi_connectivity_awake
        self.roi_connectivity_drowsy = roi_connectivity_drowsy
        
        return roi_importance, roi_pvalues

    def correlate_with_genes(self):
        gene_expression_per_roi = self.expression_data.mean(axis=1)
        
        # Overall correlation
        correlation, p_value = stats.pearsonr(self.roi_importance, gene_expression_per_roi)
        
        print(f"\nCorrelation between ROI importance and sleep gene expression:")
        print(f"  Pearson r: {correlation:.3f}")
        print(f"  P-value: {p_value:.4f}")
        
        print(f"  Expression data genes: {self.expression_data.shape[1]}")
        print(f"  Gene names loaded: {len(self.gene_names)}")
        
        
        # Per-gene correlations
        print(f"\nTop genes correlated with ROI importance:")
        gene_correlations = []
        for gene_idx in range(self.expression_data.shape[1]):
            gene_expr = self.expression_data[:, gene_idx]
            r, p = stats.pearsonr(self.roi_importance, gene_expr)
            if gene_idx < len(self.gene_names):
                gene_name = self.gene_names[gene_idx]
            else:
                gene_name = f"Gene_{gene_idx}"
            gene_correlations.append({
                'index': gene_idx,
                'name': gene_name,
                'correlation': r,
                'p_value': p
            })
        
        # Sort by absolute correlation
        gene_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
       
        
        for i, gene in enumerate(gene_correlations[:10]):
            sig = "***" if gene['p_value'] < 0.001 else "**" if gene['p_value'] < 0.01 else "*" if gene['p_value'] < 0.05 else ""
            print(f"  {i+1}. Gene {gene['name']}: r={gene['correlation']:.3f}, p={gene['p_value']:.4f} {sig}")
        
        self.gene_expression_per_roi = gene_expression_per_roi
        self.gene_correlations = gene_correlations
        
        return correlation, p_value, gene_correlations
    
    def identify_important_rois(self, importance_threshold=75, expression_threshold=60):
        imp_thresh = np.percentile(self.roi_importance, importance_threshold)
        expr_thresh = np.percentile(self.gene_expression_per_roi, expression_threshold)
        
        print(f"\nThresholds:")
        print(f"  Importance (Cohen's d): >{imp_thresh:.3f} ({importance_threshold}th percentile)")
        print(f"  Gene expression: >{expr_thresh:.3f} ({expression_threshold}th percentile)")
        
        # Identify critical ROIs (high on both)
        high_importance = self.roi_importance > imp_thresh
        high_expression = self.gene_expression_per_roi > expr_thresh
        significant = self.roi_pvalues < 0.05
        
        # Sleep-critical: high importance + high expression + significant
        critical_rois = np.where(high_importance & high_expression & significant)[0]
        
        # Also identify control groups
        high_imp_low_expr = np.where(high_importance & ~high_expression & significant)[0]
        low_imp_high_expr = np.where(~high_importance & high_expression)[0]
        
        print(f"\nROI Categories:")
        print(f"  Sleep-critical (high both): {len(critical_rois)} ROIs")
        print(f"  High importance, low genes: {len(high_imp_low_expr)} ROIs")
        print(f"  High genes, low importance: {len(low_imp_high_expr)} ROIs")
        
        if len(critical_rois) > 0:
            print(f"\nSleep-critical ROIs: {critical_rois[:20]}{'...' if len(critical_rois) > 20 else ''}")
            print(f"  Mean importance: {self.roi_importance[critical_rois].mean():.3f}")
            print(f"  Mean expression: {self.gene_expression_per_roi[critical_rois].mean():.3f}")
        
        self.critical_rois = critical_rois
        self.high_imp_low_expr = high_imp_low_expr
        self.low_imp_high_expr = low_imp_high_expr
        
        return critical_rois


    def visualize_results(self):
        # Visualize ROI importance distribution
        plt.figure(figsize=(12, 5))
        sns.histplot(self.roi_importance, bins=30, kde=True)
        plt.title("Distribution of ROI Importance (Cohen's d)")
        plt.xlabel("Cohen's d")
        plt.ylabel("Frequency")
        plt.show()
        
        # Scatter plot of importance vs gene expression
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.gene_expression_per_roi, y=self.roi_importance)
        plt.title("ROI Importance vs Sleep Gene Expression")
        plt.xlabel("Mean Sleep Gene Expression")
        plt.ylabel("ROI Importance (Cohen's d)")
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.show()


    def run_complete_analysis(self):
        self.compute_roi_importance()
        self.correlate_with_genes()
        self.identify_important_rois()
        self.visualize_results()
        
        return {
            'roi_importance': self.roi_importance,
            'roi_pvalues': self.roi_pvalues,
            'gene_correlations': self.gene_correlations,
            'critical_rois': self.critical_rois
        }


if __name__ == "__main__":
    dataset = BrainOmicsDataset(
        fmri_path='/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/60s_interval_corr_matrices/',
        expression_path='/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/gene_expression_schaefer210.npy',
        k=10,
        use_expression=True
    )

    analysis = SleepROIDiscovery(dataset, dataset.expression_data)
    
    # Run complete pipeline
    results = analysis.run_complete_analysis()
    
    print("\n Sleep-critical ROI discovery complete!")
    print(f"  Found {len(results['critical_rois'])} sleep-critical brain regions")
