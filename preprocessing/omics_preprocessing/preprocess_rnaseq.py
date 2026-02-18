import pandas as pd
import numpy as np
import abagen
from nilearn import datasets, image
import nibabel as nib


def create_combined_atlas(datadir='../sample_data'):
    # Load the Schaefer 200 atlas
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    schaefer_img = nib.load(atlas['maps'])
    schaefer_data = schaefer_img.get_fdata()

    ho_subcort = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    ho_img = ho_subcort['maps']  # Already an image object!
    ho_data = ho_img.get_fdata()
    ho_labels = ho_subcort['labels']
   
    ho_label_mapping = {
        4: 201,   # Left Thalamus
        5: 202,   # Left Caudate
        6: 203,   # Left Putamen
        9: 204,   # Left Hippocampus
        10: 205,  # Left Amygdala
        15: 206,  # Right Thalamus
        16: 207,  # Right Caudate
        17: 208,  # Right Putamen
        19: 209,  # Right Hippocampus
        20: 210,  # Right Amygdala
    }
    
    # 4. Create combined atlas
    combined_data = schaefer_data.copy()
    for ho_label, new_label in ho_label_mapping.items():
        mask = (ho_data == ho_label)
        combined_data[mask & (combined_data == 0)] = new_label
        print(f"  {new_label}: {ho_labels[ho_label]}")

    combined_img = nib.Nifti1Image(
        combined_data, 
        affine=schaefer_img.affine,
        header=schaefer_img.header
    )
    if datadir:
        nib.save(combined_img, f'{datadir}/combined_atlas.nii.gz')
    return combined_img


def create_expression_mapping(datadir='../sample_data', atlas_path=None):
    atlas = atlas_path
    expression = abagen.get_expression_data(
        atlas=atlas_path,
        return_donors=False,           # Average across donors
        donors=['9861', '10021'],      # Use both RNA-seq donors
        lr_mirror='bidirectional',     # Use both hemispheres
        missing='interpolate',         # Interpolate missing ROIs
        tolerance=2,
        probe_selection='average'      # Required for RNA-seq
    )

    # manually curated list of sleep-related genes from literature
    sleep_genes = [
        # Revisiting brain gene expression changes and protein modifications tracking homeostatic sleep pressure
        "ARC", "BDNF", "FOS", "NR4A1", "EGR1", "PER2",

        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009259
        "PER3", "PER2", "NPAS2", "CLOCK", "NFIL3",
        "BHLHE40", "CRY2", "ARNTL", "ARNTL2", "BHLHE41",
        "TIMELESS", "CRY1", "RORA", "TIPIN", "NR1D1",
        "PER1", "DBP", "CSNK1E",
    ]


    # GWAS genes added (https://www.ebi.ac.uk/gwas/efotraits/OBA_2040171)
    gwas_df = pd.read_csv(f'{datadir}/gwas_sleep_duration.tsv', sep='\t')
    gwas_genes = gwas_df['MAPPED_GENE'].dropna().unique().tolist()

    # merge the two lists and remove duplicates
    all_genes = list(set(sleep_genes + gwas_genes))
    print("Total genes in expression data:", expression.shape[1])
    print("Total sleep-related genes:", len(all_genes))

    # Filter the expression data to include only the sleep-related genes
    expression_filtered = expression[expression.columns.intersection(all_genes)]
    # Save the filtered expression data to a CSV file
    #expression_filtered.to_csv(f'{datadir}/sleep_related_gene_expression.csv')

    # save as dataloader
    np.save(f'{datadir}/gene_expression_schaefer210.npy', expression_filtered.values)
    return expression_filtered


if __name__ == "__main__":
    #create_combined_atlas("/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data")
    expression_filtered = create_expression_mapping(datadir="/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data", atlas_path="/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/combined_atlas.nii.gz")
    