import pandas as pd
import numpy as np
import abagen
from nilearn import datasets


def create_expression_mapping():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    expression = abagen.get_expression_data(
        atlas=atlas['maps'],           # Schaefer 200 atlas file
        return_donors=False,            # Average across donors
        donors=['9861'],                # Your donor (can add more: ['9861', '10021'])
        lr_mirror='bidirectional',      # Use both hemispheres
        missing='interpolate',           # Fill missing ROIs
        probe_selection='average'
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
    gwas_df = pd.read_csv('../sample_data/gwas_sleep_duration.tsv', sep='\t')
    gwas_genes = gwas_df['MAPPED_GENE'].dropna().unique().tolist()

    # merge the two lists and remove duplicates
    all_genes = list(set(sleep_genes + gwas_genes))
    print("Total genes in expression data:", expression.shape[1])
    print("Total sleep-related genes:", len(all_genes))

    # Filter the expression data to include only the sleep-related genes
    expression_filtered = expression[expression.columns.intersection(all_genes)]
    # Save the filtered expression data to a CSV file
    expression_filtered.to_csv('../sample_data/sleep_related_gene_expression.csv')



