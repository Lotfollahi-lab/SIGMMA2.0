import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Base directory
base_dir = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_xenium"
output_dir = '/lustre/scratch126/cellgen/lotfollahi/mm58/spatial_alignment/img_spatial_alignment/runs_bleep'

# Find all tissue directories (hest_1k_{tissue})
tissue_dirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) 
               and d.startswith("hest_1k_") 
               and not d.endswith("_raw") 
               and not d.endswith("_aligned")]

print(f"Found {len(tissue_dirs)} tissue directories: {tissue_dirs}")

for tissue_dir in tissue_dirs:
    # Extract tissue name (e.g., "IDC" from "hest_1k_IDC")
    tissue_name = tissue_dir.replace("hest_1k_", "")
    if tissue_name == 'IDC':
        continue
    print(f"\nProcessing tissue: {tissue_name}")
    
    tissue_path = os.path.join(base_dir, tissue_dir)
    
    # Get all sample directories
    sample_dirs = [d for d in os.listdir(tissue_path) 
                   if os.path.isdir(os.path.join(tissue_path, d))]
    
    print(f"  Found {len(sample_dirs)} samples: {sample_dirs}")
    
    # Track gene names for each sample
    all_genes = set()
        
    # Track max values
    max_values_data = []
    
    for sample_name in sample_dirs:
        h5_file_path = os.path.join(tissue_path, sample_name, "spatial_data_processed.h5")
        
        if not os.path.exists(h5_file_path):
            print(f"  Warning: {h5_file_path} does not exist, skipping {sample_name}")
            continue
        
        print(f"  Processing {sample_name}...")
        
        spatial = sc.read_h5ad(h5_file_path)
        
        # Get gene names
        gene_names = spatial.var_names.tolist()
        gene_set = set(gene_names)
        if len(all_genes) == 0:
            all_genes = gene_set
        else:
            all_genes = all_genes & gene_set
        
        # Track max value in X
        max_value = spatial.X.max()
        print(f"    Found {len(gene_set)} genes")
        print(f"    Max value in adata.X: {max_value}")
        max_values_data.append(max_value)
        
    
    # Compute intersection of gene names across all samples
    if all_genes:
        print(f"\n  Total genes in intersection: {len(all_genes)}")
        
        # Save genes to keep
        genes_df = pd.DataFrame({'gene': sorted(list(all_genes))})
        output_genes_file = os.path.join(output_dir, f"{tissue_name}_genes_to_keep.csv")
        genes_df.to_csv(output_genes_file, index=False)
        print(f"  Saved genes to: {output_genes_file}")
    else:
        print(f"  Warning: No gene sets found for {tissue_name}")
    
    # Save raw counts data
    if max_values_data:
        max_df = pd.DataFrame({
            'sample': sample_dirs,
            'adata_max': max_values_data
        })
        output_max_file = os.path.join(output_dir, f"{tissue_name}_raw_counts.csv")
        max_df.to_csv(output_max_file, index=False)
        print(f"  Saved max values to: {output_max_file}")
    else:
        print(f"  Warning: No max values found for {tissue_name}")

print("\nProcessing complete!")