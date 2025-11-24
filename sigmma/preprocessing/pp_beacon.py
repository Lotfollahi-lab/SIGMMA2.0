import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import scanpy.external as sce

# Manifest 
manifest = pd.read_csv('/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_beacon_MANIFEST.csv')
files_to_process = manifest['spatial_preprocessed'].tolist()
main_output_dir = '/lustre/scratch126/cellgen/lotfollahi/mm58/spatial_alignment/sanger_in_house_xenium'


sample_data = []

for i, file in enumerate(files_to_process):
    sample_name = f'Sample_{i}'
    h5_file_path = file
    if not os.path.exists(h5_file_path):
        print(f"  Warning: {h5_file_path} does not exist, skipping {sample_name}")
        continue

    print(f"  Loading {sample_name}...")
        
    spatial = sc.read_h5ad(h5_file_path)
        
    # Calculate number of non-zero genes per cell in original data
    num_non_zero_genes = (spatial.X != 0).sum(axis=1)
    if hasattr(num_non_zero_genes, 'A1'):  # If sparse matrix
        num_non_zero_genes = num_non_zero_genes.A1
    spatial.obs['num_non_zero_genes'] = num_non_zero_genes
        
    # Check if raw counts (max is an integer)
    max_value = spatial.X.max()
    is_raw_counts = max_value == int(max_value)
    
    print(f"    Max value in adata.X: {max_value}")
    print(f"    Is raw counts: {is_raw_counts}")
    
    # Normalize if raw counts
    if is_raw_counts:
        print(f"    Normalizing {sample_name}...")
        sc.pp.normalize_total(spatial)
        sc.pp.log1p(spatial)
     
    path_to_save = file.replace('/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium',
                                main_output_dir)

    # Create output sample directory
    output_directory= path_to_save.replace('spatial_data_processed.h5ad', '')
    os.makedirs(output_directory, exist_ok=True)
    
    # Save to disk (keeping original X and all metadata, just adding X_pca_harmony and num_non_zero_genes)
    output_file = os.path.join(output_directory, "spatial_data_lognorm.h5")
    spatial.write_h5ad(output_file)
    print(f"    Saved Sample_{i} to {output_file}")
    print(f"      Shape: {spatial.shape}")

print("\nProcessing complete!")