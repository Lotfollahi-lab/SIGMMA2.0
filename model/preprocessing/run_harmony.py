import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import scanpy.external as sce

# Base directory
base_dir = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_xenium"
output_dir = '/lustre/scratch126/cellgen/lotfollahi/mm58/spatial_alignment/public_xenium'

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
    if tissue_name != 'IDC':
        continue
    print(f"\nProcessing tissue: {tissue_name}")
    
    tissue_path = os.path.join(base_dir, tissue_dir)
    
    # Get all sample directories
    sample_dirs = [d for d in os.listdir(tissue_path) 
                   if os.path.isdir(os.path.join(tissue_path, d))]
    
    print(f"  Found {len(sample_dirs)} samples: {sample_dirs}")
    
    # Step 1: Load samples, normalize if needed, and collect gene names
    all_genes = set()
    sample_data = []
    
    for sample_name in sample_dirs:
        if 'v2' in sample_name:
            print(f"  Skipping {sample_name} (v2 sample)")
            continue
        h5_file_path = os.path.join(tissue_path, sample_name, "spatial_data_processed.h5")
        
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
        
        # Get gene names
        gene_names = spatial.var_names.tolist()
        gene_set = set(gene_names)
        
        if len(all_genes) == 0:
            all_genes = gene_set
        else:
            all_genes = all_genes & gene_set
        
        print(f"    Found {len(gene_set)} genes")
        
        # Store sample data
        sample_data.append({
            'name': sample_name,
            'adata': spatial
        })
    
    # Step 2: Filter to intersection of genes
    if not all_genes:
        print(f"  Warning: No gene intersection found for {tissue_name}, skipping")
        continue
    
    print(f"\n  Total genes in intersection: {len(all_genes)}")
    common_genes = sorted(list(all_genes))
    
    for sample in sample_data:
        sample['adata'] = sample['adata'][:, common_genes].copy()
        print(f"    Filtered {sample['name']} to {len(common_genes)} genes")
    
    # Step 3: Concatenate datasets with slide annotation
    print(f"\n  Concatenating {len(sample_data)} samples...")
    adatas_list = []
    for sample in sample_data:
        adata = sample['adata']
        adata.obs['slide'] = sample['name']
        adatas_list.append(adata)
    
    adata_combined = sc.concat(adatas_list, join='outer', label='batch', keys=[s['name'] for s in sample_data])
    print(f"    Combined shape: {adata_combined.shape}")
    
    # Step 4: Run PCA and Harmony
    print(f"\n  Running PCA...")
    sc.pp.pca(adata_combined, n_comps=50)
    
    print(f"  Running Harmony integration on 'slide'...")
    sce.pp.harmony_integrate(adata_combined, 'slide')
    
    # Step 5: Save each sample separately
    print(f"\n  Saving processed samples...")
    
    # Create output tissue directory
    output_tissue_dir = os.path.join(output_dir, f"hest_1k_{tissue_name}")
    os.makedirs(output_tissue_dir, exist_ok=True)
    
    for sample in sample_data:
        sample_name = sample['name']
        original_adata = sample['adata']
        
        # Filter combined data to this sample to get Harmony embeddings
        sample_mask = adata_combined.obs['slide'] == sample_name
        if sample_mask.sum() == 0:
            print(f"    Skipping {sample_name} (no data)")
            continue
        
        adata_sample_harmony = adata_combined[sample_mask].copy()
        
        # Check if Harmony embeddings exist
        if 'X_pca_harmony' not in adata_sample_harmony.obsm:
            print(f"    Warning: No X_pca_harmony found for {sample_name}")
            continue
        
        # Add Harmony embeddings to the original adata
        original_adata.obsm['X_pca_harmony'] = adata_sample_harmony.obsm['X_pca_harmony']
        
        # Create output sample directory
        output_sample_dir = os.path.join(output_tissue_dir, sample_name)
        os.makedirs(output_sample_dir, exist_ok=True)
        
        # Save to disk (keeping original X and all metadata, just adding X_pca_harmony and num_non_zero_genes)
        output_file = os.path.join(output_sample_dir, "spatial_data_harmony.h5")
        original_adata.write_h5ad(output_file)
        print(f"    Saved {sample_name} to {output_file}")
        print(f"      Shape: {original_adata.shape}")
        print(f"      Added X_pca_harmony with shape: {original_adata.obsm['X_pca_harmony'].shape}")

print("\nProcessing complete!")