import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def integrate_into_single_anndata(adata,
                                  cell_ids,
                                  features,
                                  key="sigmma_latent",
                                  cell_type=None):
    
    adata = adata.copy()
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    elif sp.issparse(features):
        features = features.toarray()
    elif not isinstance(features, np.ndarray):
        raise TypeError(...)

    # Initialize full feature matrix (n_cells × embedding_dim) with NaNs,
    # then place computed features at positions specified by cell_ids    
    full_features = np.full((adata.n_obs, features.shape[1]), np.nan, dtype=features.dtype)
    full_features[cell_ids.astype(int)] = features
    adata.obsm[key] = full_features
    
    if cell_type is not None:
        if not isinstance(cell_type, pd.DataFrame):
            raise TypeError(
                f"`cell_type` must be a pandas DataFrame or None, got {type(cell_type)}")

        if not cell_type.empty:
            adata.obs = adata.obs.join(cell_type, how='left')

    return adata

def remove_NaN_cells(adata, key="sigmma_latent"):
    print(f"Before: {adata.n_obs} cells")
    X = adata.obsm[key]
    valid_cells = ~np.isnan(X).any(axis=1)

    adata = adata[valid_cells].copy()
    print(f"After: {valid_cells.sum()} cells")
    print("-----")
    return adata