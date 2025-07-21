import torch
import zarr
import scanpy
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def cross_modal_retrieval(query_embedding: torch.Tensor,
                          reference_embeddings: torch.Tensor,
                          reference_ids: pd.DataFrame,
                          fov_size: int = 256,
                          k: int = 5):
    """
    Computes the similarity between query embeddings and reference embeddings.
    Args:
        query_embedding (N X D): D-dim embeddings of N queries.
        reference_embeddings (M X D): D-dim embeddings of M references.
    Returns:
        torch.Tensor: Similarity scores of shape (N, M).
    """
    # Compute cosine similairy between query and reference embeddings
    similarity = query_embedding @ reference_embeddings.T

    # Normalize the similarity scores
    query_norm = query_embedding.norm(dim=-1, keepdim=True)
    reference_norm = reference_embeddings.norm(dim=-1, keepdim=True).T
    similarity = similarity / (query_norm * reference_norm)

    # Retrieve the top-k indices based on similarity scores
    values, indices = torch.topk(similarity.squeeze(0), k)

    retrieved_embeddings = reference_embeddings[indices].detach()
    # Retrieve the corresponding reference IDs
    retrieved_instances = reference_ids.loc[lambda x:np.isin(x.data_index, indices.flatten().tolist()),:].apply(retrieve_raw_data_pair, axis=1, fov_size=fov_size).tolist()

    return values, indices, retrieved_embeddings, retrieved_instances

def retrieve_raw_data_pair(row, fov_size: int = 256):
    """
    Retrieves the raw data pair from the row of the DataFrame.
    Args:
        row (pd.Series): A row from the DataFrame containing image and spatial file paths.
    Returns:
        tuple: A tuple containing the spatial embedding and image array.
    """
    image_array = zarr.open(str(row.image_file), mode='r')
    spatial_array = scanpy.read_h5ad(row.spatial_file)
    index_array = zarr.open(str(row.index_array_file), mode='r')

    slice_args = index_array[row.data_index, :, :]
    y_start = slice_args[0, :].item()
    x_start = slice_args[1, :].item()

    points = spatial_array.obsm["spatial"]
    mask = ((x_start <= points[:, 0]) & (points[:, 0] < x_start + fov_size) & (y_start <= points[:, 1]) & (points[:, 1] < y_start + fov_size))

    spatial_exp = spatial_array[mask, :]
    fov_image = image_array[y_start : y_start + fov_size, x_start : x_start + fov_size, :]

    return spatial_exp, fov_image


def compute_pcc(retrieved: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Computes the Pearson Correlation Coefficient (PCC) between retrieved and ground truth lists.
    Args:
        retrieved (N X D): D-dim embeddings of N retrieved samples.
        ground_truth (N X D): D-dim dembeddings of N ground truth samples.
    Returns:
        float: Mean PCC of N samples.
    """
    assert retrieved.shape == ground_truth.shape, (
        "Shapes of retrieved and ground truth must match."
    )

    stats, p_val = pearsonr(retrieved.numpy(), ground_truth.numpy(), axis=1)

    return stats, p_val

def recall_at_K(retrieved: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
    """
    Computes the Recall@K metric.
    Args:
        retrieved (N X D): D-dim embeddings of N retrieved samples.
        k (int): The number of top retrieved samples to consider.
    Returns:
        float: Recall@K score.
        {\rm{Recall}}{\rm{@}}K=\frac{1}{N}\sum _{q\in Q}{\rm{I}}\left[{\rm{quantile}}\left(q\right)\le K\right],
    """
    assert retrieved.shape == ground_truth.shape, "Shapes of retrieved and ground truth must match."
    assert k > 0, "k must be a positive integer."

    # Sort the retrieved samples based on their scores
    sorted_indices = torch.argsort(retrieved, dim=1, descending=True)
    topk_indices = sorted_indices[:, :k]

    # Compare ground truth with top-K predictions
    correct = (topk_indices == sorted_indices.unsqueeze(1)).any(dim=1)
    
    recall_at_K = correct.float().mean().item()
    
    return recall_at_K
                                