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
        topk_similarity_score (N X K): Top-k similarity scores.
        topk_indices (N X K): Indices of the top-k retrieved references.
        retrieved_embeddings (N X K X D): Retrieved embeddings corresponding to the top-k indices.
        retrieved_instances (list of (image_tile, spatial_tile)): List of tuples containing spatial embedding and image array for each retrieved instance.
    """
    # Compute cosine similarity between query and reference embeddings
    similarity = cosine_similarity(query_embedding, reference_embeddings)

    # Retrieve the top-k indices based on similarity scores
    topk_similarity_score, topk_indices = torch.topk(similarity, k)

    retrieved_embeddings = reference_embeddings[topk_indices].detach()
    # Retrieve the corresponding reference IDs
    retrieved_instances = reference_ids.loc[lambda x:np.isin(x.data_index, topk_indices.flatten().tolist()),:].apply(retrieve_raw_data_pair, axis=1, fov_size=fov_size).tolist()

    return retrieved_embeddings, topk_indices, retrieved_instances

def cosine_similarity(query_embedding: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between query and reference embeddings.
    Args:
        query_embedding (N X D): D-dim embeddings of N queries.
        reference_embeddings (M X D): D-dim embeddings of M references.
    Returns:
        similarity (N X M): Cosine similarity scores between queries and references.
    """
    query_norm = query_embedding.norm(dim=-1, keepdim=True)
    reference_norm = reference_embeddings.norm(dim=-1, keepdim=True).T
    similarity = (query_embedding @ reference_embeddings.T) / (query_norm * reference_norm)

    return similarity

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


def compute_pcc(query: torch.Tensor,
                reference: torch.Tensor,
                reference_ids: pd.DataFrame,
                fov_size: int = 256,
                k: int=10) -> float:
    """
    Computes the Pearson Correlation Coefficient (PCC) between retrieved and ground truth lists.
    Args:
        query (N x D): D-dim embeddings of N queries.
        reference (M x D): D-dim embeddings of M ground truth samples.
    Returns:
        float: Mean PCC of K samples for each query.
    """
    assert query.shape[-1] == reference.shape[-1], "Dimension of query and reference embeddings must match."
    assert k > 0, "k must be a positive integer."

    retrieved_embeddings, topk_indices, retrieved_pairs = cross_modal_retrieval(query, reference, reference_ids, fov_size, k)  # N X K

    li_stats = []
    for i in range(retrieved_embeddings.shape[0]):
        li_stats_per_query = []
        for j in range(retrieved_embeddings.shape[1]):
            # Compute PCC for each retrieved sample against the ground truth
            retrieved_sample = retrieved_embeddings[i, j, :]
            ground_truth_sample = query[i, :]

            # pearsonr returns a tuple (correlation coefficient, p-value)
            stats, p_val = pearsonr(retrieved_sample.numpy(), ground_truth_sample.numpy())
            li_stats_per_query.append(stats)
        # Store the mean PCC for each query
        li_stats.append(np.mean(li_stats_per_query, axis=0))

    return torch.tensor(li_stats, dtype=torch.float32)

def recall_at_K(query: torch.Tensor,
                reference: torch.Tensor,
                reference_ids: pd.DataFrame,
                fov_size: int = 256,
                k: int=50) -> float:
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
                                