import torch
import zarr
import scanpy
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

@torch.no_grad()
def retrieval_score(model, data_loader, device, config):
    model_name = config.model.get("name")
    model.eval()

    num_datapoints = len(data_loader.dataset)
    batch_size = config.get("test.batch_size", 1)

    image_embed_li = []
    spatial_embed_li = []
    for batch in data_loader:
        image = batch["image"].to(device)
        cell_type_composition = batch["cell_type_composition"].to(device)

        if  model_name == "hierarchical_img_cellgraph":
            spatial = batch["spatial_graph"].to(device)
            inputs = [image, spatial, cell_type_composition]
        elif model_name == "base_contrastive":
            spatial = batch["spatial"].to(device)
            inputs = [image, spatial, cell_type_composition]
        elif model_name in ["openAI_CLIP", "PLIP", "OmiCLIP"]:
            spatial = batch["spatial_text"]
            inputs = [image, spatial]
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Forward pass through the model
        with torch.no_grad():
            if config.model.name == "hierarchical_img_cellgraph":
                loss_kl, loss_contrastive, loss_ce, loss, image_embed, spatial_embed = model(*inputs) # TODO: multi-resolution embeddings
            else:
                loss, image_embed, spatial_embed = model(*inputs)

        # Normalize embeddings
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        spatial_embed = spatial_embed / spatial_embed.norm(dim=-1, keepdim=True)

        image_embed_li.append(image_embed.detach().cpu())
        spatial_embed_li.append(spatial_embed.detach().cpu())
    img_embeds = torch.cat(image_embed_li, dim=0)
    spatial_embeds = torch.cat(spatial_embed_li, dim=0)

    score_matrix_img2spatial = img_embeds @ spatial_embeds.T
    score_matrix_spatial2img = spatial_embeds @ img_embeds.T

    return score_matrix_img2spatial.cpu().numpy(), score_matrix_spatial2img.cpu().numpy()

def eval_retrieval(scores_img2spatial, scores_spatial2img, query_dataloader, reference_dataloader):
    num_queries = len(query_dataloader.dataset)
    num_reference = len(reference_dataloader.dataset)
    target_len = num_reference - num_queries

    gt_indices = np.arange(num_queries) + target_len
    ranks = np.zeros(scores_img2spatial.shape[0])
    for index, score in enumerate(scores_img2spatial):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == gt_indices[index])[0][0]
    img_r10 = 100 *len(np.where(ranks < 10)[0]) / len(ranks)
    img_r50 = 100 *len(np.where(ranks < 50)[0]) / len(ranks)
    img_r100 = 100 *len(np.where(ranks < 100)[0]) / len(ranks)

    for index, score in enumerate(scores_spatial2img):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == gt_indices[index])[0][0]
    spatial_r10 = 100 *len(np.where(ranks < 10)[0]) / len(ranks)
    spatial_r50 = 100 *len(np.where(ranks < 50)[0]) / len(ranks)
    spatial_r100 = 100 *len(np.where(ranks < 100)[0]) / len(ranks)

    eval_results = {"image_recall_10": img_r10,
                     "image_recall_50": img_r50,
                     "image_recall_100": img_r100,
                     "spatial_recall_10": spatial_r10,
                     "spatial_recall_50": spatial_r50,
                     "spatial_recall_100": spatial_r100}
    return eval_results

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
        query (N x D): D-dim embeddings of N queries.
        reference (M x D): D-dim embeddings of M ground truth samples.
    Returns:
        float: recall@K score.
    """
    assert query.shape[-1] == reference.shape[-1], "Dimension of query and reference embeddings must match."
    assert k > 0, "k must be a positive integer."

    retrieved_embeddings, topk_indices, retrieved_pairs = cross_modal_retrieval(query, reference, reference_ids, fov_size, k)

    recall = None
    return recall
                                