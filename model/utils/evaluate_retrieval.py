import torch
from scipy.stats import pearsonr


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
                                