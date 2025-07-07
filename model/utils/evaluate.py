import torch
from scipy.stats import pearsonr, kentalltau


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

    return stats.mean()
