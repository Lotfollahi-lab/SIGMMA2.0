import torch


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: image embeddings, shape (B, D)
        z_j: spatial embeddings, shape (B, D)
        """
        # Filter: Only valid (non-zero) pairs
        valid_mask = z_j.norm(dim=-1) > 0.0 
        z_i = z_i[valid_mask]
        z_j = z_j[valid_mask]

        if len(z_i) == 0:
            print("No valid pairs found. Skipping loss computation.")
            return torch.tensor(0.0, device=z_i.device, requires_grad=False)

        # Compute cosine similarity
        logits = torch.mm(z_i, z_j.t()) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss


class symmetric_InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(symmetric_InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: image embeddings, shape (B, D)
        z_j: spatial embeddings, shape (B, D)
        """
        # Filter: Only valid (non-zero) pairs
        valid_mask = z_j.norm(dim=-1) > 0.0 
        z_i = z_i[valid_mask]
        z_j = z_j[valid_mask]

        if len(z_i) == 0:
            print("No valid pairs found. Skipping loss computation.")
            return torch.tensor(0.0, device=z_i.device, requires_grad=False)

        # Compute cosine similarity
        logits_xy = torch.mm(z_i, z_j.t()) / self.temperature
        logits_yx = torch.mm(z_j, z_i.t()) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(z_i.size(0), device=z_i.device)

        # Compute the loss
        loss_xy = torch.nn.functional.cross_entropy(logits_xy, labels)
        loss_yx = torch.nn.functional.cross_entropy(logits_yx, labels)

        return (loss_xy + loss_yx) / 2