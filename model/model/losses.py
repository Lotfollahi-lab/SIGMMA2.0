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

        print("z_i norm:", z_i.norm(dim=-1))  # Ideally ~1 after normalize
        print("z_j norm:", z_j.norm(dim=-1))
        print("z_i mean/std:", z_i.mean().item(), z_i.std().item())
        print("z_j mean/std:", z_j.mean().item(), z_j.std().item())

        print("z_i requires_grad:", z_i.requires_grad)
        print("z_j requires_grad:", z_j.requires_grad)

        # Filter: Only valid (non-zero) pairs
        valid_mask = z_j.norm(dim=-1) > 0.0 
        z_i = z_i[valid_mask]
        z_j = z_j[valid_mask]

        if len(z_i) == 0:
            print("No valid pairs found. Skipping loss computation.")
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)

        # Compute cosine similarity
        logits = torch.mm(z_i, z_j.t()) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss
