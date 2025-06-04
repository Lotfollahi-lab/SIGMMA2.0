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
        # Normalize the embeddings
        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)

        # Flatten in case of extra dims (e.g., (B, 1, D))
        z_i = z_i.view(z_i.size(0), -1)
        z_j = z_j.view(z_j.size(0), -1)

        # Compute cosine similarity
        logits = torch.mm(z_i, z_j.t()) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss
