import torch
from torchvision import transforms
import numpy as np


class SpatialPatchDataset(torch.utils.data.Dataset):
    def __init__(self, image, adata, patch_size=256, encoder=None):
        self.image = image
        self.adata = adata
        self.encoder = encoder
        self.patches = get_patches(image.size, patch_size)
        self.spatial_coords = adata.obsm["novae_latent"]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        bbox = self.patches[idx]
        # crop image patch
        image_patch = self.image.crop(bbox)

        # get cell latent vectors in this patch
        mask = cells_in_patch(self.spatial_coords, bbox)
        if np.sum(mask) == 0:
            latent_vecs = np.zeros((1, self.encoder.latent_dim))
        else:
            latent_vecs = self.encoder.transform(self.adata.X[mask])

        return {
            "image_patch": transforms.ToTensor()(image_patch),
            "cell_latents": torch.tensor(latent_vecs, dtype=torch.float),
            "patch_coord": torch.tensor(bbox),
        }


def cells_in_patch(spatial_coords, patch_bbox):
    x0, y0, x1, y1 = patch_bbox
    mask = (
        (spatial_coords[:, 0] >= x0)
        & (spatial_coords[:, 0] < x1)
        & (spatial_coords[:, 1] >= y0)
        & (spatial_coords[:, 1] < y1)
    )
    return mask


def get_patches(image_size, patch_size=256):
    patches = []
    for y in range(0, image_size[1], patch_size):
        for x in range(0, image_size[0], patch_size):
            patches.append((x, y, x + patch_size, y + patch_size))
    return patches
