from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import scanpy
import zarr
import torch
from torchvision.transforms import ToTensor, Resize, Compose


class SpatialPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_file: Path | str,
        spatial_file: Path | str,
        indices_file: Path | str,
        fov_size: int = 256,
    ):
        self.image_array = zarr.open(image_file, mode="r")
        self.index_array = zarr.open(indices_file, mode="r")
        self.spatial_anndata = scanpy.read_10x_h5(
            spatial_file
        )  # Must be preprocessed with `novae` to have the `novae_latent` field and `obsm["spatial"]` field.
        self.fov_size = fov_size

    def __getitem__(self, idx):
        slice_args = self.index_array[idx, :, :]
        y_start = slice_args[0, :]
        x_start = slice_args[1, :]

        # 1. spatial embedding (cells inside this FOV)
        points = self.spatial_anndata.obsm["spatial"]

        mask = (
            (x_start <= points[:, 0])
            & (points[:, 0] < x_start + self.fov_size)
            & (y_start <= points[:, 1])
            & (points[:, 1] < y_start + self.fov_size)
        )

        spatial_embedding = (
            torch.Tensor(self.spatial_anndata.obsm["novae_latent"][mask])
            .mean(dim=0)
            .unsqueeze(0)
        )

        # 2. cropped image
        fov_image = self.image_array[
            x_start : x_start + self.fov_size, y_start : y_start + self.fov_size, :
        ]
        # → Resize image for ViT input
        transform = Compose([ToTensor(), Resize((224, 224))])
        image = transform(fov_image).unsqueeze(0)

        return {
            "image": image,
            "spatial": spatial_embedding,
        }

    def __len__(self) -> int:
        return self.index_array.shape[0]
