from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from collections import Counter

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
        number_of_genes: int = 50,
        housekeeping_genes: Optional[Sequence[str]] = None,
    ):
        self.image_array = zarr.open(image_file, mode="r")
        self.index_array = zarr.open(indices_file, mode="r")
        self.spatial_anndata = scanpy.read_h5ad(
            spatial_file
        )  # Must be preprocessed with `novae` to have the `novae_latent` field and `obsm["spatial"]` field.
        self.fov_size = fov_size
        self.number_of_genes = number_of_genes
        self.housekeeping_genes = housekeeping_genes

    def __getitem__(self, idx):
        slice_args = self.index_array[idx, :, :]
        y_start = slice_args[0, :].item()
        x_start = slice_args[1, :].item()

        # 1-1. spatial embedding (cells inside this FOV)
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
        )

        # 1-2. spatial text (top K raneked genes in this FOV): same logic as loki preprocessing "https://github.com/GuangyuWangLab2021/Loki/blob/main/src/loki/preprocess.py"
        # exclude housekeeping genes
        if self.housekeeping_genes is not None:
            filtered_spatial_anndata = self.spatial_anndata[:, ~self.spatial_anndata.var.index.isin(self.housekeeping_genes)]
        else:
            filtered_spatial_anndata = self.spatial_anndata

        mean_exp = np.asarray(filtered_spatial_anndata.X[mask].mean(axis=0)).ravel()
        K = self.number_of_genes
        spatial_text = " ".join(filtered_spatial_anndata.var.index[np.argsort(mean_exp)[-K:][::-1]])

        # 2. cell type composition
        counter = Counter(self.spatial_anndata.obs["cell_types"][mask])
        value_vector = np.array([counter[key] for key in sorted(counter.keys())])
        cell_type_composition = torch.Tensor(value_vector / value_vector.sum())

        # 3. cropped image
        fov_image = self.image_array[
                    y_start : y_start + self.fov_size, 
                    x_start : x_start + self.fov_size, 
                    :
                    ]
        # → Resize image for ViT input
        transform = Compose([ToTensor(), Resize((224, 224))])
        image = transform(fov_image)

        return {
            "image": image,
            "spatial": spatial_embedding,
            "spatial_text": spatial_text,
            "cell_type_composition": cell_type_composition 
        }

    def __len__(self) -> int:
        return self.index_array.shape[0]
