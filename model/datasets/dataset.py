from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Union

import copy
import numpy as np
import pandas as pd
import collections
from itertools import product

import scanpy
import squidpy as sq
import zarr
import torch
from torchvision.transforms import ToTensor, Resize, Compose
import dgl
from torch.utils.data._utils.collate import collate, collate_tensor_fn, collate_float_fn, collate_int_fn, collate_str_fn

collate_fn_map: dict[Union[type, tuple[type, ...]], Callable] = {
    torch.Tensor: collate_tensor_fn
}
collate_fn_map[float] = collate_float_fn
collate_fn_map[int] = collate_int_fn
collate_fn_map[str] = collate_str_fn
collate_fn_map[bytes] = collate_str_fn


class MultiSectionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 manifest_path: Path | str):
        manifest =  pd.read_csv(manifest_path)
        slides = manifest['sample_name']
        li_datsets = []
        for slide in slides:
            image_file = manifest.loc[manifest['sample_name'] == slide, 'hematoxylin_eosin_image'].values[0]
            spatial_file = manifest.loc[manifest['sample_name'] == slide, 'spatial_preprocessed'].values[0]
            index_file = manifest.loc[manifest['sample_name'] == slide, 'index'].values[0]
            dataset = SpatialPatchDataset(
                image_file=Path(image_file),
                spatial_file=Path(spatial_file),
                indices_file=Path(index_file))
            li_datsets.append(dataset)
        self.modality_datasets = torch.utils.data.ConcatDataset(li_datsets)

    def __getitem__(self, idx: int):
        return self.modality_datasets[idx]

    def __len__(self):
        return len(self.modality_datasets)

class SpatialPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_file: Path | str,
        spatial_file: Path | str,
        indices_file: Path | str,
        fov_size: int = 256,
        number_of_genes: int = 50,
        num_resolutions: int = 3,
        housekeeping_genes: Optional[Sequence[str]] = None,
        hierarchical_graph: bool = True,
    ):
        self.image_file = str(image_file)
        self.indices_file = str(indices_file)
        self.spatial_file = str(spatial_file)

        self.image_array = zarr.open(image_file, mode="r")
        self.index_array = zarr.open(indices_file, mode="r")
        self.spatial_anndata = scanpy.read_h5ad(
            spatial_file
        )  # Must be preprocessed with `novae` to have the `novae_latent` field and `obsm["spatial"]` field.
        sq.gr.spatial_neighbors(self.spatial_anndata, n_neighs=10, coord_type="generic")
        self.fov_size = fov_size
        self.number_of_genes = number_of_genes
        self.num_resolutions = num_resolutions
        self.housekeeping_genes = housekeeping_genes
        self.hierarchical_graph = hierarchical_graph


    def __getitem__(self, idx):
        slice_args = self.index_array[idx, :, :]
        y_start = slice_args[0, 0]
        x_start = slice_args[1, 0]

        # 1-1. spatial embedding (cells inside this FOV)
        points = self.spatial_anndata.obsm["spatial"]

        mask = (
            (x_start <= points[:, 0])
            & (points[:, 0] < x_start + self.fov_size)
            & (y_start <= points[:, 1])
            & (points[:, 1] < y_start + self.fov_size)
        )

        spatial_embedding = (
            torch.Tensor(self.spatial_anndata.obsm["nemo_latent"][mask]).mean(dim=0)
        )

        # 1-2. spatial graph
        sparse_maxtrix = self.spatial_anndata[mask,:].obsp["spatial_connectivities"]

        spatial_graph = dgl.from_scipy(sparse_maxtrix)
        node_features = torch.Tensor(self.spatial_anndata.obsm['nemo_latent'][mask])
        spatial_graph.ndata['feat'] = node_features

        # Edge weight: 1 (intra-subgraph edges) or 0 (inter-subgraph edges)
        # Edge mask: -1 (intra-subgraph edges) or 0 (inter-subgraph edges)
        spatial_graph.edata['weight'] = torch.ones(len(spatial_graph.edges()[0]))
        spatial_graph.edata['mask'] = torch.full((len(spatial_graph.edges()[0]),), -1.0)

        if self.hierarchical_graph == True:
            grid_size = self.fov_size // 2**self.num_resolutions

            subgraphs = []

            for x_start_grid, y_start_grid in product(range(0, self.fov_size, grid_size), range(0, self.fov_size, grid_size)):

                mask_grid = (
                    (x_start + x_start_grid <= points[:, 0])
                    & (points[:, 0] < x_start + x_start_grid + grid_size)
                    & (y_start + y_start_grid <= points[:, 1])
                    & (points[:, 1] < y_start + y_start_grid + grid_size)
                    )

                sparse_maxtrix_grid = self.spatial_anndata[mask_grid,:].obsp["spatial_connectivities"]

                subgraph_grid = dgl.from_scipy(sparse_maxtrix_grid)

                node_features_grid = torch.Tensor(self.spatial_anndata[mask_grid,:].X.toarray())
                subgraph_grid.ndata['feat'] = node_features_grid

                subgraphs.append(subgraph_grid)
            
            disjoint_spatial_graph = dgl.batch(subgraphs)
            disjoint_spatial_graph.num_subgraphs = disjoint_spatial_graph.batch_size

            sub_edges = set((int(u), int(v)) for u, v in zip(*disjoint_spatial_graph.edges()))

            edge_mask = torch.zeros(spatial_graph.num_edges(), dtype=torch.int8)
            spatial_edge_map = { (int(u), int(v)): eid for eid, (u, v) in enumerate(zip(*spatial_graph.edges())) }

            for edge in sub_edges:
                if edge in spatial_edge_map:
                    edge_mask[spatial_edge_map[edge]] = -1

            spatial_graph.edata['mask'] = edge_mask
            spatial_graph.edata['weight'] = torch.where(edge_mask == -1, torch.tensor(1.0), torch.tensor(0.0))

            print("Disjoint spatial graph:")
            print(disjoint_spatial_graph.num_nodes(), disjoint_spatial_graph.num_edges())
            print("Spatial graph:")
            print(spatial_graph.num_nodes(), spatial_graph.num_edges())
        # 1-3. spatial text (top K ranked genes in this FOV): same logic as loki preprocessing "https://github.com/GuangyuWangLab2021/Loki/blob/main/src/loki/preprocess.py"
        # exclude housekeeping genes
        if self.housekeeping_genes is not None:
            filtered_spatial_anndata = self.spatial_anndata[:, ~self.spatial_anndata.var.index.isin(self.housekeeping_genes)]
        else:
            filtered_spatial_anndata = self.spatial_anndata

        mean_exp = filtered_spatial_anndata.X[mask].toarray().mean(axis=0)
        K = self.number_of_genes
        spatial_text = " ".join(filtered_spatial_anndata.var.index[np.argsort(mean_exp)[-K:][::-1]]) # TODO: exclude empty strings

        # 2. cell type composition
        cell_type_dict = collections.Counter(dict.fromkeys(self.spatial_anndata.obs["Annotation"].unique(),0))
        cell_type_counter = collections.Counter(self.spatial_anndata.obs["Annotation"][mask].dropna())
        cell_type_dict.update(cell_type_counter)
        cell_type_composition = torch.tensor([cell_type_dict[k] for k in sorted(cell_type_dict)])
        cell_type_composition = cell_type_composition.float() / cell_type_composition.sum() if cell_type_composition.sum() > 0 else cell_type_composition

        # 3. cropped image
        fov_image = self.image_array[
                    :,
                    y_start : y_start + self.fov_size, 
                    x_start : x_start + self.fov_size, 
                    ]
        # → Resize image for ViT input
        fov_image = torch.tensor(fov_image)
        transform = Resize((224, 224))
        image = transform(fov_image)

        return {
            "image": image,
            "spatial": spatial_embedding,
            "spatial_graph": spatial_graph,
            "spatial_text": spatial_text,
            "cell_type_composition": cell_type_composition,
            "image_file": self.image_file,
            "indices_file": self.indices_file,
            "spatial_file": self.spatial_file,
            "fov_size": self.fov_size,
            "index": idx,
        }

    def __len__(self) -> int:
        return self.index_array.shape[0]

def multimodal_collate(batch: collections.abc.MutableMapping):
    elem = batch[0]
    clone = copy.copy(elem)
    for key in elem:
        if type(elem[key]) in collate_fn_map.keys():
            clone.update({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)})
        elif isinstance(elem[key], dgl.DGLGraph):
            clone.update({key: dgl.batch([d[key] for d in batch])})
    return clone