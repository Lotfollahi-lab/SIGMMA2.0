from collections.abc import Sequence
from pathlib import Path
from tempfile import template
from typing import Callable, Optional, Union
import random

import numpy as np
import pandas as pd
from collections.abc import Mapping, Sequence
from itertools import product
import scanpy
import zarr
import torch
from torchvision.transforms import ToTensor, Resize, Compose
import dgl
from torch.utils.data._utils.collate import collate, collate_tensor_fn, collate_float_fn, collate_int_fn, collate_str_fn
import torchvision.transforms.functional as TF
from PIL import Image

collate_fn_map: dict[Union[type, tuple[type, ...]], Callable] = {
    torch.Tensor: collate_tensor_fn
}
collate_fn_map[float] = collate_float_fn
collate_fn_map[int] = collate_int_fn
collate_fn_map[str] = collate_str_fn
collate_fn_map[bytes] = collate_str_fn


class MultiSectionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 manifest_path: Path | str,
                 fov_size: int = 256,
                 number_of_genes: int = 50,
                 num_resolutions: int = 3,
                 num_cell_types: Optional[int] = None,
                 housekeeping_genes: Optional[Sequence[str]] = None,
                 hierarchical_graph: bool = True,
                 tissue_filtering_threshold: float = 20000,
                 pretrained_cell_embedding: bool = True,
                 type_pretrained_cell_embedding: str = 'stemo',
                 is_train: bool = False,
                 model_name: Optional[str] = None):
        manifest =  pd.read_csv(manifest_path)
        slides = manifest['sample_name']
        li_datsets = []
        for slide in slides:
            print(f"Loading slide {slide}...")
            image_file = manifest.loc[manifest['sample_name'] == slide, 'hematoxylin_eosin_image'].values[0]
            spatial_file = manifest.loc[manifest['sample_name'] == slide, 'spatial_preprocessed'].values[0]
            index_file = manifest.loc[manifest['sample_name'] == slide, 'index'].values[0]
            dataset = SpatialPatchDataset(
                image_file=Path(image_file),
                spatial_file=Path(spatial_file),
                indices_file=Path(index_file),
                fov_size=fov_size,
                number_of_genes=number_of_genes,
                num_resolutions=num_resolutions,
                num_cell_types=num_cell_types,
                housekeeping_genes=housekeeping_genes,
                hierarchical_graph=hierarchical_graph,
                tissue_filtering_threshold=tissue_filtering_threshold,
                pretrained_cell_embedding=pretrained_cell_embedding,
                type_pretrained_cell_embedding=type_pretrained_cell_embedding,
                is_train=is_train,
                model_name=model_name,
            )
            if len(dataset) > 0:
                li_datsets.append(dataset)
            else:
                print(f"Warning: Skipping slide {slide} as it has no valid patches after filtering.")
        
        if len(li_datsets) == 0:
            raise ValueError("No valid datasets found after filtering. All slides were empty.")
        
        self.modality_datasets = torch.utils.data.ConcatDataset(li_datsets)
        # Store the first dataset to access gene count
        self.first_dataset = li_datsets[0]

    def __getitem__(self, idx: int):
        return self.modality_datasets[idx]

    def __len__(self):
        return len(self.modality_datasets)
    
    def get_num_genes(self):
        """Returns the number of genes or embedding dimension depending on whether pretrained embeddings are used."""
        if self.first_dataset.pretrained_cell_embedding:
            # Return the embedding dimension instead of gene count
            if self.first_dataset.type_pretrained_cell_embedding == "stemo":
                return self.first_dataset.spatial_anndata.obsm["stemo_latent"].shape[1]
            elif self.first_dataset.type_pretrained_cell_embedding == "novae":
                return self.first_dataset.spatial_anndata.obsm["novae_latent"].shape[1]
            elif self.first_dataset.type_pretrained_cell_embedding == "X_pca_harmony":
                return self.first_dataset.spatial_anndata.obsm["X_pca_harmony"].shape[1]
        return self.first_dataset.spatial_anndata.n_vars


class SpatialPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_file: Path | str,
        spatial_file: Path | str,
        indices_file: Path | str,
        fov_size: int = 256,
        number_of_genes: int = 50,
        num_resolutions: int = 3,
        num_cell_types: Optional[int] = None,
        housekeeping_genes: Optional[Sequence[str]] = None,
        hierarchical_graph: bool = True,
        tissue_filtering_threshold: float = 20000,
        pretrained_cell_embedding: bool = True,
        type_pretrained_cell_embedding: str = 'stemo',
        is_train: bool = False,
        model_name: Optional[str] = None,
    ):
        self.image_file = str(image_file)
        self.indices_file = str(indices_file)
        self.spatial_file = str(spatial_file)

        self.image_array = zarr.open(image_file, mode="r")
        self.index_array = zarr.open(indices_file, mode="r")
        if tissue_filtering_threshold is not None:
            mask = self.index_array[:, -1] >= tissue_filtering_threshold
            valid_indices = np.where(mask)[0]
            self.index_array = self.index_array[valid_indices]
        self.spatial_anndata = scanpy.read_h5ad(
            spatial_file
        )
        self.section_id = Path(spatial_file).parts[-2]
        self.tissue_type = Path(spatial_file).parts[-3]

        assert 'stemo_latent' in self.spatial_anndata.obsm, "'stemo_latent' embedding is required in the spatial anndata obsm, please run the preprocessing script first."
        assert 'spatial_connectivities' in self.spatial_anndata.obsp, "'spatial_connectivities' is required in the spatial anndata obsp, please run the preprocessing script first."

        self.fov_size = fov_size
        self.number_of_genes = number_of_genes
        self.num_resolutions = num_resolutions
        self.num_cell_types = num_cell_types
        self.housekeeping_genes = housekeeping_genes
        self.hierarchical_graph = hierarchical_graph
        self.pretrained_cell_embedding = pretrained_cell_embedding
        self.is_train = is_train
        self.model_name = model_name
        if self.pretrained_cell_embedding == True:
            assert type_pretrained_cell_embedding in ['stemo', 'novae', 'X_pca_harmony'], "type_pretrained_cell_embedding must be either 'stemo' or 'novae'"
            self.type_pretrained_cell_embedding = type_pretrained_cell_embedding


    def __getitem__(self, idx):
        # 0. image slice
        slice_args = self.index_array[idx]
        y_start, x_start, _ = np.uint32(slice_args)

        cell_ids = self.spatial_anndata.obs.index.to_numpy()
        cell_ids_to_num = [i for i, cell_id in enumerate(cell_ids)]
        self.spatial_anndata.obs['cell_id_to_num'] = cell_ids_to_num

        # 1-1. spatial embedding (cells inside this FOV)
        points = self.spatial_anndata.obsm["spatial"]

        mask = (
            (x_start <= points[:, 0])
            & (points[:, 0] < x_start + self.fov_size)
            & (y_start <= points[:, 1])
            & (points[:, 1] < y_start + self.fov_size)
        )
        
        # For BLEEP model with PCA embeddings, check for minimum data quality
        if self.model_name == 'BLEEP' and self.pretrained_cell_embedding:
            if self.type_pretrained_cell_embedding == "X_pca_harmony":
                # Check if we have enough non-zero genes in this FOV
                if 'num_non_zero_genes' in self.spatial_anndata.obs:
                    mean_nz_genes = self.spatial_anndata.obs['num_non_zero_genes'][mask].mean()
                    if mean_nz_genes < 30:  # Require at least 30 non-zero genes on average
                        return None

        if self.pretrained_cell_embedding == True:
            if self.type_pretrained_cell_embedding == "stemo":
                cell_embeddings = torch.Tensor(self.spatial_anndata.obsm["stemo_latent"][mask])
            elif self.type_pretrained_cell_embedding == "novae":
                cell_embeddings = torch.Tensor(self.spatial_anndata.obsm["novae_latent"][mask])
            elif self.type_pretrained_cell_embedding == "X_pca_harmony":
                if 'num_non_zero_genes' in self.spatial_anndata.obs:
                    mean_nz_genes = self.spatial_anndata.obs['num_non_zero_genes'][mask].mean()
                    if mean_nz_genes < 30:  # Require at least 30 non-zero genes on average
                        return None
                cell_embeddings = torch.Tensor(self.spatial_anndata.obsm["X_pca_harmony"][mask])
        else:
            cell_embeddings = torch.Tensor(self.spatial_anndata.X[mask].toarray())
        
        spatial_embedding = cell_embeddings.mean(dim=0)
        
        if torch.isnan(spatial_embedding).any():
            spatial_embedding = spatial_embedding.nan_to_num(0.0)
            
        # 1-2. spatial graph
        sparse_matrix = self.spatial_anndata[mask,:].obsp["spatial_connectivities"]
        spatial_graph = dgl.from_scipy(sparse_matrix)

        spatial_graph.ndata['cell_id'] = torch.Tensor(self.spatial_anndata.obs['cell_id_to_num'][mask].to_numpy().astype("int64"))
        spatial_graph.ndata['feat'] = cell_embeddings
        # print('mean, std of cell embeddings', cell_embeddings.mean(), cell_embeddings.std())
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

                if self.pretrained_cell_embedding == True:
                    node_features_grid = torch.Tensor(self.spatial_anndata.obsm["stemo_latent"][mask_grid])
                else:
                    node_features_grid = torch.Tensor(self.spatial_anndata[mask_grid,:].X.toarray())
                subgraph_grid.ndata['feat'] = node_features_grid
                subgraph_grid.ndata['x_pos'] = torch.full((subgraph_grid.num_nodes(),), x_start_grid // grid_size)
                subgraph_grid.ndata['y_pos'] = torch.full((subgraph_grid.num_nodes(),), y_start_grid // grid_size)
                subgraph_grid.ndata['cell_coordinates']=torch.Tensor(self.spatial_anndata[mask_grid,:].obsm['spatial'])
                subgraphs.append(subgraph_grid)
            
            disjoint_spatial_graph = dgl.batch(subgraphs)
            disjoint_spatial_graph.num_subgraphs = disjoint_spatial_graph.batch_size

            sub_edges = set((int(u), int(v)) for u, v in zip(*disjoint_spatial_graph.edges()))

            edge_mask = torch.zeros(spatial_graph.num_edges(), dtype=torch.int8)
            spatial_edge_map = { (int(u), int(v)): eid for eid, (u, v) in enumerate(zip(*spatial_graph.edges())) }

            for edge in sub_edges:
                if edge in spatial_edge_map:
                    edge_mask[spatial_edge_map[edge]] = -1

            spatial_graph.ndata['x_pos'] = disjoint_spatial_graph.ndata['x_pos']
            spatial_graph.ndata['y_pos'] = disjoint_spatial_graph.ndata['y_pos']
            spatial_graph.ndata['cell_coordinates'] = disjoint_spatial_graph.ndata['cell_coordinates']

            spatial_graph.edata['mask'] = edge_mask
            spatial_graph.edata['weight'] = torch.where(edge_mask == -1, torch.tensor(1.0), torch.tensor(0.0))

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
        try:
            cell_type_ids = self.spatial_anndata.obs["Annotation"][mask]
            cell_type_ids_tensor = torch.tensor(cell_type_ids.tolist(), dtype=torch.int8)
            if len(cell_type_ids_tensor) != 0:
                cell_type_composition = torch.bincount(cell_type_ids_tensor, minlength=self.num_cell_types).long()
            else:
                cell_type_composition = torch.zeros(self.num_cell_types)
            cell_type_composition = cell_type_composition.float() / cell_type_composition.sum() if cell_type_composition.sum() > 0 else cell_type_composition
        except Exception as e:
            pass
            # print(f"[Warning]: Cell type composition skipped {e}")

        # 3. cropped image
        fov_image = self.image_array[
                    :,
                    y_start : y_start + self.fov_size,
                    x_start : x_start + self.fov_size,
                    ]
        
        # Apply BLEEP-specific transform if training with BLEEP model
        if self.model_name == 'BLEEP':
            # For BLEEP, apply augmentations and ImageNet normalization
            # fov_image is (C, H, W) with values in [0, 255]
            image = self.transform(fov_image)
        else:
            # For other models, just normalize to [0, 1] 
            image = torch.tensor(fov_image) / 255.0

        # Label
        label = {}
        topK_exp = mean_exp[np.argsort(mean_exp)[-K:]]

        label['topK_exp'] = torch.tensor(topK_exp, dtype=torch.float32) # TODO: add gene names
        # label['raw_exp'] = torch.tensor(mean_exp, dtype=torch.float32) # TODO: add gene names
        try:
            label['cell_type_composition'] = cell_type_composition
        except Exception as e:
            pass
            # print(f"[Warning]: Cell type composition information is not available in the spatial anndata. Please make sure cell type information is included in the spatial anndata. {e}")

        return {
            "image": image,
            "spatial": spatial_embedding,
            "spatial_graph": spatial_graph,
            "spatial_text": spatial_text,
            "label": label,
            "metadata": {"section_id": self.section_id, 
                         "tissue_type": self.tissue_type,
                         }
        }

    def __len__(self) -> int:
        return len(self.index_array)
    
    def transform(self, image):
        """
        from https://github.com/bowang-lab/BLEEP/blob/main/dataset.py
        Apply data augmentations and normalization to the input image.
        When training BLEEP
        
        Args:
            image: numpy array of shape (C, H, W) with values in [0, 255]
        
        Returns:
            torch.Tensor: Transformed image with ImageNet normalization
        """
        # Convert from (C, H, W) to (H, W, C) for PIL
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        image = Image.fromarray(image) # size (256, 256)
        
        if self.is_train:    
            # Random flipping and rotations
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            
            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)
        
        # Convert to tensor (this also converts to [0, 1] range)
        # output shape (C, H, W)
        image = TF.to_tensor(image)
                
        # Normalize using ImageNet mean and std
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
        return image

def _empty_batch_like(template: Mapping):
    """Return an empty-but-well-formed batch matching `template`'s structure."""
    out = {}
    for k, v in template.items():
        if isinstance(v, Mapping):
            out[k] = _empty_batch_like(v)
        elif isinstance(v, dgl.DGLGraph):
            # An empty batched graph: batch() on [] raises, so make a 0-node graph
            out[k] = dgl.graph(([], []))
        else:
            # Fallback: empty list; your downstream code may stack later
            out[k] = []
    return out


def multimodal_collate_dict(batch_dicts):
    # Handle None subdicts
    batch_dicts = [d for d in batch_dicts if d is not None and isinstance(d, Mapping)]
    if not batch_dicts:
        return {}

    # Drop dicts that contain None/empty graphs
    filtered = [d for d in batch_dicts if not _is_empty_sample(d)]
    if not filtered:
        return _empty_batch_like(batch_dicts[0])

    elem = filtered[0]
    out = {}
    for subkey in elem:
        values = [d.get(subkey) for d in filtered]
        if type(elem[subkey]) in collate_fn_map:
            out[subkey] = collate(values, collate_fn_map=collate_fn_map)
        elif isinstance(elem[subkey], Mapping):
            out[subkey] = multimodal_collate_dict(values)
        elif isinstance(elem[subkey], dgl.DGLGraph):
            graphs = [v for v in values if isinstance(v, dgl.DGLGraph) and v.num_nodes() > 0]
            out[subkey] = dgl.batch(graphs) if graphs else dgl.graph(([], []))
        else:
            out[subkey] = values
    return out

def _iter_dgl_graphs(x):
    if isinstance(x, dgl.DGLGraph):
        yield x
    elif isinstance(x, Mapping):
        for v in x.values():
            yield from _iter_dgl_graphs(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _iter_dgl_graphs(v)

def _is_empty_sample(sample_dict) -> bool:
    # Treat None or zero-node graphs as “empty”
    for g in _iter_dgl_graphs(sample_dict):
        if g is None or g.num_nodes() == 0:
            return True
    return False

def multimodal_collate(batch):
    # 0) Strip None samples early (your Dataset might yield None to “skip”)
    batch = [b for b in batch if b is not None and isinstance(b, Mapping)]
    if not batch:
        # No usable samples: return a safe empty structure
        # If you can, keep a global/template example to mirror shapes more precisely.
        return {}
 
    # 1) Drop samples that contain None/empty DGL graphs anywhere inside
    filtered = [b for b in batch if not _is_empty_sample(b)]
    if not filtered:
        # Build an empty structure keyed like the first non-None original sample
        return _empty_batch_like(batch[0])

    elem = filtered[0]
    out = {}

    for key in elem:
        values = [d.get(key) for d in filtered]
        if type(elem[key]) in collate_fn_map:
            out[key] = collate(values, collate_fn_map=collate_fn_map)

        elif isinstance(elem[key], dgl.DGLGraph):
            graphs = [v for v in values if isinstance(v, dgl.DGLGraph) and v.num_nodes() > 0]
            if not graphs:
                out[key] = dgl.graph(([], []))  # safe empty graph
            else:
                out[key] = dgl.batch(graphs)

        elif isinstance(elem[key], Mapping):
            out[key] = multimodal_collate_dict(values)

        else:
            out[key] = values

    return out