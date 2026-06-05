from __future__ import annotations
import bisect
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import template
from typing import Callable, Optional, OrderedDict, Union
import random

import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from itertools import product
import scanpy
import scipy.sparse as sp

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


def _load_panel_aware_svgs(
    svg_panel_dir: Path | str,
    panel_index_json: Path | str,
    k_max: int = 100,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    """Load panel-aware SVG target lists produced by
    ``scripts/compute_panel_aware_svgs.py``.

    Returns
    -------
    section_to_panel : sid -> panel_id (str)
    panel_to_genes   : panel_id -> ordered list of SVG gene names
                       (length K_panel, possibly <= k_max — short panels are
                       padded by callers to k_max with valid_mask=False)
    panel_to_int     : panel_id -> deterministic int (sorted by panel_id)
    """
    import json as _json
    svg_dir = Path(svg_panel_dir)
    section_to_panel: dict[str, str] = _json.loads(Path(panel_index_json).read_text())
    panels_in_use = sorted(set(section_to_panel.values()))
    panel_to_genes: dict[str, list[str]] = {}
    panel_to_int: dict[str, int] = {pid: i for i, pid in enumerate(panels_in_use)}
    for pid in panels_in_use:
        npy_files = sorted((svg_dir / pid).glob("svg_top*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No svg_top*.npy found for panel {pid} under {svg_dir}"
            )
        # Pick the largest-K npy that fits within k_max
        chosen = None
        for p in reversed(npy_files):  # reversed = largest K first by lexsort
            arr = np.load(p, allow_pickle=True)
            if len(arr) <= k_max:
                chosen = arr
                break
        if chosen is None:
            # fallback: truncate the largest to k_max
            arr = np.load(npy_files[-1], allow_pickle=True)
            chosen = arr[:k_max]
        panel_to_genes[pid] = [str(g) for g in chosen]
    return section_to_panel, panel_to_genes, panel_to_int


def _load_raw_gex_panel(npy_path: Path | str) -> list[str]:
    """Load a precomputed gene-name panel saved as an object-dtype npy.

    Created by ``scripts/compute_in_house_5k_hvg.py`` (or any analogous script
    that runs ``compute_top_k_hvg_by_dispersion`` once offline).
    """
    arr = np.load(str(npy_path), allow_pickle=True)
    return [str(g) for g in arr]


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
                 type_pretrained_cell_embedding: str = 'nemo',
                 is_train: bool = False,
                 model_name: Optional[str] = None,
                 hier_cache_dir: Optional[Path | str] = None,
                 raw_gex_panel_npy: Optional[Path | str] = None,
                 svg_panel_dir: Optional[Path | str] = None,
                 panel_index_json: Optional[Path | str] = None,
                 k_max: int = 100):
        manifest = pd.read_csv(manifest_path)
        # Dedupe by sample_name — some slides (e.g. Sml4-PNC) have multiple
        # rows pointing at different sequencing runs. ``keep="last"`` matches
        # the pre-computed hier_cache (built from the second/RUN2 entry).
        manifest = manifest.drop_duplicates(subset="sample_name", keep="last").reset_index(drop=True)
        slides = manifest['sample_name']
        li_datsets = []
        for slide in slides:
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"Loading slide {slide}...")
            image_file = manifest.loc[manifest['sample_name'] == slide, 'hematoxylin_eosin_image'].values[0]
            spatial_file = manifest.loc[manifest['sample_name'] == slide, 'spatial_preprocessed'].values[0]
            spatial_annotation_file = manifest.loc[manifest['sample_name'] == slide, 'annotation'].values[0]
            index_file = manifest.loc[manifest['sample_name'] == slide, 'index'].values[0]
            dataset = SpatialPatchDataset(
                image_file=Path(image_file),
                spatial_file=Path(spatial_file),
                spatial_annotation_file=spatial_annotation_file,
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
                hier_cache_dir=hier_cache_dir,
            )
            if len(dataset) > 0:
                li_datsets.append(dataset)
            else:
                print(f"Warning: Skipping slide {slide} as it has no valid patches after filtering.")
        
        if len(li_datsets) == 0:
            raise ValueError("No valid datasets found after filtering. All slides were empty.")

        # Shared selected-gene panel via gene NAMES (panel-agnostic).
        # ``raw_gex_panel_npy`` points to an object-dtype npy of gene symbols
        # produced offline by scripts/compute_in_house_5k_hvg.py (or analogous).
        # Each section maps these names to LOCAL positions in its own
        # ``_var_index``; sections lacking any panel gene fail explicitly so
        # the training cohort can be re-filtered upstream.
        # Panel-aware SVG target loading (new): per-section panel_id → SVG list
        # OR legacy shared HVG panel via raw_gex_panel_npy. Mutually exclusive.
        if svg_panel_dir is not None and panel_index_json is not None:
            if raw_gex_panel_npy is not None:
                raise ValueError(
                    "Cannot use both raw_gex_panel_npy and svg_panel_dir simultaneously."
                )
            section_to_panel, panel_to_genes, panel_to_int = _load_panel_aware_svgs(
                svg_panel_dir, panel_index_json, k_max=k_max,
            )
            self.svg_panel_dir = Path(svg_panel_dir)
            self.panel_to_int = panel_to_int
            self.selected_gene_names = None   # per-section now, not shared
            self.selected_gene_idx = None
            self.k_max = int(k_max)
            kept = []
            for d in li_datsets:
                pid = section_to_panel.get(d.section_id)
                if pid is None:
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(f"  [panel filter] skipping {d.section_id}: not in panel_index.json")
                    continue
                panel_genes = panel_to_genes[pid]
                name_to_pos = {g: i for i, g in enumerate(d._var_index)}
                # SVG genes may not all be present in this section's var.index
                # (esp. merged-panel intersections). Skip section if 0 valid.
                selected_gene_idx = np.array(
                    [name_to_pos.get(g, -1) for g in panel_genes], dtype=np.int64
                )
                if (selected_gene_idx >= 0).sum() == 0:
                    print(f"  [panel filter] skipping {d.section_id}: 0 of {len(panel_genes)} panel SVGs present")
                    continue
                d.selected_gene_idx = selected_gene_idx
                d.panel_id_str = pid
                d.panel_id_int = panel_to_int[pid]
                d.k_max = k_max
                kept.append(d)
            if not kept:
                raise ValueError(
                    "Panel-aware mode: no sections survived panel_index filter. "
                    f"Check {panel_index_json} covers the manifest."
                )
            li_datsets = kept
        elif raw_gex_panel_npy is not None:
            panel_names = _load_raw_gex_panel(raw_gex_panel_npy)
            self.selected_gene_names = np.array(panel_names)
            for d in li_datsets:
                name_to_pos = {g: i for i, g in enumerate(d._var_index)}
                missing = [g for g in panel_names if g not in name_to_pos]
                if missing:
                    raise ValueError(
                        f"raw_gex_panel_npy {raw_gex_panel_npy} contains genes not present "
                        f"in section {d.section_id}: missing {len(missing)} (first: {missing[:5]}). "
                        f"Filter the training manifest to slides that share the panel, or "
                        f"regenerate the panel against a wider cohort."
                    )
                d.selected_gene_idx = np.array([name_to_pos[g] for g in panel_names], dtype=np.int64)
            self.selected_gene_idx = li_datsets[0].selected_gene_idx
        else:
            # No raw_gex panel configured — leave per-section selected_gene_idx unset so
            # SpatialPatchDataset.__getitem__ falls back to zeros for raw_gex.
            self.selected_gene_names = None
            self.selected_gene_idx = None

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
            if self.first_dataset.type_pretrained_cell_embedding == "nemo":
                return self.first_dataset.spatial_anndata.obsm["nemo_latent"].shape[1]
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
        spatial_annotation_file: Optional[Path | str] = None,
        fov_size: int = 256,
        number_of_genes: int = 50,
        num_resolutions: int = 3,
        num_cell_types: Optional[int] = None,
        housekeeping_genes: Optional[Sequence[str]] = None,
        hierarchical_graph: bool = True,
        tissue_filtering_threshold: float = 20000,
        pretrained_cell_embedding: bool = True,
        type_pretrained_cell_embedding: str = 'nemo',
        is_train: bool = False,
        model_name: Optional[str] = None,
        hier_cache_dir: Optional[Path | str] = None,
    ):
        self.image_file = str(image_file)
        self.indices_file = str(indices_file)
        self.spatial_file = str(spatial_file)
        self.spatial_annotation_file = str(spatial_annotation_file) if spatial_annotation_file is not None else None

        adata = scanpy.read_h5ad(spatial_file)   # full load, no backed="r"

        assert 'nemo_latent' in adata.obsm, \
            "'nemo_latent' embedding is required in the spatial anndata obsm."
        assert 'spatial_connectivities' in adata.obsp, \
            "'spatial_connectivities' is required in the spatial anndata obsp."

        # Spatial coordinates  (N, 2)
        self._coords: np.ndarray = np.array(adata.obsm["spatial"], dtype=np.float32)

        # Cell embeddings
        self._nemo: np.ndarray = np.array(adata.obsm["nemo_latent"], dtype=np.float32)
        if "novae_latent" in adata.obsm:
            self._novae: np.ndarray = np.array(adata.obsm["novae_latent"], dtype=np.float32)
        else:
            self._novae = None
        if "X_pca_harmony" in adata.obsm:
            self._harmony: np.ndarray = np.array(adata.obsm["X_pca_harmony"], dtype=np.float32)
        else:
            self._harmony = None

        # Gene expression matrix  (N, G)  – keep as CSR for slicing
        if sp.issparse(adata.X):
            self._X: sp.csr_matrix = adata.X.tocsr()
        else:
            self._X: np.ndarray = np.array(adata.X, dtype=np.float32)

        # Spatial graph connectivity  (N, N)  CSR
        self._conn: sp.csr_matrix = adata.obsp["spatial_connectivities"].tocsr()

        # Gene names
        self._var_index: np.ndarray = adata.var.index.to_numpy()

        # Per-cell metadata
        self._cell_id_to_num: np.ndarray = np.arange(len(adata.obs), dtype=np.int64)

        # obs columns used downstream
        if 'num_non_zero_genes' in adata.obs.columns:
            self._num_non_zero_genes: Optional[np.ndarray] = adata.obs['num_non_zero_genes'].to_numpy()
        else:
            self._num_non_zero_genes = None

        self.section_id = Path(spatial_file).parts[-2]
        self.tissue_type = Path(spatial_file).parts[-3]

        # h5ad file is no longer needed – release the fd immediately
        del adata

        # zarr arrays (lightweight, keep open)
        self.image_array = zarr.open((str(image_file)), mode="r")
        self.index_array = zarr.open((str(indices_file)), mode="r")

        if tissue_filtering_threshold is not None:
            mask = self.index_array[:, -1] >= tissue_filtering_threshold
            self.valid_indices = np.where(mask)[0]
        else:
            self.valid_indices = None

        self.fov_size = fov_size
        self.number_of_genes = number_of_genes
        self.num_resolutions = num_resolutions
        self.num_cell_types = num_cell_types
        self.housekeeping_genes = housekeeping_genes
        self.hierarchical_graph = hierarchical_graph
        self.pretrained_cell_embedding = pretrained_cell_embedding
        self.is_train = is_train
        self.model_name = model_name

        # Optional pre-computed hierarchical-subgraph cache (one .npz per
        # slide section_id). When loaded, __getitem__ skips the per-tile
        # CSR slicing + dgl.from_scipy loop that dominates the dataloader
        # cost (profile_dataloader.py showed ~52% of per-tile time).
        self._hier_cache: Optional[dict] = None
        if hier_cache_dir is not None:
            cache_path = Path(hier_cache_dir) / f"{self.section_id}.npz"
            if cache_path.exists():
                with np.load(cache_path) as f:
                    cache = {k: f[k] for k in f.files}
                # Sanity check: cache must cover the same valid tiles this
                # dataset will iterate.
                if self.valid_indices is not None and \
                        not np.array_equal(cache['valid_indices'].astype(np.int64),
                                           self.valid_indices.astype(np.int64)):
                    raise ValueError(
                        f"hier_cache valid_indices mismatch for {self.section_id}: "
                        f"cache has {cache['valid_indices'].size} tiles, "
                        f"dataset has {self.valid_indices.size}. "
                        "Re-run precompute_hierarchical_subgraphs.py with the "
                        "same tissue_filtering_threshold."
                    )
                if int(cache['fov_size']) != fov_size or \
                        int(cache['num_resolutions']) != num_resolutions:
                    raise ValueError(
                        f"hier_cache fov/num_res mismatch for {self.section_id}: "
                        f"cache=(fov={int(cache['fov_size'])}, "
                        f"num_res={int(cache['num_resolutions'])}), "
                        f"dataset=(fov={fov_size}, num_res={num_resolutions})."
                    )
                self._hier_cache = cache

        if self.pretrained_cell_embedding:
            assert type_pretrained_cell_embedding in ['nemo', 'novae', 'X_pca_harmony'], \
                "type_pretrained_cell_embedding must be 'nemo', 'novae', or 'X_pca_harmony'"
            self.type_pretrained_cell_embedding = type_pretrained_cell_embedding

    def _get_cell_embeddings(self, mask: np.ndarray) -> np.ndarray:
        """Return (n_cells, D) float32 array for cells selected by boolean mask."""
        if self.type_pretrained_cell_embedding == "nemo":
            return self._nemo[mask]
        elif self.type_pretrained_cell_embedding == "novae":
            assert self._novae is not None, "novae_latent not found in anndata."
            return self._novae[mask]
        elif self.type_pretrained_cell_embedding == "X_pca_harmony":
            assert self._harmony is not None, "X_pca_harmony not found in anndata."
            return self._harmony[mask]
        else:
            raise ValueError(f"Unknown embedding type: {self.type_pretrained_cell_embedding}")

    def _subgraph_from_mask(self, mask: np.ndarray) -> dgl.DGLGraph:
        """Build a DGL graph from the global connectivity matrix sliced by mask."""
        sub_conn = self._conn[mask, :][:, mask]
        return dgl.from_scipy(sub_conn)

    def __getitem__(self, idx):
        # cache_pos is the position in self.valid_indices (== position in
        # the hier_cache arrays); idx is then remapped to the raw zarr index.
        cache_pos = idx
        idx = int(self.valid_indices[idx]) if self.valid_indices is not None else idx

        # 0. tile coordinates from index array
        slice_args = self.index_array[idx]
        y_start, x_start, _ = np.uint32(slice_args)

        # 1-1. cells inside this FOV
        points = self._coords   # (N, 2)  – already in memory, no h5ad access
        mask = (
            (x_start <= points[:, 0]) & (points[:, 0] < x_start + self.fov_size) &
            (y_start <= points[:, 1]) & (points[:, 1] < y_start + self.fov_size)
        )

        n_cells = int(mask.sum())
        if n_cells == 0:
            return None

        if self.pretrained_cell_embedding:
            # Optional quality filter for harmony embeddings
            if self.type_pretrained_cell_embedding == "X_pca_harmony" and self._num_non_zero_genes is not None:
                if self._num_non_zero_genes[mask].mean() < 30:
                    return None
            cell_embeddings = torch.from_numpy(self._get_cell_embeddings(mask))
        else:
            if sp.issparse(self._X):
                cell_embeddings = torch.tensor(self._X[mask].toarray(), dtype=torch.float32)
            else:
                cell_embeddings = torch.tensor(self._X[mask], dtype=torch.float32)

        spatial_embedding = cell_embeddings.mean(dim=0)
        if torch.isnan(spatial_embedding).any():
            spatial_embedding = spatial_embedding.nan_to_num(0.0)

        # 1-2. spatial graph  (full FOV)
        spatial_graph = self._subgraph_from_mask(mask)
        spatial_graph.ndata['cell_id'] = torch.from_numpy(self._cell_id_to_num[mask])
        spatial_graph.ndata['feat']  = cell_embeddings
        spatial_graph.edata['weight']  = torch.ones(spatial_graph.num_edges())
        spatial_graph.edata['mask']  = torch.full((spatial_graph.num_edges(),), -1.0)

        # 1-3. hierarchical graph (micro subgraphs)
        if self.hierarchical_graph:
            grid_size = self.fov_size // 2 ** self.num_resolutions

            if self._hier_cache is not None:
                # Cached path: rebuild the disjoint-batched graph from
                # pre-computed cell ordering + intra-sub-FOV edge lists,
                # avoiding the per-tile CSR slicing + dgl.from_scipy loop.
                cache = self._hier_cache
                c0 = int(cache['cell_offsets'][cache_pos])
                c1 = int(cache['cell_offsets'][cache_pos + 1])
                cells = cache['cell_indices'][c0:c1]
                e0 = int(cache['edge_offsets'][cache_pos])
                e1 = int(cache['edge_offsets'][cache_pos + 1])
                src = cache['edge_src'][e0:e1]
                dst = cache['edge_dst'][e0:e1]
                sf_off = cache['subfov_cell_offsets'][cache_pos]  # (n_subfov + 1,)

                n_total = int(cells.size)
                n_per_dim = self.fov_size // grid_size
                n_subfov = n_per_dim * n_per_dim

                # Reproduce disjoint.ndata['x_pos'], ['y_pos'] from sub-FOV
                # iteration order matching `product(range(0, fov, gs), ...)`
                # (x_g outer, y_g inner ⇒ sf_idx = x_idx * n_per_dim + y_idx).
                x_pos = np.empty(n_total, dtype=np.int64)
                y_pos = np.empty(n_total, dtype=np.int64)
                for sf_idx in range(n_subfov):
                    s_, e_ = int(sf_off[sf_idx]), int(sf_off[sf_idx + 1])
                    if s_ == e_:
                        continue
                    x_pos[s_:e_] = sf_idx // n_per_dim
                    y_pos[s_:e_] = sf_idx % n_per_dim

                disjoint = dgl.graph(
                    (torch.from_numpy(src.astype(np.int64)),
                     torch.from_numpy(dst.astype(np.int64))),
                    num_nodes=n_total,
                )
                disjoint.ndata['feat'] = torch.from_numpy(self._nemo[cells])
                disjoint.ndata['x_pos'] = torch.from_numpy(x_pos)
                disjoint.ndata['y_pos'] = torch.from_numpy(y_pos)
                disjoint.ndata['cell_coordinates'] = torch.from_numpy(self._coords[cells])
                disjoint.num_subgraphs = n_subfov
            else:
                subgraphs = []
                for x_g, y_g in product(range(0, self.fov_size, grid_size), range(0, self.fov_size, grid_size)):
                    mask_grid = (
                        (x_start + x_g  <= points[:, 0]) & (points[:, 0] < x_start + x_g + grid_size) &
                        (y_start + y_g   <= points[:, 1]) & (points[:, 1] < y_start + y_g + grid_size)
                    )

                    sg = self._subgraph_from_mask(mask_grid)   # no h5ad slicing!

                    # Node features – always use nemo for subgraphs (same as original)
                    sg.ndata['feat'] = torch.from_numpy(self._nemo[mask_grid])
                    sg.ndata['x_pos'] = torch.full((sg.num_nodes(),), x_g // grid_size)
                    sg.ndata['y_pos'] = torch.full((sg.num_nodes(),), y_g // grid_size)
                    sg.ndata['cell_coordinates'] = torch.from_numpy(points[mask_grid])
                    subgraphs.append(sg)

                disjoint = dgl.batch(subgraphs)
                disjoint.num_subgraphs = disjoint.batch_size

            # Compute edge mask with torch ops (no Python set/dict loops)
            src_full,  dst_full  = spatial_graph.edges()
            src_sub,   dst_sub   = disjoint.edges()
            N = spatial_graph.num_nodes()

            full_ids = src_full * N + dst_full           # unique edge IDs
            sub_ids  = src_sub  * N + dst_sub

            # isin via torch (available in torch >= 1.10)
            intra = torch.isin(full_ids, sub_ids)
            edge_mask = torch.where(intra, torch.tensor(-1, dtype=torch.int8),
                                           torch.tensor( 0, dtype=torch.int8))

            spatial_graph.ndata['x_pos'] = disjoint.ndata['x_pos']
            spatial_graph.ndata['y_pos'] = disjoint.ndata['y_pos']
            spatial_graph.ndata['cell_coordinates'] = disjoint.ndata['cell_coordinates']
            spatial_graph.edata['mask'] = edge_mask
            spatial_graph.edata['weight'] = torch.where(
                edge_mask == -1, torch.tensor(1.0), torch.tensor(0.0)
            )

        # 1-4. spatial text (top-K genes by mean expression in FOV)
        if sp.issparse(self._X):
            X_fov = self._X[mask].toarray()
        else:
            X_fov = self._X[mask]

        # 1-4b. FOV-level raw GEX over the panel. Two modes:
        # (a) legacy: shared raw_gex_panel_npy → selected_gene_idx is length-K shared,
        #     no padding/mask needed (all entries valid).
        # (b) panel-aware: per-section selected_gene_idx of length K_panel (may contain
        #     -1 for SVGs absent from this section's var.index). Pad to
        #     `k_max` (e.g. 100) with zeros; raw_gex_valid_mask marks
        #     positions where the gene is actually measured by this section.
        # When standalone (no MultiSectionDataset wrap), return placeholders
        # so probe/inference paths don't crash.
        selected_gene_idx = getattr(self, "selected_gene_idx", None)
        k_max = int(getattr(self, "k_max", 0))
        panel_id_int = int(getattr(self, "panel_id_int", -1))
        if selected_gene_idx is not None and k_max > 0:
            # Panel-aware mode
            raw_gex_fov = X_fov.sum(axis=0).ravel()
            G = raw_gex_fov.shape[0]
            valid_local = selected_gene_idx >= 0
            raw_gex_padded = np.zeros(k_max, dtype=np.float32)
            valid_mask = np.zeros(k_max, dtype=bool)
            K_panel = int(selected_gene_idx.shape[0])
            if K_panel > k_max:
                # truncate (shouldn't happen if compute_panel_aware_svgs respects k=100)
                selected_gene_use = selected_gene_idx[:k_max]
                valid_use = valid_local[:k_max]
            else:
                selected_gene_use = selected_gene_idx
                valid_use = valid_local
            valid_positions_in_kmax = np.arange(min(K_panel, k_max))
            local_idx = selected_gene_use.copy()
            local_idx[~valid_use] = 0  # safe gather; masked later
            raw_gex_padded[valid_positions_in_kmax] = raw_gex_fov[local_idx].astype(np.float32)
            # Zero out padding positions
            raw_gex_padded[valid_positions_in_kmax[~valid_use]] = 0.0
            valid_mask[valid_positions_in_kmax] = valid_use
            raw_gex_selected = raw_gex_padded
            raw_gex_valid_mask = valid_mask
        elif selected_gene_idx is not None:
            # Legacy shared-panel mode
            raw_gex_fov = X_fov.sum(axis=0).ravel()
            raw_gex_selected = raw_gex_fov[selected_gene_idx].astype(np.float32)
            raw_gex_valid_mask = np.ones(raw_gex_selected.shape[0], dtype=bool)
        else:
            raw_gex_selected = np.zeros(1, dtype=np.float32)
            raw_gex_valid_mask = np.zeros(1, dtype=bool)

        var_index = self._var_index
        if self.housekeeping_genes is not None:
            hk_set = set(self.housekeeping_genes)
            keep = np.array([g not in hk_set for g in var_index])
            X_fov = X_fov[:, keep]
            var_index = var_index[keep]

        mean_exp = X_fov.mean(axis=0).ravel()   # works for both ndarray and matrix
        K = self.number_of_genes
        top_idx = np.argsort(mean_exp)[-K:][::-1]
        spatial_text = " ".join(var_index[top_idx])

        topK_exp = mean_exp[np.argsort(mean_exp)[-K:]]

        # 2. cropped image
        fov_image = self.image_array[:, y_start: y_start + self.fov_size,
                                        x_start: x_start + self.fov_size]
        image = torch.tensor(fov_image) / 255.0

        return {
            "image": image,
            "spatial": spatial_embedding,
            "spatial_graph": spatial_graph,
            "spatial_text": spatial_text,
            "label": topK_exp,
            "raw_gex": torch.from_numpy(raw_gex_selected),
            "raw_gex_valid_mask": torch.from_numpy(raw_gex_valid_mask),
            "panel_id": panel_id_int,
            "metadata": {"section_id": self.section_id,
                         "tissue_type": self.tissue_type,
                         "annotation": self.spatial_annotation_file,
                         "y_start": int(y_start),
                         "x_start": int(x_start)}
        }

    def __len__(self) -> int:
        return len(self.valid_indices) if self.valid_indices is not None else self.index_array.shape[0]
    
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

_BATCH_TEMPLATE: Mapping = {
    "image": torch.empty(0),
    "spatial": torch.empty(0),
    "spatial_graph": dgl.graph(([], [])),
    "spatial_text": "",
    "label": torch.empty(0),
    "raw_gex": torch.empty(0),
    "raw_gex_valid_mask": torch.empty(0, dtype=torch.bool),
    "panel_id": -1,
    "metadata": {"section_id": "", "tissue_type": "", "annotation": ""},
}

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
        return _empty_batch_like(_BATCH_TEMPLATE.get("metadata", {}))

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
        # No usable samples: return a skeleton with the same key structure as a
        # real batch, so all downstream consumers can use a single empty-batch
        # guard instead of branching on dict-shape.
        return _empty_batch_like(_BATCH_TEMPLATE)
 
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
