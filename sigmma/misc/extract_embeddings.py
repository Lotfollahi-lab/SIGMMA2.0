import numpy as np
import pandas as pd
import torch
import dgl
from pathlib import Path
import scanpy

def extract_graphs(dataloader,
                   model,
                   config,
                   device,
                   group_to_id=False,
                   multi_scale=False,
                   annotation_lookup=None):
    """[Inference] Extracts graphs from the dataloader and returns them as a list of dgl.DGLGraph objects.
    ------------------------------------------------------------------------------
    Args:
    - dataloader: A PyTorch DataLoader that yields batches of data, where each batch contains an "image" tensor and a "spatial_graph" DGLGraph.
    - model: SIGMMA model
    - config: Configuration object containing model parameters, including the number of resolutions.
    - device: The device (CPU or GPU) on which the model and data are located
    - multi_scale: If True, extracts graphs at multiple resolutions (spatial scale); otherwise, only extracts at the largest scale.
    ------------------------------------------------------------------------------
    Returns:
    - graph_across_resolution: A dictionary where each key is a resolution (e.g., "scale_0", "scale_1", etc.) and each value is a list of DGLGraph objects corresponding to that resolution.
    ** dgl.DGLGraph object contains cell_id, cell_coordinates, and learned_node_feat (cell embeddings)
    """
    section_IDs = []
    mask_tiles_with_zero_cells = []

    graph_across_resolution = {}
    for k in range(config.model.num_resolutions):
        graph_across_resolution[f"scale_{k}"] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0:
                continue
            # ── 1. Filter annotation ────────────────────────────────────
            if group_to_id:
                if annotation_lookup is not None:
                    valid_mask = [
                        sid in annotation_lookup for sid in batch["metadata"]["section_id"]
                    ]
                else:
                    valid_mask = [ann is not None for ann in batch["metadata"]["annotation"]]
                if not any(valid_mask):
                    continue
                batch = add_annotation_to_batch(batch, group_to_id, annotation_lookup=annotation_lookup)
                if batch["spatial_graph"] is None:
                    continue
            else:
                batch_size = len(batch["metadata"]["section_id"])
                valid_mask = [True] * batch_size

            if not any(valid_mask):
                continue

            # ── 2. Filter tensor/list ───────────────────────────────────
            image = batch["image"][valid_mask].to(device)        # (B', C, H, W)
            spatial = batch["spatial"][valid_mask].to(device)

            graphs = dgl.unbatch(batch["spatial_graph"])
            filtered_graphs_raw = [g for g, v in zip(graphs, valid_mask) if v]

            section_ids_batch = [
                s for s, v in zip(batch["metadata"]["section_id"], valid_mask) if v
            ]

            # ── 3. Remove graphs with zero nodes ──────────────────────────────
            filtered_graphs, filtered_images, filtered_section_ids = [], [], []
            for j, g_single in enumerate(filtered_graphs_raw):
                if g_single.num_nodes() == 0:
                    mask_tiles_with_zero_cells.append((i, j))
                    continue
                filtered_graphs.append(g_single)
                filtered_images.append(image[j].unsqueeze(0))
                filtered_section_ids.append(section_ids_batch[j])

            if len(filtered_graphs) == 0:
                continue

            section_IDs.extend(filtered_section_ids)

            # ── 4. Move to device ────────────────────────────
            g = dgl.batch(filtered_graphs).to(device)
            h = g.ndata["feat"]
            image_tensor = torch.cat(filtered_images, dim=0)   # (B'', C, H, W)

            # ── 5. Multi-scale ────────────────────────────────────
            tmp = []
            with torch.amp.autocast(device.type, enabled=True):
                for k in range(config.model.num_resolutions):
                    g, h = model.spatial_encoder[k * 2](g, h, layer_id=k)
                    h_pooled = model.spatial_encoder[k * 2 + 1](g, h)
                    h_pooled_norm = torch.nn.functional.normalize(h_pooled, dim=-1)
                    tmp.append(g.clone())

            for k in range(config.model.num_resolutions):
                graph_across_resolution[f"scale_{k}"].append(tmp[k])

    if multi_scale:
        return graph_across_resolution, section_IDs
    else:
        return graph_across_resolution[f"scale_{config.model.num_resolutions - 1}"], section_IDs


def extract_cell_embeddings_and_coordinates(graphs_li,
                                            feat_key="learned_node_feat",
                                            coord_key="cell_coordinates",
                                            cell_id_key="cell_id"):
    if len(graphs_li) == 0:
        return np.array([]), np.array([]), np.array([])
    cell_ids = torch.cat([g.ndata[cell_id_key] for g in graphs_li], dim=0).cpu().numpy()
    cell_coordinates = torch.cat([g.ndata[coord_key] for g in graphs_li], dim=0).cpu().numpy()
    cell_embeddings = torch.cat([g.ndata[feat_key] for g in graphs_li], dim=0).cpu().numpy()
    return cell_ids, cell_embeddings, cell_coordinates

def add_annotation_to_batch(batch, group_to_id, annotation_lookup=None):
    """
    annotation_lookup : Optional[Dict[str, np.ndarray]]
        section_id -> positionally-indexed label array (length = num cells in the
        section's spatial anndata). Values are either group label strings or NaN.
        When provided, bypasses the on-disk CSV/h5ad join and reads from memory.
    """
    graphs = dgl.unbatch(batch['spatial_graph'])
    annotation_files = batch['metadata']['annotation']
    section_ids = batch['metadata']['section_id']

    updated_graphs = []

    for graph, section_id, annotation_file in zip(graphs, section_ids, annotation_files):
        if annotation_lookup is not None:
            section_labels = annotation_lookup.get(section_id)
            if section_labels is None:
                graph.ndata["group"] = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
                updated_graphs.append(graph)
                continue
            cell_ids = graph.ndata['cell_id'].long().cpu().numpy()
            groups = section_labels[cell_ids]
        else:
            if not annotation_file or str(annotation_file).strip().lower() == 'nan':
                graph.ndata["group"] = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
                updated_graphs.append(graph)
                continue

            anndata_file = Path(annotation_file).parent / "spatial_data_processed.h5"

            annotation = pd.read_csv(annotation_file, index_col=0)
            anndata = scanpy.read_h5ad(anndata_file)
            anndata.obs = anndata.obs.join(annotation)

            cell_ids = graph.ndata['cell_id'].long().cpu().numpy()
            groups = anndata.obs.iloc[cell_ids]["group"]

        valid_mask = torch.tensor(
            [isinstance(g, str) or not np.isnan(g) for g in groups],
            dtype=torch.bool
        )

        if valid_mask.sum() == 0:
            continue

        valid_node_ids = torch.where(valid_mask)[0]
        graph = dgl.node_subgraph(graph, valid_node_ids)

        if "_ID" in graph.ndata:
            del graph.ndata["_ID"]
        if "_ID" in graph.edata:
            del graph.edata["_ID"]

        valid_groups = [g for g, v in zip(groups, valid_mask.tolist()) if v]
        group_ids = torch.tensor(
            [group_to_id[g] for g in valid_groups],
            dtype=torch.long
        )

        graph.ndata["group"] = group_ids
        updated_graphs.append(graph)

    batch["spatial_graph"] = dgl.batch(updated_graphs) if len(updated_graphs) > 0 else None

    return batch