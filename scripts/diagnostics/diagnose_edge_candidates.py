"""Count newly-added candidate edges per StructureLearning layer at inference.

Picks one slide from a manifest, forwards a small number of FOVs through
the trained SIGMMA spatial encoder, and reports per-layer FOV-level
mean +/- std for:
  - original spatial-graph edges (mask == -1, non-self)
  - candidate edges proposed by StructureLearning (cat-3 in _rebuild_batched_graph)
  - candidate edges that pass the RelaxedBernoulli + threshold gate (mask == layer_id + 1)

The "newly added" count is the third metric: candidates that were accepted
and become real edges feeding the next-scale GRAPHLayer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, Subset

from sigmma.datasets.dataset import SpatialPatchDataset, multimodal_collate_dict
from sigmma.utils import load_model
import sigmma.utils.train_utils as _train_utils
from sigmma.model import UNI as _UNI


def fov_level_counts_from_mask(g: dgl.DGLGraph, target_mask_vals: tuple[int, ...]):
    """For each sub-graph in batched g, count edges with mask in target_mask_vals
    AND non-self. Returns a (B, len(target_mask_vals)) numpy array.
    """
    src, dst = g.edges()
    is_self = (src == dst)
    mask = g.edata["mask"]
    # Per-edge graph id (edges live within one sub-graph; use src node's gid)
    batch_num_nodes = g.batch_num_nodes()
    device = mask.device
    B = batch_num_nodes.numel()
    gid_per_node = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=batch_num_nodes.dtype),
        batch_num_nodes,
    )
    edge_gid = gid_per_node[src]

    out = np.zeros((B, len(target_mask_vals)), dtype=np.int64)
    for j, mv in enumerate(target_mask_vals):
        keep = (mask == mv) & (~is_self)
        # bincount over edge_gid, weighted by keep
        counts = torch.bincount(edge_gid[keep], minlength=B).cpu().numpy()
        out[:, j] = counts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="ckpt yaml (sigmma model config)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pth path")
    ap.add_argument("--manifest", required=True, help="manifest csv path")
    ap.add_argument("--slide_name", default=None, help="optional: slide sample_name to pick; default = first row")
    ap.add_argument("--n_fovs", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fov_size", type=int, default=256)
    ap.add_argument("--hier_cache_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ConfigDict(yaml.safe_load(open(args.config, "r")))
    cfg.model.checkpoint_path = args.ckpt
    print(f"[load] cfg from {args.config}, ckpt={args.ckpt}")

    # If the ckpt was trained with UNI v1 (image_encoder_input_dim==1024), the
    # current default UNI() builds UNI2-h and the image encoder weights mismatch.
    # We only care about the spatial encoder here, so monkey-patch the registry
    # to construct the matching variant before load_checkpoint runs.
    img_dim = int(cfg.model.image_encoder_input_dim)
    variant = "uni" if img_dim == 1024 else "uni2_h"
    print(f"[load] image_encoder_input_dim={img_dim} -> UNI variant={variant!r}")
    _train_utils._IMAGE_BACKBONE_REGISTRY["UNI"] = (
        lambda cfg, _v=variant: _UNI(checkpoint_path=cfg.model.pretrained_image_encoder_ckpt_path, variant=_v)
    )

    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=args.ckpt,
        train=False,
    )
    model = model.to(device).eval()
    num_resolutions = int(cfg.model.num_resolutions)
    print(f"[load] model loaded, num_resolutions={num_resolutions}, "
          f"edge_threshold={cfg.model.edge_threshold}, top_k={cfg.model.get('candidate_top_k', 8)}, "
          f"sim_thr={cfg.model.get('candidate_sim_threshold', 0.4)}")

    manifest = pd.read_csv(args.manifest).drop_duplicates(subset="sample_name", keep="last").reset_index(drop=True)
    if args.slide_name:
        row = manifest[manifest["sample_name"] == args.slide_name].iloc[0]
    else:
        row = manifest.iloc[0]
    print(f"[slide] picked: {row['sample_name']}")

    ds = SpatialPatchDataset(
        image_file=Path(row["hematoxylin_eosin_image"]),
        spatial_file=Path(row["spatial_preprocessed"]),
        spatial_annotation_file=row.get("annotation", None),
        indices_file=Path(row["index"]),
        fov_size=args.fov_size,
        number_of_genes=50,
        num_resolutions=num_resolutions,
        hierarchical_graph=True,
        pretrained_cell_embedding=True,
        type_pretrained_cell_embedding="nemo",
        is_train=False,
        hier_cache_dir=Path(args.hier_cache_dir) if args.hier_cache_dir else None,
    )
    print(f"[slide] {len(ds)} FOVs available; will use up to {args.n_fovs}")
    n_use = min(args.n_fovs, len(ds))
    subset = Subset(ds, list(range(n_use)))
    loader = DataLoader(subset, batch_size=args.batch_size, num_workers=0,
                        collate_fn=multimodal_collate_dict, shuffle=False)

    target_mask_vals = (-1,) + tuple(range(1, num_resolutions + 1)) + (0,)
    #                  orig    accepted-by-layer-i (i = 1..K)        rejected/unset
    per_fov_after_last = []  # collected once at end

    # We also want per-layer accepted counts measured *immediately* after that
    # layer (so we don't double-count when a later layer's mask=0 includes
    # rejected cands from layer N+1). Use a forward hook on each SLS.
    per_layer_accepted: list[list[np.ndarray]] = [[] for _ in range(num_resolutions)]
    per_layer_proposed: list[list[np.ndarray]] = [[] for _ in range(num_resolutions)]
    per_layer_orig_nonself: list[list[np.ndarray]] = [[] for _ in range(num_resolutions)]

    sls_modules = [model.spatial_encoder[i * 2].sls for i in range(num_resolutions)]
    original_forwards = [sls.forward for sls in sls_modules]

    def make_wrapped(layer_id: int, orig_fn):
        def wrapped(g_in, x, layer_id=layer_id):
            # Snapshot input: input g has been wrapped with one self-loop per node
            # at GRAPHLayer.forward, so orig non-self = E_in - N
            n_nodes_per = g_in.batch_num_nodes().cpu().numpy()
            n_edges_in_per = g_in.batch_num_edges().cpu().numpy()
            orig_nonself = n_edges_in_per - n_nodes_per

            g_out = orig_fn(g_in, x, layer_id=layer_id)

            # Per-FOV: candidates accepted by this layer = (mask == layer_id+1) & non-self
            accepted = fov_level_counts_from_mask(g_out, (layer_id + 1,))[:, 0]
            # Per-FOV: candidates proposed (cat 3) = total non-self - orig_nonself - accepted_already
            # Simpler: cat-3 count = E_out - E_in - N (since cat-1 + cat-2 add 2N self-loops, but
            # cat-1 mask=-1 self-loops replace any existing mask=-1 self-loop attribution; the
            # rebuild reconstructs all self-loops from scratch).
            #
            # Concretely _rebuild_batched_graph constructs:
            #   src_c0 (orig non-self) + src_c1 (N self mask=-1) + src_c2 (N self mask=0) + cand_src
            # E_out_per = orig_nonself + 2 * N + cand_per
            n_edges_out_per = g_out.batch_num_edges().cpu().numpy()
            proposed = n_edges_out_per - orig_nonself - 2 * n_nodes_per

            per_layer_accepted[layer_id].append(accepted)
            per_layer_proposed[layer_id].append(proposed)
            per_layer_orig_nonself[layer_id].append(orig_nonself)

            return g_out
        return wrapped

    for i, sls in enumerate(sls_modules):
        sls.forward = make_wrapped(i, original_forwards[i])

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                g = batch["spatial_graph"].to(device)
                # Replicate sigmma.get_spatial_features but only spatial encoder
                # path (no readout / no image side) for speed.
                h = g.ndata["feat"].to(device)
                for i in range(num_resolutions):
                    g, h = model.spatial_encoder[i * 2](g, h, layer_id=i)
                    # readout is skipped — does not affect mask-based counts
                # Save the final-graph mask histogram for one batch
                if bi == 0:
                    per_fov_after_last = fov_level_counts_from_mask(g, target_mask_vals)
    finally:
        for sls, ofn in zip(sls_modules, original_forwards):
            sls.forward = ofn

    # --- Print summary ---
    print(f"\n[results] slide={row['sample_name']}  n_fovs_processed={n_use}")
    print(f"  candidate_top_k={cfg.model.get('candidate_top_k', 8)}  "
          f"candidate_sim_threshold={cfg.model.get('candidate_sim_threshold', 0.4)}  "
          f"edge_threshold={cfg.model.edge_threshold}")
    for i in range(num_resolutions):
        scale_name = {0: "micro", 1: "meso", 2: "macro"}.get(i, f"layer{i}")
        accepted = np.concatenate(per_layer_accepted[i])
        proposed = np.concatenate(per_layer_proposed[i])
        orig = np.concatenate(per_layer_orig_nonself[i])
        block = 2 * (i + 1)
        print(f"\nlayer {i} ({scale_name}, hot-spot BLOCK={block}x{block} cells):")
        print(f"  per-FOV original (non-self) edges:   {orig.mean():.1f} +/- {orig.std():.1f}    (min={orig.min()}, max={orig.max()})")
        print(f"  per-FOV candidates proposed:         {proposed.mean():.1f} +/- {proposed.std():.1f}    (min={proposed.min()}, max={proposed.max()})")
        print(f"  per-FOV new edges accepted:          {accepted.mean():.1f} +/- {accepted.std():.1f}    (min={accepted.min()}, max={accepted.max()})")
        # rates (guard zero)
        prop_sum = proposed.sum()
        orig_sum = orig.sum()
        acc_sum = accepted.sum()
        if prop_sum > 0:
            print(f"  acceptance rate (accepted/proposed):           {acc_sum / prop_sum:.2%}")
        if orig_sum > 0:
            print(f"  newly-added share of orig non-self edges:      {acc_sum / orig_sum:.2%}")

    # Final-graph mask histogram (sanity check, one batch)
    if len(per_fov_after_last) > 0:
        print(f"\n[sanity] final-graph mask distribution (first batch, per-FOV mean):")
        for j, mv in enumerate(target_mask_vals):
            label = {
                -1: "orig non-self",
                0:  "candidate rejected (or unset self)",
            }.get(mv, f"accepted by layer {mv - 1}")
            print(f"  mask=={mv:>2}  ({label}): mean={per_fov_after_last[:, j].mean():.1f}")


if __name__ == "__main__":
    main()
