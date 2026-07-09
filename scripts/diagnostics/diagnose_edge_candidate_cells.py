"""Extract per-cell statistics on newly-added candidate edges.

For a single slide (with cell_groups.csv cell-type annotations), forward N FOVs
through the trained SIGMMA spatial encoder and record:
  (1) edge-level: for each accepted candidate edge, src/dst cell-type group.
  (2) cell-level: per-cell orig_degree (mask==-1 neighbors), and per-layer
      in/out degree from newly-accepted edges.

Outputs three CSVs in --out_dir:
  - edges_{slide}.csv:  fov, layer, src_obs, dst_obs, src_group, dst_group,
                        cross_type (bool)
  - cells_{slide}.csv:  obs_idx, cell_id_str, group, orig_degree,
                        new_in_deg_l{0,1,2}, new_out_deg_l{0,1,2}, new_in_total,
                        new_out_total, x_centroid, y_centroid
  - summary_{slide}.csv: per-(group, layer) frequency + cross-type rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, Subset

import sigmma.utils.train_utils as _train_utils
from sigmma.datasets.dataset import SpatialPatchDataset, multimodal_collate_dict
from sigmma.model import UNI as _UNI
from sigmma.utils import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--slide_name", required=True, help="sample_name with cell_groups annotation")
    ap.add_argument("--n_fovs", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fov_size", type=int, default=256)
    ap.add_argument("--hier_cache_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    cfg = ConfigDict(yaml.safe_load(open(args.config, "r")))
    cfg.model.checkpoint_path = args.ckpt
    img_dim = int(cfg.model.image_encoder_input_dim)
    variant = "uni" if img_dim == 1024 else "uni2_h"
    _train_utils._IMAGE_BACKBONE_REGISTRY["UNI"] = (
        lambda c, _v=variant: _UNI(checkpoint_path=c.model.pretrained_image_encoder_ckpt_path, variant=_v)
    )
    model = load_model(
        cfg, pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=args.ckpt, train=False,
    ).to(device).eval()
    num_resolutions = int(cfg.model.num_resolutions)
    print(f"[load] model loaded; num_resolutions={num_resolutions}")

    # ---- Slide row + annotations ----
    manifest = pd.read_csv(args.manifest).drop_duplicates(subset="sample_name", keep="last").reset_index(drop=True)
    row = manifest[manifest["sample_name"] == args.slide_name].iloc[0]
    print(f"[slide] {row['sample_name']}  (annotation={row.get('annotation', '<none>')})")
    assert row.get("annotation"), f"Slide {row['sample_name']} has no annotation csv"

    cell_groups = pd.read_csv(row["annotation"])  # cols: cell_id, group
    cellid_to_group: dict[str, str] = dict(zip(cell_groups["cell_id"], cell_groups["group"]))

    ad = sc.read_h5ad(row["spatial_preprocessed"])
    obs_cell_ids = ad.obs["cell_id"].to_numpy()
    obs_x = ad.obs["x_centroid"].to_numpy()
    obs_y = ad.obs["y_centroid"].to_numpy()
    obs_group = np.array([cellid_to_group.get(cid, "<unannotated>") for cid in obs_cell_ids])
    print(f"[anno] {len(obs_cell_ids)} cells, {(obs_group != '<unannotated>').sum()} annotated, "
          f"{len(set(obs_group))} unique groups")

    # ---- Dataset ----
    ds = SpatialPatchDataset(
        image_file=Path(row["hematoxylin_eosin_image"]),
        spatial_file=Path(row["spatial_preprocessed"]),
        spatial_annotation_file=row.get("annotation", None),
        indices_file=Path(row["index"]),
        fov_size=args.fov_size, number_of_genes=50,
        num_resolutions=num_resolutions, hierarchical_graph=True,
        pretrained_cell_embedding=True, type_pretrained_cell_embedding="nemo",
        is_train=False,
        hier_cache_dir=Path(args.hier_cache_dir) if args.hier_cache_dir else None,
    )
    print(f"[slide] {len(ds)} FOVs available; using {min(args.n_fovs, len(ds))}")
    n_use = min(args.n_fovs, len(ds))
    loader = DataLoader(Subset(ds, list(range(n_use))), batch_size=args.batch_size,
                        num_workers=0, collate_fn=multimodal_collate_dict, shuffle=False)

    # ---- Hook each SLS to capture accepted edges ----
    sls_modules = [model.spatial_encoder[i * 2].sls for i in range(num_resolutions)]
    original_forwards = [sls.forward for sls in sls_modules]
    accepted_records: list[dict] = []      # for edges_*.csv
    per_layer_orig_deg: list[np.ndarray] = []  # one (n_total_nodes_in_batch,) array per batch, only for layer 0
    cell_id_by_batch: list[np.ndarray] = []    # global obs idx of each node, per batch (layer 0 input order)
    nodes_per_batch: list[int] = []

    def make_wrapped(layer_id, orig_fn):
        def wrapped(g_in, x, layer_id=layer_id):
            g_out = orig_fn(g_in, x, layer_id=layer_id)
            mask = g_out.edata["mask"]
            src, dst = g_out.edges()
            sel = (mask == (layer_id + 1)) & (src != dst)
            if int(sel.sum().item()) > 0:
                u = src[sel].cpu().numpy()
                v = dst[sel].cpu().numpy()
                # Map node idx -> global obs idx via spatial_graph cell_id
                # NOTE: cell_id ndata was preserved across rebuild because
                # _rebuild_batched_graph copies all ndata over to g_new.
                cid = g_out.ndata["cell_id"].cpu().numpy()
                src_obs = cid[u]
                dst_obs = cid[v]
                for s_obs, d_obs in zip(src_obs, dst_obs):
                    accepted_records.append({
                        "layer": layer_id, "src_obs": int(s_obs), "dst_obs": int(d_obs),
                    })
            return g_out
        return wrapped

    for i, sls in enumerate(sls_modules):
        sls.forward = make_wrapped(i, original_forwards[i])

    # Per-cell stats: aggregate per global obs_idx across all FOVs
    # We accumulate as a dict keyed on obs_idx
    cell_stats: dict[int, dict[str, int]] = {}

    def stat_row(obs_idx: int) -> dict[str, int]:
        if obs_idx not in cell_stats:
            cell_stats[obs_idx] = {"orig_degree": 0, "fov_count": 0}
            for l in range(num_resolutions):
                cell_stats[obs_idx][f"new_in_l{l}"] = 0
                cell_stats[obs_idx][f"new_out_l{l}"] = 0
        return cell_stats[obs_idx]

    try:
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                g = batch["spatial_graph"].to(device)
                # orig_degree per node in this batch = count of mask==-1 incident edges
                # (Note: spatial_graph initial mask is set in dataset; mask==-1 = intra-grid)
                src0, dst0 = g.edges()
                mask0 = g.edata["mask"]
                is_orig = (mask0 == -1) & (src0 != dst0)
                src0_kept = src0[is_orig]
                dst0_kept = dst0[is_orig]
                n_nodes = int(g.num_nodes())
                deg = torch.bincount(dst0_kept, minlength=n_nodes) + torch.bincount(src0_kept, minlength=n_nodes)
                # divide by 2 since each undirected edge appears as both (u,v) and (v,u)
                deg = (deg // 2).cpu().numpy()
                cid_batch = g.ndata["cell_id"].cpu().numpy()
                # accumulate orig_degree (FOV-wise — a cell only appears in 1 FOV per non-overlap)
                for ni in range(n_nodes):
                    obs_idx = int(cid_batch[ni])
                    s = stat_row(obs_idx)
                    s["orig_degree"] += int(deg[ni])
                    s["fov_count"] += 1

                # forward (hooks record per-layer accepted endpoints into accepted_records)
                pre_len = len(accepted_records)
                h = g.ndata["feat"].to(device)
                for i in range(num_resolutions):
                    g, h = model.spatial_encoder[i * 2](g, h, layer_id=i)
                post_len = len(accepted_records)

                # Build per-cell new in/out counts from newly-appended records.
                # We have to map back through accepted_records — but we already
                # appended src_obs/dst_obs there. Accumulate now for this batch.
                for rec in accepted_records[pre_len:post_len]:
                    L = rec["layer"]
                    s_obs = rec["src_obs"]
                    d_obs = rec["dst_obs"]
                    stat_row(s_obs)[f"new_out_l{L}"] += 1
                    stat_row(d_obs)[f"new_in_l{L}"] += 1

                if bi % 4 == 0:
                    print(f"  [batch {bi}/{(n_use + args.batch_size - 1) // args.batch_size}] "
                          f"n_nodes={n_nodes}  accepted_so_far={post_len}")
    finally:
        for sls, ofn in zip(sls_modules, original_forwards):
            sls.forward = ofn

    print(f"[done] forwarded {n_use} FOVs; accepted edges total={len(accepted_records)}")

    # ---- Save edges CSV ----
    edges_df = pd.DataFrame(accepted_records)
    edges_df["src_group"] = obs_group[edges_df["src_obs"].to_numpy()]
    edges_df["dst_group"] = obs_group[edges_df["dst_obs"].to_numpy()]
    edges_df["cross_type"] = edges_df["src_group"] != edges_df["dst_group"]
    edges_path = out / f"edges_{args.slide_name}.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"[save] {edges_path}  ({len(edges_df)} rows)")

    # ---- Save per-cell CSV ----
    cells_rows = []
    for obs_idx, s in cell_stats.items():
        d = {"obs_idx": obs_idx,
             "cell_id_str": obs_cell_ids[obs_idx],
             "group": obs_group[obs_idx],
             "x_centroid": float(obs_x[obs_idx]),
             "y_centroid": float(obs_y[obs_idx]),
             **s}
        d["new_in_total"] = sum(s[f"new_in_l{l}"] for l in range(num_resolutions))
        d["new_out_total"] = sum(s[f"new_out_l{l}"] for l in range(num_resolutions))
        cells_rows.append(d)
    cells_df = pd.DataFrame(cells_rows)
    cells_path = out / f"cells_{args.slide_name}.csv"
    cells_df.to_csv(cells_path, index=False)
    print(f"[save] {cells_path}  ({len(cells_df)} cells observed)")

    # ---- Summary per (group, layer) ----
    summary_rows = []
    n_by_group = cells_df.groupby("group").size().to_dict()
    for g_ in sorted(cells_df["group"].unique()):
        sub = cells_df[cells_df["group"] == g_]
        row_ = {"group": g_, "n_cells": int(n_by_group[g_]),
                "orig_degree_mean": float(sub["orig_degree"].mean()) if len(sub) else 0.0}
        for L in range(num_resolutions):
            row_[f"new_in_l{L}_per_cell"] = float(sub[f"new_in_l{L}"].mean())
            row_[f"new_out_l{L}_per_cell"] = float(sub[f"new_out_l{L}"].mean())
        # cross-type rate from edges_df
        e_src = edges_df[edges_df["src_group"] == g_]
        if len(e_src) > 0:
            row_["cross_type_rate_as_src"] = float(e_src["cross_type"].mean())
        else:
            row_["cross_type_rate_as_src"] = float("nan")
        summary_rows.append(row_)
    summary_df = pd.DataFrame(summary_rows).sort_values("n_cells", ascending=False)
    summary_path = out / f"summary_{args.slide_name}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")
    print("\n[summary top 15 by n_cells]\n" + summary_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
