"""
Per-step timing of SpatialPatchDataset.__getitem__ on a single training slide.

Goal: locate which step (image read / spatial graph build / hierarchical
subgraph build / cell-embedding lookup / spatial text) dominates the per-tile
cost, so we can target the fix.

Usage:
    python scripts/profile_dataloader.py \
        --manifest /lustre/.../sanger_MANIFEST_train.csv \
        --n_tiles 200 \
        --fov_size 256 --num_resolutions 3
"""
from __future__ import annotations
import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy
import torch
import zarr
import dgl


def time_it(label: str, store: dict, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000  # ms
    store.setdefault(label, []).append(dt)
    return out


def profile_one_slide(row: pd.Series, n_tiles: int, fov_size: int, num_resolutions: int):
    print(f"\n=== Slide: {row['sample_name']} ===")

    # --- Eager loads (these happen once per slide at __init__) ---
    init_t = {}
    t0 = time.perf_counter()
    adata = scanpy.read_h5ad(row['spatial_preprocessed'])
    init_t['load_h5ad'] = (time.perf_counter() - t0) * 1000
    coords = np.array(adata.obsm['spatial'], dtype=np.float32)
    nemo = np.array(adata.obsm['nemo_latent'], dtype=np.float32)
    X = adata.X.tocsr() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)
    conn = adata.obsp['spatial_connectivities'].tocsr()
    var_index = adata.var.index.to_numpy()
    cell_id_to_num = np.arange(len(adata.obs), dtype=np.int64)
    init_t['load_h5ad_total'] = (time.perf_counter() - t0) * 1000
    print(f"[init] load h5ad+arrays: {init_t['load_h5ad_total']:.1f} ms  "
          f"(N_cells={coords.shape[0]}, N_genes={X.shape[1] if hasattr(X, 'shape') else len(var_index)})")
    del adata

    image_array = zarr.open(str(row['hematoxylin_eosin_image']), mode='r')
    index_array = zarr.open(str(row['index']), mode='r')
    print(f"[init] image_array shape={image_array.shape}, dtype={image_array.dtype}, "
          f"chunks={image_array.chunks}")

    # Pick valid tiles (tissue threshold == 0 here for simplicity)
    valid_idx = np.arange(index_array.shape[0])
    rng = np.random.default_rng(0)
    if valid_idx.size > n_tiles:
        sel = rng.choice(valid_idx, size=n_tiles, replace=False)
    else:
        sel = valid_idx
    print(f"[loop] tiles to profile: {sel.size}/{valid_idx.size}")

    # --- Per-tile profiling ---
    t = {}
    n_cells_seen = []
    grid_size = fov_size // (2 ** num_resolutions)
    for k, idx in enumerate(sel):
        slice_args = index_array[idx]
        y_start, x_start, _ = np.uint32(slice_args)

        # 1. cells in FOV (mask compute)
        mask = time_it('1_mask_compute', t, lambda: (
            (x_start <= coords[:, 0]) & (coords[:, 0] < x_start + fov_size) &
            (y_start <= coords[:, 1]) & (coords[:, 1] < y_start + fov_size)
        ))
        n = int(mask.sum())
        n_cells_seen.append(n)
        if n == 0:
            continue

        # 2. cell-embedding lookup (nemo)
        _ = time_it('2_nemo_lookup', t, lambda: torch.from_numpy(nemo[mask]))

        # 3. spatial-graph from CSR sub-slice + dgl.from_scipy
        def _build_graph():
            sub = conn[mask][:, mask]
            return dgl.from_scipy(sub)
        sg = time_it('3_full_graph_build', t, _build_graph)

        # 4. hierarchical subgraphs (16 sub-FOVs for num_resolutions=3)
        def _build_subgraphs():
            subgraphs = []
            for x_g, y_g in product(range(0, fov_size, grid_size),
                                    range(0, fov_size, grid_size)):
                m = (
                    (x_start + x_g  <= coords[:, 0]) & (coords[:, 0] < x_start + x_g + grid_size) &
                    (y_start + y_g  <= coords[:, 1]) & (coords[:, 1] < y_start + y_g + grid_size)
                )
                sub = conn[m][:, m]
                gg = dgl.from_scipy(sub)
                gg.ndata['feat'] = torch.from_numpy(nemo[m])
                subgraphs.append(gg)
            return dgl.batch(subgraphs)
        disjoint = time_it('4_hier_subgraphs', t, _build_subgraphs)

        # 5. edge-mask compute (the torch.isin block)
        def _edge_mask():
            src_full, dst_full = sg.edges()
            src_sub, dst_sub = disjoint.edges()
            N = sg.num_nodes()
            full_ids = src_full * N + dst_full
            sub_ids = src_sub * N + dst_sub
            return torch.isin(full_ids, sub_ids)
        time_it('5_edge_mask', t, _edge_mask)

        # 6. spatial text (top-K genes)
        def _spatial_text():
            X_fov = X[mask].toarray() if sp.issparse(X) else X[mask]
            mean_exp = X_fov.mean(axis=0).ravel()
            top_idx = np.argsort(mean_exp)[-50:][::-1]
            return " ".join(var_index[top_idx])
        time_it('6_spatial_text', t, _spatial_text)

        # 7. image crop from zarr (the suspected IO bottleneck)
        def _img_read():
            return image_array[:, y_start: y_start + fov_size,
                                  x_start: x_start + fov_size]
        time_it('7_image_zarr_read', t, _img_read)

        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{sel.size} tiles processed")

    # Summary
    print(f"\n[stats] cells per FOV: mean={np.mean(n_cells_seen):.1f}, "
          f"median={np.median(n_cells_seen):.1f}, max={np.max(n_cells_seen)}")
    print("\n=== Per-step timing (ms / tile) ===")
    print(f"{'step':<25} {'mean':>10} {'p50':>10} {'p95':>10} {'sum_s':>10}")
    rows = []
    total_mean = 0.0
    for label, vals in t.items():
        v = np.asarray(vals)
        mean_ = float(v.mean())
        med_ = float(np.median(v))
        p95_ = float(np.quantile(v, 0.95))
        sum_s = float(v.sum() / 1000)
        total_mean += mean_
        print(f"{label:<25} {mean_:>10.3f} {med_:>10.3f} {p95_:>10.3f} {sum_s:>10.2f}")
        rows.append({'step': label, 'mean_ms': mean_, 'p50_ms': med_,
                     'p95_ms': p95_, 'total_s': sum_s, 'n': v.size})
    print(f"{'TOTAL (sum of means)':<25} {total_mean:>10.3f} ms / tile")

    print("\n=== Step share (% of total mean) ===")
    for label, vals in sorted(t.items(), key=lambda kv: -np.mean(kv[1])):
        share = float(np.mean(vals)) / total_mean * 100
        print(f"  {label:<25} {share:>5.1f}%")

    return rows, total_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--n_tiles', type=int, default=200)
    ap.add_argument('--n_slides', type=int, default=2)
    ap.add_argument('--fov_size', type=int, default=256)
    ap.add_argument('--num_resolutions', type=int, default=3)
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"[load] manifest = {len(manifest)} slides")

    all_rows = []
    for i in range(min(args.n_slides, len(manifest))):
        rows, total = profile_one_slide(
            manifest.iloc[i], args.n_tiles, args.fov_size, args.num_resolutions
        )
        for r in rows:
            r['slide'] = manifest.iloc[i]['sample_name']
            r['total_mean_ms'] = total
        all_rows.extend(rows)

    if args.out_csv:
        pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
        print(f"\n[csv] wrote {args.out_csv}")


if __name__ == '__main__':
    main()
