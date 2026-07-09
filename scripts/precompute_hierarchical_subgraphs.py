"""
Pre-compute the per-tile hierarchical sub-graph structure that
SpatialPatchDataset.__getitem__ rebuilds from scratch on every batch.

Profiler showed this step (`4_hier_subgraphs`) accounts for ~52% of per-tile
dataloader time (mean ~24 ms / tile). The output of this step is a deterministic
function of the slide's spatial graph + cell coordinates and the FOV's
(x_start, y_start, fov_size, num_resolutions); caching it once per slide cuts
this cost to a flat array slice (sub-millisecond) at runtime.

Output (one .npz per slide section_id):
  cell_indices       (sum of cells across valid tiles,) int32
                     concatenation of disjoint-batched cell lists, in sub-FOV
                     iteration order ((x_g, y_g) with x_g outer, y_g inner),
                     and within each sub-FOV in original-cell-index order.
  cell_offsets       (num_valid_tiles + 1,) int64
                     prefix sums into cell_indices: tile t's cells are
                     cell_indices[cell_offsets[t]:cell_offsets[t+1]].
  subfov_cell_offsets (num_valid_tiles, n_subfov + 1) int32
                     prefix sums *within* a tile, partitioning its cells by
                     sub-FOV. n_subfov = (fov_size // grid_size) ** 2.
  edge_src           (sum of edges across valid tiles,) int32
  edge_dst           (sum of edges across valid tiles,) int32
                     edges within sub-FOVs, in disjoint-local indexing
                     (i.e. 0..tile_num_cells-1, with sub-FOV-cumulative offsets
                     already added — matches `dgl.batch(subgraphs).edges()`).
  edge_offsets       (num_valid_tiles + 1,) int64
  valid_indices      (num_valid_tiles,) int32
                     tile indices into the slide's index_array that this cache
                     covers (tissue-filtered).
  fov_size, num_resolutions, grid_size : scalars.

Usage:
    python scripts/precompute_hierarchical_subgraphs.py \\
      --manifest /lustre/.../sanger_MANIFEST_train.csv \\
      --out_dir  /lustre/.../hier_cache_fov256_res3 \\
      --fov_size 256 --num_resolutions 3 \\
      --tissue_threshold 20000 --n_workers 16
"""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scanpy
import scipy.sparse as sp
import zarr


def precompute_slide(spatial_file: str, indices_file: str,
                     fov_size: int, num_resolutions: int,
                     tissue_threshold: float, out_path: str) -> dict:
    t_start = time.perf_counter()

    adata = scanpy.read_h5ad(spatial_file)
    coords = np.array(adata.obsm['spatial'], dtype=np.float32)
    conn = adata.obsp['spatial_connectivities'].tocsr()
    del adata

    index_array = zarr.open(str(indices_file), mode='r')
    if tissue_threshold is not None and tissue_threshold > 0:
        mask = index_array[:, -1] >= tissue_threshold
        valid_indices = np.where(mask)[0].astype(np.int32)
    else:
        valid_indices = np.arange(index_array.shape[0], dtype=np.int32)

    grid_size = fov_size // (2 ** num_resolutions)
    n_per_dim = fov_size // grid_size
    n_subfov = n_per_dim * n_per_dim
    # Pre-load index_array fully (much faster than per-row zarr reads).
    all_slices = index_array[:][valid_indices]  # (num_valid, 3) y_start, x_start, _

    cell_indices_all: list[np.ndarray] = []
    edge_src_all: list[np.ndarray] = []
    edge_dst_all: list[np.ndarray] = []
    cell_offsets = np.zeros(valid_indices.size + 1, dtype=np.int64)
    edge_offsets = np.zeros(valid_indices.size + 1, dtype=np.int64)
    subfov_cell_offsets = np.zeros((valid_indices.size, n_subfov + 1), dtype=np.int32)

    for t, slice_args in enumerate(all_slices):
        y_start, x_start, _ = np.uint32(slice_args)
        local_offset = 0
        sf_off = subfov_cell_offsets[t]
        tile_cells: list[np.ndarray] = []
        tile_src: list[np.ndarray] = []
        tile_dst: list[np.ndarray] = []
        # Match __getitem__'s product(range(0, fov_size, grid_size), ...)
        # which yields (x_g, y_g) with x_g outer, y_g inner.
        for sf_idx, (x_g, y_g) in enumerate(product(range(0, fov_size, grid_size),
                                                    range(0, fov_size, grid_size))):
            m = (
                (x_start + x_g  <= coords[:, 0]) & (coords[:, 0] < x_start + x_g + grid_size) &
                (y_start + y_g  <= coords[:, 1]) & (coords[:, 1] < y_start + y_g + grid_size)
            )
            cell_ids = np.where(m)[0].astype(np.int32)
            n = cell_ids.size
            tile_cells.append(cell_ids)
            sf_off[sf_idx + 1] = sf_off[sf_idx] + n

            if n > 0:
                # Edges within this sub-FOV; conn is symmetric so we get both directions.
                sub = conn[cell_ids, :][:, cell_ids].tocoo()
                if sub.nnz > 0:
                    tile_src.append(sub.row.astype(np.int32) + local_offset)
                    tile_dst.append(sub.col.astype(np.int32) + local_offset)
            local_offset += n

        if tile_cells:
            tile_cell_arr = np.concatenate(tile_cells) if any(c.size for c in tile_cells) \
                            else np.zeros(0, dtype=np.int32)
        else:
            tile_cell_arr = np.zeros(0, dtype=np.int32)
        tile_src_arr = np.concatenate(tile_src) if tile_src else np.zeros(0, dtype=np.int32)
        tile_dst_arr = np.concatenate(tile_dst) if tile_dst else np.zeros(0, dtype=np.int32)

        cell_indices_all.append(tile_cell_arr)
        edge_src_all.append(tile_src_arr)
        edge_dst_all.append(tile_dst_arr)
        cell_offsets[t + 1] = cell_offsets[t] + tile_cell_arr.size
        edge_offsets[t + 1] = edge_offsets[t] + tile_src_arr.size

    cell_indices_flat = np.concatenate(cell_indices_all) if cell_indices_all else np.zeros(0, dtype=np.int32)
    edge_src_flat = np.concatenate(edge_src_all) if edge_src_all else np.zeros(0, dtype=np.int32)
    edge_dst_flat = np.concatenate(edge_dst_all) if edge_dst_all else np.zeros(0, dtype=np.int32)

    np.savez(
        out_path,
        cell_indices=cell_indices_flat,
        cell_offsets=cell_offsets,
        edge_src=edge_src_flat,
        edge_dst=edge_dst_flat,
        edge_offsets=edge_offsets,
        subfov_cell_offsets=subfov_cell_offsets,
        valid_indices=valid_indices,
        fov_size=np.int32(fov_size),
        num_resolutions=np.int32(num_resolutions),
        grid_size=np.int32(grid_size),
    )

    elapsed = time.perf_counter() - t_start
    return {
        "section_id": Path(spatial_file).parts[-2],
        "tiles": int(valid_indices.size),
        "cells": int(cell_indices_flat.size),
        "edges": int(edge_src_flat.size),
        "elapsed_s": float(elapsed),
        "out_path": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fov_size', type=int, default=256)
    ap.add_argument('--num_resolutions', type=int, default=3)
    ap.add_argument('--tissue_threshold', type=float, default=20000)
    ap.add_argument('--n_workers', type=int, default=8)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)

    todo = []
    for i, row in manifest.iterrows():
        section_id = Path(row['spatial_preprocessed']).parts[-2]
        out_path = Path(args.out_dir) / f"{section_id}.npz"
        if out_path.exists():
            print(f"[skip] {section_id} (cache exists)")
            continue
        todo.append({
            "spatial_file": row['spatial_preprocessed'],
            "indices_file": row['index'],
            "out_path": str(out_path),
        })
    print(f"[plan] {len(todo)} slides to process out of {len(manifest)} "
          f"(workers={args.n_workers}, fov={args.fov_size}, num_res={args.num_resolutions}, "
          f"tissue_threshold={args.tissue_threshold})")

    if not todo:
        print("[done] nothing to do")
        return

    if args.n_workers <= 1:
        for spec in todo:
            r = precompute_slide(spec['spatial_file'], spec['indices_file'],
                                 args.fov_size, args.num_resolutions,
                                 args.tissue_threshold, spec['out_path'])
            print(f"  {r['section_id']}: tiles={r['tiles']}  cells={r['cells']}  "
                  f"edges={r['edges']}  ({r['elapsed_s']:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = [
                ex.submit(precompute_slide, spec['spatial_file'], spec['indices_file'],
                          args.fov_size, args.num_resolutions,
                          args.tissue_threshold, spec['out_path'])
                for spec in todo
            ]
            for f in as_completed(futures):
                r = f.result()
                print(f"  {r['section_id']}: tiles={r['tiles']}  cells={r['cells']}  "
                      f"edges={r['edges']}  ({r['elapsed_s']:.1f}s)")
    print("[done]")


if __name__ == "__main__":
    main()
