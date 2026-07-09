"""
Verify the hierarchical-subgraph cache speedup by timing the actual
SpatialPatchDataset.__getitem__ for one slide, with and without the cache.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sigmma.datasets.dataset import SpatialPatchDataset


def time_dataset(ds: SpatialPatchDataset, n_samples: int) -> dict:
    rng = np.random.default_rng(0)
    n = len(ds)
    if n == 0:
        return {"n": 0, "mean_ms": float("nan"), "p50_ms": float("nan"),
                "p95_ms": float("nan"), "total_s": 0.0}
    sample_idx = rng.choice(n, size=min(n_samples, n), replace=False)
    times = []
    t_start = time.perf_counter()
    for i in sample_idx:
        t0 = time.perf_counter()
        out = ds[int(i)]
        if out is None:
            continue
        times.append((time.perf_counter() - t0) * 1000)
    total_s = time.perf_counter() - t_start
    arr = np.asarray(times)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else float("nan"),
        "p50_ms": float(np.median(arr)) if arr.size else float("nan"),
        "p95_ms": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "total_s": float(total_s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--slide', required=True)
    ap.add_argument('--hier_cache_dir', required=True)
    ap.add_argument('--fov_size', type=int, default=256)
    ap.add_argument('--num_resolutions', type=int, default=3)
    ap.add_argument('--tissue_threshold', type=float, default=20000)
    ap.add_argument('--n_samples', type=int, default=300)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    row = manifest[manifest['sample_name'] == args.slide].iloc[0]
    common = dict(
        image_file=Path(row['hematoxylin_eosin_image']),
        spatial_file=Path(row['spatial_preprocessed']),
        spatial_annotation_file=row['annotation'],
        indices_file=Path(row['index']),
        fov_size=args.fov_size,
        num_resolutions=args.num_resolutions,
        hierarchical_graph=True,
        tissue_filtering_threshold=args.tissue_threshold,
        pretrained_cell_embedding=True,
        type_pretrained_cell_embedding='nemo',
        is_train=False,
        model_name='sigmma',
    )

    print("=== cold (no cache) ===")
    ds_cold = SpatialPatchDataset(**common, hier_cache_dir=None)
    print(f"[load] dataset size = {len(ds_cold)}")
    cold = time_dataset(ds_cold, args.n_samples)
    print(f"  n={cold['n']}  mean={cold['mean_ms']:.3f} ms/tile  "
          f"p50={cold['p50_ms']:.3f}  p95={cold['p95_ms']:.3f}  "
          f"total={cold['total_s']:.2f} s")

    print("\n=== cached ===")
    ds_warm = SpatialPatchDataset(**common, hier_cache_dir=args.hier_cache_dir)
    assert ds_warm._hier_cache is not None, "cache failed to load"
    print(f"[load] cache loaded for {ds_warm.section_id}")
    warm = time_dataset(ds_warm, args.n_samples)
    print(f"  n={warm['n']}  mean={warm['mean_ms']:.3f} ms/tile  "
          f"p50={warm['p50_ms']:.3f}  p95={warm['p95_ms']:.3f}  "
          f"total={warm['total_s']:.2f} s")

    if cold['mean_ms'] > 0 and not np.isnan(warm['mean_ms']):
        spd = cold['mean_ms'] / warm['mean_ms']
        print(f"\n[speedup]  cached / cold = {spd:.2f}×")
        print(f"           saved per tile = {cold['mean_ms'] - warm['mean_ms']:.3f} ms")


if __name__ == '__main__':
    main()
