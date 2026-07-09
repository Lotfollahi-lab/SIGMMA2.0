#!/usr/bin/env python
"""Diagnose the normalisation state of spatial_data_processed.h5 .X for a few RCC
sections, so the angio/Teff signature score uses a CONSISTENT, correct input.

Reports per section: X dtype/min/max, integer fraction, per-cell row-sum spread
(constant => library-normalised), whether a raw-count layer / adata.raw exists,
and the same stats for any counts layer. Read-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def _stats(M, label):
    if M is None:
        print(f"    {label}: <none>")
        return
    d = M.data if sparse.issparse(M) else np.asarray(M).ravel()
    d = d[d != 0]
    if d.size == 0:
        print(f"    {label}: all-zero")
        return
    samp = d[:500000].astype(np.float64)
    int_frac = float(np.mean(np.isclose(samp, np.round(samp))))
    # per-cell row sums (library size)
    rs = np.asarray(M.sum(axis=1)).ravel()
    print(f"    {label}: dtype={d.dtype} nnz_min={samp.min():.4g} "
          f"max={samp.max():.4g} int_frac={int_frac:.3f} | "
          f"rowsum median={np.median(rs):.4g} p5={np.percentile(rs,5):.4g} "
          f"p95={np.percentile(rs,95):.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    args = ap.parse_args()
    man = pd.read_csv(args.manifest).set_index("sample_name")
    for s in args.samples:
        h5 = man.loc[s, "spatial_preprocessed"]
        print(f"\n=== {s} ===\n  {h5}")
        a = sc.read_h5ad(h5)
        print(f"  shape={a.shape}  obsm={list(a.obsm)}  layers={list(a.layers)}  "
              f"raw={'yes' if a.raw is not None else 'no'}  uns_keys={list(a.uns)[:8]}")
        _stats(a.X, "X")
        for lk in a.layers:
            _stats(a.layers[lk], f"layers[{lk}]")
        if a.raw is not None:
            _stats(a.raw.X, "raw.X")
        del a


if __name__ == "__main__":
    main()
