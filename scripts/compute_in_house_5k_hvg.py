"""Compute the 50-gene HVG panel for the 5K-cohort training set.

Uses the same algorithm as the gex_geometry_23617 diagnostic
(``compute_top_k_hvg_by_dispersion`` in ``sigmma/evaluation/transcript_binning.py``):
abundance floor (top-300 by total transcript count) → cross-slide CV ranking.

Outputs:
  data/in_house_5k_disp_hvg_{K}.npy           (np.array of gene names, dtype=object)
  data/in_house_5k_disp_hvg_{K}.meta.json     (algorithm hyperparams + n_slides used)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import h5py

from sigmma.evaluation.transcript_binning import (
    compute_top_k_hvg_by_dispersion,
    manifest_path_to_transcripts_parquet,
)


def _read_var_index(spatial_file: str) -> list[str]:
    with h5py.File(spatial_file, "r") as f:
        var = f["var"]
        idx_attr = var.attrs.get("_index", "_index")
        if isinstance(idx_attr, bytes):
            idx_attr = idx_attr.decode()
        if idx_attr not in var:
            for c in ("gene_name", "feature_name", "gene_symbol", "_index"):
                if c in var:
                    idx_attr = c
                    break
        raw = var[idx_attr][:]
    return [g.decode() if isinstance(g, (bytes, bytearray)) else str(g) for g in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train_5k_panel.csv",
    )
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--qv_min", type=float, default=20.0)
    ap.add_argument("--abundance_floor_top", type=int, default=300)
    ap.add_argument("--min_slides_present", type=int, default=3)
    ap.add_argument("--k_request_multiplier", type=int, default=4,
                    help="Request k * multiplier from dispersion to buffer against parquet→h5ad gene-name drift.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)
    m = m.drop_duplicates("sample_name", keep="last").reset_index(drop=True)
    print(f"[manifest] {len(m)} sections", flush=True)

    parquet_paths, missing = [], []
    for he in m["hematoxylin_eosin_image"].astype(str):
        p = manifest_path_to_transcripts_parquet(he)
        if p is None:
            missing.append(he)
        else:
            parquet_paths.append(p)
    print(f"[parquets] resolved {len(parquet_paths)} / {len(m)} (missing {len(missing)})", flush=True)
    if missing:
        for h in missing[:5]:
            print(f"  missing: {h}", flush=True)
    if len(parquet_paths) < args.min_slides_present:
        raise SystemExit(f"too few parquets ({len(parquet_paths)}) to compute panel")

    k_request = args.k * args.k_request_multiplier
    print(f"[hvg] computing top-{k_request} dispersion panel (oversample x{args.k_request_multiplier}) ...", flush=True)
    panel_oversample = compute_top_k_hvg_by_dispersion(
        parquet_paths,
        k=k_request,
        qv_min=args.qv_min,
        abundance_floor_top=max(args.abundance_floor_top, k_request),
        min_slides_present=args.min_slides_present,
    )
    print(f"[hvg] dispersion returned {len(panel_oversample)} candidates", flush=True)

    # Filter to genes present in the 5K cohort's h5ad var_index. The parquet's
    # gene set differs slightly from the h5ad var_index (e.g. TMEM173 in parquet
    # is absent from h5ad). Loss-side slicing uses var_index, so genes missing
    # there are unusable.
    ref_var_index = set(_read_var_index(m["spatial_preprocessed"].iloc[0]))
    print(f"[hvg] reference var_index size: {len(ref_var_index)} "
          f"(from {m['sample_name'].iloc[0]})", flush=True)
    panel_filt = [g for g in panel_oversample if g in ref_var_index]
    dropped = [g for g in panel_oversample if g not in ref_var_index]
    print(f"[hvg] {len(panel_filt)} survive var_index intersection "
          f"({len(dropped)} dropped, first 5: {dropped[:5]})", flush=True)
    if len(panel_filt) < args.k:
        raise SystemExit(
            f"only {len(panel_filt)} genes survive var_index intersection but k={args.k}. "
            f"Increase --k_request_multiplier or expand parquet coverage."
        )
    panel = panel_filt[: args.k]
    print(f"[hvg] selected {len(panel)} genes", flush=True)
    print(f"[hvg] first 10: {panel[:10]}", flush=True)

    npy_path = Path(args.out_dir) / f"in_house_5k_disp_hvg_{args.k}.npy"
    meta_path = Path(args.out_dir) / f"in_house_5k_disp_hvg_{args.k}.meta.json"
    np.save(npy_path, np.array(panel, dtype=object), allow_pickle=True)
    with open(meta_path, "w") as f:
        json.dump({
            "manifest": args.manifest,
            "n_sections": len(m),
            "n_parquets_used": len(parquet_paths),
            "n_parquets_missing": len(missing),
            "k": args.k,
            "qv_min": args.qv_min,
            "abundance_floor_top": args.abundance_floor_top,
            "min_slides_present": args.min_slides_present,
            "genes": list(panel),
            "algorithm": "compute_top_k_hvg_by_dispersion (abundance floor + cross-slide CV)",
        }, f, indent=2)
    print(f"[saved] {npy_path}\n[saved] {meta_path}", flush=True)


if __name__ == "__main__":
    main()
