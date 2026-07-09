#!/usr/bin/env python
"""Compute per-cell single-gene expression targets from in-house RCC Xenium, to
serve as regression targets for an H&E -> biomarker probe (CA-IX / PAX8).

This is the single-gene sibling of angio_teff/compute_signature_scores.py. The
angio/Teff version aggregates a *signature* (gene set) into a score; here each
target is one clinical biomarker gene, so the target is simply that gene's
depth-corrected log1p-CP10k expression per cell.

Biomarker rationale (clear-cell RCC): CA9 (CA-IX) = HIF/VHL hypoxia axis, highly
ccRCC-specific; PAX8 = renal lineage marker. Both are on the RCC Xenium 5K panel
(verified: CA9 in 19/20 sections, PAX8 in 20/20). Vimentin (VIM) is NOT on the
panel and is intentionally excluded.

The target is NOT gated by the 2,979-gene SIGMMA training intersection: it is
read from the real per-section Xenium, exactly the embedding-regression-probe
design (target from real ST, not from an imputation head).

.X normalisation follows compute_signature_scores.py: .X is log1p(RAW counts)
with NO library-size normalisation, so we RECOVER counts = expm1(X) then apply
CP(target_sum)+log1p ONCE (depth-corrected, cross-section comparable).

Per-cell output is keyed by cell_id (obs cell_id col if present, else obs_names)
AND row index, joinable to the z_sigmma_* feature HDF5 by cell_id.

Run in the sigmma poetry venv (scanpy + pyarrow).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def _nnz_sample(X) -> np.ndarray:
    sub = X[:2000] if X.shape[0] > 2000 else X
    d = sub.data if sparse.issparse(sub) else sub[sub != 0]
    return np.asarray(d[:200000], dtype=np.float64)


def _is_counts(X) -> bool:
    d = _nnz_sample(X)
    return d.size > 0 and float(np.nanmax(d)) > 30.0 and bool(np.allclose(d, np.round(d)))


def _is_log1p_counts(X) -> bool:
    d = _nnz_sample(X)
    if d.size == 0 or bool(np.allclose(d, np.round(d))):
        return False
    rec = np.expm1(d)
    return bool(np.isclose(d.min(), np.log1p(1.0), atol=1e-3)) and \
        float(np.mean(np.isclose(rec, np.round(rec), atol=1e-2))) > 0.95


def _to_cp_log1p(adata, target_sum: float) -> str:
    if _is_counts(adata.X):
        src = "raw_counts"
    elif _is_log1p_counts(adata.X):
        if sparse.issparse(adata.X):
            adata.X.data = np.rint(np.expm1(adata.X.data))
        else:
            adata.X = np.rint(np.expm1(adata.X))
        src = "expm1(log1p_counts)"
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        return "unknown->cp+log1p(assumed_counts)"
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return f"{src}->cp{target_sum:g}+log1p"


def _cell_ids(adata) -> np.ndarray:
    for k in ("cell_id", "CellID", "cell_ID", "barcode"):
        if k in adata.obs.columns:
            return adata.obs[k].astype(str).to_numpy()
    return adata.obs_names.astype(str).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="CSV with sample_name + spatial_preprocessed columns")
    ap.add_argument("--genes", nargs="+", default=["CA9", "PAX8"],
                    help="biomarker gene symbols to emit as per-cell targets")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_sum", type=float, default=1e4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "per_cell").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    genes = [g.upper() for g in args.genes]

    rows = []
    for _, r in man.iterrows():
        name = r["sample_name"]
        h5 = r.get("spatial_preprocessed", "")
        if not isinstance(h5, str) or not Path(h5).exists():
            print(f"[skip] {name}: spatial_preprocessed missing", flush=True)
            continue

        adata = sc.read_h5ad(h5)
        var_up = {}
        for g in adata.var_names:
            var_up.setdefault(str(g).upper(), str(g))  # first occurrence wins

        norm = _to_cp_log1p(adata, args.target_sum)
        if norm.startswith("unknown"):
            print(f"[warn] {name}: .X state unrecognised -> {norm}", flush=True)

        cid = _cell_ids(adata)
        cols = {"cell_id": cid, "row_idx": np.arange(adata.n_obs, dtype=np.int64)}
        rec = {"sample_name": name, "split": r.get("split", ""),
               "n_cells": int(adata.n_obs), "norm": norm}

        for g in genes:
            if g in var_up:
                sub = adata[:, [var_up[g]]].X
                sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
                expr = np.asarray(sub, dtype=np.float32).ravel()
                cols[g] = expr
                rec[f"{g}_present"] = 1
                rec[f"{g}_mean"] = float(np.nanmean(expr))
                rec[f"{g}_frac_pos"] = float(np.mean(expr > 0))
            else:
                cols[g] = np.full(adata.n_obs, np.nan, dtype=np.float32)
                rec[f"{g}_present"] = 0
                rec[f"{g}_mean"] = float("nan")
                rec[f"{g}_frac_pos"] = float("nan")

        pd.DataFrame(cols).to_parquet(out / "per_cell" / f"{name}.parquet",
                                      index=False)
        rows.append(rec)
        present = " ".join(f"{g}={rec[f'{g}_present']}" for g in genes)
        means = " ".join(f"{g}_mean={rec[f'{g}_mean']:.3f}" for g in genes)
        print(f"[ok] {name}: n={rec['n_cells']} norm={norm} {present} {means}",
              flush=True)
        del adata

    summ = pd.DataFrame(rows)
    summ_path = out / "gene_target_summary.csv"
    summ.to_csv(summ_path, index=False)
    print("\n===== summary =====", flush=True)
    print(summ.to_string(index=False), flush=True)
    print(f"\nwrote {summ_path}", flush=True)
    print(f"wrote per-cell parquet -> {out / 'per_cell'}", flush=True)


if __name__ == "__main__":
    main()
