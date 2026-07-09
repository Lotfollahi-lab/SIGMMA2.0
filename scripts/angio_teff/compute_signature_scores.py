#!/usr/bin/env python
"""Compute per-cell angiogenesis & T-effector signature scores from in-house RCC
Xenium, to serve as regression targets for an H&E -> molecular-score probe.

Clinical-outcome path 2 (see memory project_angio_teff_pipeline_design_2026_06_28):
TSR failed in RCC because it is a *morphological* ratio that UNI matches; the
angiogenesis / T-effector (Teff) axes are *molecular* and play to SIGMMA's
ST-aware strength. We use the IMmotion150/151 (McDermott 2018 Nat Med; Motzer
2020 Cancer Cell) signatures:

    Angiogenesis : VEGFA, KDR, ESM1, PECAM1, ANGPTL4, CD34
    T-effector   : CD8A, EOMES, PRF1, IFNG, CD274

NOTE on the panel: the 2,979-gene SIGMMA training INTERSECTION drops the immune
markers, but each individual section's full Xenium 5K panel measures all 11
genes, so the score TARGET (computed here from the real per-section Xenium) is
NOT gated by the intersection. This is the whole point of the embedding-
regression-probe architecture (target from real ST, not from SIGMMA's imputation
head).

IMPORTANT — .X is log1p(RAW counts) with NO library-size normalisation (verified
via diag_xstate.py: every section's min non-zero == ln2 == log1p(1); per-cell
sums vary ~35x across sections = capture-depth, not biology). This is the same
state that the `fix/gex-double-log-and-dataset-rename` branch addresses (the old
code treated this log1p X as raw -> double log). So we RECOVER counts =
expm1(X), then apply CP(target_sum)+log1p ONCE, making scores depth-corrected
and cross-section comparable. We then emit TWO per-cell scores per signature:

  *_mean  = mean log1p-CP10k expression of the signature genes. Cross-section
            comparable (same genes, same normalisation); the primary target.
  *_score = scanpy.tl.score_genes control-gene-corrected score (per-section,
            centred on a matched random control set). Secondary / robustness.

Per-cell output is keyed by cell_id (AnnData obs cell_id col if present, else
obs_names) AND row index, so it can later be joined to the z_sigmma_* feature
HDF5 by cell_id the same way train_tumour_stroma_probe.py does (cid array_equal).

Run in the sigmma poetry venv (has scanpy + pyarrow).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# IMmotion150 / IMmotion151 signatures (McDermott 2018 Nat Med; Motzer 2020).
SIGNATURES = {
    "angio": ["VEGFA", "KDR", "ESM1", "PECAM1", "ANGPTL4", "CD34"],
    "teff": ["CD8A", "EOMES", "PRF1", "IFNG", "CD274"],
}


def _nnz_sample(X) -> np.ndarray:
    sub = X[:2000] if X.shape[0] > 2000 else X
    d = sub.data if sparse.issparse(sub) else sub[sub != 0]
    return np.asarray(d[:200000], dtype=np.float64)


def _is_counts(X) -> bool:
    d = _nnz_sample(X)
    return d.size > 0 and float(np.nanmax(d)) > 30.0 and bool(np.allclose(d, np.round(d)))


def _is_log1p_counts(X) -> bool:
    """log1p(raw counts): non-integer, and expm1 recovers near-integers with
    min non-zero ~= ln2 = log1p(1)."""
    d = _nnz_sample(X)
    if d.size == 0 or bool(np.allclose(d, np.round(d))):
        return False
    rec = np.expm1(d)
    return bool(np.isclose(d.min(), np.log1p(1.0), atol=1e-3)) and \
        float(np.mean(np.isclose(rec, np.round(rec), atol=1e-2))) > 0.95


def _to_cp_log1p(adata, target_sum: float) -> str:
    """Normalise to CP(target_sum)+log1p ONCE, recovering raw counts first if .X
    is already log1p(counts). Returns a tag describing what was done."""
    if _is_counts(adata.X):
        src = "raw_counts"
    elif _is_log1p_counts(adata.X):
        # recover integer counts in place (round away float32 expm1 noise)
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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_sum", type=float, default=1e4)
    ap.add_argument("--ctrl_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "per_cell").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)

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

        for sig, genes in SIGNATURES.items():
            present = [var_up[g.upper()] for g in genes if g.upper() in var_up]
            missing = [g for g in genes if g.upper() not in var_up]
            rec[f"{sig}_n_present"] = len(present)
            rec[f"{sig}_n_total"] = len(genes)
            rec[f"{sig}_missing"] = ",".join(missing)

            if not present:
                cols[f"{sig}_mean"] = np.full(adata.n_obs, np.nan, dtype=np.float32)
                cols[f"{sig}_score"] = np.full(adata.n_obs, np.nan, dtype=np.float32)
                continue

            sub = adata[:, present].X
            sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
            mean = np.asarray(sub, dtype=np.float32).mean(axis=1).ravel()
            cols[f"{sig}_mean"] = mean
            rec[f"{sig}_mean_avg"] = float(np.nanmean(mean))

            try:
                sc.tl.score_genes(adata, present, score_name=f"_{sig}_sc",
                                  ctrl_size=args.ctrl_size, random_state=args.seed)
                cols[f"{sig}_score"] = adata.obs[f"_{sig}_sc"].to_numpy(np.float32)
                rec[f"{sig}_score_avg"] = float(np.nanmean(cols[f"{sig}_score"]))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] score_genes {sig} {name}: {exc}", flush=True)
                cols[f"{sig}_score"] = np.full(adata.n_obs, np.nan, dtype=np.float32)

        pd.DataFrame(cols).to_parquet(out / "per_cell" / f"{name}.parquet",
                                      index=False)
        rows.append(rec)
        print(f"[ok] {name}: n={rec['n_cells']} "
              f"angio={rec['angio_n_present']}/6 teff={rec['teff_n_present']}/5 "
              f"norm={norm} "
              f"angio_mean={rec.get('angio_mean_avg', float('nan')):.4f} "
              f"teff_mean={rec.get('teff_mean_avg', float('nan')):.4f}",
              flush=True)
        del adata

    summ = pd.DataFrame(rows)
    summ_path = out / "signature_score_summary.csv"
    summ.to_csv(summ_path, index=False)
    print("\n===== summary =====", flush=True)
    print(summ.to_string(index=False), flush=True)
    print(f"\nwrote {summ_path}", flush=True)
    print(f"wrote per-cell parquet -> {out / 'per_cell'}", flush=True)


if __name__ == "__main__":
    main()
