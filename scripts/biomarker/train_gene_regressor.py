#!/usr/bin/env python
"""Linear-probe (RidgeCV) regression of single-gene biomarker expression (CA9,
PAX8) from per-cell SIGMMA image embeddings, vs frozen UNI2-h (the SIGMMA-value
test). Single-gene sibling of angio_teff/train_score_regressor.py.

Two jobs in one:
  1. INTERNAL FEASIBILITY via LEAVE-ONE-SECTION-OUT CV: for each usable section,
     train RidgeCV on all OTHER sections and predict the held-out section, then
     report the per-section held-out Pearson r (mean+/-std over held-out
     sections). LOSO (not a fixed 4-section split) is used because CA9 is
     ccRCC-specific and several sections are CA9-negative; a single unlucky split
     would rest on ~1 informative section. This is the honest 'can H&E predict
     this biomarker?' answer.
  2. FROZEN PROBE for TCGA: for the sigmma_concat representation, refit each
     target on ALL usable cells and save a joblib bundle {scaler mu/sd, per-
     target RidgeCV model, n_features, keys} for predict_tcga_gene.py.

Robustness handling learned from the data:
  * Targets are decoupled: a NaN in one gene (e.g. CA9 absent from a section's
    panel) does NOT drop that section's cells for the other gene.
  * Sections whose feature cache is truncated (< --min_feat_cells) are excluded
    (CV6-KID-0-FT-5=13 cells, CV9-KID-0-FT-5=1 cell have broken caches in both
    the SIGMMA and UNI extractions).
  * A section only contributes a held-out Pearson for a gene if it has
    >= --min_eval_cells finite cells with non-zero expression variance.

Run in the sigmma poetry venv.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-1, 4, 12)
REPRS = {
    "sigmma_concat": ("sigmma", ["z_sigmma_micro", "z_sigmma_meso", "z_sigmma_macro"]),
    "uni_concat": ("uni", ["z_uni_micro", "z_uni_meso", "z_uni_macro"]),
    "uni_macro": ("uni", ["z_uni_macro"]),
}


def _decode(arr):
    return np.array([c.decode() if isinstance(c, bytes) else str(c) for c in arr])


def _load_section(sample, feat_dir, keys, score_dir, targets, max_cells, seed,
                  min_feat_cells):
    """Return (X, {target: y with NaN kept}, n_feat_cells) or None if unusable."""
    fh = Path(feat_dir) / f"{sample}.h5"
    pq = Path(score_dir) / "per_cell" / f"{sample}.parquet"
    if not fh.exists() or not pq.exists():
        return None
    with h5py.File(fh, "r") as f:
        cid = _decode(f["cell_id"][:])
        if cid.shape[0] < min_feat_cells:
            print(f"    [drop] {sample}: feature cache truncated "
                  f"({cid.shape[0]} < {min_feat_cells} cells)")
            return None
        X = np.concatenate([f[k][:] for k in keys], axis=1).astype(np.float32)
    sdf = pd.read_parquet(pq, columns=["cell_id"] + list(targets))
    sdf["cell_id"] = sdf["cell_id"].astype(str)
    smap = sdf.set_index("cell_id")
    idx = pd.Index(cid)
    present = idx.isin(smap.index)
    if present.sum() == 0:
        print(f"    [warn] {sample}: 0 cell_id join")
        return None
    X = X[present]
    sub = smap.loc[idx[present]]
    y = {t: sub[t].to_numpy(np.float32) for t in targets}  # NaN kept per-target
    n = X.shape[0]
    if max_cells and n > max_cells:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, max_cells, replace=False)
        X = X[sel]
        y = {t: y[t][sel] for t in targets}
    return X, y, int(cid.shape[0])


def _standardize_fit(X):
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = X.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1e-6
    return mu, sd


def _loso(sections, data, target, min_eval_cells):
    """Leave-one-section-out held-out Pearson for one target. data[s] = (X, y)."""
    rows = []
    for held in sections:
        Xh, yh = data[held]
        mh = np.isfinite(yh[target])
        if mh.sum() < min_eval_cells or np.std(yh[target][mh]) < 1e-8:
            continue  # held-out section has no usable/variable expression
        Xtr, ytr = [], []
        for s in sections:
            if s == held:
                continue
            Xs, ys = data[s]
            m = np.isfinite(ys[target])
            if m.sum() == 0:
                continue
            Xtr.append(Xs[m]); ytr.append(ys[target][m])
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        mu, sd = _standardize_fit(Xtr)
        model = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
        pred = model.predict((Xh[mh] - mu) / sd)
        r = float(pearsonr(pred, yh[target][mh])[0]) if np.std(pred) > 0 else float("nan")
        rho = float(spearmanr(pred, yh[target][mh])[0])
        rows.append({"held_section": held, "n_cells": int(mh.sum()),
                     "pearson": r, "spearman": rho, "alpha": float(model.alpha_)})
        print(f"      [{target} LOSO] held={held:20s} n={int(mh.sum()):5d} "
              f"r={r:+.3f} rho={rho:+.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmma_dir", required=True)
    ap.add_argument("--uni_dir", required=True)
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--targets", nargs="+", default=["CA9", "PAX8"])
    ap.add_argument("--bundle_rep", default="sigmma_concat")
    ap.add_argument("--max_cells", type=int, default=5000)
    ap.add_argument("--min_feat_cells", type=int, default=100,
                    help="drop sections whose feature cache is truncated below this")
    ap.add_argument("--min_eval_cells", type=int, default=50,
                    help="held-out section needs this many finite cells for a gene")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    samples = man["sample_name"].tolist()
    targets = list(args.targets)
    dir_map = {"sigmma": args.sigmma_dir, "uni": args.uni_dir}
    print(f"[targets] {targets}  bundle_rep={args.bundle_rep}  "
          f"n_sections={len(samples)}")

    all_rows = []
    summary = []
    for rep, (src, keys) in REPRS.items():
        fdir = dir_map[src]
        print(f"\n===== representation: {rep} ({src}, {len(keys)} keys) =====")
        data, usable = {}, []
        for s in samples:
            loaded = _load_section(s, fdir, keys, args.score_dir, targets,
                                   args.max_cells, args.seed, args.min_feat_cells)
            if loaded is None:
                continue
            X, y, _ = loaded
            data[s] = (X, y)
            usable.append(s)
        print(f"  [usable] {len(usable)}/{len(samples)} sections")

        for t in targets:
            rows = _loso(usable, data, t, args.min_eval_cells)
            for r in rows:
                r.update({"rep": rep, "target": t})
            all_rows.extend(rows)
            rs = [r["pearson"] for r in rows if np.isfinite(r["pearson"])]
            rho = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
            rec = {"rep": rep, "target": t, "n_held_sections": len(rs),
                   "loso_pearson_mean": float(np.mean(rs)) if rs else float("nan"),
                   "loso_pearson_std": float(np.std(rs)) if rs else float("nan"),
                   "loso_spearman_mean": float(np.mean(rho)) if rho else float("nan")}
            summary.append(rec)
            print(f"  [{t:8s} {rep}] LOSO pearson={rec['loso_pearson_mean']:+.3f}"
                  f"+/-{rec['loso_pearson_std']:.3f} (n={len(rs)} sections)")

        # Frozen bundle: refit each target on ALL usable cells for that target.
        if rep == args.bundle_rep:
            Xall = np.concatenate([data[s][0] for s in usable])
            mu, sd = _standardize_fit(Xall)
            Zall = (Xall - mu) / sd
            d_feat = int(Xall.shape[1])
            del Xall
            models = {}
            for t in targets:
                yall = np.concatenate([data[s][1][t] for s in usable])
                m = np.isfinite(yall)
                models[t] = RidgeCV(alphas=ALPHAS).fit(Zall[m], yall[m])
            bundle = {"rep": rep, "keys": keys, "targets": targets,
                      "mu": mu, "sd": sd, "n_features": d_feat,
                      "models": models, "train_sections": usable}
            bpath = out / "bundles" / f"{rep}.joblib"
            joblib.dump(bundle, bpath)
            print(f"\n[bundle] saved frozen probe -> {bpath} "
                  f"(targets={targets}, n_features={d_feat}, "
                  f"n_train_sections={len(usable)})")
            del Zall

    pd.DataFrame(all_rows).to_csv(out / "loso_per_section.csv", index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "gene_regression_summary.csv", index=False)
    with open(out / "gene_regression_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n===== LOSO feasibility summary (section-level Pearson mean+/-std) =====")
    piv = sdf.pivot_table(index="rep", columns="target", values="loso_pearson_mean")
    print(piv.to_string())
    print(f"\nwrote {out / 'gene_regression_summary.csv'}")


if __name__ == "__main__":
    main()
