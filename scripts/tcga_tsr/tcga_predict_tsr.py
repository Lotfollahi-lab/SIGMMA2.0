#!/usr/bin/env python
"""Stage 4: apply the RCC cell-type probe to TCGA z_sigmma features -> TSR.

For each per-slide SIGMMA feature HDF5, build z_sigmma_concat =
[micro|meso|macro] (1536d), standard-scale, predict cell type with the
frozen LogisticRegression bundle (no retraining), then aggregate to a
tumour-stroma ratio per slide.

TSR convention (user choice "Tumour-epi only = tumour"):
    tumour = {Tumour epithelium}
    stroma = {Fibroblast, Blood vasculature, Lymphatic vasculature,
              Perivascular cell}
    stroma_fraction = N_stroma / (N_tumour + N_stroma)   # classic TSR
Cells of other classes (Nephron epithelium, immune) are reported but
excluded from the ratio denominator.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

SCALES = ("micro", "meso", "macro")
# Default tumour/stroma sets for the legacy 10-class RCC level_2_cell_type
# probe. A bundle trained by train_tumour_stroma_probe.py instead carries a
# ``tsr_groups`` key ({"tumour": [...], "stroma": [...]}) whose predicted
# group names ("tumour"/"stroma", everything else = reject) override these.
TUMOUR = {"Tumour epithelium"}
STROMA = {"Fibroblast", "Blood vasculature", "Lymphatic vasculature",
          "Perivascular cell"}


def load_concat(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        parts = [f[f"z_sigmma_{s}"][:] for s in SCALES]
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--bundle", required=True, help="z_sigmma_concat joblib")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--per_cell_dir", default=None,
                    help="optional: write per-cell predictions npz here")
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    scaler, clf, classes = bundle["scaler"], bundle["clf"], list(bundle["classes_"])
    nfeat = int(bundle["n_features"])
    groups = bundle.get("tsr_groups")
    if groups:
        tumour_set = set(groups["tumour"]); stroma_set = set(groups["stroma"])
    else:
        tumour_set, stroma_set = TUMOUR, STROMA
    print(f"[probe] {args.bundle}  n_features={nfeat}  classes={len(classes)}")
    print(f"[probe] tumour={sorted(tumour_set)}  stroma={sorted(stroma_set)}  "
          f"(other classes excluded from TSR denominator)")

    rows = []
    feat_dir = Path(args.features_dir)
    for h5 in sorted(feat_dir.glob("*.h5")):
        sample = h5.stem
        X = load_concat(h5)
        if X.shape[1] != nfeat:
            print(f"[skip] {sample}: feat dim {X.shape[1]} != {nfeat}")
            continue
        pred = clf.predict(scaler.transform(X))
        counts = pd.Series(pred).value_counts()
        n_tum = int(sum(counts.get(c, 0) for c in tumour_set))
        n_str = int(sum(counts.get(c, 0) for c in stroma_set))
        denom = n_tum + n_str
        stroma_frac = n_str / denom if denom > 0 else np.nan
        row = {"sample_name": sample, "n_cells": int(len(pred)),
               "n_tumour": n_tum, "n_stroma": n_str,
               "stroma_fraction": stroma_frac,
               "tsr_ratio": (n_tum / n_str) if n_str > 0 else np.nan}
        for c in classes:
            row[f"n__{c}"] = int(counts.get(c, 0))
        rows.append(row)
        print(f"  {sample}: N={len(pred)} tum={n_tum} str={n_str} "
              f"stroma_frac={stroma_frac:.3f}", flush=True)
        if args.per_cell_dir:
            pcd = Path(args.per_cell_dir); pcd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pcd / f"{sample}.npz", pred=pred.astype("U32"))

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[tsr] {len(df)} slides -> {args.out_csv}")
    if len(df):
        print(df[["sample_name", "n_cells", "stroma_fraction"]].describe(include="all"))


if __name__ == "__main__":
    main()
