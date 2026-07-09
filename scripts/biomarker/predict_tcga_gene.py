#!/usr/bin/env python
"""Apply the frozen RCC-trained CA9/PAX8 Ridge probe to TCGA-KIRC SIGMMA
embeddings -> per-slide and per-patient predicted biomarker expression.

Single-gene regression sibling of tcga_tsr/tcga_predict_tsr.py (which applies a
LogReg cell-type bundle). Here the bundle is train_gene_regressor.py's
sigmma_concat joblib: {mu, sd, n_features, keys, targets, models[target]}.

For each of the 519 cached TCGA feature HDF5s we build z_sigmma_concat, apply the
SAME standardisation (train mu/sd), predict per-nucleus expression for each
target, and aggregate to the slide:
    <gene>_mean : slide mean over all nuclei
    <gene>_q90  : mean of the top-decile nuclei ("hotspot"; CA9/PAX8 are focal
                  tumour markers, so a slide mean over stroma/immune dilutes the
                  signal). Both are reported; the severity step can pick.

Slides roll up to patients: case_submitter_id = first 3 barcode fields
(TCGA-XX-XXXX), aggregated by mean across a patient's slides.

NOTE: TCGA has no matched ST, so these predictions cannot be validated against
ground-truth expression here — validation is the internal held-out RCC Pearson
(train_gene_regressor.py) plus the downstream severity association.

Run in the sigmma poetry venv.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd


def load_concat(h5_path: Path, keys) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        parts = [f[k][:] for k in keys]
    return np.concatenate(parts, axis=1)


def case_id(sample_name: str) -> str:
    return "-".join(sample_name.split("-")[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--bundle", required=True, help="sigmma_concat joblib")
    ap.add_argument("--out_slide_csv", required=True)
    ap.add_argument("--out_patient_csv", required=True)
    ap.add_argument("--q_hotspot", type=float, default=0.90)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    mu, sd = bundle["mu"], bundle["sd"]
    keys = bundle["keys"]
    targets = bundle["targets"]
    models = bundle["models"]
    nfeat = int(bundle["n_features"])
    print(f"[probe] {args.bundle}  n_features={nfeat}  targets={targets}  keys={keys}")

    rows = []
    feat_dir = Path(args.features_dir)
    for h5 in sorted(feat_dir.glob("*.h5")):
        sample = h5.stem
        X = load_concat(h5, keys)
        if X.shape[1] != nfeat:
            print(f"[skip] {sample}: feat dim {X.shape[1]} != {nfeat}")
            continue
        Z = (X.astype(np.float32) - mu) / sd
        row = {"sample_name": sample, "case_submitter_id": case_id(sample),
               "n_cells": int(X.shape[0])}
        for t in targets:
            pred = models[t].predict(Z)
            row[f"{t}_mean"] = float(np.mean(pred))
            thr = np.quantile(pred, args.q_hotspot)
            hot = pred[pred >= thr]
            row[f"{t}_q90"] = float(np.mean(hot)) if hot.size else float(np.mean(pred))
        rows.append(row)
        msg = " ".join(f"{t}_mean={row[f'{t}_mean']:.3f}" for t in targets)
        print(f"  {sample}: N={row['n_cells']} {msg}", flush=True)

    slide = pd.DataFrame(rows)
    Path(args.out_slide_csv).parent.mkdir(parents=True, exist_ok=True)
    slide.to_csv(args.out_slide_csv, index=False)
    print(f"\n[slide] {len(slide)} slides -> {args.out_slide_csv}")

    agg = {c: "mean" for c in slide.columns
           if c not in ("sample_name", "case_submitter_id")}
    patient = (slide.groupby("case_submitter_id", as_index=False)
               .agg({**agg, "sample_name": "count"})
               .rename(columns={"sample_name": "n_slides"}))
    patient.to_csv(args.out_patient_csv, index=False)
    print(f"[patient] {len(patient)} patients -> {args.out_patient_csv}")
    for t in targets:
        print(f"  {t}_mean: {patient[f'{t}_mean'].describe().to_dict()}")


if __name__ == "__main__":
    main()
