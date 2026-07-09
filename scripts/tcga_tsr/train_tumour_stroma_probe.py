#!/usr/bin/env python
"""Train a cross-tissue-robust tumour/stroma/other probe on RCC SIGMMA features.

The 10-class RCC ``level_2_cell_type`` probe transfers poorly to non-renal TCGA
(e.g. rectal "Nephron epithelium" calls). For external-cohort TSR we instead
collapse the renal cell types into THREE coarse groups and train a fresh
LogisticRegression on the SAME z_sigmma_concat feature space:

    tumour = {Tumour epithelium}
    stroma = {Fibroblast, Blood vasculature, Lymphatic vasculature,
              Perivascular cell}
    other  = {Nephron epithelium, B/T lymphocyte, Macrophage, Dendritic cell}
             (normal epithelium + immune)

At inference every cell is forced into one of the three; ``other`` is a reject
bucket EXCLUDED from the TSR denominator (classic Mesker TSR scores tumour vs
stroma only, ignoring inflammatory infiltrate / normal epithelium). This keeps
a clean tumour<->stroma boundary while immune/normal cells don't contaminate
the ratio.

Bundle schema matches scripts/probe_niche_cross_slide.py (scaler/clf/classes_/
n_features) plus a ``tsr_groups`` key consumed by tcga_predict_tsr.py.

Run in the sigmma poetry venv.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
)
from sklearn.preprocessing import StandardScaler

SCALES = ("micro", "meso", "macro")

# Two definitions of the tumour/stroma boundary, selected with --grouping:
#   classic  -- Mesker-style: stroma = connective + vasculature only; immune
#               infiltrate is an "other" reject bucket (the original probe).
#   tutorial -- PMC7887673 H&E TSR protocol: lymphocytic infiltration counts as
#               STROMA, and major-vessel / smooth-muscle-like compartments are
#               excluded. So immune classes move to stroma, Perivascular cell
#               (pericyte / smooth-muscle proxy) moves to other. Minor-vessel
#               endothelium (Blood/Lymphatic vasculature) stays in stroma.
GROUP_MAPS = {
    "classic": {
        "Tumour epithelium": "tumour",
        "Fibroblast": "stroma",
        "Blood vasculature": "stroma",
        "Lymphatic vasculature": "stroma",
        "Perivascular cell": "stroma",
        "Nephron epithelium": "other",
        "B lymphocyte": "other",
        "T lymphocyte": "other",
        "Macrophage": "other",
        "Dendritic cell": "other",
    },
    "tutorial": {
        "Tumour epithelium": "tumour",
        "Fibroblast": "stroma",
        "Blood vasculature": "stroma",
        "Lymphatic vasculature": "stroma",
        "B lymphocyte": "stroma",
        "T lymphocyte": "stroma",
        "Macrophage": "stroma",
        "Dendritic cell": "stroma",
        "Perivascular cell": "other",
        "Nephron epithelium": "other",
    },
    # protocol -- van Pelt / Mesker (Virchows Arch 2018, PMC6182321): the
    # stromal compartment of the TME is everything that is not tumour
    # epithelium and not normal/benign epithelium. Vessels, nerves and
    # pericytes are part of the TME, and TUMOUR-ASSOCIATED INFLAMMATORY cells
    # belong to the stromal compartment too. So immune AND perivascular move to
    # stroma; only normal (Nephron) epithelium is the "other" reject bucket.
    "protocol": {
        "Tumour epithelium": "tumour",
        "Fibroblast": "stroma",
        "Blood vasculature": "stroma",
        "Lymphatic vasculature": "stroma",
        "Perivascular cell": "stroma",
        "B lymphocyte": "stroma",
        "T lymphocyte": "stroma",
        "Macrophage": "stroma",
        "Dendritic cell": "stroma",
        "Nephron epithelium": "other",
    },
}
TSR_GROUPS = {"tumour": ["tumour"], "stroma": ["stroma"]}  # other excluded


def _load_slide(data_h5: Path, sigmma_h5: Path, label_field: str, group_map: dict):
    """Return (X = z_sigmma_concat, y_group) for one slide, cell_id-aligned."""
    with h5py.File(data_h5, "r") as f:
        cid_d = np.array([c.decode() for c in f["cell_id"][:]])
        if label_field not in f:
            return None
        raw = np.array([c.decode() for c in f[label_field][:]])
    with h5py.File(sigmma_h5, "r") as f:
        cid_s = np.array([c.decode() for c in f["cell_id"][:]])
        if cid_s.shape != cid_d.shape or not np.array_equal(cid_s, cid_d):
            raise SystemExit(f"cell_id mismatch: {data_h5.name} vs {sigmma_h5.name}")
        X = np.concatenate([f[f"z_sigmma_{s}"][:] for s in SCALES], axis=1)
    grp = np.array([group_map.get(r, "") for r in raw], dtype=object)
    keep = grp != ""          # drop cells with labels outside the known 10
    return X[keep], grp[keep].astype(str)


def _gather(manifest: pd.DataFrame, split: str, data_dir: Path,
            sigmma_dir: Path, label_field: str, group_map: dict):
    Xs, ys = [], []
    for sample in manifest[manifest["split"] == split]["sample_name"]:
        d = data_dir / f"{sample}.h5"
        s = sigmma_dir / f"{sample}.h5"
        if not d.exists() or not s.exists():
            print(f"  [skip] {sample}: missing cache")
            continue
        res = _load_slide(d, s, label_field, group_map)
        if res is None:
            print(f"  [skip] {sample}: no {label_field}")
            continue
        Xs.append(res[0]); ys.append(res[1])
    if not Xs:
        raise SystemExit(f"no slides loaded for split={split}")
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sigmma_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_bundle", required=True)
    ap.add_argument("--label_field", default="level_2_cell_type")
    ap.add_argument("--grouping", default="classic", choices=sorted(GROUP_MAPS),
                    help="cell-type -> tumour/stroma/other map (classic = "
                         "Mesker; tutorial = PMC7887673, immune in stroma).")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    group_map = GROUP_MAPS[args.grouping]
    print(f"[grouping] {args.grouping}: "
          f"{ {k: group_map[k] for k in sorted(group_map)} }")
    data_dir = Path(args.data_dir); sigmma_dir = Path(args.sigmma_dir)
    man = pd.read_csv(args.manifest)

    print("[load] train")
    X_tr, y_tr = _gather(man, "train", data_dir, sigmma_dir, args.label_field, group_map)
    print("[load] test")
    X_te, y_te = _gather(man, "test", data_dir, sigmma_dir, args.label_field, group_map)
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
    for split, y in (("train", y_tr), ("test", y_te)):
        u, c = np.unique(y, return_counts=True)
        print(f"  {split} groups: {dict(zip(u, c.tolist()))}")

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=args.C,
                             class_weight="balanced", n_jobs=-1)
    clf.fit(scaler.transform(X_tr), y_tr)

    pred = clf.predict(scaler.transform(X_te))
    labels = list(clf.classes_)
    print(f"\n[test] 3-class acc={accuracy_score(y_te, pred):.4f} "
          f"bal_acc={balanced_accuracy_score(y_te, pred):.4f} "
          f"macro_f1={f1_score(y_te, pred, average='macro'):.4f}")
    cm = confusion_matrix(y_te, pred, labels=labels)
    print("  confusion (rows=GT, cols=pred):", labels)
    for r, row in zip(labels, cm):
        print(f"    {r:8s} {row.tolist()}")
    # Binary tumour-vs-stroma view (cells whose GT is tumour or stroma).
    bin_mask = np.isin(y_te, ["tumour", "stroma"])
    yb, pb = y_te[bin_mask], pred[bin_mask]
    pb_in = np.isin(pb, ["tumour", "stroma"])
    print(f"\n[test|tumour-vs-stroma GT cells] n={bin_mask.sum()} "
          f"routed-to-other={int((~pb_in).sum())} "
          f"({100*(~pb_in).mean():.1f}%)")
    keep = pb_in
    if keep.any():
        print(f"  binary acc (excl. routed-to-other)="
              f"{accuracy_score(yb[keep], pb[keep]):.4f} "
              f"bal_acc={balanced_accuracy_score(yb[keep], pb[keep]):.4f}")

    out = Path(args.out_bundle); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "scaler": scaler, "clf": clf,
        "label_field": f"{args.label_field}__tumour_stroma_other",
        "embedding": "z_sigmma_concat",
        "classes_": clf.classes_,
        "n_features": int(X_tr.shape[1]),
        "n_train_cells": int(len(y_tr)),
        "grouping": args.grouping,
        "group_map": group_map,
        "tsr_groups": TSR_GROUPS,
    }, out)
    print(f"\n[bundle] wrote {out}")


if __name__ == "__main__":
    main()
