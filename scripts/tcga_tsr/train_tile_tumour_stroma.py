#!/usr/bin/env python
"""Train a TILE-level tumour/stroma/other head on RCC grid SIGMMA embeddings.

Option-A counterpart to ``train_tumour_stroma_probe.py`` (which is per-cell).
Here every training example is a 64um GRID TILE, not a nucleus:

  X = z_sigmma_concat of the tile (full-square pooled embedding, identical to
      what TCGA grid inference produces);
  y = majority tumour/stroma/other group of the Xenium cells that fall inside
      the tile (per-cell ``level_2_cell_type`` from the RCC data_h5, mapped
      through the same GROUP_MAPS as the per-cell probe).

A tile is kept for training only if it contains >= ``--min_cells`` labelled
cells (so the majority vote is meaningful). ``other`` (normal epithelium +,
under the chosen grouping, immune) stays a 3rd class and is excluded from the
TSR denominator at inference, exactly like the per-cell probe.

Tile<->cell alignment: grid tiles were placed at multiples of ``tile_px`` with
centre ``tx*tile_px + tile_px//2``; a cell at (cx,cy) maps to tile
(cx//tile_px, cy//tile_px), and a grid-tile centre maps back to the same key by
floor-division -- so the two are joined on integer tile coordinates, no cell_id.

Run in the sigmma poetry venv (sklearn + h5py).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tumour_stroma_probe import GROUP_MAPS, TSR_GROUPS  # reuse maps

SCALES = ("micro", "meso", "macro")


def _tile_key(x, y, tile_px):
    return (np.asarray(x) // tile_px).astype(np.int64), \
           (np.asarray(y) // tile_px).astype(np.int64)


def _section_tiles(grid_data, grid_feat, cell_data, *, tile_px, label_field,
                   group_map, min_cells):
    """Return (X tile embeddings, y tile group) for one section."""
    # tile positions + embedding (cell_id order is identical within a file)
    with h5py.File(grid_data, "r") as f:
        gx, gy = f["x_pixel"][:], f["y_pixel"][:]
    with h5py.File(grid_feat, "r") as f:
        X = np.concatenate([f[f"z_sigmma_{s}"][:] for s in SCALES], axis=1)
    if len(gx) != len(X):
        raise SystemExit(f"tile count mismatch {grid_data} vs {grid_feat}")
    gtx, gty = _tile_key(gx, gy, tile_px)

    # per-cell labels -> tile majority group
    with h5py.File(cell_data, "r") as f:
        if label_field not in f:
            return None
        cx, cy = f["x_pixel"][:], f["y_pixel"][:]
        raw = np.array([c.decode() for c in f[label_field][:]])
    grp = np.array([group_map.get(r, "") for r in raw], dtype=object)
    keep = grp != ""
    cx, cy, grp = cx[keep], cy[keep], grp[keep].astype(str)
    ctx, cty = _tile_key(cx, cy, tile_px)

    groups = ["tumour", "stroma", "other"]
    gidx = {g: i for i, g in enumerate(groups)}
    # count of each group per (tx,ty) using a dict
    from collections import defaultdict
    counts = defaultdict(lambda: np.zeros(3, np.int64))
    for tx, ty, g in zip(ctx, cty, grp):
        counts[(int(tx), int(ty))][gidx[g]] += 1

    Xs, ys = [], []
    for i, (tx, ty) in enumerate(zip(gtx, gty)):
        c = counts.get((int(tx), int(ty)))
        if c is None or c.sum() < min_cells:
            continue
        Xs.append(X[i]); ys.append(groups[int(c.argmax())])
    if not Xs:
        return None
    return np.asarray(Xs), np.asarray(ys, dtype=object)


def _gather(man, split, grid_data_dir, grid_feat_dir, cell_data_dir, **kw):
    Xs, ys = [], []
    for s in man[man["split"] == split]["sample_name"]:
        gd = grid_data_dir / f"{s}.h5"; gf = grid_feat_dir / f"{s}.h5"
        cd = cell_data_dir / f"{s}.h5"
        if not (gd.exists() and gf.exists() and cd.exists()):
            print(f"  [skip] {s}: missing grid/cell cache"); continue
        res = _section_tiles(gd, gf, cd, **kw)
        if res is None:
            print(f"  [skip] {s}: no usable tiles"); continue
        Xs.append(res[0]); ys.append(res[1])
        print(f"  {s}: {len(res[1])} tiles")
    if not Xs:
        raise SystemExit(f"no tiles for split={split}")
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_data_dir", required=True)
    ap.add_argument("--grid_feat_dir", required=True)
    ap.add_argument("--cell_data_dir", required=True,
                    help="per-cell RCC data_h5 dir (x_pixel,y_pixel,label).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_bundle", required=True)
    ap.add_argument("--label_field", default="level_2_cell_type")
    ap.add_argument("--grouping", default="classic", choices=sorted(GROUP_MAPS))
    ap.add_argument("--binary", action="store_true",
                    help="train a 2-class tumour-vs-stroma head: keep only tiles "
                         "whose majority group is tumour or stroma (drop 'other'). "
                         "At inference, ambiguous tiles are rejected by a "
                         "confidence threshold instead of a learned 'other' class.")
    ap.add_argument("--tile_um", type=float, default=64.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--min_cells", type=int, default=4,
                    help="min labelled cells in a tile to assign a majority label.")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    tile_px = int(round(args.tile_um / args.um_per_px))
    group_map = GROUP_MAPS[args.grouping]
    kw = dict(tile_px=tile_px, label_field=args.label_field,
              group_map=group_map, min_cells=args.min_cells)
    man = pd.read_csv(args.manifest)
    gdd, gfd, cdd = (Path(args.grid_data_dir), Path(args.grid_feat_dir),
                     Path(args.cell_data_dir))

    print(f"[grouping={args.grouping} tile={args.tile_um}um min_cells={args.min_cells} "
          f"binary={args.binary}]")
    print("[load] train"); X_tr, y_tr = _gather(man, "train", gdd, gfd, cdd, **kw)
    print("[load] test");  X_te, y_te = _gather(man, "test", gdd, gfd, cdd, **kw)
    if args.binary:
        # keep only genuinely tumour-or-stroma tiles; ambiguous 'other' tiles are
        # handled at inference by a confidence threshold, not a learned class.
        mtr = np.isin(y_tr, ["tumour", "stroma"]); mte = np.isin(y_te, ["tumour", "stroma"])
        X_tr, y_tr, X_te, y_te = X_tr[mtr], y_tr[mtr], X_te[mte], y_te[mte]
        print(f"  [binary] dropped 'other' tiles -> train {int(mtr.sum())} "
              f"test {int(mte.sum())}")
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
    for sp, y in (("train", y_tr), ("test", y_te)):
        u, c = np.unique(y, return_counts=True)
        print(f"  {sp} tiles: {dict(zip(u, c.tolist()))}")

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
    bmask = np.isin(y_te, ["tumour", "stroma"])
    pin = np.isin(pred[bmask], ["tumour", "stroma"])
    print(f"\n[test|tumour-vs-stroma tiles] n={int(bmask.sum())} "
          f"routed-to-other={int((~pin).sum())} ({100*(~pin).mean():.1f}%)")
    if pin.any():
        yb, pb = y_te[bmask][pin], pred[bmask][pin]
        print(f"  binary acc={accuracy_score(yb, pb):.4f} "
              f"bal_acc={balanced_accuracy_score(yb, pb):.4f}")

    out = Path(args.out_bundle); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "scaler": scaler, "clf": clf,
        "label_field": f"{args.label_field}__tile_tumour_stroma_other",
        "embedding": "z_sigmma_concat", "mask_mode": "full_square_tile",
        "classes_": clf.classes_, "n_features": int(X_tr.shape[1]),
        "n_train_tiles": int(len(y_tr)), "tile_um": float(args.tile_um),
        "min_cells": args.min_cells, "grouping": args.grouping,
        "binary": bool(args.binary),
        "group_map": group_map, "tsr_groups": TSR_GROUPS,
    }, out)
    print(f"\n[bundle] wrote {out}")


if __name__ == "__main__":
    main()
