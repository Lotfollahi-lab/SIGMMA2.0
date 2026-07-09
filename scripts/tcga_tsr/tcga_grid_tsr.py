#!/usr/bin/env python
"""Option-A TSR on a TCGA slide from the TILE-level tumour/stroma map.

Predicts every 64um grid tile's group (tumour/stroma/other) with the tile head,
then scores TSR the van Pelt/Mesker way but on a UNIFORM tile grid -- so the
"area" ratio is just a tile COUNT ratio (every tile is the same area), with NO
Voronoi tessellation and NO density gate (the sparsity problems that broke the
per-cell estimator simply do not exist here).

Per slide:
  * field = a ``field_um`` window (x10 FOV ~1.86mm) tiled over the slide;
  * a field is VALID if predicted-tumour tiles touch all four borders and it has
    >= ``min_tumour`` tumour tiles (tumour-at-all-borders, like the protocol);
  * stroma_frac of a field = n_stroma / (n_tumour + n_stroma) over tiles in a
    CENTRAL sub-window (``measure_frac`` of the side), "other" tiles excluded;
  * slide TSR = MAX over valid fields (hotspot), mean recorded too.

Also writes per-tile predictions (npz) and a whole-slide tumour/stroma map PNG.
Output csv schema matches tcga_survival.py. Run in the sigmma poetry venv.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

SCALES = ("micro", "meso", "macro")
GCOLOR = {"tumour": "#d62728", "stroma": "#1f77b4", "other": "#bbbbbb"}


def _predict(grid_feat, bundle, reject_thr=0.0):
    """Predict tile groups. For a binary tumour/stroma head, tiles whose top
    class probability is < ``reject_thr`` are rejected to 'other' (confidence
    gate) instead of being forced into tumour/stroma."""
    with h5py.File(grid_feat, "r") as f:
        X = np.concatenate([f[f"z_sigmma_{s}"][:] for s in SCALES], axis=1)
    b = joblib.load(bundle)
    Xs = b["scaler"].transform(X)
    pred = np.asarray(b["clf"].predict(Xs), dtype=object)
    if reject_thr > 0.0:
        proba = b["clf"].predict_proba(Xs)
        low = proba.max(axis=1) < reject_thr
        pred[low] = "other"
    return pred


def _field_tsr(gx, gy, grp, *, field_px, surround_px, surround_min, corner_min,
               min_tumour, measure_frac):
    """MAX-hotspot stroma fraction over valid x10 fields (tile counts).

    A field is VALID if predicted-tumour tiles are ABUNDANT in the SURROUNDING
    tiles on all 8 sides -- a ``surround_px`` band just OUTSIDE the field box: 4
    EDGE strips (left/right/top/bottom) each needing >= ``surround_min`` tumour
    tiles AND the 4 CORNER blocks each needing >= ``corner_min`` (smaller floor,
    a thin ring => corner ~1 tile), so the field is embedded in tumour on every
    side AND corner -- and it has >= min_tumour tumour tiles inside the field."""
    is_t = grp == "tumour"; is_s = grp == "stroma"
    fxv = (gx // field_px).astype(np.int64); fyv = (gy // field_px).astype(np.int64)
    lo = (1 - measure_frac) / 2; hi = (1 + measure_frac) / 2
    fracs = []
    for (fxi, fyi) in sorted(set(zip(fxv.tolist(), fyv.tolist()))):
        x0 = fxi * field_px; x1 = x0 + field_px
        y0 = fyi * field_px; y1 = y0 + field_px
        in_f = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
        if int(is_t[in_f].sum()) < min_tumour:
            continue
        xL = (gx >= x0 - surround_px) & (gx < x0)
        xR = (gx >= x1) & (gx < x1 + surround_px)
        yT = (gy >= y0 - surround_px) & (gy < y0)
        yB = (gy >= y1) & (gy < y1 + surround_px)
        yband = (gy >= y0) & (gy < y1); xband = (gx >= x0) & (gx < x1)
        sides = [is_t[xL & yband].sum(), is_t[xR & yband].sum(),
                 is_t[yT & xband].sum(), is_t[yB & xband].sum()]
        corners = [is_t[xL & yT].sum(), is_t[xR & yT].sum(),
                   is_t[xL & yB].sum(), is_t[xR & yB].sum()]
        if min(sides) < surround_min or min(corners) < corner_min:
            continue
        cin = (in_f & (gx >= x0 + lo * field_px) & (gx < x0 + hi * field_px)
               & (gy >= y0 + lo * field_px) & (gy < y0 + hi * field_px))
        nt, ns = int(is_t[cin].sum()), int(is_s[cin].sum())
        if nt + ns == 0:
            continue
        fracs.append((ns / (nt + ns), nt, ns))
    if not fracs:
        return None
    fr = max(fracs, key=lambda z: z[0])
    return dict(stroma_fraction=fr[0], n_tumour_tiles=fr[1], n_stroma_tiles=fr[2],
                n_valid_fields=len(fracs),
                stroma_fraction_mean=float(np.mean([f[0] for f in fracs])))


def _draw_map(gx, gy, grp, he_zarr, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import zarr
    he = zarr.open(he_zarr, mode="r"); _, H, W = he.shape
    s = max(1, int(max(H, W) / 1600))
    thumb = np.transpose(he[:, ::s, ::s], (1, 2, 0))
    fig, ax = plt.subplots(figsize=(7, 7 * H / W))
    ax.imshow(thumb)
    for gname, gcol in GCOLOR.items():
        sel = grp == gname
        if sel.any():
            ax.scatter(gx[sel] / s, gy[sel] / s, s=2, c=gcol,
                       edgecolors="none", alpha=0.55, label=gname)
    ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", mfc=c, mec="none",
                              ms=7, label=g) for g, c in GCOLOR.items()],
              fontsize=7, loc="lower right", frameon=False)
    n = len(gx)
    ax.set_title(f"tile tumour/stroma map  (n={n} tiles, "
                 f"tum {100*(grp=='tumour').mean():.0f}% / "
                 f"str {100*(grp=='stroma').mean():.0f}% / "
                 f"oth {100*(grp=='other').mean():.0f}%)", fontsize=8)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{fig_path}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] {fig_path}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_data_dir", required=True)
    ap.add_argument("--grid_feat_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--per_tile_dir", default=None)
    ap.add_argument("--fig_dir", default=None, help="if set, draw a map per slide.")
    ap.add_argument("--tile_um", type=float, default=64.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--field_um", type=float, default=1860.0)
    ap.add_argument("--border_frac", type=float, default=0.25,
                    help="(unused) kept for CLI compat; validity is now tumour in "
                         "the 4-side SURROUNDING band, not border bands.")
    ap.add_argument("--surround_um", type=float, default=64.0,
                    help="width of the band just OUTSIDE the field per side.")
    ap.add_argument("--surround_min", type=int, default=5,
                    help="min predicted-tumour tiles required in EACH of the 4 "
                         "side bands (>1 = field embedded in abundant tumour).")
    ap.add_argument("--corner_min", type=int, default=1,
                    help="min predicted-tumour tiles required in EACH of the 4 "
                         "corner blocks (smaller floor; thin ring => ~1 tile).")
    ap.add_argument("--min_tumour", type=int, default=10)
    ap.add_argument("--measure_frac", type=float, default=0.5)
    ap.add_argument("--reject_thr", type=float, default=0.0,
                    help="binary head: reject tiles with top class prob < thr to "
                         "'other' (0 = no confidence gate).")
    ap.add_argument("--slides", nargs="*", default=None)
    args = ap.parse_args()

    field_px = int(round(args.field_um / args.um_per_px))
    surround_px = int(round(args.surround_um / args.um_per_px))
    man = pd.read_csv(args.manifest)
    if args.slides:
        man = man[man["sample_name"].isin(args.slides)]
    gdd, gfd = Path(args.grid_data_dir), Path(args.grid_feat_dir)

    rows = []
    for _, r in man.iterrows():
        s = r["sample_name"]
        gd = gdd / f"{s}.h5"; gf = gfd / f"{s}.h5"
        if not (gd.exists() and gf.exists()):
            print(f"[skip] {s}: missing grid cache"); continue
        with h5py.File(gd, "r") as f:
            gx, gy = f["x_pixel"][:], f["y_pixel"][:]
        grp = _predict(gf, args.bundle, reject_thr=args.reject_thr)
        if args.per_tile_dir:
            Path(args.per_tile_dir).mkdir(parents=True, exist_ok=True)
            np.savez(Path(args.per_tile_dir) / f"{s}.npz", x=gx, y=gy, group=grp)
        res = _field_tsr(gx, gy, grp, field_px=field_px,
                         surround_px=surround_px, surround_min=args.surround_min,
                         corner_min=args.corner_min, min_tumour=args.min_tumour,
                         measure_frac=args.measure_frac)
        row = {"sample_name": s, "n_tiles": int(len(gx)),
               "n_cells": int(len(gx)),  # tile analogue of cell count (tcga_survival.py --min_cells/agg)
               "tile_tumour_frac": float((grp == "tumour").mean()),
               "tile_stroma_frac": float((grp == "stroma").mean()),
               "tile_other_frac": float((grp == "other").mean())}
        row.update(res if res else {"stroma_fraction": np.nan, "n_valid_fields": 0})
        rows.append(row)
        print(f"[{s.split('.')[0]}] tiles={len(gx)} "
              f"valid_fields={row.get('n_valid_fields')} "
              f"stroma_frac={row.get('stroma_fraction')}")
        if args.fig_dir:
            _draw_map(gx, gy, grp, r["hematoxylin_eosin_image"],
                      str(Path(args.fig_dir) / s.split('.')[0]))

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[grid-tsr] {len(rows)} slides -> {out}")


if __name__ == "__main__":
    main()
