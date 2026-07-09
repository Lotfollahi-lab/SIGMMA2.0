#!/usr/bin/env python
"""Option-A (tile-level) version of tcga_protocol_overlay.py.

Predicts every 64um grid tile's group (tumour/stroma/other) with the tile head,
reproduces the EXACT MAX-hotspot field selection of tcga_grid_tsr.py, and draws
two panels in the same style as the per-cell region_overlays:
  (A) whole-slide H&E thumbnail with valid candidate fields + scored fields + the
      WINNING max-stroma field (whose tile-count fraction becomes the slide TSR);
  (B) a zoom of the winning field: H&E + per-TILE dots coloured by group, the
      0.25 border bands and the central measurement window, with a caption
      giving stroma_fraction and tumour/stroma tile counts.

Self-contained: loads grid embeddings + grid coords + the tile-head bundle, so
no separate predict step is needed. Run in the sigmma poetry venv (or .venv with
zarr+joblib+sklearn).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

SCALES = ("micro", "meso", "macro")
GCOLOR = {"tumour": "#d62728", "stroma": "#1f77b4", "other": "#bbbbbb"}


def predict_tiles(grid_feat, bundle, reject_thr=0.0):
    with h5py.File(grid_feat, "r") as f:
        X = np.concatenate([f[f"z_sigmma_{s}"][:] for s in SCALES], axis=1)
    b = joblib.load(bundle)
    Xs = b["scaler"].transform(X)
    pred = np.asarray(b["clf"].predict(Xs), dtype=object)
    if reject_thr > 0.0:
        low = b["clf"].predict_proba(Xs).max(axis=1) < reject_thr
        pred[low] = "other"
    return pred


def select_fields(gx, gy, grp, *, field_px, surround_px, surround_min,
                  corner_min, min_tumour, measure_frac):
    """Mirror tcga_grid_tsr._field_tsr but RETURN the per-field table + winner.

    VALIDITY = predicted-tumour tiles abundant in the SURROUNDING tiles on all 8
    sides -- a ``surround_px`` band just OUTSIDE the field box: 4 EDGE strips
    (left/right/top/bottom) each needing >= ``surround_min`` tumour tiles, AND the
    4 CORNER blocks (NW/NE/SW/SE diagonal-outside) each needing >= ``corner_min``
    tumour tiles -- so the field is embedded in tumour on every side AND corner.
    (corner_min is a smaller floor since a thin ring makes corners ~1 tile.) Also
    require >= min_tumour tumour tiles inside the field."""
    is_t = grp == "tumour"; is_s = grp == "stroma"
    fxv = (gx // field_px).astype(np.int64); fyv = (gy // field_px).astype(np.int64)
    cells = sorted(set(zip(fxv.tolist(), fyv.tolist())))
    lo = (1 - measure_frac) / 2; hi = (1 + measure_frac) / 2
    valid_idx, scored = [], []
    for (fxi, fyi) in cells:
        x0 = fxi * field_px; x1 = x0 + field_px
        y0 = fyi * field_px; y1 = y0 + field_px
        in_f = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
        if int(is_t[in_f].sum()) < min_tumour:
            continue
        xL = (gx >= x0 - surround_px) & (gx < x0)        # left  outside column
        xR = (gx >= x1) & (gx < x1 + surround_px)        # right outside column
        yT = (gy >= y0 - surround_px) & (gy < y0)        # top    outside row
        yB = (gy >= y1) & (gy < y1 + surround_px)        # bottom outside row
        yband = (gy >= y0) & (gy < y1); xband = (gx >= x0) & (gx < x1)
        sides = [is_t[xL & yband].sum(), is_t[xR & yband].sum(),
                 is_t[yT & xband].sum(), is_t[yB & xband].sum()]
        corners = [is_t[xL & yT].sum(), is_t[xR & yT].sum(),
                   is_t[xL & yB].sum(), is_t[xR & yB].sum()]
        if min(sides) < surround_min or min(corners) < corner_min:
            continue
        valid_idx.append((fxi, fyi))
        cin = (in_f & (gx >= x0 + lo * field_px) & (gx < x0 + hi * field_px)
               & (gy >= y0 + lo * field_px) & (gy < y0 + hi * field_px))
        nt, ns = int(is_t[cin].sum()), int(is_s[cin].sum())
        if nt + ns == 0:
            continue
        scored.append((fxi, fyi, nt, ns, ns / (nt + ns)))
    return valid_idx, scored


def draw(sample, gx, gy, grp, he_zarr, *, field_px, surround_px, measure_frac,
         valid_idx, scored, fig_path):
    he = zarr.open(he_zarr, mode="r"); _, H, W = he.shape
    if not scored:
        print(f"[skip-draw] {sample}: no scored field"); return
    winner = max(scored, key=lambda f: f[4])
    wfx, wfy, wnt, wns, wfrac = winner

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.4))

    # ---- Panel A: whole-slide thumbnail + field boxes ----
    s = max(1, int(max(H, W) / 1400))
    thumb = np.transpose(he[:, ::s, ::s], (1, 2, 0))
    axA.imshow(thumb)
    for (fxi, fyi) in valid_idx:
        axA.add_patch(Rectangle((fxi * field_px / s, fyi * field_px / s),
                                field_px / s, field_px / s, fill=False,
                                ec="#888888", lw=0.6, alpha=0.7))
    for (fxi, fyi, *_r) in scored:
        axA.add_patch(Rectangle((fxi * field_px / s, fyi * field_px / s),
                                field_px / s, field_px / s, fill=False,
                                ec="#1f77b4", lw=0.9))
    axA.add_patch(Rectangle((wfx * field_px / s, wfy * field_px / s),
                            field_px / s, field_px / s, fill=False,
                            ec="#d62728", lw=2.4))
    axA.set_title(f"{sample.split('.')[0]}\n"
                  f"valid={len(valid_idx)} scored={len(scored)}", fontsize=8)
    axA.axis("off")
    legA = [Line2D([0], [0], color="#d62728", lw=2.4, label="WINNER (slide TSR)"),
            Line2D([0], [0], color="#1f77b4", lw=1.2, label="scored valid field"),
            Line2D([0], [0], color="#888888", lw=1.0, label="valid (4-side surround)")]
    axA.legend(handles=legA, fontsize=6, loc="lower right", frameon=False)

    # ---- Panel B: winning field zoom (+ surrounding band) + per-tile dots ----
    fx0, fy0 = wfx * field_px, wfy * field_px           # field box
    fx1, fy1 = fx0 + field_px, fy0 + field_px
    # expand the crop by the surround band so the 4-side surrounding tiles show
    px0 = max(fx0 - surround_px, 0); py0 = max(fy0 - surround_px, 0)
    px1 = min(fx1 + surround_px, W); py1 = min(fy1 + surround_px, H)
    crop = np.transpose(he[:, py0:py1, px0:px1], (1, 2, 0))
    axB.imshow(crop, extent=[px0, px1, py1, py0])
    inf = (gx >= px0) & (gx < px1) & (gy >= py0) & (gy < py1)
    for gname, gcol in GCOLOR.items():
        sg = inf & (grp == gname)
        if sg.any():
            axB.scatter(gx[sg], gy[sg], s=10, c=gcol, marker="s",
                        edgecolors="none", alpha=0.6)
    # the field box (validity = tumour in the band just OUTSIDE this box, 4 sides)
    axB.add_patch(Rectangle((fx0, fy0), field_px, field_px, fill=False,
                            ec="yellow", lw=1.6))
    if measure_frac < 1.0:                     # central measurement window
        cw = measure_frac * field_px; ofs = (1 - measure_frac) / 2 * field_px
        axB.add_patch(Rectangle((fx0 + ofs, fy0 + ofs), cw, cw, fill=False,
                                ec="#00cc44", lw=2.0, ls="--"))
    axB.set_xlim(px0, px1); axB.set_ylim(py1, py0)
    msr = "central window" if measure_frac < 1.0 else "whole field"
    axB.set_title(f"hotspot field — stroma_frac={wfrac:.3f} ({msr})\n"
                  f"tumour_tiles={wnt} stroma_tiles={wns} "
                  f"(field ~{field_px*0.5/1000:.2f}mm)", fontsize=8)
    axB.axis("off")
    legB = [Line2D([0], [0], marker="s", ls="", mfc=c, mec="none", ms=6, label=g)
            for g, c in GCOLOR.items()]
    legB.append(Line2D([0], [0], color="yellow", lw=1.6, label="field box"))
    axB.legend(handles=legB, fontsize=6, loc="lower right", frameon=False)

    fig.suptitle("TCGA-KIRC tile-level (Option A) TSR — region used "
                 "(max-stroma hotspot field)", fontsize=9)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{fig_path}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {fig_path}.png  winner=({wfx},{wfy}) frac={wfrac:.3f} "
          f"tum_tiles={wnt} str_tiles={wns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_data", required=True, help="grid64 data h5 (tile x/y)")
    ap.add_argument("--grid_feat", required=True, help="grid64 SIGMMA feature h5")
    ap.add_argument("--he_zarr", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--fig_path", required=True, help="output stem (no ext)")
    ap.add_argument("--field_um", type=float, default=1860.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--border_frac", type=float, default=0.25,
                    help="(unused) kept for CLI compat; validity is now "
                         "tumour in the 4-side SURROUNDING band, not border bands.")
    ap.add_argument("--surround_um", type=float, default=64.0,
                    help="width of the band JUST OUTSIDE the field, per side.")
    ap.add_argument("--surround_min", type=int, default=5,
                    help="min predicted-tumour tiles required in EACH of the 4 "
                         "side bands (>1 = field embedded in abundant tumour).")
    ap.add_argument("--corner_min", type=int, default=1,
                    help="min predicted-tumour tiles required in EACH of the 4 "
                         "corner blocks (smaller floor; thin ring => ~1 tile).")
    ap.add_argument("--min_tumour", type=int, default=10)
    ap.add_argument("--measure_frac", type=float, default=0.5)
    ap.add_argument("--reject_thr", type=float, default=0.75)
    args = ap.parse_args()

    field_px = int(round(args.field_um / args.um_per_px))   # int: zarr slice indices
    surround_px = int(round(args.surround_um / args.um_per_px))
    with h5py.File(args.grid_data, "r") as f:
        gx, gy = f["x_pixel"][:], f["y_pixel"][:]
    grp = predict_tiles(args.grid_feat, args.bundle, reject_thr=args.reject_thr)
    if len(grp) != len(gx):
        raise SystemExit(f"length mismatch: {len(grp)} preds vs {len(gx)} tiles")
    u, c = np.unique(grp, return_counts=True)
    print(f"[{args.sample}] {len(gx)} tiles  " +
          "  ".join(f"{k}={v}({100*v/len(grp):.1f}%)" for k, v in zip(u, c)))
    valid_idx, scored = select_fields(
        gx, gy, grp, field_px=field_px, surround_px=surround_px,
        surround_min=args.surround_min, corner_min=args.corner_min,
        min_tumour=args.min_tumour, measure_frac=args.measure_frac)
    draw(args.sample, gx, gy, grp, args.he_zarr, field_px=field_px,
         surround_px=surround_px, measure_frac=args.measure_frac,
         valid_idx=valid_idx, scored=scored, fig_path=args.fig_path)


if __name__ == "__main__":
    main()
