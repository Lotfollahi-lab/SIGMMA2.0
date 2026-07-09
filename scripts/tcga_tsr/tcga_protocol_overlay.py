#!/usr/bin/env python
"""Visualize the REGION actually used for the protocol TSR on TCGA-KIRC.

tcga_protocol_tsr.py reports, per slide, the stroma fraction of the single MAX
("hotspot") valid x10 field. This script reproduces that EXACT field selection
(3.46mm^2 field, tumour-at-all-four-borders, density gate, follicle gate) and
draws, per slide, two panels:
  (A) whole-slide H&E thumbnail with every VALID candidate field outlined and the
      WINNING max-stroma field (the one whose fraction becomes the slide TSR)
      highlighted -> shows WHERE on the slide TSR was measured;
  (B) a zoom of that winning field: H&E + per-cell dots coloured by Mesker group
      (tumour/stroma/immune/other) + the 0.25 border bands + a caption with the
      field's stroma_fraction and cell counts.

Field-selection logic and defaults are imported from / mirror tcga_protocol_tsr.py
so the drawn field is byte-for-byte the one that fed survival.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_protocol_tsr import (GROUP, _immune_aggregate_mask, _tile_area,
                               _recover_coords)

GCOLOR = {"tumour": "#d62728", "stroma": "#1f77b4",
          "immune": "#2ca02c", "other": "#bbbbbb"}


def select_fields(feat_h5, data_h5, he_zarr, pred, *, tile_px, area_px,
                  max_dist_px, white_thr, border_frac, min_tumour,
                  min_field_area, density_k, density_radius_px,
                  follicle_radius_px, follicle_k, follicle_min,
                  measure_frac=1.0):
    """Reproduce protocol_tsr_for_slide's field selection, but RETURN the per-field
    table + the winning (max-stroma) field so it can be drawn."""
    x, y, matched = _recover_coords(Path(feat_h5), Path(data_h5))
    if matched.sum() == 0:
        return None
    if matched.sum() < len(pred):
        pred = pred[matched]
    grp = np.array([GROUP.get(p, "other") for p in pred], dtype=object)
    is_tum = grp == "tumour"; is_imm = grp == "immune"

    tx = (x // tile_px).astype(np.int64); ty = (y // tile_px).astype(np.int64)
    ox = x - tx * tile_px; oy = y - ty * tile_px
    m = int(border_frac * tile_px)
    df = pd.DataFrame({"tx": tx, "ty": ty, "tum": is_tum,
                       "L": ox < m, "R": ox > tile_px - m,
                       "T": oy < m, "B": oy > tile_px - m})
    g = df[df.tum].groupby(["tx", "ty"])
    border = g[["L", "R", "T", "B"]].any()
    border["ntum"] = g.size()
    mask = (border["L"] & border["R"] & border["T"] & border["B"]
            & (border["ntum"] >= min_tumour)).to_numpy()
    valid = border[mask]

    follicle_fields = set()
    if not valid.empty and follicle_min >= 0:
        imm_xy = np.column_stack([x[is_imm], y[is_imm]])
        agg = _immune_aggregate_mask(imm_xy, follicle_radius_px, follicle_k)
        if agg.any():
            ax = (imm_xy[agg, 0] // tile_px).astype(np.int64)
            ay = (imm_xy[agg, 1] // tile_px).astype(np.int64)
            ac = pd.Series(1, index=pd.MultiIndex.from_arrays([ax, ay])) \
                .groupby(level=[0, 1]).size()
            for a, b in valid.index:
                if ac.get((int(a), int(b)), 0) > follicle_min:
                    follicle_fields.add((int(a), int(b)))
    kept = [ij for ij in valid.index if tuple(map(int, ij)) not in follicle_fields]

    he = zarr.open(he_zarr, mode="r")
    code = np.where(grp == "tumour", 0, np.where(grp == "stroma", 1, 2)).astype(np.int8)
    tree = cKDTree(np.column_stack([x, y]))
    ctile = max(1, int(round(measure_frac * tile_px)))
    off = (tile_px - ctile) // 2
    tum_x, tum_y = x[is_tum], y[is_tum]
    fields = []
    for (txi, tyi) in kept:
        cx0 = int(txi) * tile_px + off; cy0 = int(tyi) * tile_px + off
        if measure_frac < 1.0:
            inwin = int(((tum_x >= cx0) & (tum_x < cx0 + ctile)
                         & (tum_y >= cy0) & (tum_y < cy0 + ctile)).sum())
            if inwin < min_tumour:
                continue
        ta, sa = _tile_area(he, tree, code, cx0, cy0, ctile, area_px,
                            max_dist_px, white_thr, density_k, density_radius_px)
        if ta + sa < min_field_area:
            continue
        fields.append((int(txi), int(tyi), ta, sa, sa / (ta + sa)))
    return dict(x=x, y=y, grp=grp, he=he, m=m, tile_px=tile_px,
                ctile=ctile, off=off,
                fields=fields, follicle_fields=follicle_fields,
                valid_idx=[tuple(map(int, ij)) for ij in valid.index])


def draw(sample, sel, fig_path):
    he = sel["he"]; _, H, W = he.shape
    tile_px = sel["tile_px"]; fields = sel["fields"]
    if not fields:
        print(f"[skip-draw] {sample}: no scorable field"); return
    winner = max(fields, key=lambda f: f[4])      # max stroma_fraction
    wtx, wty, wta, wsa, wfrac = winner

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.4))

    # ---- Panel A: whole-slide thumbnail + field boxes ----
    s = max(1, int(max(H, W) / 1400))
    thumb = np.transpose(he[:, ::s, ::s], (1, 2, 0))
    axA.imshow(thumb)
    for (txi, tyi) in sel["valid_idx"]:
        col = "#ff7f0e" if (txi, tyi) in sel["follicle_fields"] else "#888888"
        axA.add_patch(Rectangle((txi * tile_px / s, tyi * tile_px / s),
                                tile_px / s, tile_px / s, fill=False,
                                ec=col, lw=0.6, alpha=0.7))
    for (txi, tyi, *_rest) in fields:                     # scored fields
        axA.add_patch(Rectangle((txi * tile_px / s, tyi * tile_px / s),
                                tile_px / s, tile_px / s, fill=False,
                                ec="#1f77b4", lw=0.9))
    axA.add_patch(Rectangle((wtx * tile_px / s, wty * tile_px / s),
                            tile_px / s, tile_px / s, fill=False,
                            ec="#d62728", lw=2.4))
    axA.set_title(f"{sample.split('.')[0]}\nvalid={len(sel['valid_idx'])} "
                  f"scored={len(fields)} follicle-excl={len(sel['follicle_fields'])}",
                  fontsize=8)
    axA.axis("off")
    legA = [Line2D([0], [0], color="#d62728", lw=2.4, label="WINNER (slide TSR)"),
            Line2D([0], [0], color="#1f77b4", lw=1.2, label="scored valid field"),
            Line2D([0], [0], color="#888888", lw=1.0, label="valid (4-border)"),
            Line2D([0], [0], color="#ff7f0e", lw=1.0, label="follicle-excluded")]
    axA.legend(handles=legA, fontsize=6, loc="lower right", frameon=False)

    # ---- Panel B: winning field zoom + cells ----
    px0, py0 = wtx * tile_px, wty * tile_px
    px1, py1 = min(px0 + tile_px, W), min(py0 + tile_px, H)
    crop = np.transpose(he[:, py0:py1, px0:px1], (1, 2, 0))
    axB.imshow(crop, extent=[px0, px1, py1, py0])
    x, y, grp = sel["x"], sel["y"], sel["grp"]
    inf = (x >= px0) & (x < px1) & (y >= py0) & (y < py1)
    for gname, gcol in GCOLOR.items():
        sg = inf & (grp == gname)
        if sg.any():
            axB.scatter(x[sg], y[sg], s=3, c=gcol, edgecolors="none", alpha=0.8)
    mm = sel["m"]
    for (rx, ry, rw, rh) in [(px0, py0, px1 - px0, mm), (px0, py1 - mm, px1 - px0, mm),
                             (px0, py0, mm, py1 - py0), (px1 - mm, py0, mm, py1 - py0)]:
        axB.add_patch(Rectangle((rx, ry), rw, rh, color="yellow", alpha=0.12))
    ct = sel.get("ctile", tile_px); ofs = sel.get("off", 0)
    if ct < tile_px:                              # central measurement window
        axB.add_patch(Rectangle((px0 + ofs, py0 + ofs), ct, ct, fill=False,
                                ec="#00cc44", lw=2.0, ls="--"))
    axB.set_xlim(px0, px1); axB.set_ylim(py1, py0)
    msr = "central window" if ct < tile_px else "whole field"
    axB.set_title(f"hotspot field — stroma_frac={wfrac:.3f} ({msr})\n"
                  f"tumour_area={wta} stroma_area={wsa} px "
                  f"(field ~{tile_px*0.5/1000:.2f}mm, measured ~{ct*0.5/1000:.2f}mm)",
                  fontsize=8)
    axB.axis("off")
    legB = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", ms=5, label=g)
            for g, c in GCOLOR.items()]
    legB.append(Line2D([0], [0], color="yellow", lw=4, alpha=0.4,
                       label="0.25 border band"))
    axB.legend(handles=legB, fontsize=6, loc="lower right", frameon=False)

    fig.suptitle("TCGA-KIRC protocol TSR — region used (max-stroma hotspot field)",
                 fontsize=9)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(f"{fig_path}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {fig_path}.png  (winner field=({wtx},{wty}) frac={wfrac:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--per_cell_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tsr_csv", required=True,
                    help="protocol TSR csv to auto-pick representative slides.")
    ap.add_argument("--samples", nargs="*", default=None,
                    help="explicit sample_name stems; overrides auto-pick.")
    ap.add_argument("--fig_dir",
                    default="figures/clinical_outcome/tso_kirc/region_overlays")
    # field/gate params — defaults match the canonical harvest call
    ap.add_argument("--field_um", type=float, default=1860.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--area_um", type=float, default=8.0)
    ap.add_argument("--max_dist_um", type=float, default=50.0)
    ap.add_argument("--white_thr", type=int, default=220)
    ap.add_argument("--border_frac", type=float, default=0.25)
    ap.add_argument("--min_tumour", type=int, default=10)
    ap.add_argument("--min_field_area", type=int, default=200)
    ap.add_argument("--measure_frac", type=float, default=1.0,
                    help="central-window side-fraction to measure (matches "
                         "tcga_protocol_tsr.py --measure_frac). 1.0 = whole field.")
    ap.add_argument("--density_k", type=int, default=5)
    ap.add_argument("--density_radius_um", type=float, default=60.0)
    ap.add_argument("--follicle_radius_um", type=float, default=50.0)
    ap.add_argument("--follicle_k", type=int, default=15)
    ap.add_argument("--follicle_min", type=int, default=30)
    args = ap.parse_args()

    tile_px = int(args.field_um / args.um_per_px)
    area_px = max(1, int(args.area_um / args.um_per_px))
    kw = dict(tile_px=tile_px, area_px=area_px,
              max_dist_px=args.max_dist_um / args.um_per_px,
              white_thr=args.white_thr, border_frac=args.border_frac,
              min_tumour=args.min_tumour, min_field_area=args.min_field_area,
              density_k=args.density_k,
              density_radius_px=args.density_radius_um / args.um_per_px,
              follicle_radius_px=args.follicle_radius_um / args.um_per_px,
              follicle_k=args.follicle_k, follicle_min=args.follicle_min,
              measure_frac=args.measure_frac)

    man = pd.read_csv(args.manifest)
    he_of = {str(r["sample_name"]).strip(): str(r["hematoxylin_eosin_image"]).strip()
             for _, r in man.iterrows()}
    data_of = {str(r["sample_name"]).strip(): str(r["data_h5"]).strip()
               for _, r in man.iterrows()}

    if args.samples:
        samples = args.samples
    else:                                   # auto-pick: low / median / high stroma
        t = pd.read_csv(args.tsr_csv).dropna(subset=["stroma_fraction"])
        t = t.sort_values("stroma_fraction")
        idx = [0, len(t) // 2, len(t) - 1]
        samples = [t.iloc[i]["sample_name"] for i in idx]
        print(f"[auto-pick] low/median/high stroma -> "
              f"{[round(t.iloc[i]['stroma_fraction'], 3) for i in idx]}")

    feat_dir = Path(args.features_dir); pcd = Path(args.per_cell_dir)
    for sample in samples:
        feat_h5 = feat_dir / f"{sample}.h5"
        npz = pcd / f"{sample}.npz"
        data_h5 = data_of.get(sample); he_zarr = he_of.get(sample)
        if not (feat_h5.exists() and npz.exists() and data_h5 and he_zarr
                and Path(data_h5).exists() and Path(he_zarr).exists()):
            print(f"[skip] {sample}: missing inputs"); continue
        pred = np.load(npz, allow_pickle=True)["pred"].astype(str)
        sel = select_fields(feat_h5, data_h5, he_zarr, pred, **kw)
        if sel is None:
            print(f"[skip] {sample}: no matched cells"); continue
        draw(sample, sel, str(Path(args.fig_dir) / sample.split('.')[0]))


if __name__ == "__main__":
    main()
