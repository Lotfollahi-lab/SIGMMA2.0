#!/usr/bin/env python
"""Protocol-faithful TSR per the van Pelt / Mesker standard (Virchows Arch 2018,
PMC6182321), as opposed to the slide-wide MEAN that tcga_voronoi_tsr.py computes.

The clinical procedure does NOT average stroma over the whole tumour. It:
  - scans at low mag for the stroma HOTSPOT,
  - scores in a single x10 field where TUMOUR CELLS ARE PRESENT AT ALL FOUR
    BORDERS of the field of view (so the field sits inside the tumour mass),
  - estimates stromal AREA % in that field,
  - and the HETEROGENEITY rule: "even if there is only ONE image field with a
    stroma-high score, this image field is decisive" => the slide TSR is the
    MAXIMUM stromal fraction over valid fields, not the mean,
  - cutoff: stroma-high = AREA > 50% = WORSE prognosis (fixed 50%, not median).

This script reuses the per-cell predictions + nearest-nucleus (Voronoi) AREA
accounting from tcga_voronoi_tsr.py, but changes the FIELD DEFINITION and the
AGGREGATION to match the protocol:

  1. tile the slide into square fields (--field_um, default 1860 um -> a
     ~3.46 mm^2 field, the area of a standard x10 ocular field; higher mag
     gives a non-representative area);
  2. a field is VALID only if predicted-tumour cells touch all four edge bands
     (--border_frac margin) -> the "tumour at all four borders" rule, which
     rejects near-acellular fibrous tiles the mean-based method wrongly counted;
  3. inside each valid field, area_stroma / (area_tumour + area_stroma) via the
     same coarse tissue-masked Voronoi tessellation ("other" territory
     excluded), now with an active low-cell-density gate (--density_k /
     --density_radius_um) that drops necrosis / acellular mucin / scar which
     pass white_thr but are too hypocellular to be true tumour stroma;
  4. slide stroma_fraction = MAX over valid fields (the hotspot/heterogeneity
     rule). Also records the mean for comparison.

Output schema matches tcga_predict_tsr.py (sample_name, n_cells, n_tumour,
n_stroma, stroma_fraction) so it feeds tcga_survival.py directly; run survival
with --stroma_cutoff 0.5 for the clinical 50% threshold.

LIMITATION (large vessels): the protocol says a field containing large blood
vessels should be skipped (or the vessel ignored) so that only true tumour
stroma is scored. We do NOT auto-detect large vessels: their acellular lumen is
removed by white_thr + the density gate, but a thick muscular WALL (perivascular
/ vasculature nuclei, classified as stroma) still counts and can inflate the
stroma fraction of a vessel-dominated field. Rare in this cohort; left as a
known limitation rather than a vessel-detection heuristic.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

# reuse the exact coord-recovery + tissue-mask helpers the mean-based path uses
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_voronoi_tsr import _recover_coords, _tissue_foreground

# Map a per-cell prediction (fine 10-class RCC level_2_cell_type OR already-coarse
# tumour/stroma/other) -> the CLASSIC Mesker TSR compartments. Immune cells are a
# SEPARATE group (not stroma): the clinical guidance is to exclude lymphocytic
# follicles/aggregates from TSR scoring (van Pelt/Mesker; "areas with lymphocytic
# follicles/aggregates ... avoided or visually ignored"). So immune is excluded
# from the tumour/stroma denominator (like "other") AND drives the follicle gate.
# Labels absent from the map fall through to "other".
GROUP = {
    "Tumour epithelium": "tumour",
    "Fibroblast": "stroma",
    "Blood vasculature": "stroma",
    "Lymphatic vasculature": "stroma",
    "Perivascular cell": "stroma",
    "B lymphocyte": "immune",
    "T lymphocyte": "immune",
    "Macrophage": "immune",
    "Dendritic cell": "immune",
    "Nephron epithelium": "other",
    # passthrough so an already-coarse npz still scores (no immune gate possible)
    "tumour": "tumour", "stroma": "stroma", "immune": "immune", "other": "other",
}


def _immune_aggregate_mask(imm_xy, radius_px, k):
    """Density-based lymphocytic-aggregate detection. An immune cell is part of a
    dense aggregate (follicle / tertiary lymphoid structure) if it has >= k immune
    neighbours within radius_px (query_ball_point counts the cell itself, so k
    includes self). Diffuse tumour-infiltrating lymphocytes stay BELOW k and are
    not flagged -> we drop follicles/aggregates, not scattered TILs."""
    if len(imm_xy) < k:
        return np.zeros(len(imm_xy), dtype=bool)
    timm = cKDTree(imm_xy)
    n_nb = timm.query_ball_point(imm_xy, r=radius_px, return_length=True)
    return np.asarray(n_nb) >= k


def _tile_area(he, tree, code, px0, py0, tile_px, area_px, max_dist_px, white_thr,
               density_k, density_radius_px):
    """Voronoi tumour/stroma AREA (coarse-pixel counts) inside one field.

    Two distance gates on every tissue-foreground coarse pixel:
      * max_dist_px  -- distance to the NEAREST nucleus (class assignment cap);
      * density_radius_px on the k-th nearest nucleus -- an active low-cell-
        density (necrosis / acellular mucin / scar) exclusion. A coarse pixel
        whose k-th nucleus is farther than density_radius_px sits in
        hypocellular tissue (coagulative necrosis with ghost cells, mucin
        pools, fibrous scar) that passes white_thr yet is not true tumour
        stroma -> dropped from BOTH tumour and stroma area symmetrically.
    """
    _, H, W = he.shape
    px1, py1 = min(px0 + tile_px, W), min(py0 + tile_px, H)
    if px1 <= px0 or py1 <= py0:
        return 0, 0
    block = np.transpose(he[:, py0:py1, px0:px1], (1, 2, 0))   # (h,w,3)
    h, w = block.shape[:2]
    ny, nx = h // area_px, w // area_px
    if ny == 0 or nx == 0:
        return 0, 0
    crop = block[:ny * area_px, :nx * area_px]
    coarse = crop.reshape(ny, area_px, nx, area_px, 3).mean(axis=(1, 3))
    tissue = _tissue_foreground(coarse.astype(np.uint8), white_thr)
    if not tissue.any():
        return 0, 0
    gy = py0 + (np.arange(ny) + 0.5) * area_px
    gx = px0 + (np.arange(nx) + 0.5) * area_px
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX[tissue], GY[tissue]])
    kq = max(1, int(density_k))
    kq = min(kq, tree.n)                      # cKDTree errors if k > n points
    d, idx = tree.query(pts, k=kq)
    if kq == 1:
        d1 = np.atleast_1d(d); idx1 = np.atleast_1d(idx); dk = d1
    else:
        d1, idx1, dk = d[:, 0], idx[:, 0], d[:, kq - 1]
    keep = (d1 <= max_dist_px) & (dk <= density_radius_px)
    cls = code[idx1[keep]]
    return int((cls == 0).sum()), int((cls == 1).sum())   # tumour, stroma


def protocol_tsr_for_slide(feat_h5, data_h5, he_zarr, pred, *,
                           tile_px, area_px, max_dist_px, white_thr,
                           border_frac, min_tumour, min_field_area,
                           density_k, density_radius_px,
                           follicle_radius_px, follicle_k, follicle_min,
                           measure_frac=1.0):
    x, y, matched = _recover_coords(Path(feat_h5), Path(data_h5))
    if matched.sum() == 0:
        return None
    if matched.sum() < len(pred):
        pred = pred[matched]

    # CLASSIC grouping: tumour / stroma / immune / other. immune is NOT stroma.
    grp = np.array([GROUP.get(p, "other") for p in pred], dtype=object)
    is_tum = grp == "tumour"
    is_imm = grp == "immune"

    tx = (x // tile_px).astype(np.int64)
    ty = (y // tile_px).astype(np.int64)
    # offset within field (0..tile_px) for the four-border test
    ox = x - tx * tile_px
    oy = y - ty * tile_px
    m = int(border_frac * tile_px)
    df = pd.DataFrame({"tx": tx, "ty": ty, "tum": is_tum,
                       "L": ox < m, "R": ox > tile_px - m,
                       "T": oy < m, "B": oy > tile_px - m})
    # a field is valid iff a TUMOUR cell touches each of the 4 border bands
    g = df[df.tum].groupby(["tx", "ty"])
    border = g[["L", "R", "T", "B"]].any()
    border["ntum"] = g.size()
    mask = (border["L"] & border["R"] & border["T"] & border["B"]
            & (border["ntum"] >= min_tumour)).to_numpy()
    valid = border[mask]
    n_border_valid = len(valid)

    # FOLLICLE GATE: drop any candidate field containing > follicle_min cells that
    # sit in a dense immune aggregate (>= follicle_k immune neighbours within
    # follicle_radius_px). Implements "avoid lymphocytic follicles/aggregates".
    n_follicle_excluded = 0
    if not valid.empty and follicle_min >= 0:
        imm_xy = np.column_stack([x[is_imm], y[is_imm]])
        agg = _immune_aggregate_mask(imm_xy, follicle_radius_px, follicle_k)
        if agg.any():
            ax = (imm_xy[agg, 0] // tile_px).astype(np.int64)
            ay = (imm_xy[agg, 1] // tile_px).astype(np.int64)
            agg_count = pd.Series(1, index=pd.MultiIndex.from_arrays([ax, ay])) \
                .groupby(level=[0, 1]).size()
            keep = np.array([agg_count.get((int(a), int(b)), 0) <= follicle_min
                             for a, b in valid.index])
            n_follicle_excluded = int((~keep).sum())
            valid = valid[keep]

    if valid.empty:
        return dict(n_tumour=0, n_stroma=0, stroma_fraction=np.nan,
                    stroma_fraction_mean=np.nan, n_valid_fields=0,
                    n_tumour_tiles=0, tumour_area_px=0, stroma_area_px=0,
                    n_border_valid=n_border_valid,
                    n_follicle_excluded=n_follicle_excluded)

    he = zarr.open(he_zarr, mode="r")                 # (3,H,W) uint8 @ 0.5 um/px
    # area accounting: tumour=0, stroma=1, everything else (immune+other)=2 excluded
    code = np.where(grp == "tumour", 0,
                    np.where(grp == "stroma", 1, 2)).astype(np.int8)
    tree = cKDTree(np.column_stack([x, y]))

    # MEASURE the ratio inside a CENTRAL window of the field (measure_frac of the
    # side, centred), NOT the whole field. The 4-border test is only a GATE that
    # the field sits inside tumour; measuring the centre avoids peripheral
    # fat/stroma pockets that merely clip the border bands from winning the MAX
    # hotspot. When measure_frac<1 a field also must have >= min_tumour predicted
    # tumour cells INSIDE that central window (i.e. the centre is genuinely
    # intratumoural, not adipose/normal). measure_frac=1.0 -> whole field (legacy).
    ctile = max(1, int(round(measure_frac * tile_px)))
    off = (tile_px - ctile) // 2
    tum_x, tum_y = x[is_tum], y[is_tum]
    he = zarr.open(he_zarr, mode="r")                 # (3,H,W) uint8 @ 0.5 um/px
    # area accounting: tumour=0, stroma=1, everything else (immune+other)=2 excluded
    code = np.where(grp == "tumour", 0,
                    np.where(grp == "stroma", 1, 2)).astype(np.int8)
    tree = cKDTree(np.column_stack([x, y]))

    fracs, tot_tum, tot_str = [], 0, 0
    n_center_drop = 0
    for (txi, tyi) in valid.index:
        cx0 = int(txi) * tile_px + off
        cy0 = int(tyi) * tile_px + off
        if measure_frac < 1.0:               # centre must itself be intratumoural
            inwin = int(((tum_x >= cx0) & (tum_x < cx0 + ctile)
                         & (tum_y >= cy0) & (tum_y < cy0 + ctile)).sum())
            if inwin < min_tumour:
                n_center_drop += 1
                continue
        ta, sa = _tile_area(he, tree, code, cx0, cy0, ctile, area_px,
                            max_dist_px, white_thr, density_k, density_radius_px)
        if ta + sa < min_field_area:        # ignore near-empty fields
            continue
        fracs.append(sa / (ta + sa))
        tot_tum += ta; tot_str += sa
    if not fracs:
        return dict(n_tumour=0, n_stroma=0, stroma_fraction=np.nan,
                    stroma_fraction_mean=np.nan, n_valid_fields=0,
                    n_tumour_tiles=0, tumour_area_px=0, stroma_area_px=0,
                    n_border_valid=n_border_valid,
                    n_follicle_excluded=n_follicle_excluded,
                    n_center_drop=n_center_drop)

    fracs = np.array(fracs)
    return dict(
        n_tumour=int(is_tum.sum()),
        n_stroma=int((grp == "stroma").sum()),
        stroma_fraction=float(fracs.max()),         # PROTOCOL: hotspot / max
        stroma_fraction_mean=float(fracs.mean()),   # for comparison
        n_valid_fields=len(fracs),
        n_tumour_tiles=len(fracs),
        tumour_area_px=tot_tum, stroma_area_px=tot_str,
        n_border_valid=n_border_valid,
        n_follicle_excluded=n_follicle_excluded,
        n_center_drop=n_center_drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--per_cell_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--field_um", type=float, default=1860.0,
                    help="x10 field-of-view side in um. Default 1860 -> a "
                         "square field of ~3.46 mm^2, the area of a standard "
                         "x10 ocular field in the van Pelt/Mesker protocol "
                         "(higher mag gives a non-representative area).")
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--area_um", type=float, default=8.0)
    ap.add_argument("--max_dist_um", type=float, default=50.0)
    ap.add_argument("--white_thr", type=int, default=220)
    ap.add_argument("--border_frac", type=float, default=0.25,
                    help="edge-band width (fraction of field) for the "
                         "tumour-at-all-four-borders test.")
    ap.add_argument("--min_tumour", type=int, default=10,
                    help="min predicted-tumour cells for a valid field.")
    ap.add_argument("--min_field_area", type=int, default=200,
                    help="min tumour+stroma coarse-pixel area for a field.")
    ap.add_argument("--measure_frac", type=float, default=1.0,
                    help="measure the stroma/tumour ratio inside a CENTRAL window "
                         "of this side-fraction (centred), not the whole field. "
                         "The 4-border test stays a gate; measuring the centre "
                         "rejects peripheral fat/stroma pockets that merely clip "
                         "the border bands. <1 also requires >= min_tumour tumour "
                         "cells in the central window. 1.0 = whole field (legacy).")
    ap.add_argument("--density_k", type=int, default=5,
                    help="active low-cell-density exclusion: keep a coarse "
                         "pixel only if its k-th nearest nucleus is within "
                         "--density_radius_um (drops necrosis / acellular "
                         "mucin / scar that passes white_thr).")
    ap.add_argument("--density_radius_um", type=float, default=60.0,
                    help="radius (um) for the k-th-nucleus density gate.")
    # --- lymphocytic-follicle / aggregate exclusion (density-based) ---
    ap.add_argument("--follicle_radius_um", type=float, default=50.0,
                    help="radius (um) to count immune neighbours when detecting "
                         "dense lymphocytic aggregates (follicles / TLS).")
    ap.add_argument("--follicle_k", type=int, default=15,
                    help="an immune cell is in an aggregate if it has >= this "
                         "many immune neighbours within --follicle_radius_um "
                         "(includes self). Diffuse TILs stay below -> kept.")
    ap.add_argument("--follicle_min", type=int, default=30,
                    help="a candidate field is dropped if it contains MORE than "
                         "this many aggregate immune cells (follicle in field). "
                         "Set -1 to disable the follicle gate.")
    args = ap.parse_args()

    tile_px = int(args.field_um / args.um_per_px)
    area_px = max(1, int(args.area_um / args.um_per_px))
    max_dist_px = args.max_dist_um / args.um_per_px
    density_radius_px = args.density_radius_um / args.um_per_px
    follicle_radius_px = args.follicle_radius_um / args.um_per_px

    man = pd.read_csv(args.manifest)
    he_of = {str(r["sample_name"]).strip(): str(r["hematoxylin_eosin_image"]).strip()
             for _, r in man.iterrows()}
    data_of = {str(r["sample_name"]).strip(): str(r["data_h5"]).strip()
               for _, r in man.iterrows()}
    feat_dir = Path(args.features_dir); pcd = Path(args.per_cell_dir)
    print(f"[protocol] field={tile_px}px (~{args.field_um/1000:.2f}mm, "
          f"{(args.field_um/1000)**2:.2f}mm^2) border_frac={args.border_frac} "
          f"min_tumour={args.min_tumour} area={area_px}px "
          f"max_dist={max_dist_px:.0f}px white_thr={args.white_thr} "
          f"density_k={args.density_k} density_radius={density_radius_px:.0f}px")
    if args.follicle_min >= 0:
        print(f"[follicle] gate ON: radius={follicle_radius_px:.0f}px "
              f"k={args.follicle_k} min_per_field={args.follicle_min} "
              f"(immune grouped via GROUP map; needs FINE 10-class npz)")
    else:
        print("[follicle] gate OFF")

    rows = []
    for npz in sorted(pcd.glob("*.npz")):
        sample = npz.stem
        feat_h5 = feat_dir / f"{sample}.h5"
        data_h5 = data_of.get(sample); he_zarr = he_of.get(sample)
        if not feat_h5.exists() or not data_h5 or not Path(data_h5).exists() \
                or not he_zarr or not Path(he_zarr).exists():
            print(f"[skip] {sample}: missing feature/data/he", flush=True)
            continue
        pred = np.load(npz, allow_pickle=True)["pred"].astype(str)
        try:
            res = protocol_tsr_for_slide(
                feat_h5, data_h5, he_zarr, pred,
                tile_px=tile_px, area_px=area_px, max_dist_px=max_dist_px,
                white_thr=args.white_thr, border_frac=args.border_frac,
                min_tumour=args.min_tumour, min_field_area=args.min_field_area,
                density_k=args.density_k, density_radius_px=density_radius_px,
                follicle_radius_px=follicle_radius_px,
                follicle_k=args.follicle_k, follicle_min=args.follicle_min,
                measure_frac=args.measure_frac)
        except Exception as e:
            print(f"[err] {sample}: {e}", flush=True)
            continue
        if res is None:
            print(f"[skip] {sample}: no cell_id matched data_h5", flush=True)
            continue
        n_cells = res["n_tumour"] + res["n_stroma"]
        rows.append({"sample_name": sample, "n_cells": n_cells, **res})
        sf = res["stroma_fraction"]
        print(f"  {sample}: valid_fields={res['n_valid_fields']} "
              f"(border_valid={res.get('n_border_valid','?')} "
              f"follicle_excl={res.get('n_follicle_excluded',0)}) "
              f"max_stroma={sf if np.isnan(sf) else round(sf,3)} "
              f"mean_stroma={res['stroma_fraction_mean'] if np.isnan(res['stroma_fraction_mean']) else round(res['stroma_fraction_mean'],3)}",
              flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[protocol-tsr] {len(df)} slides -> {args.out_csv}")
    if len(df):
        v = df.dropna(subset=["stroma_fraction"])
        print(f"valid slides={len(v)}  max_stroma median={v.stroma_fraction.median():.3f} "
              f"[{v.stroma_fraction.quantile(.25):.3f},{v.stroma_fraction.quantile(.75):.3f}]  "
              f">0.5: {(v.stroma_fraction>0.5).mean()*100:.0f}%")
        if "n_follicle_excluded" in df:
            tot_excl = int(df["n_follicle_excluded"].fillna(0).sum())
            n_aff = int((df["n_follicle_excluded"].fillna(0) > 0).sum())
            print(f"[follicle] excluded {tot_excl} fields across {n_aff}/{len(df)} "
                  f"slides")


if __name__ == "__main__":
    main()
