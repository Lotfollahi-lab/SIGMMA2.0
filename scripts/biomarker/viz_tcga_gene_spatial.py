#!/usr/bin/env python
"""Visualise, for TCGA-KIRC H&E slides, the SIGMMA-predicted 2D biomarker
(PAX8/CA9) expression next to the H&E — the frozen in-house-RCC probe applied to
segmented nuclei on H&E alone.

Per slide, two panels side by side:
  left  : H&E thumbnail (strip-read + strided downsample of the .he.zarr).
  right : predicted expression as a spatial grid heatmap. Segmented-nucleus
          predictions (from predict = frozen Ridge probe on z_sigmma_concat) are
          binned on the thumbnail grid; each bin = mean predicted expression of
          its nuclei (>= --min_bin nuclei), overlaid on a faint greyscale H&E for
          anatomical context. Empty bins are transparent.

Nuclei coords come from data.h5 (x_pixel/y_pixel @0.5um/px, same frame as the
zarr). Predictions join to coords by cell_id. vmin/vmax are shared across slides
(global 2/98 pct) so colours are comparable.

Run in the sigmma poetry venv (zarr 3.x + sklearn + h5py + matplotlib).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def predict_slide(feat_h5, bundle, gene):
    with h5py.File(feat_h5, "r") as f:
        cid = np.array([c.decode() if isinstance(c, bytes) else str(c)
                        for c in f["cell_id"][:]])
        X = np.concatenate([f[k][:] for k in bundle["keys"]], axis=1).astype(np.float32)
    Z = (X - bundle["mu"]) / bundle["sd"]
    pred = bundle["models"][gene].predict(Z)
    return dict(zip(cid, pred))


def load_coords(data_h5):
    with h5py.File(data_h5, "r") as f:
        cid = np.array([c.decode() if isinstance(c, bytes) else str(c)
                        for c in f["cell_id"][:]])
        x = f["x_pixel"][:].astype(np.float64)
        y = f["y_pixel"][:].astype(np.float64)
    return cid, x, y


def thumbnail(zpath, target_px=1600, strip=2048):
    z = zarr.open(zpath, mode="r")
    C, H, W = z.shape
    stride = max(1, int(round(max(H, W) / target_px)))
    rows = []
    for y0 in range(0, H, strip):
        blk = np.asarray(z[:, y0:min(y0 + strip, H), :])       # (3, h, W)
        rows.append(blk[:, ::stride, ::stride])
        del blk
    thumb = np.concatenate(rows, axis=1)                        # (3, h', w')
    return np.transpose(thumb, (1, 2, 0)), stride              # (h', w', 3)


def grid_mean(x, y, vals, stride, shape_hw, bin_px, min_bin):
    """Mean predicted expression per bin on the thumbnail grid."""
    h, w = shape_hw
    nby, nbx = int(np.ceil(h / bin_px)), int(np.ceil(w / bin_px))
    tx = (x / stride / bin_px).astype(int).clip(0, nbx - 1)
    ty = (y / stride / bin_px).astype(int).clip(0, nby - 1)
    flat = ty * nbx + tx
    ssum = np.bincount(flat, weights=vals, minlength=nby * nbx)
    scnt = np.bincount(flat, minlength=nby * nbx)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / scnt
    mean[scnt < min_bin] = np.nan
    return mean.reshape(nby, nbx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--slide_csv", default=None, help="for per-slide mean in titles")
    ap.add_argument("--gene", default="PAX8")
    ap.add_argument("--bin_px", type=int, default=6, help="thumbnail px per grid bin")
    ap.add_argument("--min_bin", type=int, default=2)
    ap.add_argument("--marker_size", type=float, default=9.0,
                    help="scatter marker area for per-nucleus predictions")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    gene = args.gene
    man = pd.read_csv(args.manifest).set_index("sample_name")
    smean = {}
    if args.slide_csv and Path(args.slide_csv).exists():
        sdf = pd.read_csv(args.slide_csv).set_index("sample_name")
        smean = sdf[f"{gene}_mean"].to_dict()

    # pass 1: predictions + coords + H&E thumbnail per slide (shared colour scale)
    data = {}
    allvals = []
    for s in args.samples:
        p = predict_slide(Path(args.feat_dir) / f"{s}.h5", bundle, gene)
        cid, x, y = load_coords(man.loc[s, "data_h5"])
        keep = np.isin(cid, np.fromiter(p.keys(), dtype=cid.dtype))
        vals = np.array([p[c] for c in cid[keep]])
        thumb, stride = thumbnail(man.loc[s, "hematoxylin_eosin_image"])
        data[s] = {"x": x[keep], "y": y[keep], "vals": vals,
                   "thumb": thumb, "stride": stride}
        allvals.append(vals)
        print(f"[{s}] {int(keep.sum())} nuclei, {gene} mean={vals.mean():.3f}, "
              f"thumb={thumb.shape}", flush=True)
    allvals = np.concatenate(allvals)
    vmin, vmax = np.percentile(allvals, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.magma

    def draw(ax0, ax1, s):
        """H&E (ax0) + per-nucleus predicted expression scatter over faint H&E (ax1)."""
        d = data[s]
        thumb, stride = d["thumb"], d["stride"]
        h, w = thumb.shape[:2]
        ext = [0, w, h, 0]
        ax0.imshow(thumb, extent=ext)
        ax0.set_title(f"{s.split('.')[0]}  |  H&E", fontsize=9)
        ax1.imshow(thumb.mean(axis=2), extent=ext, cmap="gray",
                   alpha=0.40, vmin=0, vmax=255)
        sc = ax1.scatter(d["x"] / stride, d["y"] / stride, c=d["vals"],
                         s=args.marker_size, cmap=cmap, norm=norm,
                         edgecolors="none", alpha=0.9)
        ax1.set_xlim(0, w); ax1.set_ylim(h, 0)
        title = f"predicted {gene}"
        if s in smean:
            title += f"  (slide mean={smean[s]:.2f})"
        ax1.set_title(title, fontsize=9)
        for ax in (ax0, ax1):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        return sc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # (a) one legible figure per slide, sized to that slide's aspect ratio
    for s in args.samples:
        h, w = data[s]["thumb"].shape[:2]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 13 / 2 * (h / w) + 0.8))
        sc = draw(ax0, ax1, s)
        cb = fig.colorbar(sc, ax=ax1, fraction=0.035, pad=0.02)
        cb.set_label(f"{gene} (log1p-CP10k)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.suptitle(f"TCGA-KIRC  |  H&E -> SIGMMA-predicted {gene} "
                     f"(frozen in-house RCC probe)", fontsize=10)
        fig.tight_layout()
        ps = out.with_name(out.stem + f"_{s.split('.')[0]}.png")
        fig.savefig(ps, dpi=250, bbox_inches="tight")
        fig.savefig(str(ps).replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"[out] {ps}", flush=True)

    # (b) combined overview, rows scaled to each slide's aspect ratio
    ratios = [data[s]["thumb"].shape[0] / data[s]["thumb"].shape[1]
              for s in args.samples]
    figw = 12
    fig, axes = plt.subplots(len(args.samples), 2,
                             figsize=(figw, sum(figw / 2 * r for r in ratios) + 1.4),
                             gridspec_kw={"height_ratios": ratios}, squeeze=False)
    for i, s in enumerate(args.samples):
        sc = draw(axes[i][0], axes[i][1], s)
        cb = fig.colorbar(sc, ax=axes[i][1], fraction=0.035, pad=0.02)
        cb.set_label(f"{gene}", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"TCGA-KIRC  |  H&E -> SIGMMA-predicted {gene} "
                 f"(frozen in-house RCC probe)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[out] {out}", flush=True)


if __name__ == "__main__":
    main()
