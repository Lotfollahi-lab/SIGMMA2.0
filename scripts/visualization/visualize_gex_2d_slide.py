"""Slide-level GEX prediction visualization.

Loads a trained probing head, predicts on every cached FOV of a slide,
stitches predictions into a slide-coordinate grid, and plots:
  [H&E thumbnail] | [GT gene-X] | [UNI pred gene-X] | [SIGMMA pred gene-X]

Target grid is taken from CLI; cache native grid is detected per file.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
import matplotlib.pyplot as plt

from sigmma.evaluation.gex_2d_probing import (
    ProbingHead2D,
    assemble_features,
    _downsample_count_target,
)
from sigmma.evaluation.gex_cache_io import iter_section_fovs


def load_slide_fovs(data_cache_dir: Path, cache_root: Path, sample: str) -> list[dict]:
    """Merge the per-section caches into one FOV dict list keyed implicitly by
    creation order. GT + bbox come from the data cache; UNI / SIGMMA features
    are joined in by fov_id (a FOV missing a feature cache simply lacks that
    key, mirroring the old single-npz behaviour)."""
    uni_feats = {
        d["fov_id"]: d["features"]
        for d in iter_section_fovs(cache_root / "uni", sample, fields=["features"])
    }
    sig_feats = {
        d["fov_id"]: d
        for d in iter_section_fovs(cache_root / "sigmma", sample,
                                   fields=["micro", "meso", "macro"])
    }
    fovs: list[dict] = []
    for d in iter_section_fovs(data_cache_dir, sample, fields=["target", "mask"]):
        fid = d["fov_id"]
        target = d["target"].astype(np.float32)
        fov = {
            "target_raw": target,
            "mask_raw": (d["mask"].astype(np.float32) if "mask" in d
                         else np.ones(target.shape[1:], dtype=np.float32)),
            "x_start": d["x_start"],
            "y_start": d["y_start"],
            "fov_size": d["fov_size"],
        }
        if fid in uni_feats:
            fov["features"] = uni_feats[fid].astype(np.float32)
        if fid in sig_feats:
            s = sig_feats[fid]
            fov["micro"] = s["micro"].astype(np.float32)
            fov["meso"] = s["meso"].astype(np.float32)
            fov["macro"] = s["macro"].astype(np.float32)
        fovs.append(fov)
    return fovs


def predict_fov(head, fov, input_repr, target_grid, device):
    if input_repr == "uni":
        feats = torch.from_numpy(fov["features"]).unsqueeze(0)
        x = assemble_features([feats], "uni", target_grid)
    else:
        scales = [torch.from_numpy(fov[k]).unsqueeze(0) for k in ("micro", "meso", "macro")]
        x = assemble_features(scales, input_repr, target_grid)
    with torch.no_grad():
        pred = head(x.to(device))[0]  # (G, target_grid, target_grid)
    return pred.cpu().numpy()


def stitch_to_slide(
    fovs: list[dict],
    pred_maps: list[np.ndarray] | None,
    target_grid: int,
    H_px: int,
    W_px: int,
    gene_idx: int,
    use_gt: bool = False,
):
    """Place each FOV's (G, target_grid, target_grid) map at slide-coord bins.

    Returns a (Hb, Wb) array at bin resolution (bin_px = fov_size / target_grid),
    with NaN where no FOV covers the bin.
    """
    fov_size = fovs[0]["fov_size"]
    bin_px = fov_size / target_grid
    Hb = int(np.ceil(H_px / bin_px))
    Wb = int(np.ceil(W_px / bin_px))
    out = np.full((Hb, Wb), np.nan, dtype=np.float32)
    for i, fov in enumerate(fovs):
        if use_gt:
            target_pool, _ = _downsample_count_target(
                torch.from_numpy(fov["target_raw"]),
                torch.from_numpy(fov["mask_raw"]),
                target_grid,
            )
            tile = np.log1p(target_pool[gene_idx].numpy())
        else:
            tile = pred_maps[i][gene_idx]
        x0 = int(round(fov["x_start"] / bin_px))
        y0 = int(round(fov["y_start"] / bin_px))
        # Place tile at [y0:y0+target_grid, x0:x0+target_grid]
        y1 = min(y0 + target_grid, Hb)
        x1 = min(x0 + target_grid, Wb)
        out[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
    return out


def read_thumbnail(zarr_path: Path, target_long_side: int = 1500):
    z = zarr.open(str(zarr_path), mode="r")
    C, H, W = z.shape
    step = max(1, int(np.ceil(max(H, W) / target_long_side)))
    thumb = z[:, ::step, ::step][:]
    if thumb.dtype != np.uint8:
        thumb = thumb.astype(np.uint8)
    return np.transpose(thumb, (1, 2, 0)), step, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--gene", required=True)
    ap.add_argument("--cache_root", required=True,
                    help="gex_2d cache root holding uni/ and sigmma/ subdirs of "
                         "per-section .h5 feature caches")
    ap.add_argument("--data_cache_dir", required=True,
                    help="data cache dir holding per-section <sample>.h5 with GT "
                         "targets + FOV bboxes")
    ap.add_argument("--panel_dir", required=True)
    ap.add_argument("--head_root", required=True,
                    help="dir containing heads/<repr>/best.pt")
    ap.add_argument("--manifest", required=True,
                    help="manifest CSV to look up the H&E zarr path")
    ap.add_argument("--target_grid", type=int, default=14)
    ap.add_argument("--reprs", nargs="+", default=["uni", "concat"],
                    help="which input_reprs to include in the plot")
    ap.add_argument("--cmap", default="viridis",
                    help="matplotlib colormap. magma blends 0 into black; "
                         "viridis/plasma/cividis give a cleaner dark→bright "
                         "ramp that is distinct from the NaN colour.")
    ap.add_argument("--nan_color", default="match",
                    help="colour for bins with no FOV coverage. 'match' (default) "
                         "uses cmap(0) so NaN blends with the zero-expression "
                         "floor; pass any matplotlib colour name to override.")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--style", choices=["default", "nature"], default="default",
                    help="figure styling. 'nature' uses sans-serif, small "
                         "fonts, no top/right spines, bold panel letters, "
                         "and 300 dpi output suitable for Nature-family "
                         "manuscripts.")
    args = ap.parse_args()

    # ---- panel + gene index
    panel = json.load(open(Path(args.panel_dir) / "gene_panel.json"))["panel"]
    if args.gene not in panel:
        raise SystemExit(f"gene {args.gene!r} not in panel ({len(panel)} entries)")
    gene_idx = panel.index(args.gene)
    G = len(panel)
    print(f"[panel] gene={args.gene}  idx={gene_idx}/{G}")

    # ---- find H&E zarr
    manifest = pd.read_csv(args.manifest)
    row = manifest[manifest["sample_name"] == args.sample]
    if row.empty:
        # try train manifest fallback later if needed
        for alt in [
            "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv",
            "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_val.csv",
        ]:
            df = pd.read_csv(alt)
            row = df[df["sample_name"] == args.sample]
            if not row.empty:
                break
    if row.empty:
        raise SystemExit(f"sample {args.sample!r} not in manifest")
    he_zarr = row.iloc[0]["hematoxylin_eosin_image"]
    print(f"[he]  {he_zarr}")

    # ---- thumbnail
    thumb, step, H_px, W_px = read_thumbnail(Path(he_zarr))
    print(f"[thumb]  H={H_px}px W={W_px}px (step={step})")

    # ---- load cached FOVs: GT + bbox from the data cache, UNI / SIGMMA
    # features joined in by fov_id from the gex_2d cache root.
    cache_root = Path(args.cache_root)
    fovs = load_slide_fovs(Path(args.data_cache_dir), cache_root, args.sample)
    print(f"[cache] {len(fovs)} FOVs "
          f"(with uni={sum('features' in f for f in fovs)} "
          f"sigmma={sum('macro' in f for f in fovs)})")
    if not fovs:
        raise SystemExit("no data cache for this sample")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- predict per repr
    pred_maps_per_repr = {}
    for repr_name in args.reprs:
        in_ch = 3072 if repr_name == "concat" else 1024
        head = ProbingHead2D(in_channels=in_ch, num_genes=G, hidden_dim=256)
        ckpt = Path(args.head_root) / repr_name / "best.pt"
        if not ckpt.exists():
            print(f"  [skip] {repr_name}: no checkpoint at {ckpt}")
            continue
        head.load_state_dict(torch.load(ckpt, map_location="cpu"))
        head = head.to(device).eval()
        preds = [predict_fov(head, fov, repr_name, args.target_grid, device) for fov in fovs]
        pred_maps_per_repr[repr_name] = preds
        print(f"  [{repr_name}] predicted on {len(preds)} FOVs (head={ckpt})")

    # ---- stitch
    gt_slide = stitch_to_slide(fovs, None, args.target_grid, H_px, W_px,
                                gene_idx, use_gt=True)
    pred_slides = {
        rn: stitch_to_slide(
            fovs, pred_maps_per_repr[rn], args.target_grid, H_px, W_px,
            gene_idx, use_gt=False,
        )
        for rn in pred_maps_per_repr
    }

    # Shared vmin/vmax across GT + preds (excluding NaN), for fair comparison.
    stack = [gt_slide] + list(pred_slides.values())
    vals = np.concatenate([s[~np.isnan(s)] for s in stack])
    vmin = float(np.nanpercentile(vals, 1))
    vmax = float(np.nanpercentile(vals, 99))
    print(f"[viz] vmin={vmin:.3f} vmax={vmax:.3f}")

    # ---- plot
    if args.style == "nature":
        # Nature-style: sans-serif, small fonts, no spines, panel letters
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    ncols = 2 + len(pred_slides)
    panel_w = 1.6 if args.style == "nature" else 5
    panel_h = 1.6 if args.style == "nature" else 5
    fig, axes = plt.subplots(1, ncols, figsize=(panel_w * ncols + 0.3, panel_h),
                              gridspec_kw={"wspace": 0.05})
    axes[0].imshow(thumb)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # vmin clipped at 0 for cleaner ramp; NaN colour either user-set or matched
    # to cmap(0) so 0-bins and out-of-tissue bins blend visually.
    cmap = plt.get_cmap(args.cmap).copy()
    vmin = max(0.0, vmin)
    if args.nan_color == "match":
        cmap.set_bad(cmap(0.0))
    else:
        cmap.set_bad(args.nan_color)

    axes[1].imshow(gt_slide, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    for i, (rn, sl) in enumerate(pred_slides.items()):
        ax = axes[2 + i]
        im = ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])

    if args.style == "nature":
        # Panel titles below the image (compact) + bold letters a/b/c/...
        labels = ["H&E", "Ground truth", *[f"{rn} pred" for rn in pred_slides]]
        for j, (ax, lbl) in enumerate(zip(axes, labels)):
            ax.text(0.0, 1.02, chr(ord("a") + j), transform=ax.transAxes,
                    fontsize=8, fontweight="bold", va="bottom", ha="left")
            ax.text(0.5, -0.04, lbl, transform=ax.transAxes,
                    fontsize=7, va="top", ha="center")
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.01,
                             shrink=0.85)
        cbar.set_label(f"log1p({args.gene})", fontsize=7)
        cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
        cbar.outline.set_linewidth(0.5)
    else:
        axes[0].set_title(f"H&E  ({args.sample})", fontsize=11)
        axes[1].set_title(f"GT  log1p({args.gene})", fontsize=11)
        for i, (rn, _) in enumerate(pred_slides.items()):
            axes[2 + i].set_title(f"{rn}  pred", fontsize=11)
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.01)
        cbar.set_label(f"log1p(count) — bins of {args.target_grid * args.target_grid}")
        fig.suptitle(
            f"Slide-level GEX prediction — {args.gene} on {args.sample}  "
            f"(target_grid={args.target_grid})",
            y=1.02, fontsize=12,
        )

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=300 if args.style == "nature" else 150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {args.out_path}")


if __name__ == "__main__":
    main()
