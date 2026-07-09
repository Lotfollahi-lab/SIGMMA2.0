"""Show 2D GEX prediction (y_hat) for ONE section, SIGMMA concat head.

Reuses the canonical prediction path (`predict_all`) + slide stitching
(`_stitch_per_slide`) so the maps match exactly what the eval JSON metrics
were computed on. For the chosen section it:
  - runs the trained SIGMMA (concat) probe head on every cached FOV,
  - stitches per-FOV (G,K,K) preds + GT into slide canvases,
  - picks the top-`n_genes` genes by GT spatial variance (most structure),
  - plots GT vs y_hat side by side per gene (each row self-normalised to its
    own [p1,p99] so the spatial pattern is comparable despite the magnitude
    shrinkage of the MSE-trained head),
  - annotates each with the pooled per-gene Pearson r.

Paths default to the svg0529 step2000 SIGMMA a01 run (the corrected pipeline).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.probe_gex_2d import build_head, in_channels_for, list_fov_keys
from sigmma.evaluation.gex_2d_probing import (
    predict_all, _stitch_per_slide, per_gene_pearson,
)

CACHE = "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
DEFAULTS = dict(
    data_cache_dir=f"{CACHE}/data/gex_2d/svg0529_panel_35a7bd78801c_g14_fov256",
    uni_cache_dir=f"{CACHE}/uni/gex_2d/uni2h_svg0529_fov256_p14",
    sigmma_cache_dir=f"{CACHE}/sigmma/gex_2d/36852_step2000",
    panel_dir="outputs/gex_2d/svg0529_panel_35a7bd78801c",
    head_ckpt="outputs/gex_2d/svg0529_step2000/sigmma_a01_step2000/heads/concat/best_seed0.pt",
    target_grid=14,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="BK24-SKI-28-FO-1-s3")
    ap.add_argument("--n_genes", type=int, default=6)
    ap.add_argument("--genes", nargs="+", default=None,
                    help="explicit gene names; overrides automatic selection")
    ap.add_argument("--rank_by", choices=["pearson", "variance"], default="pearson",
                    help="auto-select top-n_genes by pooled per-gene Pearson r "
                         "(default) or by GT spatial variance")
    ap.add_argument("--out_path", default="figures/gex_2d/show_section.png")
    ap.add_argument("--shared_scale", action="store_true",
                    help="use the GT [p1,p99] colour range for BOTH rows (one "
                         "shared colourbar/gene) so y and y_hat are directly "
                         "comparable; reveals the MSE-head magnitude shrinkage. "
                         "default = per-row self-normalisation.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    args = ap.parse_args()

    panel = json.load(open(Path(args.panel_dir) / "gene_panel.json"))["panel"]
    G = len(panel)
    keys = list_fov_keys(args.data_cache_dir, [args.sample])
    if not keys:
        raise SystemExit(f"no cached FOVs for {args.sample}")
    print(f"[info] {args.sample}: {len(keys)} FOVs, panel={G} genes")

    in_ch = in_channels_for("concat", sigmma_cache_dir=args.sigmma_cache_dir,
                            sample_key=keys[0])
    head = build_head("mlp", in_ch, G, 256)
    head.load_state_dict(torch.load(args.head_ckpt, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] in_ch={in_ch} device={device} head={args.head_ckpt}")

    out = predict_all(
        head, keys, target_grid=int(args.target_grid), device=device,
        input_repr="concat",
        data_cache_dir=args.data_cache_dir,
        uni_cache_dir=args.uni_cache_dir,
        sigmma_cache_dir=args.sigmma_cache_dir,
        batch_size=64, num_workers=0,
    )
    pred, target, mask = out["pred"], out["target"], out["mask"]
    print(f"[info] pred shape (N,G,K,K) = {pred.shape}")

    # pooled per-gene Pearson on this section
    r = per_gene_pearson(pred, target, mask)

    # stitch to slide canvases
    slides = _stitch_per_slide(
        pred, target, mask, out["sample_ids"], out["x_starts"], out["y_starts"],
        out["fov_size"], int(args.target_grid),
    )
    pcan, tcan, mcan = slides[args.sample]  # (G,Hb,Wb),(G,Hb,Wb),(Hb,Wb)
    tissue = mcan > 0

    # choose genes
    if args.genes:
        gidx = [panel.index(g) for g in args.genes]
    elif args.rank_by == "pearson":
        score = np.where(np.isfinite(r), r, -np.inf)
        gidx = list(np.argsort(score)[::-1][: args.n_genes])
    else:
        var = np.array([
            np.nanvar(tcan[g][tissue]) if np.isfinite(r[g]) else -1.0
            for g in range(G)
        ])
        gidx = list(np.argsort(var)[::-1][: args.n_genes])
    print("[genes] " + ", ".join(f"{panel[g]}(r={r[g]:.2f})" for g in gidx))

    # plot: 2 rows (GT, pred) x n_genes cols
    n = len(gidx)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.0),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    if n == 1:
        axes = axes.reshape(2, 1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))
    for j, g in enumerate(gidx):
        gt = np.where(np.isnan(tcan[g]), np.nan, tcan[g])
        pr = np.where(np.isnan(pcan[g]), np.nan, pcan[g])
        gvmin = max(0.0, float(np.nanpercentile(gt[tissue], 1)))
        gvmax = float(np.nanpercentile(gt[tissue], 99))
        if args.shared_scale:
            # Both rows on the GT [p1, p99] range so y and y_hat are directly
            # comparable on the SAME colourbar (shows the MSE-head shrinkage).
            pvmin, pvmax = gvmin, gvmax
        else:
            # Per-row self-normalisation: spatial pattern comparable despite the
            # magnitude shrinkage of the MSE head.
            pvmin = max(0.0, float(np.nanpercentile(pr[tissue], 1)))
            pvmax = float(np.nanpercentile(pr[tissue], 99))
        im_gt = axes[0, j].imshow(gt, cmap=cmap, vmin=gvmin, vmax=gvmax)
        im_pr = axes[1, j].imshow(pr, cmap=cmap, vmin=pvmin, vmax=pvmax)
        axes[0, j].set_title(f"{panel[g]}\nr={r[g]:.2f}", fontsize=9)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([])
        if args.shared_scale:
            # one shared colourbar spanning both rows
            fig.colorbar(im_gt, ax=[axes[0, j], axes[1, j]],
                         fraction=0.046, pad=0.02)
        else:
            # separate colourbar per row so each shows its own value range
            fig.colorbar(im_gt, ax=axes[0, j], fraction=0.046, pad=0.02)
            fig.colorbar(im_pr, ax=axes[1, j], fraction=0.046, pad=0.02)
    axes[0, 0].set_ylabel("Ground truth  y", fontsize=10)
    axes[1, 0].set_ylabel(r"SIGMMA pred  $\hat{y}$", fontsize=10)
    fig.suptitle(f"2D GEX prediction — {args.sample}  "
                 f"(log1p, {args.target_grid}x{args.target_grid} bins/FOV)",
                 y=0.98, fontsize=12)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .pdf)")
    print(f"[summary] median per-gene Pearson on section = "
          f"{np.nanmedian(r):.3f}  (n_genes_finite={np.isfinite(r).sum()})")


if __name__ == "__main__":
    main()
