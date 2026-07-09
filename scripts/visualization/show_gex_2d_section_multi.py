"""Show 2D GEX prediction (y_hat) for ONE section across MULTIPLE methods.

Companion to ``show_gex_2d_section.py`` but stacks several methods as rows:
row 0 = ground-truth, then one row per method (SIGMMA + FT baselines + UNI2-h).
Every method reuses the canonical probe path (``predict_all`` + per-FOV head)
reading from its OWN cached features, so the maps match the eval-JSON metrics.

Each (row, gene) panel is self-normalised to its own ``[p1, p99]`` so the
spatial pattern is comparable across methods despite the magnitude shrinkage
of the MSE-trained heads. Per-gene Pearson r (pooled on this section) is
annotated under each method's panel title.

Paths default to the svg0529 step2000 corrected-pipeline runs.
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
DATA_CACHE = f"{CACHE}/data/gex_2d/svg0529_panel_35a7bd78801c_g14_fov256"
OUT_ROOT = "outputs/gex_2d/svg0529_step2000"

# Method registry: each runs predict_all with its own (input_repr, head, cache).
#   feat_cache → uni_cache_dir for input_repr='uni', else sigmma_cache_dir.
METHODS = [
    dict(label="SIGMMA", input_repr="concat", head_type="mlp",
         head=f"{OUT_ROOT}/sigmma_a01_step2000/heads/concat/best_seed0.pt",
         feat_cache=f"{CACHE}/sigmma/gex_2d/36852_step2000"),
    dict(label="BLEEP", input_repr="uni", head_type="linear",
         head=f"{OUT_ROOT}/bleep_uni_step2000/heads_linear_sweep/uni/best_seed0.pt",
         feat_cache=f"{CACHE}/uni/gex_2d/bleep_uni_svg0529_step2000"),
    dict(label="CLIP", input_repr="uni", head_type="linear",
         head=f"{OUT_ROOT}/clip_uni_step2000/heads_linear_sweep/uni/best_seed0.pt",
         feat_cache=f"{CACHE}/uni/gex_2d/clip_uni_svg0529_step2000"),
    dict(label="PLIP", input_repr="uni", head_type="linear",
         head=f"{OUT_ROOT}/plip_uni_step2000/heads_linear_sweep/uni/best_seed0.pt",
         feat_cache=f"{CACHE}/uni/gex_2d/plip_uni_svg0529_step2000"),
    dict(label="OmiCLIP", input_repr="uni", head_type="linear",
         head=f"{OUT_ROOT}/omiclip_uni_step2000/heads_linear_sweep/uni/best_seed0.pt",
         feat_cache=f"{CACHE}/uni/gex_2d/omiclip_uni_svg0529_step2000"),
    dict(label="UNI2-h", input_repr="uni", head_type="linear",
         head=f"{OUT_ROOT}/uni2h_no_ft/heads_linear_sweep/uni/best_seed0.pt",
         feat_cache=f"{CACHE}/uni/gex_2d/uni2h_svg0529_fov256_p14"),
]


def run_method(m, keys, panel_G, target_grid, device):
    """predict_all + stitch for one method → (pred_canvas, target_canvas, mask)."""
    is_uni = m["input_repr"] == "uni"
    uni_dir = m["feat_cache"] if is_uni else None
    sig_dir = None if is_uni else m["feat_cache"]
    in_ch = in_channels_for(m["input_repr"], uni_cache_dir=uni_dir,
                            sigmma_cache_dir=sig_dir, sample_key=keys[0])
    head = build_head(m["head_type"], in_ch, panel_G, 256)
    head.load_state_dict(torch.load(m["head"], map_location="cpu"))
    out = predict_all(
        head, keys, target_grid=target_grid, device=device,
        input_repr=m["input_repr"],
        data_cache_dir=DATA_CACHE, uni_cache_dir=uni_dir, sigmma_cache_dir=sig_dir,
        batch_size=64, num_workers=0,
    )
    r = per_gene_pearson(out["pred"], out["target"], out["mask"])
    slides = _stitch_per_slide(
        out["pred"], out["target"], out["mask"], out["sample_ids"],
        out["x_starts"], out["y_starts"], out["fov_size"], target_grid,
    )
    return slides, r, in_ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="BK24-SKI-28-FO-3-S3")
    ap.add_argument("--genes", nargs="+",
                    default=["EEF1G", "ZNF683", "NBL1", "ITGB4", "APP", "P4HB"])
    ap.add_argument("--panel_dir",
                    default="outputs/gex_2d/svg0529_panel_35a7bd78801c")
    ap.add_argument("--target_grid", type=int, default=14)
    ap.add_argument("--out_path",
                    default="figures/gex_2d/show_section_multi.png")
    args = ap.parse_args()

    panel = json.load(open(Path(args.panel_dir) / "gene_panel.json"))["panel"]
    G = len(panel)
    keys = list_fov_keys(DATA_CACHE, [args.sample])
    if not keys:
        raise SystemExit(f"no cached FOVs for {args.sample}")
    gidx = [panel.index(g) for g in args.genes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] {args.sample}: {len(keys)} FOVs, panel={G}, device={device}")

    # run every method; GT canvas is identical across methods → grab from first
    pcans, rs = [], []
    tcan = mcan = None
    for m in METHODS:
        slides, r, in_ch = run_method(m, keys, G, args.target_grid, device)
        pc, tc, mc = slides[args.sample]
        pcans.append(pc); rs.append(r)
        if tcan is None:
            tcan, mcan = tc, mc
        print(f"[ok] {m['label']:8s} in_ch={in_ch} "
              f"median_r={np.nanmedian(r):.3f}")
    tissue = mcan > 0

    # layout: rows = GT + methods, cols = genes
    nrow = 1 + len(METHODS)
    ncol = len(gidx)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.3 * nrow),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    if ncol == 1:
        axes = axes.reshape(nrow, 1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))

    def show(ax, canvas, with_cbar=True):
        img = np.where(np.isnan(canvas), np.nan, canvas)
        vmin = max(0.0, float(np.nanpercentile(img[tissue], 1)))
        vmax = float(np.nanpercentile(img[tissue], 99))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if with_cbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for j, g in enumerate(gidx):
        # row 0: ground truth
        show(axes[0, j], tcan[g])
        axes[0, j].set_title(panel[g], fontsize=10)
        # rows 1..: methods
        for i, m in enumerate(METHODS):
            show(axes[i + 1, j], pcans[i][g])
            axes[i + 1, j].set_title(f"r={rs[i][g]:.2f}", fontsize=8)

    axes[0, 0].set_ylabel("Ground truth", fontsize=10)
    for i, m in enumerate(METHODS):
        axes[i + 1, 0].set_ylabel(m["label"], fontsize=10)
    fig.suptitle(f"2D GEX prediction — {args.sample}  "
                 f"(log1p, {args.target_grid}x{args.target_grid} bins/FOV; "
                 f"each panel self-normalised)", y=0.995, fontsize=12)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ .pdf)")


if __name__ == "__main__":
    main()
