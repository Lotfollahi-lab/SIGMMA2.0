"""Visualize failure-mode patches: BLEEP-ft vs SIGMMA s2 on global PQ.

Top 3 patches where BLEEP wins by largest ΔPQ + bottom 3 where SIGMMA wins.
For each: H&E, GT instance map, BLEEP pred, SIGMMA pred, R1/R2/R3 overlay,
per-pixel correctness Δ (BLEEP_correct − SIGMMA_correct).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from skimage.morphology import binary_dilation, disk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eva_critique.data import build_regions, iter_monusac_test
from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.metrics_seg import label_components

DATA_ROOT = "/nfs/team361/dj16/projects/SIGMMA/data/monusac"
BLEEP_NPZ = "outputs/eva_critique/bleep_uni_ft_21166/convdec_image/test_artifacts.npz"
SIGMMA_NPZ = "outputs/eva_critique/sigmma_scale2/convdec_image/test_artifacts.npz"
OUT_PNG = "figures/eva_critique/bleep_vs_sigmma_s2_failure_patches.png"
OUT_PDF = "figures/eva_critique/bleep_vs_sigmma_s2_failure_patches.pdf"
TAU = 0.5

TOP3 = [
    "TCGA-DW-7838-01Z-00-DX1_5",
    "TCGA-2Z-A9JG-01Z-00-DX1_2",
    "TCGA-A2-A04X-01Z-00-DX1_003",
]
BOTTOM3 = [
    "TCGA-DW-7963-01Z-00-DX1_2",
    "TCGA-F9-A8NY-01Z-00-DX1_4",
    "TCGA-G9-6367-01Z-00-DX1_4",
]
SELECTED = TOP3 + BOTTOM3


def colorize_instances(inst: np.ndarray) -> np.ndarray:
    """Map int32 instance label map to a (H, W, 3) RGB image with per-instance colors."""
    rng = np.random.default_rng(0)
    n = int(inst.max())
    if n == 0:
        return np.ones((*inst.shape, 3), dtype=np.float32)  # white
    colors = rng.uniform(0.2, 1.0, size=(n + 1, 3)).astype(np.float32)
    colors[0] = (1.0, 1.0, 1.0)  # bg = white
    return colors[inst]


def correctness_delta(pred_b: np.ndarray, pred_s: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Δ = (BLEEP correct) − (SIGMMA correct), in {-1, 0, +1} per pixel."""
    cb = (pred_b == gt).astype(np.int8)
    cs = (pred_s == gt).astype(np.int8)
    return cb - cs


def region_overlay(R1, R2, R3) -> np.ndarray:
    rgb = np.ones((*R1.shape, 3), dtype=np.float32)
    rgb[R3] = (1.0, 0.85, 0.85)  # OOT light red
    rgb[R2] = (1.0, 0.95, 0.55)  # lipid light yellow
    rgb[R1] = (0.65, 1.0, 0.65)  # cell-dense light green
    return rgb


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(x) for x in d["image_id"]]
    return {ids[i]: dict(prob=d["prob"][i].astype(np.float32),
                          pred=d["pred"][i].astype(np.uint8),
                          gt=d["gt"][i].astype(np.uint8)) for i in range(len(ids))}


def main():
    bleep = load_npz(BLEEP_NPZ)
    sigmma = load_npz(SIGMMA_NPZ)
    # Need RGB per FOV — build a dict only for selected ids.
    needed = set(SELECTED)
    rgb_dict = {}
    for tile in iter_monusac_test(DATA_ROOT):
        if tile.image_id in needed:
            rgb_dict[tile.image_id] = tile.image_rgb
            if len(rgb_dict) == len(needed):
                break
    missing = needed - set(rgb_dict)
    if missing:
        print(f"WARNING: missing RGB for {missing}")

    # PQ values for annotation
    alt = pd.read_csv("outputs/eva_critique/alternative_metrics.csv")
    alt = alt[alt["tau"] == TAU]
    pq_b = alt[alt["model_scale"] == "bleep_uni_ft"].set_index("image_id")["PQ"]
    pq_s = alt[alt["model_scale"] == "sigmma_s2"].set_index("image_id")["PQ"]

    n_rows = len(SELECTED)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.6))

    col_titles = [
        "H&E (224×224)",
        "GT instances",
        "BLEEP-ft pred",
        "SIGMMA s2 pred",
        "Regions\nR1 green / R2 yellow / R3 red",
        "Correctness Δ\nblue=BLEEP win, red=SIGMMA win",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9)

    delta_cmap = ListedColormap(["#d62728", "#dddddd", "#1f77b4"])  # -1 red, 0 grey, +1 blue

    for r, iid in enumerate(SELECTED):
        if iid not in rgb_dict or iid not in bleep or iid not in sigmma:
            for c in range(n_cols):
                axes[r, c].axis("off")
                axes[r, c].text(0.5, 0.5, f"missing\n{iid}", ha="center", va="center")
            continue
        rgb = rgb_dict[iid]
        b = bleep[iid]; s = sigmma[iid]
        gt = b["gt"].astype(bool)
        R1, R2, R3, R4, _ = build_regions(rgb, gt)
        gt_inst = label_components(gt)
        bleep_inst = prob_to_instances(b["prob"], threshold=TAU)
        sigmma_inst = prob_to_instances(s["prob"], threshold=TAU)
        dpq = float(pq_b.get(iid, np.nan) - pq_s.get(iid, np.nan))
        tag = "BLEEP↑" if dpq > 0 else "SIGMMA↑"

        # H&E with row label on left
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(
            f"{iid}\nPQ_B={pq_b.get(iid, float('nan')):.3f}  "
            f"PQ_S={pq_s.get(iid, float('nan')):.3f}\nΔPQ={dpq:+.3f} ({tag})",
            fontsize=7, rotation=0, ha="right", va="center", labelpad=60,
        )

        axes[r, 1].imshow(colorize_instances(gt_inst))
        axes[r, 2].imshow(colorize_instances(bleep_inst))
        axes[r, 3].imshow(colorize_instances(sigmma_inst))
        axes[r, 4].imshow(region_overlay(R1, R2, R3))
        delta = correctness_delta(b["pred"], s["pred"], gt.astype(np.uint8))
        axes[r, 5].imshow(delta, cmap=delta_cmap, vmin=-1, vmax=1, interpolation="nearest")

        for c in range(n_cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    plt.suptitle(
        "BLEEP-ft vs SIGMMA s2 (global PQ): top 3 BLEEP-wins + bottom 3 SIGMMA-wins",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
