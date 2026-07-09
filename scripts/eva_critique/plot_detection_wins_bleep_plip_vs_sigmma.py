"""Visualize FOVs where SIGMMA beats BLEEP-ft AND PLIP-ft on detection (DQ/PQ)
while staying within 0.05 SQ — i.e., pixel-level segmentation is similar but
SIGMMA correctly splits instances that baselines merge.

Selection criteria (per FOV, seed-averaged metrics on UNI2-h step1000):
  - |ΔSQ| < 0.05   (pixel-level similar)
  - ΔDQ > 0.05      (SIGMMA detection wins) vs both BLEEP-ft AND PLIP-ft
  - ΔPQ > 0         (sanity)

Layout: 5 rows × 5 columns = (FOV) × (H&E, GT, BLEEP-ft, PLIP-ft, SIGMMA).
Row labels show per-FOV ΔSQ + ΔDQ + ΔPQ vs each baseline.

Uses seed=42 predictions (representative); FOV selection uses seed-averaged
metrics across {42, 2, 6, 14, 21}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eva_critique.data import iter_monusac_test
from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.metrics_seg import label_components

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
DATA_ROOT = ROOT / "data/monusac"

BLEEP_NPZ  = ROOT / "outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft/seed42/bleep_uni_ft/convdec_image/test_artifacts.npz"
PLIP_NPZ   = ROOT / "outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed42/plip_uni_ft/convdec_image/test_artifacts.npz"
SIGMMA_NPZ = ROOT / "outputs/eva_critique_gex_sweep_step1000/a01/seed42/sigmma_shared_lora/convdec_image/test_artifacts.npz"

OUT_PNG = ROOT / "figures/eva_critique/sigmma_detection_wins_vs_bleep_plip.png"
OUT_PDF = ROOT / "figures/eva_critique/sigmma_detection_wins_vs_bleep_plip.pdf"

TAU = 0.5

# Top FOVs where SIGMMA wins detection over BOTH BLEEP-ft AND PLIP-ft, ranked by
# combined ΔDQ. Annotations (ΔSQ / ΔDQ / ΔPQ) computed from seed-averaged metrics.
SELECTED = [
    "TCGA-DW-7963-01Z-00-DX1_4",
    "TCGA-DW-7838-01Z-00-DX1_4",
    "TCGA-DW-7838-01Z-00-DX1_2",
    "TCGA-DW-7963-01Z-00-DX1_2",
    "TCGA-MP-A4T7-01Z-00-DX1_005",
]
ROW_ANNOT = {
    "TCGA-DW-7963-01Z-00-DX1_4":    {"bleep": (+0.017, +0.196, +0.147), "plip": (-0.007, +0.120, +0.087)},
    "TCGA-DW-7838-01Z-00-DX1_4":    {"bleep": (-0.020, +0.067, +0.046), "plip": (-0.031, +0.133, +0.098)},
    "TCGA-DW-7838-01Z-00-DX1_2":    {"bleep": (+0.008, +0.163, +0.104), "plip": (+0.017, +0.133, +0.088)},
    "TCGA-DW-7963-01Z-00-DX1_2":    {"bleep": (+0.033, +0.166, +0.129), "plip": (+0.025, +0.059, +0.052)},
    "TCGA-MP-A4T7-01Z-00-DX1_005":  {"bleep": (-0.009, +0.134, +0.092), "plip": (-0.030, +0.104, +0.063)},
}


def colorize_instances(inst: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(inst.max())
    if n == 0:
        return np.ones((*inst.shape, 3), dtype=np.float32)
    colors = rng.uniform(0.2, 1.0, size=(n + 1, 3)).astype(np.float32)
    colors[0] = (1.0, 1.0, 1.0)
    return colors[inst]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(x) for x in d["image_id"]]
    return {ids[i]: dict(prob=d["prob"][i].astype(np.float32),
                          pred=d["pred"][i].astype(np.uint8),
                          gt=d["gt"][i].astype(np.uint8)) for i in range(len(ids))}


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    bleep  = load_npz(BLEEP_NPZ)
    plip   = load_npz(PLIP_NPZ)
    sigmma = load_npz(SIGMMA_NPZ)

    needed = set(SELECTED)
    rgb_dict = {}
    for tile in iter_monusac_test(str(DATA_ROOT)):
        if tile.image_id in needed:
            rgb_dict[tile.image_id] = tile.image_rgb
            if len(rgb_dict) == len(needed):
                break
    missing = needed - set(rgb_dict)
    if missing:
        print(f"WARNING: missing RGB for {missing}")

    n_rows = len(SELECTED)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.3))

    col_titles = ["H&E (224×224)", "GT instances", "BLEEP-ft pred", "PLIP-ft pred", "SIGMMA pred"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9)

    for r, iid in enumerate(SELECTED):
        if iid not in rgb_dict or iid not in bleep or iid not in plip or iid not in sigmma:
            for c in range(n_cols):
                axes[r, c].axis("off")
                axes[r, c].text(0.5, 0.5, f"missing\n{iid}", ha="center", va="center")
            continue
        rgb = rgb_dict[iid]
        b = bleep[iid]; p = plip[iid]; s = sigmma[iid]
        gt = b["gt"].astype(bool)
        gt_inst = label_components(gt)
        bleep_inst  = prob_to_instances(b["prob"], threshold=TAU)
        plip_inst   = prob_to_instances(p["prob"], threshold=TAU)
        sigmma_inst = prob_to_instances(s["prob"], threshold=TAU)

        b_dsq, b_ddq, b_dpq = ROW_ANNOT[iid]["bleep"]
        p_dsq, p_ddq, p_dpq = ROW_ANNOT[iid]["plip"]

        short = iid.replace("TCGA-", "").replace("-01Z-00-DX1", "")
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(
            f"{short}\n"
            f"vs BLEEP: ΔSQ={b_dsq:+.2f}\n"
            f"  ΔDQ={b_ddq:+.2f}  ΔPQ={b_dpq:+.2f}\n"
            f"vs PLIP : ΔSQ={p_dsq:+.2f}\n"
            f"  ΔDQ={p_ddq:+.2f}  ΔPQ={p_dpq:+.2f}",
            fontsize=6.5, rotation=0, ha="right", va="center", labelpad=70,
        )

        axes[r, 1].imshow(colorize_instances(gt_inst))
        axes[r, 2].imshow(colorize_instances(bleep_inst))
        axes[r, 3].imshow(colorize_instances(plip_inst))
        axes[r, 4].imshow(colorize_instances(sigmma_inst))

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.suptitle(
        "FOVs where SIGMMA wins instance separation (|ΔSQ|<0.05, ΔDQ>0.05 vs both baselines)",
        fontsize=10, y=0.995,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
