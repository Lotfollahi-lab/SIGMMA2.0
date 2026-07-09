"""Visualize FOVs where SIGMMA wins on detection vs each of the 4 contrastive-FT
baselines (BLEEP-ft, CLIP-ft, PLIP-ft, OmiCLIP-ft). One unique FOV per baseline,
selected by max ΔDQ subject to |ΔSQ|<0.05, ΔDQ>0.05, ΔPQ>0.

Outlier FOVs (n_cells ≤ 2) excluded.

Layout: 4 rows × 4 cols = (baseline) × (H&E, GT, baseline pred, SIGMMA pred).
Row labels show baseline + Δ metrics.

Output: figures/eva_critique/sigmma_wins_vs_2ndbest.{pdf,png}
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

# All N=5 seed test_artifacts paths (seed 42 used for representative viz)
NPZ_BY_MODEL = {
    "UNI2-h":     ROOT / "outputs/eva_critique_uni2h/seed42/uni/convdec_image/test_artifacts.npz",
    "BLEEP-ft":   ROOT / "outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft/seed42/bleep_uni_ft/convdec_image/test_artifacts.npz",
    "CLIP-ft":    ROOT / "outputs/eva_critique_baselines_uni2h_step1000/clip_uni_ft/seed42/clip_uni_ft/convdec_image/test_artifacts.npz",
    "PLIP-ft":    ROOT / "outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed42/plip_uni_ft/convdec_image/test_artifacts.npz",
    "OmiCLIP-ft": ROOT / "outputs/eva_critique_baselines_uni2h_step1000/omiclip_uni_ft/seed42/omiclip_uni_ft/convdec_image/test_artifacts.npz",
}
SIGMMA_NPZ = ROOT / "outputs/eva_critique_gex_sweep_step1000/a01/seed42/sigmma_shared_lora/convdec_image/test_artifacts.npz"

OUT_PNG = ROOT / "figures/eva_critique/sigmma_wins_vs_2ndbest.png"
OUT_PDF = ROOT / "figures/eva_critique/sigmma_wins_vs_2ndbest.pdf"

TAU = 0.5

# One unique FOV per baseline (4 baselines × 1 FOV). Selection criteria:
# outliers excluded, |ΔSQ|<0.05, ΔDQ>0.05, ΔPQ>0; ranked by ΔDQ then made unique.
SELECTED = [
    ("TCGA-A2-A04X-01Z-00-DX1_002", "BLEEP-ft",   (+0.008, +0.096, +0.065)),  # Breast n=192
    ("TCGA-DW-7838-01Z-00-DX1_2",   "CLIP-ft",    (+0.005, +0.123, +0.078)),  # Kidney n=205
    ("TCGA-55-7573-01Z-00-DX1_004", "PLIP-ft",    (+0.027, +0.148, +0.104)),  # Lung   n=311
    ("TCGA-DW-7963-01Z-00-DX1_2",   "OmiCLIP-ft", (+0.042, +0.209, +0.161)),  # Kidney n=5 (marginal)
]


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

    # Load only models that appear in SELECTED
    needed_models = set(b for _, b, _ in SELECTED)
    model_data = {m: load_npz(NPZ_BY_MODEL[m]) for m in needed_models}
    sigmma = load_npz(SIGMMA_NPZ)

    needed = set(iid for iid, _, _ in SELECTED)
    rgb_dict = {}
    for tile in iter_monusac_test(str(DATA_ROOT)):
        if tile.image_id in needed:
            rgb_dict[tile.image_id] = tile.image_rgb
            if len(rgb_dict) == len(needed):
                break

    n_rows = len(SELECTED)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.3))

    col_titles = ["H&E (224×224)", "GT instances", "Baseline pred", "SIGMMA pred"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9)

    for r, (iid, best_bl, deltas) in enumerate(SELECTED):
        if iid not in rgb_dict:
            for c in range(n_cols):
                axes[r, c].axis("off")
                axes[r, c].text(0.5, 0.5, f"missing\n{iid}", ha="center", va="center")
            continue
        rgb = rgb_dict[iid]
        b = model_data[best_bl][iid]
        s = sigmma[iid]
        gt = b["gt"].astype(bool)
        gt_inst = label_components(gt)
        baseline_inst = prob_to_instances(b["prob"], threshold=TAU)
        sigmma_inst  = prob_to_instances(s["prob"], threshold=TAU)

        dsq, ddq, dpq = deltas
        short = iid.replace("TCGA-", "").replace("-01Z-00-DX1", "")

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(
            f"{short}\nvs {best_bl}\n"
            f"ΔSQ={dsq:+.2f}  ΔDQ={ddq:+.2f}\n"
            f"ΔPQ={dpq:+.2f}",
            fontsize=6.5, rotation=0, ha="right", va="center", labelpad=65,
        )

        axes[r, 1].imshow(colorize_instances(gt_inst))
        axes[r, 2].imshow(colorize_instances(baseline_inst))
        axes[r, 2].set_title(best_bl, fontsize=7, color="#555555") if r > 0 else None
        # For r==0 the column title already says "Best non-SIGMMA baseline pred";
        # add baseline name as a small sub-annotation under the panel.
        axes[r, 2].text(0.02, 0.98, best_bl, transform=axes[r, 2].transAxes,
                        ha="left", va="top", fontsize=6.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none",
                                  boxstyle="round,pad=0.2"))
        axes[r, 3].imshow(colorize_instances(sigmma_inst))

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.suptitle(
        "One representative FOV per baseline where SIGMMA wins on detection\n"
        "|ΔSQ| < 0.05, ΔDQ > 0.05, ΔPQ > 0; outlier FOVs (n_cells ≤ 2) excluded",
        fontsize=9, y=0.995,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
