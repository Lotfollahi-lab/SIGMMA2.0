"""Visualize the two Breast FOVs where SIGMMA α=0.1 catastrophically loses
to PLIP-ft on PQ (TCGA-F9-A8NY_4: Δ −0.320; TCGA-A04X-01Z-00-DX1_003: Δ −0.173).
Both are 100% Macrophage/Neutrophil according to MoNuSAC XML annotations.

Layout: 2 rows × 4 columns.
  Row = FOV, Cols = [H&E image | GT instances | SIGMMA pred | PLIP-ft pred]

Output: figures/eva_critique/breast_failure_fovs.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from skimage.morphology import binary_dilation, disk

import sys
ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
sys.path.insert(0, str(ROOT))
from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.metrics_seg import label_components

FOVS = ["TCGA-F9-A8NY-01Z-00-DX1_4", "TCGA-A2-A04X-01Z-00-DX1_003"]
SEED = 42  # consistent across panels

SIGMMA_NPZ = ROOT / f"outputs/eva_critique_gex_sweep_step1000/a01/seed{SEED}/sigmma_shared_lora/convdec_image/test_artifacts.npz"
PLIP_NPZ   = ROOT / f"outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed{SEED}/plip_uni_ft/convdec_image/test_artifacts.npz"
TIF_DIR    = ROOT / "data/monusac/MoNuSAC Testing Data and Annotations"
OUT_DIR    = ROOT / "figures/eva_critique"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {str(i): (d["prob"][k].astype(np.float32), d["gt"][k])
            for k, i in enumerate(d["image_id"])}


def load_image(fov_id):
    slide = fov_id.rsplit("_", 1)[0]  # strip trailing _N
    path = TIF_DIR / slide / f"{fov_id}.tif"
    img = Image.open(path).convert("RGB")
    return np.array(img)


def color_instances(inst, image=None, bg_color=(1, 1, 1)):
    """Use label2rgb with distinct colors per instance."""
    return label2rgb(inst, image=image, bg_label=0, bg_color=bg_color,
                     alpha=0.45, image_alpha=1.0)


def main():
    sigmma_data = load_artifacts(SIGMMA_NPZ)
    plip_data   = load_artifacts(PLIP_NPZ)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(len(FOVS), 4, figsize=(8.6, 4.4))

    for row, fov in enumerate(FOVS):
        prob_s, gt = sigmma_data[fov]
        prob_p, _  = plip_data[fov]
        H, W = gt.shape

        # Load + resize source TIF to match 224×224 prediction grid
        img = load_image(fov)
        if img.shape[:2] != (H, W):
            img = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        # GT instances + predicted instances (current defaults: sd=7, ms=32)
        gt_inst   = label_components(gt.astype(bool))
        sig_inst  = prob_to_instances(prob_s, threshold=0.5)
        plip_inst = prob_to_instances(prob_p, threshold=0.5)

        # Render
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{fov.split('-',2)[2]}\nH&E (224×224)")
        axes[row, 1].imshow(color_instances(gt_inst, img/255.0))
        axes[row, 1].set_title(f"GT instances (n={gt_inst.max()})")
        axes[row, 2].imshow(color_instances(sig_inst, img/255.0))
        axes[row, 2].set_title(f"SIGMMA α=0.1 (n={sig_inst.max()})")
        axes[row, 3].imshow(color_instances(plip_inst, img/255.0))
        axes[row, 3].set_title(f"PLIP-ft (n={plip_inst.max()})")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    # Per-row label
    annotations = [
        ("Δ PQ = −0.320 (worst SIGMMA loss)", "100% Mac/Neut annotated"),
        ("Δ PQ = −0.173 (2nd worst)",         "100% Mac/Neut annotated"),
    ]
    for row, (a, b) in enumerate(annotations):
        axes[row, 0].set_ylabel(f"{a}\n{b}", fontsize=7.5, color="#666666",
                                rotation=0, ha="right", va="center", labelpad=6)

    fig.suptitle("Breast FOVs where SIGMMA α=0.1 catastrophically loses to PLIP-ft (PQ, Global)",
                 fontsize=9.5, y=1.00)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"breast_failure_fovs.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
