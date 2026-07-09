"""Visualize SIGMMA success/failure vs the best model (UNI2-h) under STOCK
watershed params (min_size=16, seed_min_distance=3) — the oversegmenting config.

Best model under sd3/ms16 = UNI2-h (global PQ 0.3245); SIGMMA = full-rerun step500.
Per-FOV PQ computed at sd3/ms16. Top-3 FOVs where SIGMMA beats UNI2-h (success)
+ top-3 where SIGMMA loses to UNI2-h (failure).
Panels: H&E | GT instances | UNI2-h pred | SIGMMA pred.
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
from scripts.eva_critique.metrics_seg import label_components, panoptic_quality

DATA_ROOT = "/nfs/team361/dj16/projects/SIGMMA/data/monusac"
UNI_NPZ = "outputs/eva_step2000/uni2h/seed42/uni/convdec_image/test_artifacts.npz"
SIG_NPZ = "outputs/eva_step2000/a01_36852_step500/seed42/sigmma_shared_lora/convdec_image/test_artifacts.npz"
OUT_PNG = "figures/eva_critique/uni2h_vs_sigmma_stock_sd3ms16_cases.png"
OUT_PDF = "figures/eva_critique/uni2h_vs_sigmma_stock_sd3ms16_cases.pdf"

TAU = 0.5
MS = 16          # stock min_size
SD = 3           # stock seed_min_distance
MIN_GT_INST = 8  # skip near-empty tiles so the viz is meaningful
K = 3            # top-K per direction


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
                         gt=d["gt"][i].astype(np.uint8)) for i in range(len(ids))}


def pq_of(prob, gt):
    pred = prob_to_instances(prob, threshold=TAU, min_size=MS, seed_min_distance=SD)
    gti = label_components(gt.astype(bool))
    PQ, *_ = panoptic_quality(pred, gti, iou_thr=0.5)
    return PQ, pred, gti


def main():
    uni = load_npz(UNI_NPZ)
    sig = load_npz(SIG_NPZ)
    ids = sorted(set(uni) & set(sig))

    rows = []
    cache = {}
    for iid in ids:
        gt = uni[iid]["gt"]
        gti = label_components(gt.astype(bool))
        if int(gti.max()) < MIN_GT_INST:
            continue
        pq_u, pred_u, _ = pq_of(uni[iid]["prob"], gt)
        pq_s, pred_s, _ = pq_of(sig[iid]["prob"], gt)
        cache[iid] = dict(gt_inst=gti, pred_u=pred_u, pred_s=pred_s, pq_u=pq_u, pq_s=pq_s)
        rows.append((iid, pq_s - pq_u, pq_u, pq_s))

    rows.sort(key=lambda x: x[1])
    failures = rows[:K]          # SIGMMA loses most (negative ΔPQ)
    successes = rows[-K:][::-1]  # SIGMMA wins most (positive ΔPQ)
    selected = [("SUCCESS", r) for r in successes] + [("FAILURE", r) for r in failures]

    needed = {r[0] for _, r in selected}
    rgb_dict = {}
    for tile in iter_monusac_test(DATA_ROOT):
        if tile.image_id in needed:
            rgb_dict[tile.image_id] = tile.image_rgb
            if len(rgb_dict) == len(needed):
                break

    n_rows = len(selected)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.7))
    col_titles = ["H&E (224×224)", "GT instances",
                  "UNI2-h pred (best)", "SIGMMA pred (step500)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9)

    for r, (kind, (iid, dpq, pq_u, pq_s)) in enumerate(selected):
        cc = cache[iid]
        rgb = rgb_dict.get(iid)
        if rgb is not None:
            axes[r, 0].imshow(rgb)
        else:
            axes[r, 0].text(0.5, 0.5, "no RGB", ha="center", va="center")
        tag = "SIGMMA↑" if dpq > 0 else "UNI2-h↑"
        axes[r, 0].set_ylabel(
            f"[{kind}] {iid}\nPQ_UNI={pq_u:.3f}  PQ_SIG={pq_s:.3f}\nΔ={dpq:+.3f} ({tag})",
            fontsize=7, rotation=0, ha="right", va="center", labelpad=64,
        )
        axes[r, 1].imshow(colorize_instances(cc["gt_inst"]))
        axes[r, 2].imshow(colorize_instances(cc["pred_u"]))
        axes[r, 3].imshow(colorize_instances(cc["pred_s"]))
        for c in range(n_cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    plt.suptitle(
        "STOCK sd3/ms16 (oversegment): SIGMMA success (top) vs failure (bottom) vs best model UNI2-h\n"
        f"per-FOV global PQ, seed42, GT≥{MIN_GT_INST} instances",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")

    print("\nSelected FOVs:")
    for kind, (iid, dpq, pq_u, pq_s) in selected:
        print(f"  [{kind}] {iid}  PQ_UNI={pq_u:.3f} PQ_SIG={pq_s:.3f} Δ={dpq:+.3f}")


if __name__ == "__main__":
    main()
