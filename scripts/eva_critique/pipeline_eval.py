"""Run components 1, 2, 3, 5 from cached test artifacts (per-tile prob + attn).

Inputs:
  --uni_artifacts     {dir}/test_artifacts.npz      (from train_decoder linear)
  --sigmma_artifacts  {dir}/test_artifacts.npz
  --uni_attention     {dir}/attention.npz
  --sigmma_attention  {dir}/attention.npz
  --data_root         MoNuSAC root (for raw RGB + GT to rebuild regions)

Outputs (under --out_dir):
  region_stratified_metrics.csv   (Component 1)
  alternative_metrics.csv         (Component 2)
  instance_metrics.csv            (Component 3 — sweep tau, then pick tau* per model)
  attention_consistency.csv       (Component 5)
  figures/attention_grid_<tile>.png  (Component 5 figure for ~20 tiles)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from skimage.measure import find_contours

from scripts.eva_critique.data import iter_monusac_test
from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.metrics_seg import (
    aji,
    auprc_in_region,
    boundary_f1,
    detection_f1_hungarian,
    dice,
    ece_in_region,
    fp_count,
    fp_rate,
    hd95_asd,
    iou,
    label_components,
    mean_prob,
    panoptic_quality,
    recall,
)


TAU_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)


# ============================================================
# Helpers
# ============================================================
def _load_artifacts(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    return {
        "image_id": z["image_id"].astype(str),
        "prob": z["prob"].astype(np.float32),     # (N, 224, 224) — fg prob
        "pred": z["pred"],                        # (N, 224, 224) — argmax class
        "gt":   z["gt"],                          # (N, 224, 224) — semantic
        "entropy": z["entropy"].astype(np.float32),
    }


def _index_by_id(arts):
    return {iid: i for i, iid in enumerate(arts["image_id"])}


def _load_attention(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    return {
        "image_id": z["image_id"].astype(str),
        "attn": z["attn"].astype(np.float32),     # (N, 16, 14, 14)
        "attn_mean": z["attn_mean"].astype(np.float32),
    }


def _upsample_attn(a14: np.ndarray) -> np.ndarray:
    """Bilinear upsample (..., 14, 14) -> (..., 224, 224) using PIL-style."""
    import torch
    t = torch.from_numpy(a14).unsqueeze(0).unsqueeze(0)  # (1,1,14,14) per call
    out = torch.nn.functional.interpolate(t, size=(224, 224),
                                          mode="bilinear", align_corners=False)
    return out.squeeze().numpy()


def _norm_unit(a: np.ndarray) -> np.ndarray:
    s = a.sum()
    return a / s if s > 0 else a


# ============================================================
# Component 1: region-stratified metrics
# ============================================================
def component1(tile, prob_uni, prob_sig, tau: float = 0.5) -> Dict:
    pred_uni = prob_uni >= tau
    pred_sig = prob_sig >= tau
    nuc = tile.nuclei_gt

    rows = []
    for model_name, pred, prob in (("uni", pred_uni, prob_uni),
                                   ("sigmma", pred_sig, prob_sig)):
        rows.append({
            "image_id": tile.image_id, "model": model_name, "tau": tau,
            "R1_dice": dice(pred, nuc, mask=tile.R1),
            "R1_iou":  iou(pred, nuc, mask=tile.R1),
            "R1_recall": recall(pred, nuc, mask=tile.R1),
            "R2_fp_rate": fp_rate(pred, tile.R2),
            "R2_fp_count": fp_count(pred, tile.R2),
            "R2_meanp": mean_prob(prob, tile.R2),
            "R3_fp_rate": fp_rate(pred, tile.R3),
            "R3_fp_count": fp_count(pred, tile.R3),
            "R3_meanp": mean_prob(prob, tile.R3),
            "R4_dice": dice(pred, nuc, mask=tile.R4),
        })
    return rows


# ============================================================
# Component 2: alternative metrics
# ============================================================
def component2(tile, prob_uni, prob_sig, full_probs_uni, full_probs_sig,
               tau: float = 0.5) -> List[Dict]:
    nuc = tile.nuclei_gt
    rows = []
    for name, prob, full in (("uni", prob_uni, full_probs_uni),
                             ("sigmma", prob_sig, full_probs_sig)):
        pred = (prob >= tau).astype(np.uint8)
        gt_inst = label_components(nuc)
        pr_inst = label_components(pred.astype(bool))
        AJI = aji(pr_inst, gt_inst)
        PQ, SQ, DQ, _, _, _ = panoptic_quality(pr_inst, gt_inst, iou_thr=0.5)
        hd95, asd = hd95_asd(pred.astype(bool), nuc)
        bf = {f"BF1_d{d}": boundary_f1(pred.astype(bool), nuc, d) for d in (2, 3, 5)}
        det = detection_f1_hungarian(pr_inst, gt_inst,
                                     iou_thresholds=(0.5, 0.7, 0.9))
        row = {"image_id": tile.image_id, "model": name, "tau": tau,
               "AJI": AJI, "PQ": PQ, "SQ": SQ, "DQ": DQ,
               "HD95": hd95, "ASD": asd, **bf,
               "DetF1@0.5": det["f1@0.5"],
               "DetF1@0.7": det["f1@0.7"],
               "DetF1@0.9": det["f1@0.9"]}
        # Per-region AUPRC, ECE, mean entropy.
        for rk, mask in (("R1", tile.R1), ("R2", tile.R2), ("R3", tile.R3)):
            row[f"AUPRC_{rk}"] = auprc_in_region(prob, nuc, mask)
            row[f"ECE_{rk}"]   = ece_in_region(prob, nuc, mask)
            from scripts.eva_critique.metrics_seg import mean_entropy_in_region
            row[f"H_{rk}"]     = mean_entropy_in_region(full, mask)
        rows.append(row)
    return rows


# ============================================================
# Component 3: object-level pipeline (sweep tau)
# ============================================================
def component3(tile, prob_uni, prob_sig) -> List[Dict]:
    nuc = tile.nuclei_gt
    gt_inst = label_components(nuc)
    rows = []
    for name, prob in (("uni", prob_uni), ("sigmma", prob_sig)):
        for tau in TAU_GRID:
            inst = prob_to_instances(prob, threshold=tau)
            det = detection_f1_hungarian(inst, gt_inst,
                                         iou_thresholds=(0.5, 0.7, 0.9))
            row = {"image_id": tile.image_id, "model": name, "tau": tau,
                   "DetF1@0.5": det["f1@0.5"], "DetF1@0.7": det["f1@0.7"],
                   "DetF1@0.9": det["f1@0.9"], "mIoU_TPs@0.5": det["miou_tps@0.5"],
                   "n_split": det["n_split"], "n_merge": det["n_merge"]}
            rows.append(row)
    return rows


# ============================================================
# Component 5: attention–prediction consistency
# ============================================================
def component5(tile, prob_uni, prob_sig, attn_uni, attn_sig) -> List[Dict]:
    """attn_{uni,sig}: (16, 14, 14) per-head."""
    rows = []
    M_R1 = tile.R1.astype(bool); M_R2 = tile.R2.astype(bool); M_R3 = tile.R3.astype(bool)
    M_nuc = tile.nuclei_gt.astype(bool)

    for name, attn, prob in (("uni", attn_uni, prob_uni),
                             ("sigmma", attn_sig, prob_sig)):
        # Head-mean (1, 14, 14) and per-head — upsample each.
        heads = list(attn) + [attn.mean(axis=0)]
        head_names = [f"h{h}" for h in range(attn.shape[0])] + ["head_mean"]
        for hname, h14 in zip(head_names, heads):
            A = _upsample_attn(h14)
            A = _norm_unit(A.astype(np.float64))
            P = prob.astype(np.float64)
            align_gt = float((A[M_nuc].sum())) if M_nuc.sum() > 0 else float("nan")
            waste_R2 = float(A[M_R2].sum()) if M_R2.sum() > 0 else float("nan")
            waste_R3 = float(A[M_R3].sum()) if M_R3.sum() > 0 else float("nan")
            try:
                pr = float(pearsonr(A.ravel(), P.ravel())[0])
            except Exception:
                pr = float("nan")
            try:
                sp = float(spearmanr(A.ravel(), P.ravel())[0])
            except Exception:
                sp = float("nan")
            d = {"image_id": tile.image_id, "model": name, "head": hname,
                 "AlignGT": align_gt, "Waste_R2": waste_R2, "Waste_R3": waste_R3,
                 "Pearson_r": pr, "Spearman_rho": sp}
            for k, mask in (("R1", M_R1), ("R2", M_R2), ("R3", M_R3)):
                if mask.sum() > 0:
                    d[f"Delta_{k}"] = float(A[mask].mean() - P[mask].mean())
                else:
                    d[f"Delta_{k}"] = float("nan")
            rows.append(d)
    return rows


def figure_attention_grid(tile, attn_uni, attn_sig, prob_uni, prob_sig,
                          out_path: str, n_heads_show: int = 6):
    """input | head1..6 | head-mean ;  rows: SIGMMA, UNI ; blue GT contour."""
    contours = find_contours(tile.nuclei_gt.astype(np.float32), level=0.5)
    cols = 1 + n_heads_show + 1
    fig, axes = plt.subplots(2, cols, figsize=(2.0 * cols, 4.4))
    for r, (name, attn) in enumerate((("SIGMMA", attn_sig), ("UNI", attn_uni))):
        axes[r, 0].imshow(tile.image_rgb)
        axes[r, 0].set_ylabel(name, fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("input", fontsize=8)
        for ci in range(n_heads_show):
            ax = axes[r, 1 + ci]
            A = _upsample_attn(attn[ci])
            ax.imshow(tile.image_rgb)
            ax.imshow(A, cmap="inferno", alpha=0.55,
                      vmin=A.min(), vmax=A.max())
            for c in contours:
                ax.plot(c[:, 1], c[:, 0], color="#1f77ff", lw=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"head {ci+1}", fontsize=8)
        ax = axes[r, -1]
        A = _upsample_attn(attn.mean(axis=0))
        ax.imshow(tile.image_rgb)
        ax.imshow(A, cmap="inferno", alpha=0.55,
                  vmin=A.min(), vmax=A.max())
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color="#1f77ff", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("head-mean", fontsize=8)
    fig.suptitle(f"{tile.image_id}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _pick_figure_tiles(tiles, n: int = 20):
    """Pick ~n tiles spanning R1-, R2-, R3-dominant cases (by area fraction)."""
    scored = []
    for t in tiles:
        f1 = t.R1.mean(); f2 = t.R2.mean(); f3 = t.R3.mean()
        scored.append((t, f1, f2, f3))
    pick = []
    for which in (1, 2, 3):
        ranked = sorted(scored, key=lambda r: -r[which])
        pick.extend(t for t, *_ in ranked[: n // 3])
    seen = set(); out = []
    for t in pick:
        if t.image_id in seen: continue
        seen.add(t.image_id); out.append(t)
        if len(out) >= n:
            break
    return out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uni_artifacts", required=True)
    ap.add_argument("--sigmma_artifacts", required=True)
    ap.add_argument("--uni_attention", required=True)
    ap.add_argument("--sigmma_attention", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_figure_tiles", type=int, default=20)
    ap.add_argument(
        "--sigmma_label", default="sigmma",
        help="Override the model tag written for the second-slot artifacts "
             "(e.g. 'clip_uni_ft' when --sigmma_artifacts points at a "
             "baseline-UNI EVA run, not SIGMMA).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    arts_u = _load_artifacts(args.uni_artifacts)
    arts_s = _load_artifacts(args.sigmma_artifacts)
    attn_u_pack = _load_attention(args.uni_attention)
    attn_s_pack = _load_attention(args.sigmma_attention)
    idx_u = _index_by_id(arts_u); idx_s = _index_by_id(arts_s)
    iattn_u = {iid: i for i, iid in enumerate(attn_u_pack["image_id"])}
    iattn_s = {iid: i for i, iid in enumerate(attn_s_pack["image_id"])}

    rows_c1, rows_c2, rows_c3, rows_c5 = [], [], [], []
    tiles_kept = []
    n_processed = 0
    for tile in iter_monusac_test(os.path.join(args.data_root, "monusac")):
        if tile.image_id not in idx_u or tile.image_id not in idx_s:
            continue
        i_u, i_s = idx_u[tile.image_id], idx_s[tile.image_id]
        prob_u = arts_u["prob"][i_u]; prob_s = arts_s["prob"][i_s]
        # Reconstruct full softmax (C, H, W) for entropy: only fg vs bg is stored,
        # so we approximate with the 2-class entropy on (1-fg, fg). For the
        # default linear/MLP/UNet decoder this still captures predictive
        # uncertainty; if the cached `entropy` is present, prefer that.
        ent_u = arts_u["entropy"][i_u]
        ent_s = arts_s["entropy"][i_s]
        full_u = np.stack([1 - prob_u, prob_u], axis=0)
        full_s = np.stack([1 - prob_s, prob_s], axis=0)
        # Use cached multi-class entropy as a second column for reference.
        rows_c1.extend(component1(tile, prob_u, prob_s, tau=0.5))
        rows_c2.extend(component2(tile, prob_u, prob_s, full_u, full_s, tau=0.5))
        rows_c3.extend(component3(tile, prob_u, prob_s))

        if tile.image_id in iattn_u and tile.image_id in iattn_s:
            a_u = attn_u_pack["attn"][iattn_u[tile.image_id]]
            a_s = attn_s_pack["attn"][iattn_s[tile.image_id]]
            rows_c5.extend(component5(tile, prob_u, prob_s, a_u, a_s))

        tiles_kept.append(tile)
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"[pipeline_eval] processed {n_processed} tiles")

    # Rewrite the "sigmma" model tag if the second slot was a baseline-UNI
    # variant rather than SIGMMA proper.
    if args.sigmma_label != "sigmma":
        for rows in (rows_c1, rows_c2, rows_c3, rows_c5):
            for row in rows:
                if row.get("model") == "sigmma":
                    row["model"] = args.sigmma_label

    pd.DataFrame(rows_c1).to_csv(os.path.join(args.out_dir,
                                              "region_stratified_metrics.csv"), index=False)
    pd.DataFrame(rows_c2).to_csv(os.path.join(args.out_dir,
                                              "alternative_metrics.csv"), index=False)
    pd.DataFrame(rows_c3).to_csv(os.path.join(args.out_dir,
                                              "instance_metrics.csv"), index=False)
    pd.DataFrame(rows_c5).to_csv(os.path.join(args.out_dir,
                                              "attention_consistency.csv"), index=False)

    # Figures: ~20 tiles, head-mean + first 6 heads, both rows.
    fig_tiles = _pick_figure_tiles(tiles_kept, n=args.n_figure_tiles)
    for tile in fig_tiles:
        if tile.image_id not in iattn_u or tile.image_id not in iattn_s:
            continue
        a_u = attn_u_pack["attn"][iattn_u[tile.image_id]]
        a_s = attn_s_pack["attn"][iattn_s[tile.image_id]]
        i_u, i_s = idx_u[tile.image_id], idx_s[tile.image_id]
        figure_attention_grid(tile, a_u, a_s,
                              arts_u["prob"][i_u], arts_s["prob"][i_s],
                              out_path=os.path.join(fig_dir,
                                                    f"attention_grid_{tile.image_id}.png"))
    print(f"[pipeline_eval] done — processed {n_processed} tiles, "
          f"saved {len(fig_tiles)} attention figures")


if __name__ == "__main__":
    main()
