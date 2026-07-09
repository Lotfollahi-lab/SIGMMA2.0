"""Pixel-, region-, and instance-level metrics used by the critique pipeline.

All functions take numpy arrays and operate on a single image (H, W). Probability
maps are nuclei-vs-background (float, [0,1]). Predicted binary masks come from
thresholding that probability or argmax over class channels (caller's choice).

Metrics implemented:
  - dice / iou / recall / fp_rate / fp_count / mean_prob (per region)
  - aji  (aggregated Jaccard index, Kumar 2017)
  - panoptic_quality (PQ = SQ * DQ; nucleus class only)
  - hd95 / asd  (boundary distances, both directions)
  - boundary_f1 at distance d
  - detection_f1 at IoU thresholds (instance-level, Hungarian match)
  - auprc, ece (15 bins), predictive entropy — restricted to a region mask
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label as nd_label
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score


# ============================================================
# Pixel-level
# ============================================================
def dice(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    if mask is not None:
        m = mask.astype(bool)
        p = p & m; g = g & m
    s = p.sum() + g.sum()
    return float(2 * (p & g).sum() / s) if s > 0 else float("nan")


def iou(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    if mask is not None:
        m = mask.astype(bool)
        p = p & m; g = g & m
    u = (p | g).sum()
    return float((p & g).sum() / u) if u > 0 else float("nan")


def recall(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    if mask is not None:
        m = mask.astype(bool)
        p = p & m; g = g & m
    return float((p & g).sum() / g.sum()) if g.sum() > 0 else float("nan")


def fp_rate(pred: np.ndarray, region: np.ndarray) -> float:
    """Fraction of `region` pixels that the model marks foreground."""
    r = region.astype(bool)
    if r.sum() == 0:
        return float("nan")
    return float((pred.astype(bool) & r).sum() / r.sum())


def fp_count(pred: np.ndarray, region: np.ndarray) -> int:
    return int((pred.astype(bool) & region.astype(bool)).sum())


def mean_prob(prob: np.ndarray, region: np.ndarray) -> float:
    r = region.astype(bool)
    if r.sum() == 0:
        return float("nan")
    return float(prob[r].mean())


# ============================================================
# Calibration / AUPRC / entropy
# ============================================================
def auprc_in_region(prob: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    r = region.astype(bool)
    if r.sum() == 0:
        return float("nan")
    y = gt.astype(np.uint8)[r]
    if y.min() == y.max():
        # Degenerate: only one class present. Return prevalence as a baseline.
        return float(y.mean())
    return float(average_precision_score(y, prob[r]))


def ece_in_region(prob: np.ndarray, gt: np.ndarray, region: np.ndarray, n_bins: int = 15) -> float:
    r = region.astype(bool)
    if r.sum() == 0:
        return float("nan")
    p = prob[r]; y = gt.astype(np.float32)[r]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    n = p.size
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        conf = p[sel].mean()
        acc = y[sel].mean()
        ece += (sel.sum() / n) * abs(conf - acc)
    return float(ece)


def mean_entropy_in_region(probs_multiclass: np.ndarray, region: np.ndarray) -> float:
    """Mean per-pixel predictive entropy over `region` for a (C, H, W) softmax map."""
    r = region.astype(bool)
    if r.sum() == 0:
        return float("nan")
    p = np.clip(probs_multiclass, 1e-12, 1.0)
    H = -(p * np.log(p)).sum(axis=0)  # (H, W)
    return float(H[r].mean())


# ============================================================
# Boundary metrics
# ============================================================
def _boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def hd95_asd(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """HD95 and average symmetric surface distance, in pixels."""
    p = pred.astype(bool); g = gt.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0, 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("nan"), float("nan")
    pb = _boundary(p); gb = _boundary(g)
    if pb.sum() == 0 or gb.sum() == 0:
        return float("nan"), float("nan")
    dist_to_g = distance_transform_edt(~gb)
    dist_to_p = distance_transform_edt(~pb)
    d_pg = dist_to_g[pb]
    d_gp = dist_to_p[gb]
    asd = float(0.5 * (d_pg.mean() + d_gp.mean()))
    all_d = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(all_d, 95))
    return hd95, asd


def boundary_f1(pred: np.ndarray, gt: np.ndarray, d: int) -> float:
    """Boundary F1 at tolerance d pixels (Csurka 2013)."""
    p = pred.astype(bool); g = gt.astype(bool)
    pb = _boundary(p); gb = _boundary(g)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    prec = float((dt_g[pb] <= d).mean())
    rec = float((dt_p[gb] <= d).mean())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ============================================================
# Instance-level — needs labelled instances (int arrays)
# ============================================================
def _pairwise_iou(pred_lab: np.ndarray, gt_lab: np.ndarray) -> np.ndarray:
    """Return IoU matrix shape (n_pred, n_gt) for connected-component labels."""
    p_ids = [i for i in np.unique(pred_lab) if i != 0]
    g_ids = [i for i in np.unique(gt_lab) if i != 0]
    iou_mat = np.zeros((len(p_ids), len(g_ids)), dtype=np.float64)
    for i, pi in enumerate(p_ids):
        pm = pred_lab == pi
        psum = pm.sum()
        for j, gj in enumerate(g_ids):
            gm = gt_lab == gj
            inter = (pm & gm).sum()
            if inter == 0:
                continue
            iou_mat[i, j] = inter / (psum + gm.sum() - inter)
    return iou_mat, p_ids, g_ids


def aji(pred_lab: np.ndarray, gt_lab: np.ndarray) -> float:
    """Aggregated Jaccard Index (Kumar 2017)."""
    if pred_lab.max() == 0 and gt_lab.max() == 0:
        return 1.0
    if pred_lab.max() == 0 or gt_lab.max() == 0:
        return 0.0
    iou_mat, p_ids, g_ids = _pairwise_iou(pred_lab, gt_lab)
    # For each GT, pick the pred with the largest IoU.
    used_pred = set()
    inter_sum = 0
    union_sum = 0
    if iou_mat.size:
        best_p_for_g = iou_mat.argmax(axis=0)
        best_iou = iou_mat.max(axis=0)
    else:
        best_p_for_g = np.array([], dtype=int)
        best_iou = np.array([])
    for gi, g in enumerate(g_ids):
        gm = gt_lab == g
        if best_iou.size and best_iou[gi] > 0:
            pi = p_ids[best_p_for_g[gi]]
            used_pred.add(pi)
            pm = pred_lab == pi
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            inter_sum += inter
            union_sum += union
        else:
            union_sum += gm.sum()
    # Unmatched preds add their full area to union.
    for pi in p_ids:
        if pi not in used_pred:
            union_sum += int((pred_lab == pi).sum())
    return float(inter_sum / union_sum) if union_sum > 0 else 0.0


def panoptic_quality(pred_lab: np.ndarray, gt_lab: np.ndarray, iou_thr: float = 0.5):
    """Single-class panoptic quality. Returns (PQ, SQ, DQ, TP, FP, FN)."""
    if pred_lab.max() == 0 and gt_lab.max() == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    iou_mat, p_ids, g_ids = _pairwise_iou(pred_lab, gt_lab)
    matched = iou_mat >= iou_thr
    # PQ assumes each pred matches at most one gt with IoU>=0.5; greedy works.
    tp_pairs: List[Tuple[int, int]] = []
    used_p, used_g = set(), set()
    if iou_mat.size:
        order = np.argsort(-iou_mat, axis=None)
        for flat in order:
            i, j = np.unravel_index(flat, iou_mat.shape)
            if not matched[i, j]:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j)
            tp_pairs.append((i, j))
    tp = len(tp_pairs)
    fp = len(p_ids) - tp
    fn = len(g_ids) - tp
    if tp == 0:
        return 0.0, 0.0, 0.0, tp, fp, fn
    sq = float(np.mean([iou_mat[i, j] for i, j in tp_pairs]))
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    pq = sq * dq
    return float(pq), float(sq), float(dq), tp, fp, fn


def detection_f1_hungarian(
    pred_lab: np.ndarray, gt_lab: np.ndarray, iou_thresholds=(0.5, 0.7, 0.9)
) -> Dict[str, float]:
    """Hungarian-matched detection F1 at multiple IoU thresholds.

    Returns dict {"f1@0.5": ..., "f1@0.7": ..., "f1@0.9": ..., "miou_tps@0.5": ...}.
    Also returns split / merge counts (relative to GT) computed at IoU>=0.1.
    """
    out: Dict[str, float] = {}
    iou_mat, p_ids, g_ids = _pairwise_iou(pred_lab, gt_lab)
    n_p, n_g = len(p_ids), len(g_ids)
    if n_p == 0 and n_g == 0:
        for t in iou_thresholds:
            out[f"f1@{t}"] = 1.0
            out[f"miou_tps@{t}"] = float("nan")
        out["n_split"] = 0
        out["n_merge"] = 0
        return out

    # Hungarian on -IoU; pad the matrix to square.
    if n_p == 0 or n_g == 0:
        for t in iou_thresholds:
            out[f"f1@{t}"] = 0.0
            out[f"miou_tps@{t}"] = float("nan")
        out["n_split"] = 0
        out["n_merge"] = 0
        return out
    cost = -iou_mat
    row, col = linear_sum_assignment(cost)
    matched_iou = iou_mat[row, col]

    for t in iou_thresholds:
        tp_mask = matched_iou >= t
        tp = int(tp_mask.sum())
        fp = n_p - tp
        fn = n_g - tp
        if tp == 0:
            out[f"f1@{t}"] = 0.0
            out[f"miou_tps@{t}"] = float("nan")
        else:
            f1 = 2 * tp / (2 * tp + fp + fn)
            out[f"f1@{t}"] = float(f1)
            out[f"miou_tps@{t}"] = float(matched_iou[tp_mask].mean())

    # Split/merge counts at IoU>=0.1 (lenient overlap):
    overlap = iou_mat >= 0.1
    n_pred_per_gt = overlap.sum(axis=0)  # how many preds overlap each gt
    n_gt_per_pred = overlap.sum(axis=1)  # how many gts overlap each pred
    out["n_split"] = int((n_pred_per_gt >= 2).sum())
    out["n_merge"] = int((n_gt_per_pred >= 2).sum())
    return out


def label_components(binary: np.ndarray) -> np.ndarray:
    lab, _ = nd_label(binary.astype(bool))
    return lab.astype(np.int32)
