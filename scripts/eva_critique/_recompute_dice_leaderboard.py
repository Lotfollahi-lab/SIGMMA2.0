"""Recompute MoNuSAC Dice on saved test_artifacts.npz using EVA leaderboard's
MonaiDiceScore definition (include_background=True, num_classes=6,
ignore_index=5, reduction='none' then nanmean across classes).

Compare to our existing `_eval_dice` (include_background=False, num_classes=5).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from monai.metrics import DiceMetric


def recompute(npz_path: str, num_classes_real: int = 5, ignore_idx: int = 5):
    z = np.load(npz_path, allow_pickle=False)
    preds = torch.from_numpy(z["pred"]).long()        # (N, 224, 224) — argmax of probs
    gts = torch.from_numpy(z["gt"]).long()            # (N, 224, 224)

    # Map ambiguous (5) -> background (0) in both pred and gt to match
    # `_eval_dice` behaviour and the EVA `ignore_index=5` semantics.
    pred_masked = preds.clone()
    gt_masked = gts.clone()
    amb = gts == ignore_idx
    pred_masked[amb] = 0
    gt_masked[amb] = 0

    results = {}

    # (A) Our metric: include_background=False, num_classes=5, reduction='mean'
    m_ours = DiceMetric(
        include_background=False, reduction="mean",
        ignore_empty=True, num_classes=num_classes_real,
    )
    m_ours(y_pred=pred_masked.unsqueeze(1), y=gt_masked.unsqueeze(1))
    results["ours_include_bg_False"] = m_ours.aggregate().item()
    m_ours.reset()

    # (B) Leaderboard-style: include_background=True, num_classes=5, reduction='mean'
    m_lb = DiceMetric(
        include_background=True, reduction="mean",
        ignore_empty=True, num_classes=num_classes_real,
    )
    m_lb(y_pred=pred_masked.unsqueeze(1), y=gt_masked.unsqueeze(1))
    results["leaderboard_include_bg_True"] = m_lb.aggregate().item()
    m_lb.reset()

    # (C) Per-class breakdown (include_background=True, reduction='none' -> mean per class)
    m_pc = DiceMetric(
        include_background=True, reduction="none",
        ignore_empty=True, num_classes=num_classes_real,
    )
    m_pc(y_pred=pred_masked.unsqueeze(1), y=gt_masked.unsqueeze(1))
    raw = m_pc.aggregate()
    # raw shape: (N, num_classes) with NaN where GT empty for that class.
    per_class = torch.nanmean(raw, dim=0).cpu().numpy()
    class_names = ["background", "epithelial", "lymphocyte", "neutrophil", "macrophage"]
    results["per_class_dice"] = {n: float(v) for n, v in zip(class_names, per_class)}
    m_pc.reset()

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to test_artifacts.npz")
    args = ap.parse_args()
    out = recompute(args.npz)
    print(f"Ours (include_bg=False):        {out['ours_include_bg_False']:.4f}")
    print(f"Leaderboard (include_bg=True):  {out['leaderboard_include_bg_True']:.4f}")
    print("Per-class Dice:")
    for k, v in out["per_class_dice"].items():
        print(f"  {k:12s} = {v:.4f}")
