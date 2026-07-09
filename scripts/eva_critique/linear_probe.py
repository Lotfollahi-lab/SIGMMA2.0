"""Patch-level nuclei linear probe: ridge regression on (CLS + mean) tokens.

For each MoNuSAC training image we tile the per-pixel nuclei GT into 16x16
patches (matching the ViT patch grid), label each patch by its mean nuclei
fraction > 0.5, and fit a ridge classifier on the per-image (CLS + mean)
features. We use *image-level* GroupKFold (5 folds) to avoid leakage from
patches of the same WSI region. AUROC is computed per fold and averaged.

Output: {out_dir}/linear_probe_results.csv with rows:
   model, fold, n_train, n_val, auroc
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from scripts.eva_critique.data import _AMBIGUOUS_IDX
from scripts.eva_critique.models import load_sigmma, load_uni
from scripts.eva_critique.train_decoder import _MoNuSACTransformed


def _patch_label(mask_224: np.ndarray) -> np.ndarray:
    """Per-patch (14x14) binary nuclei label from a 224x224 mask."""
    nuc = ((mask_224 >= 1) & (mask_224 != _AMBIGUOUS_IDX)).astype(np.float32)
    # Block-mean over 16x16 patches.
    h = nuc.reshape(14, 16, 14, 16).mean(axis=(1, 3))
    return (h > 0.5).astype(np.uint8).reshape(-1)


def _extract_features(encoder, loader, device):
    feats, labels, groups = [], [], []
    with torch.no_grad():
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            d = encoder.forward_cls_mean(x)
            cls = d["cls"].cpu().numpy()
            mean = d["mean"].cpu().numpy()
            f_img = np.concatenate([cls, mean], axis=1)  # (B, 2D)
            for i, fpath in enumerate(fn):
                # Per-image label vector; use the same per-image feature for
                # every patch in that image (CLS + mean are global tokens).
                msk = y[i].numpy()
                lab = _patch_label(msk)
                feats.append(np.broadcast_to(f_img[i], (lab.size, f_img.shape[1])).copy())
                labels.append(lab)
                groups.extend([fpath] * lab.size)
    return np.concatenate(feats), np.concatenate(labels), np.array(groups)


def _decision_scores(clf, X):
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmma_config", required=True)
    ap.add_argument("--scale", type=int, default=0)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    train_ds = _MoNuSACTransformed(os.path.join(args.data_root, "monusac"),
                                   "train", train=False)
    test_ds = _MoNuSACTransformed(os.path.join(args.data_root, "monusac"),
                                  "test", train=False)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    for model_name in ("uni", "sigmma"):
        if model_name == "uni":
            encoder = load_uni(args.sigmma_config, device)
        else:
            encoder = load_sigmma(args.sigmma_config, device, scale_idx=args.scale)
        print(f"[probe] extracting {model_name} features...")
        X_tr, y_tr, g_tr = _extract_features(encoder, train_loader, device)

        # Single-class fold guard.
        if len(np.unique(y_tr)) < 2:
            print(f"[probe] {model_name}: degenerate labels, skipping")
            continue

        kf = GroupKFold(n_splits=5)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(X_tr, y_tr, groups=g_tr)):
            if len(np.unique(y_tr[idx_va])) < 2:
                continue
            clf = RidgeClassifier(alpha=args.alpha)
            clf.fit(X_tr[idx_tr], y_tr[idx_tr])
            scores = _decision_scores(clf, X_tr[idx_va])
            auroc = float(roc_auc_score(y_tr[idx_va], scores))
            rows.append({"model": model_name, "split": "train_cv",
                         "fold": fold, "n_train": len(idx_tr),
                         "n_val": len(idx_va), "auroc": auroc})
            print(f"[probe] {model_name} fold {fold}: AUROC={auroc:.4f}")

        # Held-out test (single split): train on full train, eval on test.
        X_te, y_te, _ = _extract_features(encoder, test_loader, device)
        if len(np.unique(y_te)) >= 2:
            clf = RidgeClassifier(alpha=args.alpha).fit(X_tr, y_tr)
            scores = _decision_scores(clf, X_te)
            auroc = float(roc_auc_score(y_te, scores))
            rows.append({"model": model_name, "split": "test",
                         "fold": -1, "n_train": len(X_tr),
                         "n_val": len(X_te), "auroc": auroc})
            print(f"[probe] {model_name} TEST: AUROC={auroc:.4f}")

        del encoder
        torch.cuda.empty_cache()

    out = os.path.join(args.out_dir, "linear_probe_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
