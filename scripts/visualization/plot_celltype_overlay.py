"""Standalone re-render of the cell-type overlay (GT + per-embedding predictions
on one test slide), reading SAVED ``predictions.npz`` + the DATA cache rather
than re-running the probe — so figure styling can be iterated cheaply.

Predictions in the npz are stacked across all test slides (kept cells only,
manifest order). This script replays the probe's kept-mask offline to slice the
viz slide's per-cell predictions, and asserts the reconstructed ``y_true`` is
byte-identical to the npz ``y_true`` before trusting the slicing.

Styling vs the in-probe overlay:
  * Tumour epithelium is fixed to #ca7695.
  * RARE classes are drawn LAST (on top) with a larger marker, so sparse immune
    populations (B/T lymphocyte, dendritic, lymphatic) don't vanish under the
    dominant tumour/stroma points.
  * Legend on the far right, one cell type per row.

Usage:
  python scripts/plot_celltype_overlay.py \
      --data_dir <cache>/data/cell_niche_rcc/scales_64_128_256 \
      --manifest configs/cell_niche_evaluable_slides_rcc.csv \
      --predictions outputs/cell_niche_rcc/51136/predictions.npz \
      --out_dir figures/clinical_outcome/51136
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_niche_cross_slide import (  # noqa: E402
    _read_thumbnail, _load_centroids, EMB_TITLE, CELLTYPE_COLOURS,
)

DEFAULT_EMB = ["z_uni_macro", "z_uni_macro_cls",
               "z_sigmma_micro", "z_sigmma_meso", "z_sigmma_macro", "z_sigmma_concat"]


def _slide_labels(data_h5: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(data_h5, "r") as f:
        cid = np.array([c.decode() if isinstance(c, bytes) else c for c in f["cell_id"][:]])
        lab = np.array([c.decode() if isinstance(c, bytes) else c for c in f[label][:]])
    return cid, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--label", default="level_2_cell_type")
    ap.add_argument("--embeddings", nargs="*", default=DEFAULT_EMB)
    ap.add_argument("--slide", default=None, help="Viz slide; default = test slide with most labels.")
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    man = pd.read_csv(args.manifest)
    d = np.load(args.predictions, allow_pickle=True)

    # Train classes = non-empty labels across train slides (probe's LogReg
    # support; test cells of unseen classes were dropped).
    train_classes = set()
    for s in man[man.split == "train"]["sample_name"]:
        h5 = data_dir / f"{s}.h5"
        if h5.exists():
            _, lab = _slide_labels(h5, args.label)
            train_classes.update(x for x in lab if x != "")

    # Rebuild the stacked test order (manifest order), per-slide kept counts,
    # and verify against the npz y_true.
    test_rows = man[man.split == "test"]["sample_name"].tolist()
    slide_cells, kept_counts, kept_label_concat = {}, {}, []
    order = []
    for s in test_rows:
        h5 = data_dir / f"{s}.h5"
        if not h5.exists():
            continue
        cid, lab = _slide_labels(h5, args.label)
        keep = np.isin(lab, list(train_classes)) & (lab != "")
        slide_cells[s] = (cid, lab, keep)
        kept_counts[s] = int(keep.sum())
        kept_label_concat.append(lab[keep])
        order.append(s)
    y_true_recon = np.concatenate(kept_label_concat)
    ref_key = f"{args.label}/{args.embeddings[0]}/test/y_true"
    if not np.array_equal(y_true_recon, d[ref_key]):
        raise SystemExit("Reconstruction mismatch vs npz y_true — aborting.")
    print(f"[verify] reconstruction matches npz ({len(y_true_recon):,} kept cells)")

    # Viz slide.
    if args.slide:
        viz = args.slide
    else:
        viz = max(order, key=lambda s: int((slide_cells[s][1] != "").sum()))
    print(f"[viz] slide = {viz}")
    offsets = {}
    cur = 0
    for s in order:
        offsets[s] = cur
        cur += kept_counts[s]

    cid, gt, keep = slide_cells[viz]
    ncells = len(cid)
    o = offsets[viz]
    n_keep = kept_counts[viz]
    keep_idx = np.flatnonzero(keep)

    # Per-embedding full-slide prediction vectors ("" where dropped/unlabeled).
    preds_per_emb, titles = [], []
    for e in args.embeddings:
        k = f"{args.label}/{e}/test/y_pred"
        if k not in d.files:
            print(f"  [skip] {e}: not in npz")
            continue
        y_pred_slice = d[k][o:o + n_keep]
        full = np.array([""] * ncells, dtype=object)
        full[keep_idx] = y_pred_slice
        preds_per_emb.append(full)
        titles.append(EMB_TITLE.get(e, e))

    # Colours: tab20 + fixed tumour.
    classes = set(c for c in gt if c)
    for v in preds_per_emb:
        classes.update(c for c in v if c)
    classes = sorted(classes)
    cmap = plt.get_cmap("tab20", max(len(classes), 1))
    colour = {c: CELLTYPE_COLOURS.get(c, cmap(i)) for i, c in enumerate(classes)}

    # Global abundance → draw order (rare last/on top) + sizes.
    counts = Counter()
    for v in [gt] + preds_per_emb:
        counts.update(c for c in v if c)
    draw_order = sorted(classes, key=lambda c: counts.get(c, 0), reverse=True)

    # Geometry.
    row = man[man.sample_name == viz].iloc[0]
    thumb, step = _read_thumbnail(Path(row["hematoxylin_eosin_image"]))
    cents = _load_centroids(row["spatial_preprocessed"], args.um_per_px)
    xy = np.stack([cents.loc[c, ["x_pixel", "y_pixel"]].to_numpy()
                   if c in cents.index else (np.nan, np.nan) for c in cid])
    have = ~np.isnan(xy[:, 0])
    xs, ys = xy[:, 0] / step, xy[:, 1] / step

    panels = [(f"Ground truth (n={int((gt!='').sum()):,})", gt)] + list(zip(titles, preds_per_emb))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (title, vec) in zip(axes, panels):
        ax.imshow(thumb, alpha=0.2)  # faint H&E background
        vec = np.asarray(vec)
        # draw_order is abundant-first, so the rarest classes are scattered last
        # and sit on top (reinforced by zorder) — uniform marker size, no boost.
        for z, c in enumerate(draw_order):
            m = have & (vec == c)
            if not m.any():
                continue
            ax.scatter(xs[m], ys[m], c=[colour[c]], s=3, linewidths=0,
                       alpha=0.85, zorder=2 + z)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Legend order: Tumour -> stroma -> rest (CELLTYPE_COLOURS insertion order),
    # unknown classes appended last. Independent of the scatter draw order.
    legend_order = ([c for c in CELLTYPE_COLOURS if c in classes]
                    + [c for c in classes if c not in CELLTYPE_COLOURS])
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=colour[c], markeredgecolor="none",
                          markersize=8, label=c) for c in legend_order]
    fig.legend(handles=handles, loc="center left", ncol=1, frameon=False,
               bbox_to_anchor=(1.0, 0.5), title=args.label, fontsize=9)
    fig.suptitle(f"Cell-type overlay ({args.label}) — test slide {viz} "
                 f"(rare classes drawn on top)", y=1.02, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    stem = f"{args.label}_overlay"
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[fig] wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
