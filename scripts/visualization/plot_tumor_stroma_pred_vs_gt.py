"""Predicted-vs-ground-truth tumour-stroma composition for the RCC probe.

Reads ``predictions.npz`` written by ``probe_niche_cross_slide.py`` (per
(label, embedding, split) ``y_true`` / ``y_pred`` arrays) and compares the
COMPOSITION of the evaluated cells: an "Actual" bar from ``y_true`` next to one
bar per embedding from its ``y_pred``. Each bar is a stacked 3-way proportion of

    Tumor  = {Tumour epithelium}
    Stroma = {Fibroblast, Blood vasculature, Lymphatic vasculature,
              Perivascular cell}
    Other  = everything else (immune + Nephron epithelium)

normalised over all evaluated cells (sums to 100%), so over-/under-prediction of
each compartment is directly visible. Note this is COMPOSITION agreement, not
per-cell accuracy — errors can cancel. ``y_true`` is identical across embeddings
(same kept-cell set), so a single Actual bar is shown.

Usage:
  python scripts/plot_tumor_stroma_pred_vs_gt.py \
      --predictions outputs/cell_niche_rcc/51051/predictions.npz \
      --split test --out_dir figures/clinical_outcome
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TUMOR = ["Tumour epithelium"]
STROMA = ["Fibroblast", "Blood vasculature", "Lymphatic vasculature",
          "Perivascular cell"]

C_TUMOR = "#ca7695"
C_STROMA = "#a9c6d3"
C_OTHER = "#cfcfcf"

# Short display titles, matching probe_niche_cross_slide.EMB_TITLE.
EMB_TITLE = {
    "z_uni_macro": "UNI@256-dense",
    "z_uni_macro_cls": "UNI@256-CLS",
    "z_sigmma_micro": "SIGMMA-micro",
    "z_sigmma_meso": "SIGMMA-meso",
    "z_sigmma_macro": "SIGMMA-macro",
    "z_sigmma_concat": "SIGMMA-concat",
}


def _group_fracs(labels: np.ndarray) -> tuple[float, float, float]:
    n = len(labels)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    t = np.isin(labels, TUMOR).sum()
    s = np.isin(labels, STROMA).sum()
    o = n - t - s
    return (t / n, s / n, o / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="probe predictions.npz")
    ap.add_argument("--label", default="level_2_cell_type")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--embeddings", nargs="*", default=None,
                    help="Embedding keys to show (default: all present, in a "
                         "fixed UNI->SIGMMA order).")
    ap.add_argument("--figsize", default="9,5")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.predictions, allow_pickle=True)

    prefix = f"{args.label}/"
    suffix = f"/{args.split}/y_pred"
    present = [k[len(prefix):-len(suffix)]
               for k in d.files if k.startswith(prefix) and k.endswith(suffix)]
    order = list(EMB_TITLE.keys())
    embs = ([e for e in (args.embeddings or order) if e in present])
    if not embs:
        raise SystemExit(f"No matching embeddings for split={args.split}. "
                         f"Present: {present}")

    # y_true identical across embeddings -> take the first.
    y_true = d[f"{prefix}{embs[0]}/{args.split}/y_true"]
    rows = [("Actual (GT)", *_group_fracs(y_true), len(y_true))]
    for e in embs:
        y_pred = d[f"{prefix}{e}/{args.split}/y_pred"]
        rows.append((EMB_TITLE.get(e, e), *_group_fracs(y_pred), len(y_pred)))

    comp = pd.DataFrame(rows, columns=["bar", "tumor", "stroma", "other", "n"])
    csv_path = out_dir / f"tumor_stroma_pred_vs_gt_{args.split}.csv"
    comp.to_csv(csv_path, index=False)
    print(f"[csv] wrote {csv_path}")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- figure ----
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "svg.fonttype": "none"})
    figsize = tuple(float(v) for v in args.figsize.split(","))
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, comp["tumor"], width=0.72, color=C_TUMOR, label="Tumour epithelium",
           edgecolor="white", linewidth=0.4)
    ax.bar(x, comp["stroma"], width=0.72, bottom=comp["tumor"], color=C_STROMA,
           label="Stroma", edgecolor="white", linewidth=0.4)
    ax.bar(x, comp["other"], width=0.72,
           bottom=comp["tumor"] + comp["stroma"], color=C_OTHER,
           label="Other (immune + nephron)", edgecolor="white", linewidth=0.4)

    # % labels inside each segment (skip tiny ones).
    for xi, r in zip(x, comp.itertuples()):
        for frac, base, col in (
            (r.tumor, 0.0, "white"),
            (r.stroma, r.tumor, "0.25"),
            (r.other, r.tumor + r.stroma, "0.35"),
        ):
            if frac > 0.06:
                ax.text(xi, base + frac / 2, f"{100*frac:.0f}", ha="center",
                        va="center", fontsize=6.5, color=col)

    # Separate the Actual bar from the predictions with a light divider.
    ax.axvline(0.5, color="0.8", lw=0.8, ls="--", zorder=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"Proportion of evaluated cells ({args.split}, n={len(y_true):,})")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["bar"], rotation=40, ha="right", fontsize=7)
    ax.set_xlim(-0.7, len(comp) - 0.3)
    ax.legend(loc="lower center", frameon=False, ncol=3, fontsize=7,
              bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"RCC kidney — predicted vs GT tumour-stroma composition "
                 f"({args.split})", fontsize=9, y=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    for ext in ("png", "pdf"):
        p = out_dir / f"tumor_stroma_pred_vs_gt_{args.split}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fig] wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
