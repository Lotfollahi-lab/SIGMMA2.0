"""Per-section predicted-vs-GT tumour-stroma composition for the RCC probe.

``predictions.npz`` stores per-split ``y_true``/``y_pred`` stacked across that
split's slides with no per-cell section id. This script reconstructs the section
boundaries by replaying the probe's masking offline from the cached per-slide
DATA HDF5s, then VERIFIES the reconstruction by asserting the rebuilt ``y_true``
is byte-identical to the npz ``y_true`` (per split) before slicing predictions
per section. It aborts on mismatch rather than emit a misaligned plot.

Only held-out splits have honest predictions, so train sections are excluded.
For each section that CONTAINS tumour epithelium in GT, it draws a GT bar next
to a Predicted bar:

  * mode=tumor_stroma (default): 2-way Tumour vs Stroma, each normalised to
    (Tumour + Stroma) cells — the tumour-stroma ratio. Stroma = {Fibroblast,
    Blood vasculature, Lymphatic vasculature, Perivascular cell}.
  * mode=three_way: Tumour / Stroma / Other over all evaluated cells.

Usage:
  python scripts/plot_tumor_stroma_pred_vs_gt_per_section.py \
      --data_dir <cache>/data/cell_niche_rcc/scales_64_128_256 \
      --manifest configs/cell_niche_evaluable_slides_rcc.csv \
      --predictions outputs/cell_niche_rcc/51136/predictions.npz \
      --embedding z_sigmma_concat --splits test val --out_dir figures/clinical_outcome
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
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


def _slide_labels(data_h5: Path, label: str) -> np.ndarray:
    with h5py.File(data_h5, "r") as f:
        return np.array([c.decode() if isinstance(c, bytes) else c
                         for c in f[label][:]])


def _counts(labels: np.ndarray) -> tuple[int, int, int]:
    n = len(labels)
    t = int(np.isin(labels, TUMOR).sum())
    s = int(np.isin(labels, STROMA).sum())
    return t, s, n - t - s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--embedding", default="z_sigmma_concat")
    ap.add_argument("--label", default="level_2_cell_type")
    ap.add_argument("--splits", nargs="*", default=["test", "val"],
                    help="Held-out splits with honest predictions (no train).")
    ap.add_argument("--mode", default="tumor_stroma",
                    choices=["tumor_stroma", "three_way"])
    ap.add_argument("--min_tumor_cells", type=int, default=1,
                    help="Keep sections whose GT has >= this many tumour cells.")
    ap.add_argument("--min_tumor_frac", type=float, default=0.10,
                    help="Keep sections whose GT tumour fraction (of all "
                         "evaluated cells) exceeds this. Default 0.10.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--figsize", default="10,6")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    man = pd.read_csv(args.manifest)
    d = np.load(args.predictions, allow_pickle=True)

    # Train classes = non-empty labels across train slides (LogReg support).
    train_classes = set()
    for s in man[man.split == "train"]["sample_name"]:
        h5 = data_dir / f"{s}.h5"
        if h5.exists():
            train_classes.update(x for x in _slide_labels(h5, args.label) if x != "")

    rows = []
    for split in args.splits:
        split_rows = man[man.split == split]["sample_name"].tolist()
        sections, kept_labels, counts = [], [], []
        for s in split_rows:
            h5 = data_dir / f"{s}.h5"
            if not h5.exists():
                continue
            lab = _slide_labels(h5, args.label)
            keep = np.isin(lab, list(train_classes)) & (lab != "")
            sections.append(s)
            kept_labels.append(lab[keep])
            counts.append(int(keep.sum()))
        if not sections:
            print(f"  [skip] no {split} slides")
            continue
        y_true_recon = np.concatenate(kept_labels)
        k_true = f"{args.label}/{args.embedding}/{split}/y_true"
        k_pred = f"{args.label}/{args.embedding}/{split}/y_pred"
        if k_true not in d.files:
            print(f"  [skip] {split}: {k_true} not in npz")
            continue
        if not np.array_equal(y_true_recon, d[k_true]):
            raise SystemExit(f"Reconstruction mismatch for split={split} — aborting.")
        print(f"[verify] {split}: reconstruction matches npz "
              f"(n={len(y_true_recon):,}, {len(sections)} sections)")
        y_pred = d[k_pred]
        offs = np.cumsum([0] + counts)
        for i, sec in enumerate(sections):
            gt = kept_labels[i]
            pr = y_pred[offs[i]:offs[i + 1]]
            gt_t, gt_s, gt_o = _counts(gt)
            pr_t, pr_s, pr_o = _counts(pr)
            rows.append({"section": sec, "split": split, "n": counts[i],
                         "gt_t": gt_t, "gt_s": gt_s, "gt_o": gt_o,
                         "pr_t": pr_t, "pr_s": pr_s, "pr_o": pr_o})

    comp = pd.DataFrame(rows)
    # Keep sections with a meaningful tumour-epithelium presence in GT:
    # >= min_tumor_cells AND GT tumour fraction (of all evaluated cells) above
    # min_tumor_frac.
    before = len(comp)
    gt_tumor_frac_all = comp["gt_t"] / (comp["gt_t"] + comp["gt_s"] + comp["gt_o"])
    comp = comp[(comp["gt_t"] >= args.min_tumor_cells)
                & (gt_tumor_frac_all > args.min_tumor_frac)].copy()
    dropped = before - len(comp)
    if dropped:
        print(f"[filter] dropped {dropped} section(s): GT tumour < "
              f"{args.min_tumor_cells} cells or <= {args.min_tumor_frac:.0%} of cells")

    if args.mode == "tumor_stroma":
        comp["gt_den"] = (comp["gt_t"] + comp["gt_s"]).replace(0, np.nan)
        comp["pr_den"] = (comp["pr_t"] + comp["pr_s"]).replace(0, np.nan)
        comp["gt_tumor"] = comp["gt_t"] / comp["gt_den"]
        comp["gt_stroma"] = comp["gt_s"] / comp["gt_den"]
        comp["pr_tumor"] = comp["pr_t"] / comp["pr_den"]
        comp["pr_stroma"] = comp["pr_s"] / comp["pr_den"]
        segs = [("tumor", C_TUMOR, "Tumour epithelium"),
                ("stroma", C_STROMA, "Stroma")]
        ynote = "Proportion of (Tumour + Stroma) cells"
    else:
        for who in ("gt", "pr"):
            tot = (comp[f"{who}_t"] + comp[f"{who}_s"] + comp[f"{who}_o"]).replace(0, np.nan)
            comp[f"{who}_tumor"] = comp[f"{who}_t"] / tot
            comp[f"{who}_stroma"] = comp[f"{who}_s"] / tot
            comp[f"{who}_other"] = comp[f"{who}_o"] / tot
        segs = [("tumor", C_TUMOR, "Tumour epithelium"),
                ("stroma", C_STROMA, "Stroma"),
                ("other", C_OTHER, "Other (immune + nephron)")]
        ynote = "Proportion of evaluated cells"

    comp = comp.sort_values("gt_tumor", ascending=False).reset_index(drop=True)
    csv_path = out_dir / f"tumor_stroma_pred_vs_gt_per_section_{args.embedding}_{args.mode}.csv"
    comp.to_csv(csv_path, index=False)
    print(f"[csv] wrote {csv_path}")
    print(comp[["section", "split", "n", "gt_tumor", "pr_tumor"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- figure: per section, GT bar then Pred bar ----
    plt.rcParams.update({"font.size": 13, "font.family": "sans-serif",
                         "svg.fonttype": "none"})
    figsize = tuple(float(v) for v in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=figsize)
    xpos, xticklab, centers = [], [], []
    cur = 0.0
    for i in range(len(comp)):
        gx, px = cur, cur + 1.0
        for x, who in ((gx, "gt"), (px, "pr")):
            base = 0.0
            for key, col, _ in segs:
                frac = comp.iloc[i][f"{who}_{key}"]
                frac = 0.0 if pd.isna(frac) else frac
                ax.bar(x, frac, width=0.85, bottom=base, color=col,
                       edgecolor="white", linewidth=0.4)
                if frac > 0.07:
                    ax.text(x, base + frac / 2, f"{100*frac:.0f}", ha="center",
                            va="center", fontsize=12, color="white" if key == "tumor" else "0.25")
                base += frac
            xpos.append(x)
            xticklab.append("GT" if who == "gt" else "Pred")
        centers.append((gx + px) / 2)
        cur = px + 1.4

    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ynote)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xticklab, fontsize=12)
    ax.set_xlim(-0.7, xpos[-1] + 0.7)
    for c, (_, r) in zip(centers, comp.iterrows()):
        ax.text(c, -0.14, f"{r['section']}\n(n={int(r['n']):,})",
                ha="center", va="top", fontsize=11,
                transform=ax.get_xaxis_transform())

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in segs]
    ax.legend(handles, [lab for _, _, lab in segs], loc="lower center",
              frameon=False, ncol=len(segs), fontsize=12, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"RCC kidney — per-section GT vs predicted tumour-stroma "
                 f"composition ({args.embedding}; tumour-containing sections)",
                 fontsize=13, y=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    stem = f"tumor_stroma_pred_vs_gt_per_section_{args.embedding}_{args.mode}"
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fig] wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
