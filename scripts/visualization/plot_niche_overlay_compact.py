"""Redraw 2D niche-transfer overlays for paper, keeping only
UNI + SIGMMA micro/meso/macro columns (plus GT).

Reuses cached predictions.npz from probe_niche_cross_slide.py — no re-fitting.

Nature-style: sans-serif, compact, double-column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

SCALES = ("micro", "meso", "macro")
LABEL_FIELDS = ("niche_category", "niches_terra")

# Embeddings to keep (matches scripts/plot_niche_transfer_bars.py).
KEEP_EMBS = ("z_uni_macro", "z_sigmma_micro", "z_sigmma_meso", "z_sigmma_macro")
EMB_TITLE = {
    "z_uni_macro": "UNI",
    "z_sigmma_micro":  "SIGMMA-micro",
    "z_sigmma_meso":   "SIGMMA-meso",
    "z_sigmma_macro":  "SIGMMA-macro",
}

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _read_thumbnail(zarr_path: Path, target_long_side: int = 1500):
    z = zarr.open(str(zarr_path), mode="r")
    C, H, W = z.shape
    step = max(1, int(np.ceil(max(H, W) / target_long_side)))
    thumb = z[:, ::step, ::step][:]
    if thumb.dtype != np.uint8:
        thumb = thumb.astype(np.uint8)
    return np.transpose(thumb, (1, 2, 0)), step


def _load_centroids(preproc_path: Path) -> pd.DataFrame:
    # Cell centroids in H&E-image pixel space. The preprocessed h5 stores these
    # in obsm/spatial (the obs/x_pixel,y_pixel fields were dropped when coords
    # moved to µm); obsm/spatial stays aligned to the H&E pixel grid.
    with h5py.File(preproc_path, "r") as f:
        cell_ids = [c.decode() if isinstance(c, bytes) else c
                    for c in f["obs/cell_id"][:]]
        spatial = f["obsm/spatial"][:]
        x = spatial[:, 0]
        y = spatial[:, 1]
    return pd.DataFrame(
        {"cell_id": cell_ids, "x_pixel": x, "y_pixel": y}
    ).set_index("cell_id")


def _load_slide_meta(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        return {
            "cell_id": np.array([c.decode() for c in f["cell_id"][:]]),
            "niche_category": np.array([c.decode() for c in f["niche_category"][:]]),
            "niches_terra":   np.array([c.decode() for c in f["niches_terra"][:]]),
            "sample": f.attrs["sample_name"],
        }


def _gather_test_slides(features_dir: Path, manifest: pd.DataFrame):
    out = []
    for sample in manifest.loc[manifest["split"] == "test", "sample_name"]:
        p = features_dir / f"{sample}.h5"
        if p.exists():
            out.append(_load_slide_meta(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path,
                    default=Path("outputs/cell_niche/21052"))
    ap.add_argument("--features_dir", type=Path,
                    default=Path("/lustre/scratch126/cellgen/lotfollahi/dj16/"
                                 "projects/sigmma_outputs/cache/cell_niche/"
                                 "features/19044"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("configs/cell_niche_evaluable_slides.csv"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("figures/cell_niche/21052"))
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    test_slides = _gather_test_slides(args.features_dir, manifest)
    if not test_slides:
        raise SystemExit("no test slides found")

    # Pick the slide with most labelled cells (same rule as probe script).
    best_idx, best_n, best_sample = -1, 0, None
    for i, s in enumerate(test_slides):
        n = sum(int((s[lf] != "").sum()) for lf in LABEL_FIELDS)
        if n > best_n:
            best_idx, best_n, best_sample = i, n, s["sample"]

    # Offset of the target slide within the concatenated test set.
    offsets = np.cumsum([0] + [len(s["cell_id"]) for s in test_slides])
    ts, te = offsets[best_idx], offsets[best_idx + 1]
    print(f"[viz] target slide={best_sample}  cells[{ts}:{te}]")

    target = test_slides[best_idx]
    row = manifest[manifest["sample_name"] == best_sample].iloc[0]
    thumb, step = _read_thumbnail(Path(row["hematoxylin_eosin_image"]))
    centroids = _load_centroids(Path(row["spatial_preprocessed"]))
    cents_xy = np.stack([
        centroids.loc[c, ["x_pixel", "y_pixel"]].to_numpy()
        if c in centroids.index else (np.nan, np.nan)
        for c in target["cell_id"]
    ])
    have_centroid = ~np.isnan(cents_xy[:, 0])

    preds = np.load(args.run_dir / "predictions.npz", allow_pickle=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(NATURE_RC):
        for label_field in LABEL_FIELDS:
            gt_target = target[label_field]
            preds_per_emb = []
            for emb in KEEP_EMBS:
                key = f"{label_field}/{emb}/test/y_pred"
                if key not in preds.files:
                    raise SystemExit(f"missing key in predictions.npz: {key}")
                preds_per_emb.append(preds[key][ts:te])

            keep = have_centroid & (gt_target != "")
            xs = cents_xy[keep, 0] / step
            ys = cents_xy[keep, 1] / step
            gt = gt_target[keep]
            preds_kept = [p[keep] for p in preds_per_emb]

            classes = sorted({c for c in gt if c} |
                             {c for vec in preds_kept for c in vec if c})
            cmap = plt.get_cmap("tab20", max(len(classes), 1))
            colour = {c: cmap(i) for i, c in enumerate(classes)}

            panels = [("Ground truth", gt)] + list(zip(
                [EMB_TITLE[e] for e in KEEP_EMBS], preds_kept))
            n = len(panels)

            # Nature double-column ~7.2in; 5 panels in a row.
            fig, axes = plt.subplots(1, n, figsize=(1.45 * n, 1.9))
            for ax, (title, vec) in zip(axes, panels):
                ax.imshow(thumb)
                cs = [colour[c] if c else (0, 0, 0, 0) for c in vec]
                ax.scatter(xs, ys, c=cs, s=0.8, linewidths=0, alpha=0.85)
                ax.set_title(title, pad=2)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

            handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                  markerfacecolor=colour[c],
                                  markeredgecolor="none",
                                  markersize=4, label=c) for c in classes]
            ncol = min(8, len(classes))
            fig.legend(handles=handles, loc="lower center", ncol=ncol,
                       frameon=False, bbox_to_anchor=(0.5, -0.04),
                       handletextpad=0.4, columnspacing=0.8)

            fig.tight_layout(w_pad=0.4)
            out_png = args.out_dir / f"{label_field}_overlay.png"
            out_pdf = args.out_dir / f"{label_field}_overlay.pdf"
            fig.savefig(out_png, dpi=600, bbox_inches="tight")
            fig.savefig(out_pdf, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
