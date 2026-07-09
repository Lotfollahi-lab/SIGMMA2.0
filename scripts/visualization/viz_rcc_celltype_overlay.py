#!/usr/bin/env python
"""Re-draw RCC cell-type overlays per test section WITHOUT re-running the probe.

predictions.npz stores, per embedding, the test-split y_true/y_pred concatenated
over labelled cells in test-slide order (n_dropped==0 for level_2_cell_type, so
"kept" == "label != ''"). We reload each test slide's data h5 (cell_id, label,
x_pixel, y_pixel) in the SAME manifest order, split the flat predictions back per
slide, assert the reconstructed y_true matches predictions.npz, then overlay
GT vs predictions on the H&E thumbnail — one row per section, section ID labelled.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr

LABEL = "level_2_cell_type"


def _read_thumbnail(zarr_path, target_long_side=1500):
    z = zarr.open(str(zarr_path), mode="r")
    C, H, W = z.shape
    step = max(1, int(np.ceil(max(H, W) / target_long_side)))
    thumb = z[:, ::step, ::step][:]
    if thumb.dtype != np.uint8:
        thumb = thumb.astype(np.uint8)
    return np.transpose(thumb, (1, 2, 0)), step


def _load_slide_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        return {
            "cell_id": np.array([c.decode() for c in f["cell_id"][:]]),
            "label": np.array([c.decode() for c in f[LABEL][:]]),
            "x": f["x_pixel"][:].astype(float),
            "y": f["y_pixel"][:].astype(float),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embeddings", nargs="+",
                    default=["z_uni_macro", "z_sigmma_concat"])
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    test_rows = manifest[manifest["split"] == "test"]
    test_samples = list(test_rows["sample_name"])
    print(f"[viz] {len(test_samples)} test sections: {test_samples}")

    # Load each test slide's data h5 in manifest order; keep labelled cells.
    slides = []
    for s in test_samples:
        d = _load_slide_h5(Path(args.data_dir) / f"{s}.h5")
        keep = d["label"] != ""
        slides.append({
            "sample": s,
            "label": d["label"][keep],
            "x": d["x"][keep],
            "y": d["y"][keep],
            "n": int(keep.sum()),
        })

    npz = np.load(args.predictions, allow_pickle=True)

    # Verify ordering against stored y_true for the first embedding.
    ref_emb = args.embeddings[0]
    y_true_ref = npz[f"{LABEL}/{ref_emb}/test/y_true"].astype(str)
    recon_true = np.concatenate([sl["label"] for sl in slides])
    assert len(recon_true) == len(y_true_ref), (
        f"length mismatch: recon {len(recon_true)} vs npz {len(y_true_ref)}")
    n_mismatch = int((recon_true != y_true_ref).sum())
    if n_mismatch:
        raise SystemExit(
            f"[FATAL] reconstructed y_true mismatches npz at {n_mismatch} cells "
            "— positional split is unsafe, aborting.")
    print(f"[viz] y_true ordering verified ({len(recon_true)} cells, 0 mismatch)")

    # Per-slide offsets in the flat prediction arrays.
    offsets, cur = [], 0
    for sl in slides:
        offsets.append((cur, cur + sl["n"]))
        cur += sl["n"]

    # Global colour map over the union of GT + all predicted classes.
    classes = set(recon_true)
    for emb in args.embeddings:
        classes.update(npz[f"{LABEL}/{emb}/test/y_pred"].astype(str))
    classes = sorted(c for c in classes if c)
    cmap = plt.get_cmap("tab20", max(len(classes), 1))
    colour = {c: cmap(i) for i, c in enumerate(classes)}

    EMB_TITLE = {"z_uni_macro": "UNI2-h", "z_sigmma_concat": "SIGMMA (concat)",
                 "z_sigmma_macro": "SIGMMA macro", "z_sigmma_meso": "SIGMMA meso",
                 "z_sigmma_micro": "SIGMMA micro", "z_uni_macro_cls": "UNI2-h CLS"}
    col_titles = ["Ground truth"] + [EMB_TITLE.get(e, e) for e in args.embeddings]
    ncol = len(col_titles)
    nrow = len(slides)

    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    for r, (sl, (st, en)) in enumerate(zip(slides, offsets)):
        he_zarr = Path(test_rows.iloc[r]["hematoxylin_eosin_image"])
        thumb, step = _read_thumbnail(he_zarr)
        xs, ys = sl["x"] / step, sl["y"] / step
        panels = [("Ground truth", sl["label"])]
        for emb in args.embeddings:
            panels.append((emb, npz[f"{LABEL}/{emb}/test/y_pred"].astype(str)[st:en]))

        for c, (title, vec) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(thumb)
            cs = [colour.get(v, (0, 0, 0, 0)) for v in vec]
            ax.scatter(xs, ys, c=cs, s=2, linewidths=0, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=13)
        # Section ID as bold row label on the left-most panel.
        axes[r, 0].set_ylabel(sl["sample"], fontsize=13, fontweight="bold",
                              rotation=90, labelpad=10)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=colour[c], markeredgecolor="none",
                          markersize=9, label=c) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(classes)),
               frameon=False, bbox_to_anchor=(0.5, -0.02 - 0.01 * (len(classes) // 5)))
    fig.suptitle(f"RCC cell-type prediction ({LABEL}) — test sections",
                 y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[viz] wrote {args.out}")


if __name__ == "__main__":
    main()
