"""Per-section MCC + Moran's I of prediction correctness for niche transfer.

Loads predictions.npz from the three probe runs (UNI2-h baseline, FT baselines,
SIGMMA α=0.1) and reconstructs per-section slices by replaying the same
section ordering + label-filter logic the probe used. Computes:

  * Matthews Correlation Coefficient (multi-class generalisation, sklearn).
  * Moran's I of the per-cell correctness indicator on a kNN spatial weight
    matrix built from cell centroids (k=8 neighbours, row-standardised).

Writes ``outputs/cell_niche/extra_metrics_per_section.csv`` and runs paired
Wilcoxon signed-rank (SIGMMA α=0.1 vs each comparator) for n=6 sections.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wilcoxon
from scipy.sparse import csr_matrix

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
CACHE = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/data/cell_niche/scales_64_128_256"
)
MANIFEST_CSV = ROOT / "configs/cell_niche_evaluable_slides.csv"

PRED_PATHS = {
    "UNI2-h":       (ROOT / "outputs/cell_niche/uni2h_25629/predictions.npz",            "z_uni_macro"),
    "BLEEP":        (ROOT / "outputs/cell_niche/27663/predictions.npz",                  "z_bleep_macro"),
    "CLIP":         (ROOT / "outputs/cell_niche/27663/predictions.npz",                  "z_clip_macro"),
    "PLIP":         (ROOT / "outputs/cell_niche/27663/predictions.npz",                  "z_plip_macro"),
    "OmiCLIP":      (ROOT / "outputs/cell_niche/27663/predictions.npz",                  "z_omiclip_macro"),
    "SIGMMA α=0.1": (ROOT / "outputs/cell_niche/gex_sweep_step1000/a01_698686/predictions.npz", "z_sigmma_macro_dense_noproj"),
}
LABELS = ("niches_terra", "niche_category")
SPLITS = ("test", "val")
K_NEIGH = 8


def load_section(sample: str):
    """Return cell_ids, x, y, niche_terra labels, niche_category labels."""
    p = CACHE / f"{sample}.h5"
    with h5py.File(p, "r") as f:
        cell_ids = np.array([c.decode() for c in f["cell_id"][:]])
        x = f["x_pixel"][:]
        y = f["y_pixel"][:]
        nt = np.array([c.decode() for c in f["niches_terra"][:]])
        nc = np.array([c.decode() for c in f["niche_category"][:]])
    return cell_ids, x, y, nt, nc


def build_offsets(manifest: pd.DataFrame, split: str):
    """Replay the probe's section stacking for one split."""
    rows = manifest[manifest["split"] == split]
    offsets = []
    cursor = 0
    for sample in rows["sample_name"]:
        h5 = CACHE / f"{sample}.h5"
        if not h5.exists():
            continue
        cids, x, y, nt, nc = load_section(sample)
        n = len(cids)
        offsets.append({
            "sample": sample, "split": split,
            "start": cursor, "end": cursor + n,
            "cell_ids": cids, "x": x, "y": y,
            "niches_terra": nt, "niche_category": nc,
        })
        cursor += n
    return offsets


def collect_train_classes(manifest: pd.DataFrame, label: str):
    rows = manifest[manifest["split"] == "train"]
    cls = set()
    for sample in rows["sample_name"]:
        h5 = CACHE / f"{sample}.h5"
        if not h5.exists():
            continue
        with h5py.File(h5, "r") as f:
            lbl = np.array([c.decode() for c in f[label][:]])
        cls.update(c for c in np.unique(lbl) if c != "")
    return cls


def morans_I(values: np.ndarray, coords: np.ndarray, k: int = K_NEIGH) -> float:
    """Global Moran's I with row-standardised kNN spatial weight (k=8, no self-loop).

    I = (n / W) * (Σ_ij w_ij (z_i - z̄)(z_j - z̄)) / Σ_i (z_i - z̄)²
    """
    n = len(values)
    if n < k + 2:
        return float("nan")
    z = values.astype(float)
    z_centred = z - z.mean()
    denom = float(np.dot(z_centred, z_centred))
    if denom == 0.0:
        return float("nan")  # all correct or all wrong → no variance
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)  # drop self
    w = np.full(rows.shape, 1.0 / k)  # row-standardised
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    Wz = W @ z_centred
    numer = float(np.dot(z_centred, Wz))
    W_sum = float(w.sum())  # equals n with row standardisation
    return (n / W_sum) * (numer / denom)


def main():
    manifest = pd.read_csv(MANIFEST_CSV)
    test_offsets = build_offsets(manifest, "test")
    val_offsets = build_offsets(manifest, "val")
    print(f"[load] test sections: {len(test_offsets)}, val sections: {len(val_offsets)}")

    train_classes = {lbl: collect_train_classes(manifest, lbl) for lbl in LABELS}
    print(f"[load] train_classes: niches_terra={len(train_classes['niches_terra'])}, "
          f"niche_category={len(train_classes['niche_category'])}")

    rows = []
    for label in LABELS:
        # Build per-section keep masks (uniform across embeddings)
        for split, offsets in (("test", test_offsets), ("val", val_offsets)):
            # Step 1: per-section "valid" mask = (label != "") & (label in train_classes)
            for sec in offsets:
                lbl_full = sec[label]  # length n_cells
                valid_mask = np.array([c != "" and c in train_classes[label] for c in lbl_full])
                sec[f"valid_{label}_{split}"] = valid_mask

            # Step 2: cumulative npz index offsets (sum of valid per prior section)
            cum = 0
            for sec in offsets:
                sec[f"npz_start_{label}_{split}"] = cum
                cum += int(sec[f"valid_{label}_{split}"].sum())
            n_total_npz = cum
            print(f"[{label}/{split}] total cells in npz = {n_total_npz}")

        # Per-model evaluation
        for model_name, (npz_path, emb_key) in PRED_PATHS.items():
            npz = np.load(npz_path, allow_pickle=True)
            for split, offsets in (("test", test_offsets), ("val", val_offsets)):
                key_true = f"{label}/{emb_key}/{split}/y_true"
                key_pred = f"{label}/{emb_key}/{split}/y_pred"
                if key_true not in npz.files:
                    print(f"  [skip] {model_name}: missing {key_true}")
                    continue
                y_true_all = np.asarray(npz[key_true])
                y_pred_all = np.asarray(npz[key_pred])
                if len(y_true_all) != int(sum(sec[f"valid_{label}_{split}"].sum()
                                              for sec in offsets)):
                    print(f"  [warn] {model_name}: npz len mismatch "
                          f"({len(y_true_all)} vs reconstructed)")
                for sec in offsets:
                    s = sec[f"npz_start_{label}_{split}"]
                    n = int(sec[f"valid_{label}_{split}"].sum())
                    if n < 10:
                        continue
                    yt = y_true_all[s : s + n]
                    yp = y_pred_all[s : s + n]
                    # MCC + support-weighted F1
                    mcc = float(matthews_corrcoef(yt, yp))
                    wf1 = float(f1_score(yt, yp, average="weighted"))
                    # Moran's I of correctness indicator
                    valid_mask = sec[f"valid_{label}_{split}"]
                    coords = np.column_stack([
                        sec["x"][valid_mask],
                        sec["y"][valid_mask],
                    ])
                    correct = (yt == yp).astype(int)
                    mi = morans_I(correct, coords)
                    rows.append({
                        "label": label,
                        "split": split,
                        "sample": sec["sample"],
                        "model": model_name,
                        "n_cells_scored": n,
                        "n_correct": int(correct.sum()),
                        "mcc": mcc,
                        "weighted_f1": wf1,
                        "morans_I_correctness": mi,
                    })

    out_df = pd.DataFrame(rows)
    out_csv = ROOT / "outputs/cell_niche/extra_metrics_per_section.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[wrote] {out_csv} ({len(out_df)} rows)")

    # weighted-F1 companion CSV consumed by plot_niche_category_mcc_wf1.py
    wf1_csv = ROOT / "outputs/cell_niche/weighted_f1_per_section.csv"
    out_df[["label", "split", "sample", "model", "n_cells_scored",
            "n_correct", "weighted_f1"]].to_csv(wf1_csv, index=False)
    print(f"[wrote] {wf1_csv}")

    # =============== Aggregation + paired Wilcoxon ===============
    section_order = [
        "BK46-SKI-27-FO-2-s3", "BK39-SKI-27-FO-1-S3",
        "BK23-SKI-27-FO-1-S6-C2", "BK30-SKI-27-FO-4-S3",
        "BK50-SKI-27-FO-1-s2", "BK46-SKI-27-FO-2-s2",
    ]
    for label in LABELS:
        for metric in ("mcc", "weighted_f1", "morans_I_correctness"):
            print(f"\n========== {label} | {metric} ==========")
            sub = out_df[out_df["label"] == label]
            pivot = (sub.pivot_table(index="sample", columns="model", values=metric)
                        .reindex(section_order))
            print(pivot.round(4).to_string())
            print(f"  mean: {pivot.mean().round(4).to_dict()}")
            print(f"  std:  {pivot.std().round(4).to_dict()}")
            print(f"  Paired Wilcoxon (SIGMMA α=0.1 vs):")
            sv = pivot["SIGMMA α=0.1"].values
            for ref in ["UNI2-h", "BLEEP", "CLIP", "PLIP", "OmiCLIP"]:
                rv = pivot[ref].values
                if np.isnan(sv).any() or np.isnan(rv).any():
                    print(f"    vs {ref}: NaN, skip")
                    continue
                diff = sv - rv
                w = int((diff > 0).sum())
                l = int((diff < 0).sum())
                try:
                    _, p2 = wilcoxon(sv, rv, alternative="two-sided", zero_method="wilcox")
                    _, p1 = wilcoxon(sv, rv, alternative="greater", zero_method="wilcox")
                    print(f"    vs {ref:<8s} Δ̄={diff.mean():+.4f}  W/L={w}/{l}  "
                          f"p2={p2:.4f}  p1={p1:.4f}")
                except Exception as e:
                    print(f"    vs {ref}: {e}")


if __name__ == "__main__":
    main()
