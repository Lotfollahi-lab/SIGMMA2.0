"""Diagnostic: GEX cosine-similarity structure used by GEXAwareSoftContrastiveLoss
under three gene-axis choices — ALL genes / HVG-50 / SVG-100 — on 5K-panel sections.

The loss builds its soft target from the FOV x FOV cosine similarity of the
preprocessed raw-GEX vectors (log1p -> per-row z-score -> L2-normalize ->
x_n @ x_n.T; see sigmma/model/losses.py preprocess_gex / build_gex_soft_targets).
This script reconstructs the per-FOV raw-GEX vector exactly as the dataset does
(sum of in-FOV cell counts over the gene axis; dataset.py:537) for the SAME 5K
gene panel, then computes that cosine similarity under each gene set and plots:

  row 1 : FOV x FOV similarity heatmap (FOVs ordered by section -> block diagonal
          = within-section pairs)
  row 2 : distribution of off-diagonal cosine sims, split within-section vs
          cross-section (a useful soft target should separate the two).

Output: figures/diagnostics/gex_cossim_gene_panels_5k.{png,pdf}
Run:    .venv/bin/python scripts/plot_gex_cossim_gene_panels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc
import zarr
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
MANIFEST = ROOT / "outputs/gex_2d/svg_panel_2d22ba31/MANIFEST_panel_2d22ba31.csv"
HVG_NPY = ROOT / "data/in_house_5k_disp_hvg_50.npy"
SVG_JSON = ROOT / "outputs/gex_2d/svg_panel_2d22ba31/gene_panel.json"
OUT = ROOT / "figures/diagnostics"

N_SECTIONS = 10          # sections sampled from the 5K cohort
FOVS_PER_SECTION = 120   # FOVs sampled per section
MIN_CELLS_PER_FOV = 10
FOV_SIZE = 256
TISSUE_THR = 20000.0
SEED = 0

# gex_aware_soft_loss default preprocess = "log1p_zscore"
def preprocess_gex(g: np.ndarray) -> np.ndarray:
    x = np.log1p(np.clip(g, 0, None))
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + 1e-8)


def cosine_offdiag(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    xn = mat / (n + 1e-8)
    return xn @ xn.T


def fix_index_path(p: str) -> str:
    # manifest regen wrote nonexistent *_v2.zarr index paths (known bug)
    cand = p.replace("_v2.zarr", ".zarr")
    return cand if Path(cand).exists() else p


def collect_fov_gex(sf: str, idx_path: str, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_fov, n_gene) all-gene FOV vectors + the section's var.index."""
    adata = sc.read_h5ad(sf)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    var_index = adata.var.index.to_numpy()
    X = adata.X.tocsr() if sp.issparse(adata.X) else np.asarray(adata.X)

    z = zarr.open(fix_index_path(idx_path), mode="r")
    index_array = np.asarray(z[:])
    valid = np.where(index_array[:, -1] >= TISSUE_THR)[0]
    if valid.size > FOVS_PER_SECTION:
        valid = rng.choice(valid, size=FOVS_PER_SECTION, replace=False)

    vecs = []
    for i in valid:
        y0, x0, _ = np.uint32(index_array[i])
        m = ((x0 <= coords[:, 0]) & (coords[:, 0] < x0 + FOV_SIZE)
             & (y0 <= coords[:, 1]) & (coords[:, 1] < y0 + FOV_SIZE))
        if int(m.sum()) < MIN_CELLS_PER_FOV:
            continue
        sub = X[m]
        v = np.asarray(sub.sum(axis=0)).ravel() if sp.issparse(sub) else sub.sum(axis=0)
        vecs.append(v.astype(np.float32))
    del adata, X
    if not vecs:
        return np.zeros((0, len(var_index)), np.float32), var_index
    return np.vstack(vecs), var_index


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hvg = [str(g) for g in np.load(HVG_NPY, allow_pickle=True)]
    svg = [str(g) for g in json.load(open(SVG_JSON))["panel"]]
    print(f"[panels] HVG={len(hvg)}  SVG={len(svg)}", flush=True)

    df = pd.read_csv(MANIFEST)
    sel = df.iloc[rng.choice(len(df), size=min(N_SECTIONS, len(df)), replace=False)]
    print(f"[plan] {len(sel)}/{len(df)} sections", flush=True)

    per_section_vecs = []      # list of (V_section, n_all_genes) on common axis
    section_labels = []
    common_axis = None
    raw_blocks, raw_var = [], []
    for _, r in sel.iterrows():
        try:
            V, vidx = collect_fov_gex(r["spatial_preprocessed"], r["index"], rng)
        except Exception as e:
            print(f"  {r['sample_name']}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if V.shape[0] == 0:
            print(f"  {r['sample_name']}: no FOVs", flush=True)
            continue
        raw_blocks.append((r["sample_name"], V, vidx))
        common_axis = set(vidx) if common_axis is None else (common_axis & set(vidx))
        print(f"  {r['sample_name']}: {V.shape[0]} FOVs, {V.shape[1]} genes", flush=True)

    if not raw_blocks:
        print("no data collected"); return
    common = sorted(common_axis)
    print(f"[axis] common genes across sections: {len(common)}", flush=True)

    # Re-project each section's FOV matrix onto the common gene axis.
    allmat, labels = [], []
    for name, V, vidx in raw_blocks:
        pos = {g: j for j, g in enumerate(vidx)}
        cols = np.array([pos[g] for g in common])
        allmat.append(V[:, cols])
        labels += [name] * V.shape[0]
    A = np.vstack(allmat)                      # (F, n_common) all genes
    labels = np.asarray(labels)
    cpos = {g: j for j, g in enumerate(common)}
    hvg_cols = [cpos[g] for g in hvg if g in cpos]
    svg_cols = [cpos[g] for g in svg if g in cpos]
    print(f"[axis] HVG on axis: {len(hvg_cols)}/{len(hvg)}  "
          f"SVG on axis: {len(svg_cols)}/{len(svg)}  | total FOVs={A.shape[0]}", flush=True)

    reps = {
        f"ALL genes (n={len(common)})": A,
        f"HVG-50 (n={len(hvg_cols)})": A[:, hvg_cols],
        f"SVG-100 (n={len(svg_cols)})": A[:, svg_cols],
    }

    # within vs cross-section pair mask (upper triangle, off-diagonal)
    same = labels[:, None] == labels[None, :]
    F = A.shape[0]
    iu = np.triu_indices(F, k=1)
    same_u = same[iu]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = {"within": "#d62728", "cross": "#4477AA"}
    stats = {}
    for c, (title, mat) in enumerate(reps.items()):
        gpre = preprocess_gex(mat)
        S = cosine_offdiag(gpre)
        # heatmap (FOVs already ordered by section in `labels`)
        ax = axes[0, c]
        im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine sim")

        svals = S[iu]
        win = svals[same_u]; crs = svals[~same_u]
        ax2 = axes[1, c]
        bins = np.linspace(-1, 1, 80)
        ax2.hist(crs, bins=bins, density=True, alpha=0.55, color=colors["cross"],
                 label=f"cross-section (med {np.median(crs):.2f})")
        ax2.hist(win, bins=bins, density=True, alpha=0.55, color=colors["within"],
                 label=f"within-section (med {np.median(win):.2f})")
        ax2.axvline(np.median(crs), color=colors["cross"], ls="--", lw=1)
        ax2.axvline(np.median(win), color=colors["within"], ls="--", lw=1)
        ax2.set_xlabel("FOV–FOV cosine similarity")
        if c == 0:
            ax2.set_ylabel("density")
        ax2.set_title(f"separation (within−cross med) = "
                      f"{np.median(win)-np.median(crs):+.3f}", fontsize=9)
        ax2.legend(fontsize=7, loc="upper left")
        stats[title] = {
            "within_med": float(np.median(win)), "cross_med": float(np.median(crs)),
            "within_mean": float(win.mean()), "cross_mean": float(crs.mean()),
            "separation_med": float(np.median(win) - np.median(crs)),
            "n_within_pairs": int(same_u.sum()), "n_cross_pairs": int((~same_u).sum()),
        }

    fig.suptitle(
        "GEX cosine similarity in GEXAwareSoftContrastiveLoss — 5K-panel sections\n"
        f"{len(raw_blocks)} sections, {F} FOVs; preprocess = log1p + per-FOV z-score (loss default)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"gex_cossim_gene_panels_5k.{ext}", dpi=150, bbox_inches="tight")
    (OUT / "gex_cossim_gene_panels_5k_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n[done] wrote", OUT / "gex_cossim_gene_panels_5k.png")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
