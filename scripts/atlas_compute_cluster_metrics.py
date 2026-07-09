"""Step 3 of the cross-tissue atlas pipeline.

Combine `cluster_assignments.parquet` (Step 2) with the per-tissue niche
composition caches (Step 1) to produce a per-cluster metric table:

    cluster_id, n_patches, n_labelled_patches, n_tissues_represented,
    H_tissue_norm,                # Shannon / log(8)
    H_niche_norm,                 # mean across labelled tissues
                                  # of (H_t / log(K_t)), weighted by
                                  # labelled-patch count
    dominant_niche, dominant_niche_frac,
    tissue_breakdown_json,
    score = H_tissue_norm - H_niche_norm

H_niche requires at least MIN_LABELLED_PATCHES labelled patches in the
cluster; otherwise H_niche_norm = NaN and score = NaN.

Usage:
    python -m scripts.atlas_compute_cluster_metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.atlas_aggregate_niche_composition import (
    CACHE_DIR as NICHE_CACHE_DIR,
    LABELLED_TISSUES,
)
from scripts.cross_tissue_extract_patches import TISSUES

ATLAS_DIR = Path("outputs/cross_tissue_retrieval/atlas")
MIN_LABELLED_PATCHES = 30
N_TISSUES_GLOBAL = len(TISSUES)


def _load_tissue_composition_df(tissue: str, cache_dir: Path) -> pd.DataFrame | None:
    path = cache_dir / f"{tissue}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        labelled = bool(f.attrs["labelled"])
        if not labelled:
            return None
        section_id = np.array([s.decode() for s in f["section_id"][:]])
        y_start = f["y_start"][:]
        x_start = f["x_start"][:]
        composition = f["composition"][:]
        valid = f["valid_mask"][:]
        cats = [c.decode() for c in f["niche_categories"][:]]
    df = pd.DataFrame({
        "section_id": section_id,
        "y_start": y_start,
        "x_start": x_start,
        "valid_mask": valid,
    })
    for i, name in enumerate(cats):
        df[f"_n_{i}"] = composition[:, i]
    df.attrs_niche_cats = cats  # type: ignore[attr-defined]
    return df


def _shannon(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= eps:
        return float("nan")
    p = p / s
    safe = np.where(p > eps, p, 1.0)
    return float(-np.sum(p * np.log(safe)))


def _normalize_entropy(h: float, k: int) -> float:
    if not np.isfinite(h) or k <= 1:
        return float("nan")
    return h / np.log(k)


def compute_metrics(
    assignments: pd.DataFrame,
    composition_dfs: dict[str, pd.DataFrame],
    niche_cats: dict[str, list[str]],
) -> pd.DataFrame:
    cluster_ids = sorted(assignments["cluster_id"].unique())
    records = []
    for cid in cluster_ids:
        cluster = assignments[assignments["cluster_id"] == cid]
        n_patches = len(cluster)
        tissue_counts = cluster["tissue"].value_counts()
        tissue_prob = (tissue_counts / tissue_counts.sum()).to_numpy()
        h_tissue = _shannon(tissue_prob)
        h_tissue_norm = _normalize_entropy(h_tissue, N_TISSUES_GLOBAL)
        n_tissues_present = int((tissue_counts > 0).sum())
        tissue_breakdown = {
            t: int(c) for t, c in tissue_counts.items()
        }

        # Per-tissue H_niche normalized then averaged
        h_niche_terms: list[tuple[float, float, str, float]] = []  # (h_norm, weight, dominant_label, dominant_frac)
        total_labelled = 0
        per_tissue_dominant: list[tuple[str, str, float, int]] = []  # (tissue, label, frac, n_labelled)
        for tissue, comp_df in composition_dfs.items():
            cluster_t = cluster[cluster["tissue"] == tissue]
            if cluster_t.empty:
                continue
            merged = cluster_t.merge(
                comp_df,
                on=["section_id", "y_start", "x_start"],
                how="inner",
            )
            valid = merged[merged["valid_mask"]]
            n_valid = len(valid)
            if n_valid == 0:
                continue
            cats = niche_cats[tissue]
            K_t = len(cats)
            comp_cols = [f"_n_{i}" for i in range(K_t)]
            comp_sum = valid[comp_cols].to_numpy().sum(axis=0)
            h_t = _shannon(comp_sum)
            h_t_norm = _normalize_entropy(h_t, K_t)
            h_niche_terms.append((h_t_norm, n_valid))
            total_labelled += n_valid
            dom_idx = int(np.argmax(comp_sum))
            dom_frac = float(comp_sum[dom_idx] / comp_sum.sum()) if comp_sum.sum() > 0 else 0.0
            per_tissue_dominant.append((tissue, cats[dom_idx], dom_frac, n_valid))

        if total_labelled >= MIN_LABELLED_PATCHES and h_niche_terms:
            weights = np.array([w for _, w in h_niche_terms], dtype=np.float64)
            values = np.array([v for v, _ in h_niche_terms], dtype=np.float64)
            h_niche_norm = float(np.average(values, weights=weights))
            score = h_tissue_norm - h_niche_norm
        else:
            h_niche_norm = float("nan")
            score = float("nan")

        if per_tissue_dominant:
            best = max(per_tissue_dominant, key=lambda r: (r[3], r[2]))
            dominant_tissue = best[0]
            dominant_niche = best[1]
            dominant_niche_frac = best[2]
        else:
            dominant_tissue = ""
            dominant_niche = ""
            dominant_niche_frac = float("nan")

        records.append({
            "cluster_id": int(cid),
            "n_patches": int(n_patches),
            "n_labelled_patches": int(total_labelled),
            "n_tissues_represented": n_tissues_present,
            "tissue_breakdown_json": json.dumps(tissue_breakdown, sort_keys=True),
            "H_tissue_norm": h_tissue_norm,
            "H_niche_norm": h_niche_norm,
            "dominant_tissue": dominant_tissue,
            "dominant_niche": dominant_niche,
            "dominant_niche_frac": dominant_niche_frac,
            "score": score,
        })
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="meso",
                   choices=["micro", "meso", "macro", "concat"],
                   help="Picks cluster_assignments_<scale>.parquet and writes cluster_metrics_<scale>.csv.")
    p.add_argument("--assignments", default=None,
                   help="Optional override (else derived from --scale).")
    p.add_argument("--cache_dir", default=str(NICHE_CACHE_DIR))
    p.add_argument("--out_csv", default=None,
                   help="Optional override (else derived from --scale).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assignments_path = (
        args.assignments
        if args.assignments
        else str(ATLAS_DIR / f"cluster_assignments_{args.scale}.parquet")
    )
    out_csv = (
        args.out_csv
        if args.out_csv
        else str(ATLAS_DIR / f"cluster_metrics_{args.scale}.csv")
    )
    assignments = pd.read_parquet(assignments_path)
    cache_dir = Path(args.cache_dir)
    composition_dfs: dict[str, pd.DataFrame] = {}
    niche_cats: dict[str, list[str]] = {}
    for tissue in LABELLED_TISSUES:
        df = _load_tissue_composition_df(tissue, cache_dir)
        if df is None:
            print(f"[{tissue}] no labelled cache; skip from H_niche.")
            continue
        composition_dfs[tissue] = df
        with h5py.File(cache_dir / f"{tissue}.h5", "r") as f:
            niche_cats[tissue] = [c.decode() for c in f["niche_categories"][:]]
        print(f"[{tissue}] {len(df)} cache rows, K = {len(niche_cats[tissue])}")

    metrics = compute_metrics(assignments, composition_dfs, niche_cats)
    metrics = metrics.sort_values("score", ascending=False, na_position="last")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(metrics)} clusters)")
    print("\nTop 10 by score (NaN = insufficient labelled coverage):")
    print(
        metrics[["cluster_id", "n_patches", "n_labelled_patches",
                 "n_tissues_represented", "H_tissue_norm",
                 "H_niche_norm", "dominant_tissue", "dominant_niche",
                 "dominant_niche_frac", "score"]].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
