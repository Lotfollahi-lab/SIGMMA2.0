"""Step 2 of the cross-tissue atlas pipeline.

Load SIGMMA H&E patch embeddings across all 8 tissues, cluster with
MiniBatchKMeans on a few K values, pick K by silhouette score on a
subsample, and dump:
    - outputs/cross_tissue_retrieval/atlas/k_selection.csv
        per-(scale, K) silhouette + inertia.
    - outputs/cross_tissue_retrieval/atlas/cluster_assignments.parquet
        one row per patch (tissue, section_id, y_start, x_start, cluster_id)
        for the chosen scale + chosen K.

Default scale = `image_emb_meso`. A `concat(micro,meso,macro)` row is
recorded in the sweep for diagnostic comparison but cluster_assignments
is only emitted for the chosen scale.

Usage:
    python -m scripts.atlas_cluster_embeddings
    python -m scripts.atlas_cluster_embeddings --k_grid 10,20,30,50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from scripts.cross_tissue_extract_patches import TISSUES

EMB_DIR = Path("outputs/cross_tissue_retrieval/embeddings")
OUT_DIR = Path("outputs/cross_tissue_retrieval/atlas")
SILHOUETTE_SAMPLE = 50_000
RANDOM_STATE = 42


def _stack_tissue(scale_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate embeddings from every tissue HDF5, return matrix + provenance."""
    chunks: list[np.ndarray] = []
    rows: list[pd.DataFrame] = []
    for tissue, _ in TISSUES:
        path = EMB_DIR / f"{tissue}.h5"
        if not path.exists():
            print(f"[stack/{scale_key}] {tissue}.h5 missing; skip.")
            continue
        with h5py.File(path, "r") as f:
            if scale_key == "concat":
                emb = np.concatenate(
                    [f["image_emb_micro"][:], f["image_emb_meso"][:], f["image_emb_macro"][:]],
                    axis=1,
                ).astype(np.float32)
            else:
                emb = f[scale_key][:].astype(np.float32)
            section = np.array([s.decode() for s in f["section_id"][:]])
            ys = f["y_start"][:]
            xs = f["x_start"][:]
        chunks.append(emb)
        rows.append(pd.DataFrame({
            "tissue": tissue,
            "section_id": section,
            "y_start": ys,
            "x_start": xs,
        }))
        print(f"[stack/{scale_key}] {tissue}: {emb.shape[0]} patches, dim={emb.shape[1]}")
    if not chunks:
        raise SystemExit("no tissue HDF5s found; run Gate 0 first.")
    X = np.concatenate(chunks, axis=0)
    prov = pd.concat(rows, axis=0, ignore_index=True)
    return X, prov


def _silhouette_subsample(X: np.ndarray, labels: np.ndarray, n: int, rng: np.random.Generator) -> float:
    if X.shape[0] > n:
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return float(silhouette_score(X[idx], labels[idx], metric="euclidean"))
    return float(silhouette_score(X, labels, metric="euclidean"))


def run_sweep(X: np.ndarray, k_grid: list[int], scale_key: str) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Run MiniBatchKMeans for each K; return per-K metrics and label arrays."""
    rng = np.random.default_rng(RANDOM_STATE)
    records = []
    label_cache: dict[int, np.ndarray] = {}
    for k in k_grid:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=4096,
            max_iter=300,
            n_init=5,
        )
        labels = km.fit_predict(X)
        sil = _silhouette_subsample(X, labels, SILHOUETTE_SAMPLE, rng)
        records.append({
            "scale": scale_key,
            "K": k,
            "silhouette": sil,
            "inertia": float(km.inertia_),
            "n_samples": int(X.shape[0]),
        })
        label_cache[k] = labels
        print(f"  [{scale_key}] K={k}: silhouette={sil:.4f}, inertia={km.inertia_:.1f}")
    return pd.DataFrame(records), label_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="image_emb_meso",
                   choices=["image_emb_micro", "image_emb_meso", "image_emb_macro", "concat"],
                   help="Scale used for the final cluster_assignments output.")
    p.add_argument("--k_grid", default="10,20,30,50")
    p.add_argument("--include_concat_in_sweep", action="store_true", default=True)
    p.add_argument("--out_dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = [int(k) for k in args.k_grid.split(",")]

    sweep_rows: list[pd.DataFrame] = []
    chosen_labels: np.ndarray | None = None
    chosen_prov: pd.DataFrame | None = None
    chosen_k: int | None = None

    # Primary scale: full sweep, keep all label arrays.
    print(f"[primary scale = {args.scale}]")
    X_primary, prov_primary = _stack_tissue(args.scale)
    sweep_primary, labels_primary = run_sweep(X_primary, k_grid, args.scale)
    sweep_rows.append(sweep_primary)

    chosen_k = int(sweep_primary.loc[sweep_primary["silhouette"].idxmax(), "K"])
    chosen_labels = labels_primary[chosen_k]
    chosen_prov = prov_primary
    print(f"  chosen K (max silhouette) = {chosen_k}")

    # Ablation: concat sweep (diagnostic only; not used for assignments).
    if args.include_concat_in_sweep and args.scale != "concat":
        print("[ablation scale = concat]")
        X_concat, _ = _stack_tissue("concat")
        sweep_concat, _ = run_sweep(X_concat, k_grid, "concat")
        sweep_rows.append(sweep_concat)

    sweep_df = pd.concat(sweep_rows, axis=0, ignore_index=True)
    suffix = args.scale.replace("image_emb_", "")
    sweep_path = out_dir / f"k_selection_{suffix}.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"wrote {sweep_path}")

    assignments = chosen_prov.copy()
    assignments["cluster_id"] = chosen_labels.astype(np.int32)
    assignments["scale"] = args.scale
    assignments["K"] = chosen_k
    suffix = args.scale.replace("image_emb_", "")
    asg_path = out_dir / f"cluster_assignments_{suffix}.parquet"
    assignments.to_parquet(asg_path, index=False)
    print(f"wrote {asg_path} ({len(assignments)} patches, K={chosen_k}, scale={args.scale})")


if __name__ == "__main__":
    main()
