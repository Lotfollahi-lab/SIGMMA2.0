"""Step 1 of the cross-tissue atlas pipeline.

For each patch in `outputs/cross_tissue_retrieval/embeddings/<tissue>.h5`,
compute a per-tissue niche / cell-type composition vector by intersecting
the FOV bounding box with cell centroids + niche labels from the matching
`labels.h5ad`. The label space is tissue-specific (see TISSUE_CONFIG);
H_niche is normalised downstream by log(K) so clusters spanning multiple
tissues can be averaged on a comparable [0, 1] scale.

Curated tissues only carry signal; the 5 unlabelled tissues produce
zero-filled stubs aligned by row so downstream code can union by index.

Cache:
    /lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/
    cache/cross_tissue_atlas/niche_composition/<tissue>.h5

Usage:
    python -m scripts.atlas_aggregate_niche_composition --tissue renal_cell_carcinoma
    python -m scripts.atlas_aggregate_niche_composition --tissue all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd

from sigmma.application.cell_niche_transfer.extract import _load_preproc_centroids
from sigmma.evaluation.cluster_atlas import safe_l1_normalize
from scripts.cross_tissue_extract_patches import (
    DATA_ROOT,
    TISSUE_BY_NAME,
    TISSUES,
    manifest_path_for,
)

EMB_DIR = Path("outputs/cross_tissue_retrieval/embeddings")
CACHE_DIR = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/"
    "cache/cross_tissue_atlas/niche_composition"
)
FOV_SIZE = 256
MIN_LABELLED_CELLS = 5

# Tissue-specific configuration for sample-ID matching + niche column.
# match_kind:
#   "normalized_equal" — strip "-S\d+" and lowercase-then-upper both sides, exact match
#   "substring"        — manifest sample_name appears as substring in labels sample column
TISSUE_CONFIG: dict[str, dict] = {
    "skin": {
        "sample_column": "Sample ID",
        "niche_column": "niche_category",
        "match_kind": "normalized_equal",
    },
    "renal_cell_carcinoma": {
        "sample_column": "batch",
        "niche_column": "level_2_cell_type",
        "match_kind": "normalized_equal",
    },
    "whole_embryo": {
        "sample_column": "sample_id",
        "niche_column": "niche19",
        "match_kind": "substring",
    },
}
LABELLED_TISSUES = set(TISSUE_CONFIG.keys())


def _normalize_sample(section_id: str) -> str:
    """Strip the run/section suffix so manifest names match labels' batch keys."""
    s = section_id.upper()
    s = re.sub(r"-FO-0+(\d)", r"-FO-\1", s)
    s = re.sub(r"-S\d+(-[A-Z]\d+)?$", "", s)
    return s


def _labels_h5ad_for_tissue(tissue: str) -> Path:
    prefix = TISSUE_BY_NAME[tissue]
    return DATA_ROOT / prefix / "labels.h5ad"


def _read_niche_categories(labels_path: Path, niche_column: str) -> list[str]:
    with h5py.File(labels_path, "r") as f:
        cats = [c.decode() for c in f[f"obs/{niche_column}/categories"][:]]
    return cats


def _select_sample_rows(
    labels_path: Path,
    sample_column: str,
    manifest_sample_name: str,
    match_kind: str,
) -> np.ndarray:
    """Return a boolean mask over labels.h5ad rows that belong to one sample."""
    with h5py.File(labels_path, "r") as f:
        cats = [c.decode() for c in f[f"obs/{sample_column}/categories"][:]]
        codes = f[f"obs/{sample_column}/codes"][:]
    if match_kind == "normalized_equal":
        target = _normalize_sample(manifest_sample_name)
        target_codes = {
            i for i, c in enumerate(cats) if c and _normalize_sample(c) == target
        }
    elif match_kind == "substring":
        target_codes = {i for i, c in enumerate(cats) if manifest_sample_name in c}
    else:
        raise ValueError(f"unknown match_kind: {match_kind}")
    return np.isin(codes, list(target_codes))


def _load_sample_labels(
    labels_path: Path,
    sample_column: str,
    niche_column: str,
    manifest_sample_name: str,
    match_kind: str,
) -> pd.DataFrame:
    """Return a per-cell DataFrame indexed by cell_id, with one niche-code column."""
    in_sample = _select_sample_rows(labels_path, sample_column, manifest_sample_name, match_kind)
    if not in_sample.any():
        return pd.DataFrame()
    with h5py.File(labels_path, "r") as f:
        cid_cats = [c.decode() for c in f["obs/cell_id/categories"][:]]
        cid_codes = f["obs/cell_id/codes"][:][in_sample]
        n_cats = [c.decode() for c in f[f"obs/{niche_column}/categories"][:]]
        n_codes = f[f"obs/{niche_column}/codes"][:][in_sample]
    cell_ids = np.array(
        [cid_cats[c] if 0 <= c < len(cid_cats) else "" for c in cid_codes]
    )
    df = pd.DataFrame({
        "cell_id": cell_ids,
        "niche_code": n_codes,
    })
    df = df[df["cell_id"] != ""]
    df = df.drop_duplicates(subset="cell_id", keep="first").set_index("cell_id")
    return df


def _build_slide_cells(
    section_id: str,
    manifest: pd.DataFrame,
    tissue: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_pixel, y_pixel, niche_code) for labelled cells of one slide."""
    row = manifest.loc[manifest["sample_name"] == section_id]
    if row.empty:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
    preproc_path = row.iloc[0]["spatial_preprocessed"]
    centroids = _load_preproc_centroids(preproc_path)

    cfg = TISSUE_CONFIG[tissue]
    labels_path = _labels_h5ad_for_tissue(tissue)
    labels = _load_sample_labels(
        labels_path=labels_path,
        sample_column=cfg["sample_column"],
        niche_column=cfg["niche_column"],
        manifest_sample_name=section_id,
        match_kind=cfg["match_kind"],
    )
    if labels.empty:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)

    merged = centroids.join(labels, how="inner")
    merged = merged[merged["niche_code"] >= 0]
    return (
        merged["x_pixel"].to_numpy(dtype=np.float32),
        merged["y_pixel"].to_numpy(dtype=np.float32),
        merged["niche_code"].to_numpy(dtype=np.int64),
    )


def aggregate_labelled_tissue(tissue: str, out_path: Path) -> None:
    emb_path = EMB_DIR / f"{tissue}.h5"
    if not emb_path.exists():
        print(f"[{tissue}] embedding HDF5 not found at {emb_path}; skip.")
        return

    cfg = TISSUE_CONFIG[tissue]
    labels_path = _labels_h5ad_for_tissue(tissue)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.h5ad missing for {tissue}: {labels_path}")
    niche_categories = _read_niche_categories(labels_path, cfg["niche_column"])
    n_classes = len(niche_categories)

    manifest = pd.read_csv(manifest_path_for(tissue, "test"))

    with h5py.File(emb_path, "r") as f:
        section_id = np.array([s.decode() for s in f["section_id"][:]])
        y_start = f["y_start"][:].astype(np.int32)
        x_start = f["x_start"][:].astype(np.int32)
    n_patches = len(section_id)

    composition = np.zeros((n_patches, n_classes), dtype=np.float32)
    n_labelled_cells = np.zeros(n_patches, dtype=np.int32)

    slide_cache: dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid in np.unique(section_id):
        slide_cache[sid] = _build_slide_cells(sid, manifest, tissue)
        x, _, _ = slide_cache[sid]
        print(f"  [{tissue}/{sid}] {len(x)} labelled cells")

    for i in range(n_patches):
        sid = section_id[i]
        x, y, code = slide_cache[sid]
        if x.size == 0:
            continue
        mask = (
            (x_start[i] <= x) & (x < x_start[i] + FOV_SIZE)
            & (y_start[i] <= y) & (y < y_start[i] + FOV_SIZE)
        )
        in_patch = code[mask]
        if in_patch.size == 0:
            continue
        bincount = np.bincount(in_patch, minlength=n_classes).astype(np.float32)
        composition[i] = bincount
        n_labelled_cells[i] = int(in_patch.size)

    valid_mask = n_labelled_cells >= MIN_LABELLED_CELLS
    composition_norm = np.zeros_like(composition)
    composition_norm[valid_mask] = safe_l1_normalize(composition[valid_mask], axis=-1)

    _write_cache(
        out_path=out_path,
        tissue=tissue,
        labelled=True,
        niche_column=cfg["niche_column"],
        section_id=section_id,
        y_start=y_start,
        x_start=x_start,
        composition=composition_norm,
        n_labelled_cells=n_labelled_cells,
        valid_mask=valid_mask,
        niche_categories=niche_categories,
    )
    print(
        f"[{tissue}] wrote {out_path}: {n_patches} patches, "
        f"{int(valid_mask.sum())} with ≥{MIN_LABELLED_CELLS} labelled cells "
        f"(label column = {cfg['niche_column']}, K = {n_classes})."
    )


def aggregate_unlabelled_stub(tissue: str, out_path: Path) -> None:
    emb_path = EMB_DIR / f"{tissue}.h5"
    if not emb_path.exists():
        print(f"[{tissue}] embedding HDF5 not found at {emb_path}; skip stub.")
        return
    with h5py.File(emb_path, "r") as f:
        section_id = np.array([s.decode() for s in f["section_id"][:]])
        y_start = f["y_start"][:].astype(np.int32)
        x_start = f["x_start"][:].astype(np.int32)
    n_patches = len(section_id)
    _write_cache(
        out_path=out_path,
        tissue=tissue,
        labelled=False,
        niche_column="",
        section_id=section_id,
        y_start=y_start,
        x_start=x_start,
        composition=np.zeros((n_patches, 0), dtype=np.float32),
        n_labelled_cells=np.zeros(n_patches, dtype=np.int32),
        valid_mask=np.zeros(n_patches, dtype=bool),
        niche_categories=[],
    )
    print(f"[{tissue}] wrote stub {out_path}: {n_patches} patches, all unlabelled.")


def _write_cache(
    *,
    out_path: Path,
    tissue: str,
    labelled: bool,
    niche_column: str,
    section_id: np.ndarray,
    y_start: np.ndarray,
    x_start: np.ndarray,
    composition: np.ndarray,
    n_labelled_cells: np.ndarray,
    valid_mask: np.ndarray,
    niche_categories: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["tissue"] = tissue
        f.attrs["labelled"] = bool(labelled)
        f.attrs["niche_column"] = niche_column
        f.attrs["fov_size"] = FOV_SIZE
        f.attrs["min_labelled_cells"] = MIN_LABELLED_CELLS
        f.create_dataset("composition", data=composition.astype(np.float32))
        f.create_dataset("n_labelled_cells", data=n_labelled_cells.astype(np.int32))
        f.create_dataset("valid_mask", data=valid_mask)
        f.create_dataset("section_id", data=np.array(section_id, dtype="S64"))
        f.create_dataset("y_start", data=y_start)
        f.create_dataset("x_start", data=x_start)
        f.create_dataset(
            "niche_categories",
            data=np.array(niche_categories, dtype="S128") if niche_categories
            else np.empty(0, dtype="S128"),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tissue", default="all", help="One tissue or 'all'.")
    p.add_argument("--cache_dir", default=str(CACHE_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if args.tissue == "all":
        tissues = [t for t, _ in TISSUES]
    elif args.tissue in TISSUE_BY_NAME:
        tissues = [args.tissue]
    else:
        raise SystemExit(f"unknown tissue: {args.tissue}")

    for t in tissues:
        out_path = cache_dir / f"{t}.h5"
        if t in LABELLED_TISSUES:
            aggregate_labelled_tissue(t, out_path)
        else:
            aggregate_unlabelled_stub(t, out_path)


if __name__ == "__main__":
    main()
