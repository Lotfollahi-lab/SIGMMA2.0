"""2D Xenium transcript binning for H&E → ST GEX probing (Framing A Task 1).

Bin per-slide `transcripts.parquet` (Xenium output) into a `(G, K, K)` count
tensor for an arbitrary FOV bbox in H&E pixel coordinates. Used by the 2D
GEX probing pipeline as supervision; encoder code path is untouched.

Path layout (in-house Sanger cohort):
    H&E zarr: /lustre/.../sanger_beacon/<run_id>/<sample>/hematoxylin_eosin.zarr
    Xenium:   /nfs/team361/data/data_platform/data_staging/sanger_beacon/
              <run_id>/<sample>/output-XETG*/transcripts.parquet

~30% of manifest slides do not have transcripts on /nfs/team361 — caller must
handle the `None` return.

Coordinates: H&E zarr is in pixels at ``um_per_px`` (= 0.5 for the in-house
H&E scans). Xenium ``x_location``/``y_location`` are in µm. The FOV bbox is
expressed in pixels (matching the dataset's index array) and converted to µm
internally.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

LUSTRE_DATA_ROOT = os.environ.get(
    "SIGMMA_IN_HOUSE_DATA_ROOT",
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium",
)
PUBLIC_DATA_ROOT = os.environ.get(
    "SIGMMA_PUBLIC_DATA_ROOT",
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_xenium",
)
NFS_DATA_ROOT = os.environ.get(
    "SIGMMA_NFS_DATA_ROOT", "/nfs/team361/data/data_platform/data_staging"
)

# Back-compat aliases (other modules may import these by name).
LUSTRE_HE_ROOT = LUSTRE_DATA_ROOT + "/sanger_beacon"
NFS_XENIUM_ROOT = NFS_DATA_ROOT + "/sanger_beacon"


def manifest_path_to_transcripts_parquet(hematoxylin_zarr_path: str) -> Optional[str]:
    """Translate a manifest H&E zarr path to its Xenium transcripts.parquet.

    ``transcripts.parquet`` is co-located with ``hematoxylin_eosin.zarr`` in the
    SAME sample directory for both the in-house Sanger cohort
    (``sanger_in_house_xenium/<tissue>/<sgp>/<sample>/``) and the HEST-1k public
    cohort (``public_xenium/<hest_panel>/<sample>/``). Resolving against the
    zarr's parent directory therefore handles both layouts uniformly. Returns
    ``None`` if no ``transcripts.parquet`` sits next to the zarr (~30% of
    in-house slides lack transcripts — caller must handle ``None``).
    """
    sample_dir = Path(str(hematoxylin_zarr_path)).parent
    parquet = sample_dir / "transcripts.parquet"
    return str(parquet) if parquet.is_file() else None


def load_slide_transcripts(
    parquet_path: str,
    gene_panel: Sequence[str],
    qv_min: float = 20.0,
    um_per_px: float = 0.5,
) -> dict:
    """Load and pre-filter a slide's transcripts for repeated FOV binning.

    Returns a dict with pixel-coordinate x/y arrays and a gene index per
    transcript (-1 for genes outside the panel). Filters:
      - ``is_gene == True`` (drops Xenium codeword controls)
      - ``qv >= qv_min`` (default 20 — Xenium's standard high-confidence cut)
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    panel_to_idx = {g: i for i, g in enumerate(gene_panel)}

    # Some Xenium parquets (e.g. Sanger pediatric ovary, panel_1ca7fe57390a)
    # lack the `is_gene` column that flags real genes vs codeword controls.
    # Detect it from the schema so we only request columns that exist —
    # otherwise `pq.read_table(columns=[..., "is_gene"])` raises ArrowInvalid.
    schema_names = set(pq.read_schema(parquet_path).names)
    has_is_gene = "is_gene" in schema_names

    read_cols = ["feature_name", "x_location", "y_location", "qv"]
    if has_is_gene:
        read_cols.append("is_gene")
    table = pq.read_table(parquet_path, columns=read_cols)

    # Filter in pyarrow first (zero-copy bitmap) so we never materialize the
    # full ~600M-row feature_name column as Python strings on huge sections
    # like BRA / WEM — which OOMs at 32-128 GB. After filter only the kept
    # ~5-15% rows are materialized as Python list / numpy.
    mask = pc.greater_equal(table.column("qv"), qv_min)
    if has_is_gene:
        # `is_gene` drops Xenium codeword controls in one cheap boolean AND.
        mask = pc.and_(table.column("is_gene"), mask)
    else:
        # No `is_gene` flag: restrict to panel genes directly via membership.
        # This both drops codeword controls (never in the panel) and bounds
        # memory the same way `is_gene` did — only the ~panel rows survive
        # before to_pylist(). Codeword controls are dropped anyway by the
        # panel_to_idx match below, so this only changes *how early* we cut.
        # Only valid when feature_name is a (large_)string column; binary
        # feature_name (e.g. HEST-1k IDC) ships with `is_gene`, taking the
        # branch above.
        feat_type = table.schema.field("feature_name").type
        if pa.types.is_string(feat_type) or pa.types.is_large_string(feat_type):
            panel_arr = pa.array(gene_panel, type=feat_type)
            mask = pc.and_(
                pc.is_in(table.column("feature_name"), value_set=panel_arr),
                mask,
            )
    filtered = table.filter(mask)
    feature_name = filtered.column("feature_name").to_pylist()
    # Some Xenium parquets (e.g. HEST-1k IDC) store `feature_name` as a binary
    # column → bytes values that never match the str gene panel. Decode them.
    if feature_name and isinstance(feature_name[0], (bytes, bytearray)):
        feature_name = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else n
            for n in feature_name
        ]
    x_um = filtered.column("x_location").to_numpy(zero_copy_only=False)
    y_um = filtered.column("y_location").to_numpy(zero_copy_only=False)

    gene_idx = np.full(len(feature_name), -1, dtype=np.int32)
    for i, name in enumerate(feature_name):
        idx = panel_to_idx.get(name, -1)
        if idx >= 0:
            gene_idx[i] = idx
    in_panel = gene_idx >= 0
    return {
        "x_px": (x_um[in_panel] / um_per_px).astype(np.float32),
        "y_px": (y_um[in_panel] / um_per_px).astype(np.float32),
        "gene_idx": gene_idx[in_panel].astype(np.int32),
        "num_genes": len(gene_panel),
        "um_per_px": um_per_px,
    }


def bin_fov(
    loaded: dict,
    x_start_px: int,
    y_start_px: int,
    fov_size_px: int,
    target_grid: int,
) -> np.ndarray:
    """Bin a pre-loaded slide's panel transcripts into a single FOV.

    Returns ``(G, target_grid, target_grid)`` float32 transcript counts.
    Bin edges divide the FOV evenly in pixel space; transcript pixel
    coordinates are clamped to ``[x_start_px, x_start_px + fov_size_px)``
    via half-open interval semantics (``np.histogram2d`` would otherwise
    push the right/bottom edge into bin K-1, which we want).
    """
    G = loaded["num_genes"]
    x_lo = float(x_start_px)
    x_hi = float(x_start_px + fov_size_px)
    y_lo = float(y_start_px)
    y_hi = float(y_start_px + fov_size_px)

    x = loaded["x_px"]
    y = loaded["y_px"]
    g = loaded["gene_idx"]
    in_bbox = (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
    x = x[in_bbox]
    y = y[in_bbox]
    g = g[in_bbox]

    if x.size == 0:
        return np.zeros((G, target_grid, target_grid), dtype=np.float32)

    # Map (x, y) px → bin index in [0, target_grid).
    bin_w = fov_size_px / target_grid
    xb = np.minimum(((x - x_lo) / bin_w).astype(np.int32), target_grid - 1)
    yb = np.minimum(((y - y_lo) / bin_w).astype(np.int32), target_grid - 1)
    # Flat index across (G, target_grid, target_grid).
    flat = (g.astype(np.int64) * target_grid + yb.astype(np.int64)) * target_grid + xb.astype(np.int64)
    counts = np.bincount(flat, minlength=G * target_grid * target_grid)
    return counts.reshape(G, target_grid, target_grid).astype(np.float32)


def bin_transcripts(
    parquet_path: str,
    x_start_px: int,
    y_start_px: int,
    fov_size_px: int,
    target_grid: int,
    gene_panel: Sequence[str],
    um_per_px: float = 0.5,
    qv_min: float = 20.0,
) -> np.ndarray:
    """One-shot binning for a single FOV. Convenience wrapper around
    ``load_slide_transcripts`` + ``bin_fov`` — load+bin per call. For multiple
    FOVs from the same slide use the two-step API.
    """
    loaded = load_slide_transcripts(parquet_path, gene_panel, qv_min=qv_min, um_per_px=um_per_px)
    return bin_fov(loaded, x_start_px, y_start_px, fov_size_px, target_grid)


def iter_fov_bboxes(
    index_zarr_path: str,
    tissue_filtering_threshold: float = 20000.0,
):
    """Yield ``(fov_id, x_start_px, y_start_px, tissue_mass)`` for valid FOVs
    of a slide, matching the dataset's ``valid_indices`` selection rule
    (``index_array[:, -1] >= threshold``). ``fov_id`` is the row index into
    the raw zarr (matches ``__getitem__`` after the valid_indices remap).
    """
    import zarr
    arr = zarr.open(index_zarr_path, mode="r")
    data = np.asarray(arr[:])
    mask = data[:, -1] >= tissue_filtering_threshold
    valid = np.where(mask)[0]
    for idx in valid:
        y_start, x_start, tissue = data[idx]
        yield int(idx), int(x_start), int(y_start), float(tissue)


def _per_slide_gene_counts(
    parquet_paths: Sequence[str], qv_min: float
) -> tuple[list[dict[str, int]], set[str]]:
    """Collect per-slide gene → count dicts and the union of gene names.

    Filters: ``is_gene == True`` and ``qv >= qv_min`` (same as caching).
    """
    per_slide: list[dict[str, int]] = []
    union: set[str] = set()
    for p in parquet_paths:
        table = pq.read_table(p, columns=["feature_name", "qv", "is_gene"])
        is_gene = table.column("is_gene").to_numpy(zero_copy_only=False)
        qv = table.column("qv").to_numpy(zero_copy_only=False)
        keep = is_gene & (qv >= qv_min)
        names = np.asarray(table.column("feature_name").to_pylist())[keep]
        u, c = np.unique(names, return_counts=True)
        sc = {name: int(n) for name, n in zip(u, c)}
        per_slide.append(sc)
        union.update(sc.keys())
    return per_slide, union


def compute_top_k_hvg_by_mean(
    parquet_paths: Sequence[str],
    k: int = 50,
    qv_min: float = 20.0,
) -> list[str]:
    """Compute top-k genes by total transcript count across the supplied
    parquet files. Used to define the gene panel from the training set.

    Note: "HVG by mean expression" reduces to "by total count divided by
    constant tissue area," so the ranking is equivalent to ranking by total
    transcript count when slides have comparable scan area. Caller is
    expected to supply only training-split paths.
    """
    per_slide, union = _per_slide_gene_counts(parquet_paths, qv_min)
    totals = {name: sum(s.get(name, 0) for s in per_slide) for name in union}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def compute_top_k_hvg_by_dispersion(
    parquet_paths: Sequence[str],
    k: int = 50,
    qv_min: float = 20.0,
    abundance_floor_top: int = 300,
    min_slides_present: int = 3,
) -> list[str]:
    """Two-stage panel selection: abundance floor + cross-slide dispersion.

    Motivation: top-by-count selects housekeeping/structural genes that
    saturate the H&E-only baseline (UNI), leaving no headroom for SIGMMA's
    ST-aware multi-scale information to lift the score. Ranking by CV =
    std/mean across slides, restricted to abundance-eligible candidates,
    selects genes whose *amount* varies sample-to-sample — i.e. those with
    real biological heterogeneity rather than ubiquitous expression.

    Stage 1: keep ``abundance_floor_top`` genes by total count (removes
    transcript-floor noise — Pearson r is ill-defined when 100% of bins
    are zero).
    Stage 2: among those, rank by coefficient of variation
    ``std(slide_counts) / mean(slide_counts)``. Genes present in fewer
    than ``min_slides_present`` slides are dropped (CV is unstable).
    """
    per_slide, union = _per_slide_gene_counts(parquet_paths, qv_min)
    n_slides = len(per_slide)

    rows = []
    for name in union:
        counts = np.asarray([s.get(name, 0) for s in per_slide], dtype=np.float64)
        rows.append((name, int(counts.sum()), counts))
    rows.sort(key=lambda r: -r[1])  # by total descending
    candidates = rows[:abundance_floor_top]

    scored = []
    for name, total, counts in candidates:
        present = int((counts > 0).sum())
        if present < min_slides_present:
            continue
        mean = float(counts.mean())
        std = float(counts.std())
        cv = std / (mean + 1e-12)
        scored.append((name, cv, total, mean, std, present))
    scored.sort(key=lambda r: -r[1])  # by CV descending
    return [r[0] for r in scored[:k]]
