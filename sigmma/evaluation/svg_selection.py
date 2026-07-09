"""Panel-aware iSCALE-style spatially variable gene (SVG) selection.

Replaces the section-level total-count CV "HVG" approach (which failed for
two reasons: (a) the original compute resolved only sanger_beacon parquets,
making the panel SKI-only, and (b) total-count CV does not test spatial
structure and zero-pads missing genes across heterogeneous panels). Instead,
this module computes gene-wise Moran's I from tile-binned normalized
expression *within each section*, then aggregates per panel (var.index hash).

Sections are grouped by ``panel_hash``; small groups (n_sec < threshold) are
merged with similar larger panels by Jaccard / min-overlap thresholds, with
the merged pseudo-panel's gene axis = INTERSECTION of constituent var.indexes
(so each contributing section can be reindexed onto the merged axis without
zero-filling absent genes).

Per-panel selection writes ordered top-K SVG gene lists + diagnostic
metadata (per-gene tissue contribution, detection frac, expression mean,
spatial-structure stability).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse


# ---------- Authoritative section → tissue lookup ----------
# Tissue labels are read from the curated section_to_gene_panel.csv
# (columns: sample_name, tissue, run_id, panel_name, ...), NOT parsed from the
# 3-letter section-name code. Name-code parsing was lossy: it silently dropped
# adrenocortical_carcinoma (no code in the name), split pediatric_ovary across
# OVR/GON/RPT sub-codes, and conflated beacon vs CTCL under "SKI".
SECTION_PANEL_CSV = os.environ.get(
    "SIGMMA_SECTION_PANEL_CSV",
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
    "gene_panel_info/section_to_gene_panel.csv",
)

_SECTION_TISSUE_CACHE: Optional[dict] = None

# Legacy section-name code → authoritative tissue label. Used ONLY as a
# fallback when the curated CSV is unavailable. SKI cannot be disambiguated
# (beacon vs CTCL) from the name alone, so it degrades to "skin".
_LEGACY_CODE_TO_TISSUE = {
    "KID": "renal_cell_carcinoma",
    "LIV": "liver",
    "SKI": "skin",
    "WEM": "whole_embryo",
    "BRA": "glioblastoma",
    "OVR": "pediatric_ovary",
    "GON": "pediatric_ovary",
    "RPT": "pediatric_ovary",
    "PNC": "pancreas",
    "LUN": "lung",
}


def _load_section_tissue_map(csv_path: Optional[str] = None) -> dict:
    """Lazy-load + cache the authoritative {sample_name: tissue} map from
    section_to_gene_panel.csv. Returns {} (triggering the legacy name-code
    fallback in ``tissue_of``) when the CSV file is missing."""
    global _SECTION_TISSUE_CACHE
    if csv_path is None and _SECTION_TISSUE_CACHE is not None:
        return _SECTION_TISSUE_CACHE
    path = csv_path or SECTION_PANEL_CSV
    mapping: dict = {}
    if os.path.exists(path):
        df = pd.read_csv(path, usecols=["sample_name", "tissue"])
        mapping = dict(zip(df["sample_name"].astype(str), df["tissue"].astype(str)))
    if csv_path is None:
        _SECTION_TISSUE_CACHE = mapping
    return mapping


def _tissue_of_from_name(sample_name: str) -> Optional[str]:
    """Legacy fallback: derive a tissue label from the section-name code.
    TB-prefixed names (TB16/TB17/TB18/TB21..., '-' or '_' separator) are lung."""
    if re.match(r"TB\d+[-_]", sample_name):
        return "lung"
    for code, tissue in _LEGACY_CODE_TO_TISSUE.items():
        if f"-{code}-" in sample_name:
            return tissue
    return None


def tissue_of(sample_name: str, csv_path: Optional[str] = None) -> Optional[str]:
    """Authoritative tissue label for an in-house section.

    Looks the section up in the curated section_to_gene_panel.csv (path
    overridable via the ``csv_path`` arg or the ``SIGMMA_SECTION_PANEL_CSV``
    env var) and returns its ``tissue`` value, e.g. ``pediatric_ovary``,
    ``renal_cell_carcinoma``, ``adrenocortical_carcinoma``, ``beacon``,
    ``CTCL``, ``glioblastoma``, ``lung``, ``liver``, ``pancreas``,
    ``whole_embryo``.

    Falls back to legacy section-name code parsing only when the CSV is
    unavailable; returns None if neither resolves.
    """
    mapping = _load_section_tissue_map(csv_path)
    if sample_name in mapping:
        return mapping[sample_name]
    if not mapping:
        return _tissue_of_from_name(sample_name)
    return None


# ---------- Panel-hash group merging ----------
@dataclass
class MergedPanel:
    """A panel group after small-panel merging.

    `merged_var_index` is the intersection of all member var.indexes;
    every member section is guaranteed to contain every gene in this list,
    so reindexing per-section diagnostics onto it never produces NaN.
    """
    panel_id: str
    merged_from_hashes: list[str]
    sample_names: list[str]
    merged_var_index: list[str]
    n_sections: int
    jaccard_min: float = 1.0
    min_overlap_min: float = 1.0
    notes: str = ""


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _min_overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def merge_small_panels(
    panel_groups: list[dict],
    min_sections: int = 5,
    jaccard_threshold: float = 0.90,
    min_overlap_threshold: float = 0.95,
    consolidate_stable_jaccard: float | None = None,
) -> list[MergedPanel]:
    """Greedy single-pass merge: small groups attempt to attach to the
    largest already-stable panel that satisfies both thresholds.

    Strategy:
      1. Sort all panel_groups by n_sections desc.
      2. Initial stable set = groups with n_sections >= min_sections (each
         becomes its own MergedPanel with merged_var_index = own var.index).
      3. For each small group (n_sections < min_sections), find the stable
         group with HIGHEST Jaccard satisfying Jaccard >= jaccard_threshold
         AND min_overlap >= min_overlap_threshold. Attach if found, else
         leave as a standalone "unmerged" MergedPanel (will be excluded from
         training but kept for diagnostics).
      4. After all attaches resolved, recompute each merged panel's
         merged_var_index = intersection of every constituent's gene set.

    Two-level small-vs-small chaining is NOT supported (kept simple).

    If `consolidate_stable_jaccard` is set, a final pass also merges
    near-duplicate STABLE panels (Jaccard >= threshold) into one. This fixes
    the case where a single assay (e.g. the Xenium 5K panel) fragments into
    several >=min_sections hashes because per-section gene pruning gave each
    cluster a distinct var.index — without it those clusters stay as separate
    stable panels. Distinct assays (5K vs <500-gene targeted) have Jaccard
    <0.15 so a threshold of ~0.5 consolidates only true duplicates.
    """
    # Pre-extract gene sets for each input group
    gset = {g["panel_hash"]: set(g["gene_names_sorted"]) for g in panel_groups}
    groups_sorted = sorted(panel_groups, key=lambda g: -g["n_sections"])

    stable: list[MergedPanel] = []
    member_hashes: dict[str, list[str]] = {}   # panel_id -> list of source hashes
    member_var_index: dict[str, list[set]] = {}  # panel_id -> list of var.index sets

    # Pass 1: seed stable panels (n_sections >= min_sections each)
    for g in groups_sorted:
        if g["n_sections"] < min_sections:
            continue
        pid = f"panel_{g['panel_hash']}"
        member_hashes[pid] = [g["panel_hash"]]
        member_var_index[pid] = [gset[g["panel_hash"]]]
        stable.append(MergedPanel(
            panel_id=pid,
            merged_from_hashes=[g["panel_hash"]],
            sample_names=list(g["sample_names"]),
            merged_var_index=sorted(gset[g["panel_hash"]]),
            n_sections=g["n_sections"],
        ))

    # Pass 2: try to attach small groups
    unmerged_small: list[MergedPanel] = []
    for g in groups_sorted:
        if g["n_sections"] >= min_sections:
            continue
        small_set = gset[g["panel_hash"]]
        best = None  # (jaccard, target_idx)
        for i, sp in enumerate(stable):
            # Candidate target's CURRENT merged_var_index (may have shrunk)
            target_set = set(sp.merged_var_index)
            j = _jaccard(small_set, target_set)
            m = _min_overlap(small_set, target_set)
            if j >= jaccard_threshold and m >= min_overlap_threshold:
                if best is None or j > best[0]:
                    best = (j, i, m)
        if best is not None:
            j, idx, m = best
            sp = stable[idx]
            member_hashes[sp.panel_id].append(g["panel_hash"])
            member_var_index[sp.panel_id].append(small_set)
            sp.sample_names.extend(g["sample_names"])
            sp.n_sections += g["n_sections"]
            sp.merged_from_hashes.append(g["panel_hash"])
            sp.jaccard_min = min(sp.jaccard_min, float(j))
            sp.min_overlap_min = min(sp.min_overlap_min, float(m))
            # Shrink merged_var_index to intersection
            new_inter = set(sp.merged_var_index) & small_set
            sp.merged_var_index = sorted(new_inter)
        else:
            # Standalone small panel (kept for diagnostics, not training)
            pid = f"panel_{g['panel_hash']}_singleton"
            unmerged_small.append(MergedPanel(
                panel_id=pid,
                merged_from_hashes=[g["panel_hash"]],
                sample_names=list(g["sample_names"]),
                merged_var_index=sorted(small_set),
                n_sections=g["n_sections"],
                notes=f"unmerged (no stable panel met Jaccard≥{jaccard_threshold} "
                      f"& min_overlap≥{min_overlap_threshold})",
            ))

    # Pass 3 (optional): consolidate near-duplicate stable panels. A single
    # assay that fragmented into several >=min_sections hashes (e.g. the 5K
    # panel under per-section pruning) is re-merged here; the larger panel
    # absorbs the smaller and merged_var_index shrinks to their intersection.
    if consolidate_stable_jaccard is not None:
        merged_any = True
        while merged_any and len(stable) > 1:
            merged_any = False
            stable.sort(key=lambda sp: -sp.n_sections)  # largest absorbs
            for i in range(len(stable)):
                for j in range(i + 1, len(stable)):
                    a = set(stable[i].merged_var_index)
                    b = set(stable[j].merged_var_index)
                    j_sim = len(a & b) / max(1, len(a | b))
                    if j_sim >= consolidate_stable_jaccard:
                        big, small = stable[i], stable[j]
                        m_ov = len(a & b) / max(1, min(len(a), len(b)))
                        big.merged_from_hashes.extend(small.merged_from_hashes)
                        big.sample_names.extend(small.sample_names)
                        big.n_sections += small.n_sections
                        big.jaccard_min = min(big.jaccard_min, float(j_sim))
                        big.min_overlap_min = min(big.min_overlap_min, float(m_ov))
                        big.merged_var_index = sorted(a & b)
                        big.notes = (big.notes + "; " if big.notes else "") + \
                            f"consolidated {small.panel_id} (J={j_sim:.3f})"
                        stable.pop(j)
                        merged_any = True
                        break
                if merged_any:
                    break

    return stable + unmerged_small


# ---------- Per-section binning ----------

def _section_bounds_um(parquet_path: str) -> tuple[float, float, float, float]:
    """Two-pass approach: get (xmin, xmax, ymin, ymax) cheaply from parquet
    row-group statistics so we can pre-allocate a dense accumulator.
    Falls back to streaming min/max if statistics missing."""
    pf = pq.ParquetFile(parquet_path)
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for rg_i in range(pf.num_row_groups):
        rg = pf.metadata.row_group(rg_i)
        for col_i in range(rg.num_columns):
            col = rg.column(col_i)
            name = col.path_in_schema
            if name not in ("x_location", "y_location"):
                continue
            stats = col.statistics
            if stats is None or not stats.has_min_max:
                # Fallback: streaming pass
                return _section_bounds_um_streaming(parquet_path)
            if name == "x_location":
                xmin = min(xmin, float(stats.min))
                xmax = max(xmax, float(stats.max))
            else:
                ymin = min(ymin, float(stats.min))
                ymax = max(ymax, float(stats.max))
    return xmin, xmax, ymin, ymax


def _section_bounds_um_streaming(parquet_path: str) -> tuple[float, float, float, float]:
    pf = pq.ParquetFile(parquet_path)
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for batch in pf.iter_batches(batch_size=4_000_000,
                                  columns=["x_location", "y_location"]):
        x = batch.column("x_location").to_numpy(zero_copy_only=False)
        y = batch.column("y_location").to_numpy(zero_copy_only=False)
        xmin = min(xmin, float(x.min()))
        xmax = max(xmax, float(x.max()))
        ymin = min(ymin, float(y.min()))
        ymax = max(ymax, float(y.max()))
    return xmin, xmax, ymin, ymax


def bin_section_expression(
    parquet_path: str,
    var_index: list[str],
    bin_size_px: int = 256,
    qv_min: float = 20.0,
    um_per_px: float = 0.5,
    min_bin_transcripts: int = 5,
    batch_size: int = 2_000_000,
) -> tuple[sparse.csr_matrix, np.ndarray, tuple[int, int]]:
    """Bin all transcripts on `var_index` into a sparse (N_tissue_bins, G_panel)
    CSR matrix via dense-accumulator streaming.

    Memory pattern (vs naive ``pq.read_table``):
      - Naive:   peak ~ 20-30 GB on 500M-row sections (loads all rows then COO).
      - Streaming list-accumulate: still ~7-10 GB (keeps int32 row/col/gene
        triplets in RAM until concat).
      - Dense accumulator (this): bounded ~ (H*W*G*4 bytes) ≈ 200 MB for
        BRA-sized sections, plus per-batch transients ~200 MB. Stable.

    Implementation:
      1. Two-pass: first determine H, W from x/y bounds (cheap row-group
         statistics, or one streaming pass if stats missing).
      2. Allocate dense int32 grid (H*W, G).
      3. Stream parquet in batches; per batch, filter + map names → gene_idx
         vectorized via pandas; do ``np.add.at(grid_flat, (bin_flat, gene_idx), 1)``.
      4. Sparsify by tissue mask.

    Returns
    -------
    expr_sparse : csr_matrix (N_tissue, G_panel)
    bin_coords : ndarray (N_tissue, 2) int64
    grid_shape : (H, W)
    """
    G = len(var_index)
    panel_map = pd.Series(np.arange(G, dtype=np.int32), index=var_index)

    # Schema check: older Xenium parquets (pre-2025) lack `is_gene`.
    # In that case we filter only by `qv >= qv_min` and rely on the panel
    # gene-name lookup to drop control codewords (their feature_name strings
    # like "NegControl_*" / "BLANK_*" won't be in the gene panel).
    pf_schema = pq.read_schema(parquet_path)
    schema_names = set(pf_schema.names)
    has_is_gene = "is_gene" in schema_names
    cols_needed = ["feature_name", "x_location", "y_location", "qv"]
    if has_is_gene:
        cols_needed.append("is_gene")

    # Pass 1: section spatial bounds → (H, W)
    xmin, xmax, ymin, ymax = _section_bounds_um(parquet_path)
    H = int(ymax / um_per_px // bin_size_px) + 1
    W = int(xmax / um_per_px // bin_size_px) + 1
    # Defensive guard: catch malformed parquets
    if H <= 0 or W <= 0 or (H * W) > 4_000_000:
        # Fallback to streaming-list approach (older path) — should not
        # trigger on Xenium sections (typical H*W < 200k).
        raise ValueError(
            f"Unreasonable grid {H}x{W} for {parquet_path}; check bin_size_px or um_per_px."
        )

    N_bins_full = H * W
    # Dense accumulator: int32 (H*W, G).
    # Memory: 200x200 × 451 × 4 = 72 MB (BRA); 200x200 × 4952 × 4 = 800 MB (4952-panel WEM).
    grid = np.zeros((N_bins_full, G), dtype=np.int32)

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=cols_needed,
    ):
        qv = batch.column("qv").to_numpy(zero_copy_only=False)
        if has_is_gene:
            is_gene = batch.column("is_gene").to_numpy(zero_copy_only=False)
            keep = is_gene & (qv >= qv_min)
        else:
            # Older Xenium schema: panel-gene filter happens via
            # feature_name vs `var_index` lookup; control codewords have
            # names like "NegControl_*"/"BLANK_*" which won't match.
            keep = (qv >= qv_min)
        if not keep.any():
            continue
        names = batch.column("feature_name").to_pandas()
        # Some Xenium parquets (e.g. HEST-1k IDC) store `feature_name` as a
        # binary column, so values arrive as bytes and never match the str
        # gene panel (→ 0 in-panel transcripts → 0 bins). Decode to str.
        if names.size and isinstance(names.iloc[0], (bytes, bytearray)):
            names = names.str.decode("utf-8")
        names = names.where(keep).dropna()
        if names.empty:
            continue
        gene_idx = panel_map.reindex(names.values).to_numpy()
        in_panel_mask = ~np.isnan(gene_idx)
        if not in_panel_mask.any():
            continue
        gene_idx = gene_idx[in_panel_mask].astype(np.int32)
        x_um_arr = batch.column("x_location").to_numpy(zero_copy_only=False)
        y_um_arr = batch.column("y_location").to_numpy(zero_copy_only=False)
        x_um_kept = x_um_arr[keep][in_panel_mask]
        y_um_kept = y_um_arr[keep][in_panel_mask]
        row_idx = (y_um_kept / um_per_px // bin_size_px).astype(np.int32)
        col_idx = (x_um_kept / um_per_px // bin_size_px).astype(np.int32)
        # Clamp to grid (defensive against rounding edge cases)
        row_idx = np.clip(row_idx, 0, H - 1)
        col_idx = np.clip(col_idx, 0, W - 1)
        bin_flat = row_idx.astype(np.int64) * W + col_idx.astype(np.int64)
        # In-place increment
        np.add.at(grid, (bin_flat, gene_idx), 1)

    # Tissue mask + sparsify
    bin_totals = grid.sum(axis=1)
    tissue_bins = np.where(bin_totals >= min_bin_transcripts)[0]
    if tissue_bins.size == 0:
        return sparse.csr_matrix((0, G), dtype=np.int32), \
            np.zeros((0, 2), dtype=np.int64), (H, W)

    expr_tissue = sparse.csr_matrix(grid[tissue_bins, :])
    del grid  # release
    bin_coords = np.stack([tissue_bins // W, tissue_bins % W], axis=1).astype(np.int64)
    return expr_tissue, bin_coords, (H, W)


def _build_rook_adjacency(
    bin_coords: np.ndarray,
    grid_shape: tuple[int, int],
) -> sparse.csr_matrix:
    """Symmetric rook (4-neighbor) adjacency over surviving tissue bins.

    Each surviving bin is mapped to its local index 0..N-1. An edge exists
    between two surviving bins if they are 4-neighbors on the original grid.
    """
    H, W = grid_shape
    if bin_coords.shape[0] == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    flat = bin_coords[:, 0] * W + bin_coords[:, 1]
    flat_to_local = -np.ones(H * W, dtype=np.int64)
    flat_to_local[flat] = np.arange(flat.size)

    src_local = []
    dst_local = []
    rows = bin_coords[:, 0]
    cols = bin_coords[:, 1]
    # Right neighbor (col + 1)
    mask = cols < W - 1
    nbr_flat = (rows[mask]) * W + (cols[mask] + 1)
    nbr_loc = flat_to_local[nbr_flat]
    keep = nbr_loc >= 0
    src_local.append(np.where(mask)[0][keep])
    dst_local.append(nbr_loc[keep])
    # Down neighbor (row + 1)
    mask = rows < H - 1
    nbr_flat = (rows[mask] + 1) * W + (cols[mask])
    nbr_loc = flat_to_local[nbr_flat]
    keep = nbr_loc >= 0
    src_local.append(np.where(mask)[0][keep])
    dst_local.append(nbr_loc[keep])

    src = np.concatenate(src_local)
    dst = np.concatenate(dst_local)
    # Symmetric: stack both directions
    rows_all = np.concatenate([src, dst])
    cols_all = np.concatenate([dst, src])
    data = np.ones(rows_all.size, dtype=np.float32)
    N = bin_coords.shape[0]
    W_mat = sparse.coo_matrix((data, (rows_all, cols_all)), shape=(N, N)).tocsr()
    return W_mat


def section_morans_i(
    expr_sparse: sparse.csr_matrix,
    bin_coords: np.ndarray,
    grid_shape: tuple[int, int],
    normalize: Literal["lib_size_log1p", "log1p"] = "lib_size_log1p",
    min_present_bins: int = 50,
    gene_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-gene Moran's I over tissue-mask bins with rook adjacency.

    Returns
    -------
    morans_i : (G,) float32, NaN where bins_present < min_present_bins
    bins_present : (G,) int32, count of bins where count > 0
    mean_log1p_expression : (G,) float32, mean of normalized expression
        across tissue bins (whether or not nonzero)
    """
    N, G = expr_sparse.shape
    out_I = np.full(G, np.nan, dtype=np.float32)
    bins_present = np.zeros(G, dtype=np.int32)
    mean_expr = np.zeros(G, dtype=np.float32)

    if N == 0:
        return out_I, bins_present, mean_expr

    # Library-size normalize: divide each row by row sum, then log1p
    lib_size = np.asarray(expr_sparse.sum(axis=1)).ravel().astype(np.float64)
    lib_size = np.where(lib_size > 0, lib_size, 1.0)
    expr_csr = expr_sparse.tocsr()

    # bins_present per gene (via nnz per column)
    bins_present[:] = np.diff(expr_csr.tocsc().indptr).astype(np.int32)

    # Build rook adjacency once
    W_mat = _build_rook_adjacency(bin_coords, grid_shape)
    W_sum = float(W_mat.sum())
    if W_sum < 1e-12:
        return out_I, bins_present, mean_expr

    # Process gene chunks to bound memory
    for g_start in range(0, G, gene_chunk):
        g_end = min(G, g_start + gene_chunk)
        # Slice columns
        sub = expr_csr[:, g_start:g_end].toarray().astype(np.float64)  # (N, Gc)
        if normalize == "lib_size_log1p":
            sub = sub / lib_size[:, None]
        sub = np.log1p(sub)
        mean_expr[g_start:g_end] = sub.mean(axis=0).astype(np.float32)
        xb = sub - sub.mean(axis=0, keepdims=True)
        denom = (xb ** 2).sum(axis=0)
        wx = W_mat @ xb        # (N, Gc) — sparse @ dense
        num = (xb * wx).sum(axis=0)
        # Moran's I
        I_chunk = (N / W_sum) * num / (denom + 1e-12)
        out_I[g_start:g_end] = I_chunk.astype(np.float32)

    # NaN out genes with insufficient detection
    out_I[bins_present < min_present_bins] = np.nan
    return out_I, bins_present, mean_expr


# ---------- Panel-level aggregation ----------

def aggregate_panel_svgs(
    panel_var_index: list[str],
    per_section: list[dict],
    min_detection_frac: float = 0.7,
    min_morans_i_floor: float = 0.05,
    min_morans_i_floor_frac: float = 0.5,
    min_mean_log1p_expr: float = 0.01,
    k: int = 100,
    weight_by: Literal["n_cells", "n_bins", "equal"] = "n_cells",
    lambda_blend: float = 0.0,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Aggregate per-section Moran's I onto the merged panel var.index.

    Each `per_section` dict has keys:
      sid, tissue, var_index (section's gene order), morans_i, bins_present,
      mean_log1p_expression, n_cells_used, n_bins_used.

    `lambda_blend` mixes the spatial-structure axis (weighted_mean_I, the
    HVG-divergent SVG criterion) with an expression-variability axis to make
    the panel composition more HVG-like:

        score = (1 - λ)·z(weighted_mean_I) + λ·z(dispersion)

    with λ=0 reproducing the pure-SVG ranking exactly. The dispersion axis
    used here is the ACROSS-SECTION Fano factor of per-section mean log1p
    expression (var/mean over detected sections) — a cohort-variability proxy
    computable from the cached diagnostics. NOTE: this is NOT scanpy's
    cell-level HVG dispersion (which would require re-binning to recover the
    bin×gene variance); it is the cheapest faithful variability axis available
    from the per-section cache. To bring genuinely low-spatial HVG genes into
    contention, relax `min_morans_i_floor(_frac)` alongside a higher λ.

    Returns (top_k_genes, kept_diag_df, rejected_diag_df).
    """
    G = len(panel_var_index)
    name_to_pos = {g: i for i, g in enumerate(panel_var_index)}

    n_sec = len(per_section)
    I_mat = np.full((n_sec, G), np.nan, dtype=np.float32)
    bins_mat = np.zeros((n_sec, G), dtype=np.int32)
    expr_mat = np.full((n_sec, G), np.nan, dtype=np.float32)
    tissues = []
    weights = np.zeros(n_sec, dtype=np.float64)

    for si, sec in enumerate(per_section):
        sec_vi = sec["var_index"]
        # Build mapping: panel position -> section position (vectorized).
        sec_name_to_pos = {g: i for i, g in enumerate(sec_vi)}
        sec_pos = np.fromiter(
            (sec_name_to_pos.get(g, -1) for g in panel_var_index),
            dtype=np.int64, count=G,
        )
        if (sec_pos < 0).any():
            # Merged panel construction guarantees gene presence; flag if not
            missing = int((sec_pos < 0).sum())
            raise ValueError(
                f"Section {sec['sid']}: var.index missing "
                f"{missing} of {G} panel genes — merged panel "
                f"intersection should preclude this."
            )
        I_mat[si] = sec["morans_i"][sec_pos]
        bins_mat[si] = sec["bins_present"][sec_pos]
        expr_mat[si] = sec["mean_log1p_expression"][sec_pos]
        tissues.append(sec.get("tissue") or "UNK")
        if weight_by == "n_cells":
            weights[si] = float(sec.get("n_cells_used", 1))
        elif weight_by == "n_bins":
            weights[si] = float(sec.get("n_bins_used", 1))
        else:
            weights[si] = 1.0

    weights = np.where(weights > 0, weights, 1.0)
    tissues_arr = np.asarray(tissues)

    # ---- Filters ----
    detected = (bins_mat > 0) & (~np.isnan(I_mat))
    detection_frac = detected.mean(axis=0)
    morans_above_floor = (I_mat >= min_morans_i_floor) & detected
    morans_above_floor_frac = np.where(
        detection_frac > 0,
        morans_above_floor.sum(axis=0) / np.maximum(detected.sum(axis=0), 1),
        0.0,
    )
    # Mean log1p expr across DETECTED sections only
    masked_expr = np.where(detected, expr_mat, np.nan)
    mean_log1p = np.nanmean(masked_expr, axis=0)
    std_log1p = np.nanstd(masked_expr, axis=0)
    # Reasons
    pass_detection = detection_frac >= min_detection_frac
    pass_structure = morans_above_floor_frac >= min_morans_i_floor_frac
    pass_expression = (mean_log1p >= min_mean_log1p_expr)
    pass_all = pass_detection & pass_structure & pass_expression

    # ---- Weighted-mean aggregation (over detected sections only) ----
    I_for_mean = np.where(detected, I_mat, 0.0)
    w_per_pos = np.where(detected, weights[:, None], 0.0)
    sum_w = w_per_pos.sum(axis=0)
    weighted_mean_I = np.where(sum_w > 0, (I_for_mean * weights[:, None]).sum(axis=0) / sum_w, np.nan)
    mean_I = np.nanmean(np.where(detected, I_mat, np.nan), axis=0)
    std_I = np.nanstd(np.where(detected, I_mat, np.nan), axis=0)
    n_detected = detected.sum(axis=0)

    # Per-tissue weighted mean (only over detected sections of that tissue)
    unique_tissues = sorted(set(tissues_arr.tolist()))
    tissue_contrib = []
    n_tissues_with_det = np.zeros(G, dtype=np.int32)
    top_tissue = np.array([""] * G, dtype=object)
    top_tissue_I = np.full(G, np.nan, dtype=np.float32)
    for g in range(G):
        contrib = {}
        for t in unique_tissues:
            mask = (tissues_arr == t) & detected[:, g]
            if mask.sum() == 0:
                continue
            w = weights[mask]
            I_t = I_mat[mask, g]
            contrib[t] = float((I_t * w).sum() / max(w.sum(), 1e-12))
        tissue_contrib.append(json.dumps(contrib))
        n_tissues_with_det[g] = len(contrib)
        if contrib:
            top_t = max(contrib.items(), key=lambda kv: kv[1])
            top_tissue[g] = top_t[0]
            top_tissue_I[g] = top_t[1]

    # ---- Expression-variability (HVG-like) axis: across-section Fano ----
    # Fano = var(mean_log1p) / mean(mean_log1p) over detected sections.
    _eps = 1e-8
    dispersion = (std_log1p ** 2) / (mean_log1p + _eps)

    def _zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        z = np.full_like(x, np.nan)
        if finite.sum() > 1:
            mu = x[finite].mean()
            sd = x[finite].std()
            z[finite] = (x[finite] - mu) / (sd + _eps)
        return z

    blend_score = (
        (1.0 - lambda_blend) * _zscore(weighted_mean_I)
        + lambda_blend * _zscore(dispersion)
    )

    base_df = pd.DataFrame({
        "gene": panel_var_index,
        "weighted_mean_I": weighted_mean_I,
        "dispersion_fano": dispersion,
        "blend_score": blend_score,
        "mean_I": mean_I,
        "std_I": std_I,
        "n_sections_detected": n_detected,
        "detection_frac": detection_frac,
        "morans_above_floor_frac": morans_above_floor_frac,
        "mean_log1p_expression": mean_log1p,
        "mean_log1p_expression_std": std_log1p,
        "tissue_contribution_json": tissue_contrib,
        "n_tissues_with_detection": n_tissues_with_det,
        "top_tissue": top_tissue,
        "top_tissue_I": top_tissue_I,
        "pass_detection": pass_detection,
        "pass_structure": pass_structure,
        "pass_expression": pass_expression,
        "pass_all": pass_all,
    })

    sort_col = "weighted_mean_I" if lambda_blend == 0.0 else "blend_score"
    kept = base_df[pass_all].sort_values(
        sort_col, ascending=False, na_position="last"
    ).reset_index(drop=True)
    rejected = base_df[~pass_all].copy()
    # Reason column
    reasons = []
    for _, row in rejected.iterrows():
        rr = []
        if not row["pass_detection"]:
            rr.append(f"detection<{min_detection_frac}")
        if not row["pass_structure"]:
            rr.append(f"morans_floor<{min_morans_i_floor_frac}")
        if not row["pass_expression"]:
            rr.append(f"expr<{min_mean_log1p_expr}")
        reasons.append("|".join(rr))
    rejected = rejected.assign(rejection_reason=reasons).reset_index(drop=True)

    top_k = kept.head(k)
    top_k_genes = top_k["gene"].tolist()
    top_k = top_k.assign(rank=range(1, len(top_k) + 1))
    cols = ["rank"] + [c for c in top_k.columns if c != "rank"]
    top_k = top_k[cols]

    return top_k_genes, top_k, rejected
