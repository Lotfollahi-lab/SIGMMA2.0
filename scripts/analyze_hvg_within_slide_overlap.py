"""Sanity check (NO code change): how much does a within-slide seurat_v3 HVG
panel overlap the existing cross-slide `dispersion` HVG panel?

Motivation: the current `compute_top_k_hvg_by_dispersion` aggregates each slide
to a single per-gene total count, then ranks by the CV of those totals ACROSS
slides — i.e. it measures inter-sample abundance variability, not within-tissue
spot/cell heterogeneity. The reference protocol instead computes HVGs WITHIN
each slide (spot/cell-level variance, mean-variance corrected) via
`scanpy.pp.highly_variable_genes(flavor='seurat_v3')` and then takes a
cross-slide consensus.

This script builds a true cell x gene count matrix per section (group Xenium
transcripts by `cell_id`), applies the reference's QC filters, runs seurat_v3
per section, and reports:
  - per-section overlap of the within-slide HVG-50 vs the existing disp panel
  - a cross-slide CONSENSUS HVG-50 (genes ranked by #sections-HVG) vs disp panel
  - inter-section agreement (how stable per-section HVG is)

Run: .venv/bin/python scripts/analyze_hvg_within_slide_overlap.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import scipy.sparse as sp
import scanpy as sc

sys.path.insert(0, ".")
from sigmma.evaluation.transcript_binning import manifest_path_to_transcripts_parquet

TRAIN_MANIFEST = (
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
    "sanger_MANIFEST_train.csv"
)
DISP_PANEL = "outputs/gex_2d/disp/gene_panel.json"
K = 50
QV_MIN = 20.0
N_SECTIONS = 24          # sample for a quick-but-representative check
SEED = 0
TARGET_READ_ROWS = 15_000_000   # cap transcripts read per section (memory + IO)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def section_seurat_v3_hvg(parquet_path: str, k: int = K) -> list[str] | None:
    """Build cell x gene counts from a slide's transcripts and return the
    top-k seurat_v3 HVGs (within-slide), or None if too sparse to rank."""
    import pyarrow as pa
    schema = set(pq.read_schema(parquet_path).names)
    cols = ["feature_name", "cell_id", "qv"]
    has_is_gene = "is_gene" in schema
    if has_is_gene:
        cols.append("is_gene")

    # Sections range ~1M–1B transcripts; reading a 1B-row brain slide whole OOMs.
    # Read a uniformly-spaced subset of row groups to hit ~TARGET_READ_ROWS, and
    # scale the count-based QC thresholds by the sampled fraction below.
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    n_rg = pf.num_row_groups
    frac = min(1.0, TARGET_READ_ROWS / max(total, 1))
    stride = max(1, round(1.0 / frac)) if frac < 1.0 else 1
    rg_ids = list(range(0, n_rg, stride))

    kept_cell, kept_gene = [], []
    read_rows = 0
    for rg in rg_ids:
        t = pf.read_row_group(rg, columns=cols)
        read_rows += t.num_rows
        mask = pc.greater_equal(t.column("qv"), QV_MIN)
        if has_is_gene:
            mask = pc.and_(t.column("is_gene"), mask)
        mask = pc.and_(pc.not_equal(t.column("cell_id"), "UNASSIGNED"), mask)
        t = t.filter(mask)
        if t.num_rows:
            kept_cell.append(t.column("cell_id").combine_chunks())
            kept_gene.append(t.column("feature_name").combine_chunks())
    if not kept_cell:
        return None
    eff_frac = read_rows / max(total, 1)

    # Dictionary-encode the two string columns → int32 codes + small uniques.
    # Avoids materializing the kept rows as Python str objects (prev OOM cause).
    cell_dict = pa.chunked_array(kept_cell).combine_chunks().dictionary_encode()
    gene_dict = pa.chunked_array(kept_gene).combine_chunks().dictionary_encode()
    cell_codes = cell_dict.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    gene_codes = gene_dict.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    gene_uniques = gene_dict.dictionary.to_pylist()
    if gene_uniques and isinstance(gene_uniques[0], (bytes, bytearray)):
        gene_uniques = [
            g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else g
            for g in gene_uniques
        ]
    gene_names = np.asarray(gene_uniques, dtype=object)
    n_cells = int(cell_codes.max()) + 1
    n_genes = int(gene_codes.max()) + 1
    X = sp.coo_matrix(
        (np.ones(len(cell_codes), dtype=np.float32), (cell_codes, gene_codes)),
        shape=(n_cells, n_genes),
    ).tocsr()

    adata = sc.AnnData(X=X)
    adata.var_names = [str(g) for g in gene_names]

    # Reference QC (thresholds scaled by the sampled fraction so the cut matches
    # the full slide): spots(cells) >=20 counts; genes >=200 counts & detected
    # in >=10% of cells.
    min_cell_counts = max(1, int(round(20 * eff_frac)))
    min_gene_counts = max(1, int(round(200 * eff_frac)))
    sc.pp.filter_cells(adata, min_counts=min_cell_counts)
    if adata.n_obs == 0:
        return None
    sc.pp.filter_genes(adata, min_counts=min_gene_counts)
    sc.pp.filter_genes(adata, min_cells=int(0.10 * adata.n_obs))
    if adata.n_vars < k:
        return None

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=k)
    hvg = adata.var.sort_values("highly_variable_rank").index[:k].tolist()
    return hvg


def main():
    disp = set(json.load(open(DISP_PANEL))["panel"])
    print(f"[ref] existing disp panel: {len(disp)} genes\n")

    df = pd.read_csv(TRAIN_MANIFEST)
    rows = []
    for _, r in df.iterrows():
        p = manifest_path_to_transcripts_parquet(r["hematoxylin_eosin_image"])
        if p:
            rows.append((r["sample_name"], p, r.get("annotation", "")))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rows), size=min(N_SECTIONS, len(rows)), replace=False)
    sample = [rows[i] for i in sorted(idx)]
    print(f"[plan] {len(sample)}/{len(rows)} sections sampled (seed={SEED})\n")

    per_section_sets = []
    vote = Counter()
    print(f"{'section':32s} {'nHVG':>4s} {'∩disp':>5s} {'Jaccard':>7s}")
    for name, p, _ann in sample:
        try:
            hvg = section_seurat_v3_hvg(p)
        except Exception as e:
            print(f"{name:32s}  ERROR: {type(e).__name__}: {e}")
            continue
        if hvg is None:
            print(f"{name:32s}  (skipped — too sparse)")
            continue
        s = set(hvg)
        per_section_sets.append((name, s))
        vote.update(s)
        inter = len(s & disp)
        print(f"{name:32s} {len(s):4d} {inter:5d} {jaccard(s, disp):7.3f}")

    if not per_section_sets:
        print("\nno sections produced HVG — aborting")
        return

    # Per-section overlap summary
    inters = [len(s & disp) for _, s in per_section_sets]
    jacs = [jaccard(s, disp) for _, s in per_section_sets]
    print(f"\n[per-section vs disp]  mean ∩ = {np.mean(inters):.1f}/{K}  "
          f"(min {min(inters)}, max {max(inters)})   mean Jaccard = {np.mean(jacs):.3f}")

    # Cross-slide CONSENSUS panel: genes by #sections-HVG (batch_key analog)
    consensus = [g for g, _ in vote.most_common(K)]
    cons_set = set(consensus)
    print(f"\n[consensus HVG-{K} vs disp]  ∩ = {len(cons_set & disp)}/{K}  "
          f"Jaccard = {jaccard(cons_set, disp):.3f}")
    print("  consensus∖disp (in new, not old), first 20:",
          [g for g in consensus if g not in disp][:20])
    print("  disp∖consensus (in old, not new), first 20:",
          [g for g in disp if g not in cons_set][:20])

    # Inter-section agreement (stability of within-slide HVG)
    pj = []
    for i in range(len(per_section_sets)):
        for j in range(i + 1, len(per_section_sets)):
            pj.append(jaccard(per_section_sets[i][1], per_section_sets[j][1]))
    print(f"\n[inter-section stability]  mean pairwise Jaccard = {np.mean(pj):.3f}  "
          f"(n_pairs={len(pj)})")

    # how many genes appear in >=50% of sections
    ns = len(per_section_sets)
    robust = [g for g, c in vote.items() if c >= 0.5 * ns]
    print(f"[consensus core]  {len(robust)} genes HVG in >=50% of sections; "
          f"{len(set(robust) & disp)} of them are in disp")


if __name__ == "__main__":
    main()
