"""Panel inventory diagnostic for the in-house Xenium training cohort.

Answers the five questions from the GEX-aware soft contrastive loss planning:
  1. Union size of gene names across all training sections.
  2. Intersection size across all training sections.
  3. Per-tissue intersection size.
  4. Panel-group sizes (clusters of sections that share an identical ``var.index``).
  5. For a 50-gene panel chosen by ``compute_top_k_hvg_by_dispersion`` over the
     training transcript parquets, how many of those 50 are present in each
     section's ``var.index`` (and how many sections are missing zero / >0).

Reads ONLY ``var.index`` from each section's h5ad via ``anndata.read_h5ad(backed='r')``
— no full X load — so ~150 sections finish in minutes, not hours.

Outputs:
  outputs/diagnostics/panel_inventory/panel_inventory.json   (summary)
  outputs/diagnostics/panel_inventory/per_section.csv        (per-section row)
  outputs/diagnostics/panel_inventory/panel_groups.json      (group_id -> [sample names], var_index)
  outputs/diagnostics/panel_inventory/chosen_50_genes.json   (panel + per-section presence count)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sigmma.utils.tissue_map import tissue_of, GROUP_OF


def _tissue_from_sample_name(sample_name: str) -> str:
    """Authoritative display group for a section (sample_id).

    Uses sample_metadata.csv (single source of truth) and collapses the two
    skin cohorts (beacon + CTCL) into one 'skin' group. Falls back to the raw
    tissue_type if it has no display group, or 'UNKNOWN' if the section is not
    in the metadata.

    Examples
    --------
    BK25-SKI-27-FO-1-S3        -> skin
    DF29-LIV-0-FFPE-1-S1-i     -> liver
    CV2-KID-0-FO-2             -> renal_cell_carcinoma
    SOB26-WEM-FFPE-1-S183      -> whole_embryo
    """
    t = tissue_of(sample_name)
    if t is None:
        return "UNKNOWN"
    return GROUP_OF.get(t, t)


def _read_var_index(spatial_file: str) -> list[str]:
    """Read the gene-name index from an h5ad/h5 file using h5py only.

    Avoids anndata's backed='r' mode which loads obs/var DataFrames fully
    (OOMs on large embryo/brain sections with millions of cells).
    """
    import h5py

    with h5py.File(spatial_file, "r") as f:
        var_grp = f["var"]
        idx_attr = var_grp.attrs.get("_index", "_index")
        if isinstance(idx_attr, bytes):
            idx_attr = idx_attr.decode()
        if idx_attr not in var_grp:
            # Fall back to common gene-name dataset keys
            for k in ("gene_name", "feature_name", "gene_symbol", "_index"):
                if k in var_grp:
                    idx_attr = k
                    break
            else:
                raise KeyError(f"No gene-name dataset found in var/ of {spatial_file}")
        raw = var_grp[idx_attr][:]
    return [g.decode() if isinstance(g, (bytes, bytearray)) else str(g) for g in raw]


def _process_section(row: tuple[str, str]) -> Optional[dict]:
    sample, spatial_file = row
    try:
        var_idx = _read_var_index(spatial_file)
    except Exception as e:
        return {
            "sample_name": sample,
            "tissue": _tissue_from_sample_name(sample),
            "error": f"{type(e).__name__}: {e}",
        }
    h = hashlib.sha1("|".join(var_idx).encode()).hexdigest()[:12]
    return {
        "sample_name": sample,
        "tissue": _tissue_from_sample_name(sample),
        "var_index": var_idx,
        "panel_hash": h,
        "n_genes": len(var_idx),
    }


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/diagnostics/panel_inventory",
    )
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--skip_panel", action="store_true",
                    help="Skip the 50-gene panel computation (Q5). Useful for "
                         "a fast first pass (Q1-Q4 only).")
    ap.add_argument("--panel_k", type=int, default=50)
    ap.add_argument("--panel_qv_min", type=float, default=20.0)
    ap.add_argument("--panel_abundance_floor_top", type=int, default=300)
    ap.add_argument("--panel_min_slides_present", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest = manifest.drop_duplicates(subset="sample_name", keep="last").reset_index(drop=True)
    print(f"[manifest] {len(manifest)} unique sections in {args.manifest}", flush=True)

    rows = list(zip(manifest["sample_name"].astype(str),
                    manifest["spatial_preprocessed"].astype(str)))

    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_process_section, r): r[0] for r in rows}
        for i, fut in enumerate(as_completed(futures)):
            sample = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"sample_name": sample, "error": str(e)}
            results.append(r)
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                print(f"  [scan] {i + 1}/{len(rows)} sections read", flush=True)

    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]
    if errors:
        print(f"[WARN] {len(errors)} sections failed to read var.index. "
              f"See per_section.csv for details.", flush=True)

    # ---------- per-section CSV ----------
    per_section_rows = []
    for r in results:
        per_section_rows.append({
            "sample_name": r["sample_name"],
            "tissue": r.get("tissue", "UNKNOWN"),
            "panel_hash": r.get("panel_hash"),
            "n_genes": r.get("n_genes"),
            "error": r.get("error"),
        })
    pd.DataFrame(per_section_rows).to_csv(out_dir / "per_section.csv", index=False)

    if not good:
        print("[FATAL] No sections readable.", flush=True)
        sys.exit(1)

    # ---------- Q1: union ----------
    union_set = set()
    for r in good:
        union_set.update(r["var_index"])
    union_size = len(union_set)
    print(f"[Q1] union size across {len(good)} sections: {union_size}", flush=True)

    # ---------- Q2: intersection ----------
    intersection_set = set(good[0]["var_index"])
    for r in good[1:]:
        intersection_set &= set(r["var_index"])
    intersection_size = len(intersection_set)
    print(f"[Q2] intersection size: {intersection_size}", flush=True)

    # ---------- Q3: per-tissue intersection ----------
    by_tissue: dict[str, list[dict]] = defaultdict(list)
    for r in good:
        by_tissue[r["tissue"]].append(r)
    per_tissue_intersection = {}
    for tissue, rs in by_tissue.items():
        inter = set(rs[0]["var_index"])
        for r in rs[1:]:
            inter &= set(r["var_index"])
        per_tissue_intersection[tissue] = {
            "n_sections": len(rs),
            "intersection_size": len(inter),
            "union_size": len(set().union(*(r["var_index"] for r in rs))),
        }
    print(f"[Q3] per-tissue: " + ", ".join(
        f"{t}({d['n_sections']} sec, ∩{d['intersection_size']})"
        for t, d in sorted(per_tissue_intersection.items())
    ), flush=True)

    # ---------- Q4: panel groups ----------
    by_panel_hash: dict[str, list[dict]] = defaultdict(list)
    for r in good:
        by_panel_hash[r["panel_hash"]].append(r)
    panel_groups = []
    for hash_, rs in sorted(by_panel_hash.items(), key=lambda kv: -len(kv[1])):
        rs_sorted = sorted(rs, key=lambda r: r["sample_name"])
        panel_groups.append({
            "panel_hash": hash_,
            "n_sections": len(rs),
            "panel_size": rs_sorted[0]["n_genes"],
            "tissues": sorted({r["tissue"] for r in rs}),
            "sample_names": [r["sample_name"] for r in rs_sorted],
        })
    print(f"[Q4] distinct panels: {len(panel_groups)}", flush=True)
    for g in panel_groups[:10]:
        print(f"     panel={g['panel_hash']} size={g['panel_size']:>5} "
              f"sections={g['n_sections']:>3} tissues={g['tissues']}", flush=True)
    if len(panel_groups) > 10:
        print(f"     ... {len(panel_groups) - 10} more groups", flush=True)

    # Save panel-group catalogue (without the full var_index list, only canonical name list)
    panel_groups_full = []
    for g in panel_groups:
        canonical = sorted(by_panel_hash[g["panel_hash"]][0]["var_index"])
        panel_groups_full.append({**g, "gene_names_sorted": canonical})
    _write_json(out_dir / "panel_groups.json", panel_groups_full)

    summary = {
        "manifest": args.manifest,
        "n_sections_total": len(rows),
        "n_sections_readable": len(good),
        "n_sections_failed": len(errors),
        "q1_union_size": union_size,
        "q2_intersection_size": intersection_size,
        "q3_per_tissue": per_tissue_intersection,
        "q4_n_panel_groups": len(panel_groups),
        "q4_panel_group_sizes": [g["n_sections"] for g in panel_groups],
    }

    # ---------- Q5: chosen 50 genes presence ----------
    if not args.skip_panel:
        print("[Q5] computing 50-gene dispersion panel from training parquets …", flush=True)
        from sigmma.evaluation.transcript_binning import (
            compute_top_k_hvg_by_dispersion,
            manifest_path_to_transcripts_parquet,
        )
        # Build parquet path list from manifest's H&E zarr paths
        parquet_paths = []
        missing_parquet = []
        for he_zarr in manifest["hematoxylin_eosin_image"].astype(str):
            pq_path = manifest_path_to_transcripts_parquet(he_zarr)
            if pq_path is None:
                missing_parquet.append(he_zarr)
            else:
                parquet_paths.append(pq_path)
        print(f"     parquet paths resolved: {len(parquet_paths)} / {len(manifest)} "
              f"(missing: {len(missing_parquet)})", flush=True)
        if len(parquet_paths) == 0:
            print("[Q5] no parquet paths resolved — skipping Q5.", flush=True)
        else:
            chosen = compute_top_k_hvg_by_dispersion(
                parquet_paths,
                k=args.panel_k,
                qv_min=args.panel_qv_min,
                abundance_floor_top=args.panel_abundance_floor_top,
                min_slides_present=args.panel_min_slides_present,
            )
            chosen_set = set(chosen)
            print(f"     chosen panel size: {len(chosen)} (requested k={args.panel_k})", flush=True)

            per_section_presence = []
            histogram = defaultdict(int)
            for r in good:
                var_set = set(r["var_index"])
                present = chosen_set & var_set
                missing = chosen_set - var_set
                per_section_presence.append({
                    "sample_name": r["sample_name"],
                    "tissue": r["tissue"],
                    "panel_hash": r["panel_hash"],
                    "present_count": len(present),
                    "missing_count": len(missing),
                    "missing_genes": sorted(missing),
                })
                histogram[len(present)] += 1

            # Write CSV
            pd.DataFrame(
                [{k: (v if k != "missing_genes" else ";".join(v)) for k, v in row.items()}
                 for row in per_section_presence]
            ).to_csv(out_dir / "chosen_50_per_section.csv", index=False)

            # Distribution print
            print(f"     present_count histogram (n_sections by # of chosen present):", flush=True)
            for cnt in sorted(histogram, reverse=True):
                print(f"       {cnt:>3} / {len(chosen)} present → {histogram[cnt]} sections", flush=True)

            _write_json(out_dir / "chosen_50_genes.json", {
                "panel": chosen,
                "panel_k_requested": args.panel_k,
                "panel_size_actual": len(chosen),
                "panel_qv_min": args.panel_qv_min,
                "panel_abundance_floor_top": args.panel_abundance_floor_top,
                "panel_min_slides_present": args.panel_min_slides_present,
                "n_parquets_used": len(parquet_paths),
                "n_parquets_missing": len(missing_parquet),
                "presence_histogram": dict(sorted(histogram.items())),
                "n_sections_with_all_present":
                    sum(1 for r in per_section_presence if r["missing_count"] == 0),
                "n_sections_with_any_missing":
                    sum(1 for r in per_section_presence if r["missing_count"] > 0),
            })
            summary["q5_panel_size"] = len(chosen)
            summary["q5_n_parquets_used"] = len(parquet_paths)
            summary["q5_n_sections_with_all_present"] = \
                sum(1 for r in per_section_presence if r["missing_count"] == 0)
            summary["q5_presence_histogram"] = dict(sorted(histogram.items()))

    _write_json(out_dir / "panel_inventory.json", summary)
    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
