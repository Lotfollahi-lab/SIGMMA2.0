"""Compute panel-aware SVG (Moran's I) target lists for the in-house Xenium
training cohort.

Replaces the old `scripts/compute_in_house_5k_hvg.py` (dispersion CV) target
selection, which was bugged (resolved only 26 sanger_beacon parquets) AND
ill-suited for cross-panel cohorts.

Pipeline
--------
1. Group all train manifest sections by var.index sha1 (`panel_hash`) using
   the precomputed `outputs/diagnostics/panel_inventory/panel_groups.json`.
2. Merge small (n_sec < min_sections_per_panel) panels into similar larger
   panels by Jaccard / min-overlap thresholds. Merged pseudo-panel's gene
   axis = intersection of constituents.
3. For each section: bin transcripts → sparse (N_bins, G_panel) →
   library-size-normalize + log1p → rook-adjacency Moran's I per gene
   over tissue-mask bins. Save per-section diagnostic npz.
4. Per merged panel: reindex each section's Moran's I onto merged var.index,
   apply detection / spatial-structure / expression filters, weighted-mean
   aggregate, take top-K. Write svg_top{K}.npy + diagnostic CSV + meta.json.

Outputs (under `out_dir`, one subdir per merged panel)
----------------------------------------------------
- svg_top{K}.npy            ordered gene list (object array)
- svg_top{K}.meta.json      panel metadata + hyperparams + git commit
- svg_top{K}.csv            top-K diagnostic frame (rank/gene/scores/tissue)
- sections.csv              audit trail (per section)
- rejected_genes.csv        genes that failed filters (with reason)
- panel_index.json          (root-level) maps section_id -> panel_id
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# Make sigmma importable when running as `python scripts/...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigmma.evaluation.transcript_binning import (
    manifest_path_to_transcripts_parquet,
)
from sigmma.evaluation.svg_selection import (
    MergedPanel,
    aggregate_panel_svgs,
    bin_section_expression,
    merge_small_panels,
    section_morans_i,
    tissue_of,
)


def _read_var_index(spatial_h5_path: str) -> list[str]:
    """Read gene-name index from an h5ad/h5 file using h5py only."""
    with h5py.File(spatial_h5_path, "r") as f:
        var_grp = f["var"]
        idx_attr = var_grp.attrs.get("_index", "_index")
        if isinstance(idx_attr, bytes):
            idx_attr = idx_attr.decode()
        if idx_attr not in var_grp:
            for k in ("gene_name", "feature_name", "gene_symbol", "_index"):
                if k in var_grp:
                    idx_attr = k
                    break
            else:
                raise KeyError(f"No gene-name dataset in var/ of {spatial_h5_path}")
        raw = var_grp[idx_attr][:]
    return [g.decode() if isinstance(g, (bytes, bytearray)) else str(g) for g in raw]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "-C", str(ROOT), "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------- Per-section worker ----------

def _process_section(
    sid: str,
    he_zarr: str,
    spatial_h5: str,
    panel_hash: str,
    out_dir: str,
    bin_size_px: int,
    qv_min: float,
    min_bin_transcripts: int,
    min_present_bins: int,
    gene_chunk: int,
    force_rerun: bool = False,
) -> dict:
    """Run binning + Moran's I for one section. Saves per-section npz, returns
    a small dict summary (no large arrays in the return path).

    Skips re-computation if the per-section diagnostic npz already exists at
    `{out_dir}/_diag/{panel_hash}/{sid}.npz`, unless `force_rerun=True`.
    Idempotent re-runs: workers that died (BrokenProcessPool) can be picked
    up cleanly by re-launching the driver.
    """
    try:
        # Idempotency: skip if diagnostic already on disk.
        expected_npz = Path(out_dir) / f"_diag/{panel_hash}/{sid}.npz"
        if expected_npz.exists() and not force_rerun:
            try:
                z = np.load(expected_npz, allow_pickle=True)
                n_bins = int(z["n_bins_used"])
                n_tx = int(z["n_transcripts_used"])
                tissue = str(z["tissue"])
                return {
                    "sid": sid, "status": "ok",
                    "panel_hash": panel_hash,
                    "tissue": tissue,
                    "n_bins_used": n_bins,
                    "n_transcripts_used": n_tx,
                    "var_size": int(z["var_index"].size),
                    "resumed_from_disk": True,
                }
            except Exception as _e:
                # Corrupt npz — fall through to recompute
                pass

        parquet = manifest_path_to_transcripts_parquet(he_zarr)
        if parquet is None:
            return {"sid": sid, "status": "skip", "reason": "parquet_unresolved"}
        var_index = _read_var_index(spatial_h5)

        expr_sparse, bin_coords, grid_shape = bin_section_expression(
            parquet, var_index, bin_size_px=bin_size_px,
            qv_min=qv_min, min_bin_transcripts=min_bin_transcripts,
        )
        n_bins = int(expr_sparse.shape[0])
        if n_bins < min_present_bins:
            return {"sid": sid, "status": "skip", "reason": f"too_few_bins({n_bins})"}

        morans, bins_present, mean_log1p = section_morans_i(
            expr_sparse, bin_coords, grid_shape,
            min_present_bins=min_present_bins, gene_chunk=gene_chunk,
        )

        n_cells_used = int(expr_sparse.sum())  # total transcripts (proxy for cell count)
        tissue = tissue_of(sid) or "UNK"

        # Save per-section diagnostic
        sec_dir = Path(out_dir) / f"_diag/{panel_hash}"
        sec_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sec_dir / f"{sid}.npz",
            morans_i=morans.astype(np.float32),
            bins_present=bins_present.astype(np.int32),
            mean_log1p_expression=mean_log1p.astype(np.float32),
            var_index=np.array(var_index, dtype=object),
            n_bins_used=np.int64(n_bins),
            n_transcripts_used=np.int64(n_cells_used),
            grid_shape=np.array(grid_shape, dtype=np.int64),
            tissue=tissue,
        )
        return {
            "sid": sid, "status": "ok",
            "panel_hash": panel_hash,
            "tissue": tissue,
            "n_bins_used": n_bins,
            "n_transcripts_used": n_cells_used,
            "var_size": len(var_index),
        }
    except Exception as e:
        return {
            "sid": sid, "status": "fail",
            "panel_hash": panel_hash,
            "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


# ---------- Per-merged-panel aggregator ----------

def _aggregate_one_panel(
    panel: MergedPanel,
    per_section_dir: Path,
    section_audit_rows: list[dict],
    out_root: Path,
    K: int,
    weight_by: str,
    min_detection_frac: float,
    min_morans_i_floor: float,
    min_morans_i_floor_frac: float,
    min_mean_log1p_expr: float,
) -> Optional[dict]:
    """Load per-section diagnostics for this panel's member sections,
    reindex onto merged_var_index, run `aggregate_panel_svgs`. Returns a
    summary dict on success."""
    if len(panel.merged_var_index) == 0:
        return {"panel_id": panel.panel_id, "status": "skip", "reason": "empty_merged_var_index"}

    per_section = []
    for sid in panel.sample_names:
        candidate_dirs = [per_section_dir / h for h in panel.merged_from_hashes]
        npz_path = None
        for d in candidate_dirs:
            p = d / f"{sid}.npz"
            if p.exists():
                npz_path = p
                break
        if npz_path is None:
            # section was skipped at binning stage
            continue
        z = np.load(npz_path, allow_pickle=True)
        per_section.append({
            "sid": sid,
            "tissue": str(z["tissue"]),
            "var_index": list(z["var_index"]),
            "morans_i": z["morans_i"],
            "bins_present": z["bins_present"],
            "mean_log1p_expression": z["mean_log1p_expression"],
            "n_cells_used": int(z["n_transcripts_used"]),
            "n_bins_used": int(z["n_bins_used"]),
        })

    if len(per_section) == 0:
        return {"panel_id": panel.panel_id, "status": "skip", "reason": "no_sections_with_diagnostics"}

    top_genes, top_df, rejected_df = aggregate_panel_svgs(
        panel.merged_var_index, per_section,
        min_detection_frac=min_detection_frac,
        min_morans_i_floor=min_morans_i_floor,
        min_morans_i_floor_frac=min_morans_i_floor_frac,
        min_mean_log1p_expr=min_mean_log1p_expr,
        k=K, weight_by=weight_by,
    )

    # Outputs
    panel_dir = out_root / panel.panel_id
    panel_dir.mkdir(parents=True, exist_ok=True)
    npy_path = panel_dir / f"svg_top{K}.npy"
    np.save(npy_path, np.array(top_genes, dtype=object), allow_pickle=True)
    top_df.to_csv(panel_dir / f"svg_top{K}.csv", index=False)
    rejected_df.to_csv(panel_dir / "rejected_genes.csv", index=False)

    # Sections audit
    sec_rows = [r for r in section_audit_rows if r["sid"] in panel.sample_names]
    if sec_rows:
        pd.DataFrame(sec_rows).to_csv(panel_dir / "sections.csv", index=False)

    meta = {
        "panel_id": panel.panel_id,
        "merged_from_hashes": panel.merged_from_hashes,
        "n_sections_in_aggregate": len(per_section),
        "n_sections_in_panel": panel.n_sections,
        "merged_var_index_size": len(panel.merged_var_index),
        "jaccard_min": panel.jaccard_min,
        "min_overlap_min": panel.min_overlap_min,
        "notes": panel.notes,
        "K": K,
        "weight_by": weight_by,
        "min_detection_frac": min_detection_frac,
        "min_morans_i_floor": min_morans_i_floor,
        "min_morans_i_floor_frac": min_morans_i_floor_frac,
        "min_mean_log1p_expr": min_mean_log1p_expr,
        "merged_var_index_sample": panel.merged_var_index[:10],
    }
    with open(panel_dir / f"svg_top{K}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "panel_id": panel.panel_id,
        "status": "ok",
        "n_top_genes": len(top_genes),
        "n_rejected_genes": int(len(rejected_df)),
        "n_sections": len(per_section),
        "top1": top_genes[0] if top_genes else None,
        "median_top_K_I": float(top_df["weighted_mean_I"].median()) if len(top_df) else float("nan"),
    }


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_manifest",
        default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv",
    )
    ap.add_argument(
        "--panel_groups_json",
        default="outputs/diagnostics/panel_inventory/panel_groups.json",
    )
    ap.add_argument("--out_dir", default="data/svg_panels")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--bin_size_px", type=int, default=256)
    ap.add_argument("--qv_min", type=float, default=20.0)
    ap.add_argument("--min_bin_transcripts", type=int, default=5)
    ap.add_argument("--force_rerun", action="store_true",
                    help="Recompute per-section diagnostics even if their npz "
                         "already exists. Default: skip-if-exists for idempotent "
                         "resume after partial runs.")
    ap.add_argument("--min_present_bins", type=int, default=50)
    ap.add_argument("--gene_chunk", type=int, default=256)
    ap.add_argument("--min_sections_per_panel", type=int, default=5)
    ap.add_argument("--jaccard_threshold", type=float, default=0.90)
    ap.add_argument("--min_overlap_threshold", type=float, default=0.95)
    ap.add_argument("--consolidate_stable_jaccard", type=float, default=None,
                    help="If set, merge near-duplicate STABLE panels with "
                         "pairwise Jaccard >= this value into one (e.g. 0.5 to "
                         "re-unify a 5K assay fragmented by per-section pruning).")
    ap.add_argument("--weight_by", default="n_cells",
                    choices=["n_cells", "n_bins", "equal"])
    ap.add_argument("--min_detection_frac", type=float, default=0.7)
    ap.add_argument("--min_morans_i_floor", type=float, default=0.05)
    ap.add_argument("--min_morans_i_floor_frac", type=float, default=0.5)
    ap.add_argument("--min_mean_log1p_expr", type=float, default=0.01)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--limit_sections", type=int, default=0,
                    help="Smoke-test cap on # sections processed (0 = all)")
    ap.add_argument("--limit_panels", type=str, default=None,
                    help="Comma-separated panel_hash list to restrict (smoke)")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load manifest + panel groups, merge small panels -----
    manifest = pd.read_csv(args.train_manifest)
    manifest = manifest.drop_duplicates("sample_name", keep="last")
    sid_to_he = dict(zip(manifest["sample_name"], manifest["hematoxylin_eosin_image"]))
    sid_to_spatial = dict(zip(manifest["sample_name"], manifest["spatial_preprocessed"]))
    print(f"[manifest] {len(manifest)} unique sections", flush=True)

    with open(args.panel_groups_json) as f:
        panel_groups = json.load(f)
    if args.limit_panels:
        wanted = set(args.limit_panels.split(","))
        panel_groups = [g for g in panel_groups if g["panel_hash"] in wanted]
        print(f"[smoke] limiting to {len(panel_groups)} panel groups: {wanted}", flush=True)

    merged_panels = merge_small_panels(
        panel_groups,
        min_sections=args.min_sections_per_panel,
        jaccard_threshold=args.jaccard_threshold,
        min_overlap_threshold=args.min_overlap_threshold,
        consolidate_stable_jaccard=args.consolidate_stable_jaccard,
    )
    print(f"[merge] {len(panel_groups)} input groups -> {len(merged_panels)} merged panels", flush=True)
    for mp in merged_panels:
        kind = "STABLE" if "_singleton" not in mp.panel_id else "UNMERGED"
        print(
            f"  [{kind}] {mp.panel_id}  n_sec={mp.n_sections}  "
            f"merged_var={len(mp.merged_var_index)}  hashes={mp.merged_from_hashes}  "
            f"Jmin={mp.jaccard_min:.3f}  overlap_min={mp.min_overlap_min:.3f}",
            flush=True,
        )

    # ----- 2. Per-section binning + Moran's I (parallel) -----
    # Build per-section task list with the panel_hash they came from
    section_panel = {}  # sid -> panel_hash (original, not merged)
    for g in panel_groups:
        for sid in g["sample_names"]:
            section_panel[sid] = g["panel_hash"]

    task_args = []
    for sid, panel_hash in section_panel.items():
        if sid not in sid_to_he:
            continue
        task_args.append((sid, sid_to_he[sid], sid_to_spatial[sid], panel_hash))
    if args.limit_sections > 0:
        task_args = task_args[: args.limit_sections]
    print(f"[sections] processing {len(task_args)} sections with {args.n_workers} workers", flush=True)

    section_results: list[dict] = []
    failed: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(
                _process_section,
                sid, he, spatial, panel_hash, str(out_root),
                args.bin_size_px, args.qv_min, args.min_bin_transcripts,
                args.min_present_bins, args.gene_chunk,
                args.force_rerun,
            ): sid
            for sid, he, spatial, panel_hash in task_args
        }
        for i, fut in enumerate(as_completed(futures)):
            sid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"sid": sid, "status": "fail", "reason": f"{type(e).__name__}: {e}"}
            section_results.append(r)
            if r["status"] != "ok":
                failed.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == len(task_args):
                ok = sum(1 for x in section_results if x["status"] == "ok")
                skip = sum(1 for x in section_results if x["status"] == "skip")
                fail = sum(1 for x in section_results if x["status"] == "fail")
                print(f"  [progress] {i+1}/{len(task_args)}  ok={ok}  skip={skip}  fail={fail}", flush=True)

    pd.DataFrame(section_results).to_csv(out_root / "_section_status.csv", index=False)
    if failed:
        print(f"[WARN] {len(failed)} sections failed; first 3:", flush=True)
        for r in failed[:3]:
            print(f"  {r['sid']}: {r.get('reason')}", flush=True)

    # ----- 3. Per merged panel: aggregate -----
    per_section_dir = out_root / "_diag"
    panel_index: dict[str, str] = {}
    summaries: list[dict] = []

    for panel in merged_panels:
        if "_singleton" in panel.panel_id and panel.n_sections < args.min_sections_per_panel:
            # Aggregating an unmerged tiny panel: still emit (n=1..4 may be noisy)
            pass
        s = _aggregate_one_panel(
            panel, per_section_dir, section_results, out_root,
            args.k, args.weight_by,
            args.min_detection_frac, args.min_morans_i_floor,
            args.min_morans_i_floor_frac, args.min_mean_log1p_expr,
        )
        summaries.append(s)
        if s and s.get("status") == "ok":
            for sid in panel.sample_names:
                panel_index[sid] = panel.panel_id

    with open(out_root / "panel_index.json", "w") as f:
        json.dump(panel_index, f, indent=2)

    pd.DataFrame([s for s in summaries if s]).to_csv(
        out_root / "_panel_summary.csv", index=False
    )

    # Write a run-level meta file
    with open(out_root / "_run_meta.json", "w") as f:
        json.dump({
            "git_commit": _git_commit(),
            "argv": vars(args),
            "n_panel_groups_input": len(panel_groups),
            "n_merged_panels": len(merged_panels),
            "n_sections_processed": len(section_results),
            "n_sections_ok": sum(1 for r in section_results if r["status"] == "ok"),
            "n_sections_failed": len(failed),
        }, f, indent=2)

    print(f"\n[done] outputs in {out_root}", flush=True)
    print(f"[summary] panel-level results:", flush=True)
    for s in summaries:
        if not s:
            continue
        if s.get("status") == "ok":
            print(f"  {s['panel_id']:<45s} n_sec={s['n_sections']:>3d}  "
                  f"top1={s['top1']:<10s}  median_top_K_I={s['median_top_K_I']:.4f}",
                  flush=True)
        else:
            print(f"  {s.get('panel_id'):<45s} SKIP ({s.get('reason')})", flush=True)


if __name__ == "__main__":
    main()
