"""Phase 0c — extract each unseen panel's MEASURED gene set for strict zero-shot (B).

For Analysis B we evaluate the frozen main 94-gene head on genes shared between its
output and each unseen panel's MEASURED genes. "Measured" = the gene symbols
physically present in the panel's evaluation sections' ``transcripts.parquet`` (the
assay design), NOT the SVG-selected genes (which were chosen using expression and,
for the public panels, the held-out test sections themselves → leak).

The GEX-2D data cache bins directly from transcripts.parquet
(``load_slide_transcripts(parquet, gene_panel)``), so any physically measured gene is
evaluable regardless of preprocessing pruning. We therefore read the parquet
feature_name set per eval section and intersect across the panel's eval sections, so
every reported shared gene is present in every evaluated section.

Writes ``outputs/gex_2d/measured_var_index/<HASH>.json`` =
  {panel_hash, eval_sections, n_measured, measured, n_shared, shared}
``shared`` is in main-panel gene order (the head's output order).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

from sigmma.evaluation.transcript_binning import manifest_path_to_transcripts_parquet

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
SVG0529 = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels_2026_05_29")
OUT_DIR = ROOT / "outputs/gex_2d/measured_var_index"

MAIN_PANEL = np.load(SVG0529 / "panel_35a7bd78801c" / "svg_top100.npy", allow_pickle=True).tolist()

# Unseen panels for B. For the public-mixed panels evaluate on TENX (public) only.
PANELS = {
    "1ca7fe57390a": None,    # ovary  — all test sections
    "a87c22a4bcb9": None,    # lung
    "2df22e67cd31": None,    # brain
    "11bbd69265ed": "TENX",  # PAAD/LUAD public
    "80b4f72e3da8": "TENX",  # SKCM public
}


def section_feature_names(parquet: str, batch_size: int = 2_000_000, max_batches: int = 12) -> set:
    """Distinct feature_name set, read in batches and stopped once stable
    (2 consecutive batches add nothing — Xenium transcripts are spatially mixed
    so all genes appear within the first row groups)."""
    pf = pq.ParquetFile(parquet)
    seen: set = set()
    stable = 0
    for i, b in enumerate(pf.iter_batches(columns=["feature_name"], batch_size=batch_size)):
        u = pc.unique(b.column("feature_name")).to_pylist()
        u = [(x.decode() if isinstance(x, (bytes, bytearray)) else x) for x in u]
        before = len(seen)
        seen.update(u)
        stable = stable + 1 if len(seen) == before else 0
        if i + 1 >= max_batches or stable >= 2:
            break
    return seen


def eval_sections(panel_hash: str, prefix: str | None) -> list[dict]:
    rows = list(csv.DictReader(open(
        ROOT / f"outputs/gex_2d/svg0529_panel_{panel_hash}/MANIFEST_panel_{panel_hash}_test.csv")))
    if prefix:
        rows = [r for r in rows if r["sample_name"].startswith(prefix)]
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_set = set(MAIN_PANEL)
    print(f"main panel = {len(MAIN_PANEL)} genes\n")
    for h, prefix in PANELS.items():
        rows = eval_sections(h, prefix)
        per_sec = {}
        measured = None
        for r in rows:
            pqp = manifest_path_to_transcripts_parquet(r["hematoxylin_eosin_image"])
            if pqp is None:
                print(f"  [warn] {h} {r['sample_name']}: no parquet"); continue
            g = section_feature_names(pqp)
            per_sec[r["sample_name"]] = len(g)
            measured = g if measured is None else (measured & g)
        if not measured:
            print(f"  [skip] {h}: no measured genes resolved"); continue
        shared = [g for g in MAIN_PANEL if g in measured]   # main-order
        out = {
            "panel_hash": h,
            "eval_sections": list(per_sec.keys()),
            "n_eval_sections": len(per_sec),
            "per_section_n_features": per_sec,
            "n_measured": len(measured),
            "measured": sorted(measured),
            "n_shared": len(shared),
            "shared": shared,
        }
        (OUT_DIR / f"{h}.json").write_text(json.dumps(out, indent=2))
        print(f"  {h}: eval_sec={len(per_sec)}  measured={len(measured)}  "
              f"shared={len(shared)}  -> {shared}")


if __name__ == "__main__":
    main()
