"""Build per-panel GEX-2D artifacts (gene_panel.json + split manifests) from the
2026-05-29 panel-aware SVG output.

For each viable panel (train>=5 AND test>=1 sections) under
``svg_panels_2026_05_29``, this writes a panel_dir under
``outputs/gex_2d/svg0529_<panel>/`` containing:
  - gene_panel.json     (same schema probe_gex_2d.py expects: a "panel" gene
                          list + provenance fields, mapped from the SVG meta)
  - MANIFEST_<panel>.csv         (all sections — used by the cache phase)
  - MANIFEST_<panel>_{train,val,test}.csv  (sanger-split-filtered subsets)

Manifests are produced by filtering the canonical sanger_MANIFEST_{split}.csv by
sample_name membership, so they inherit the exact columns + section->path
mapping (no leakage: split membership comes straight from the sanger splits).

Mirrors the layout of the legacy outputs/gex_2d/svg_panel_2d22ba31/.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

SVG_ROOT = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels_2026_05_29"
)
SANGER = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium")
OUT_ROOT = Path("/nfs/team361/dj16/projects/SIGMMA/outputs/gex_2d")

MIN_TRAIN = 5
MIN_TEST = 1


def read_split(split: str) -> "list[dict]":
    with open(SANGER / f"sanger_MANIFEST_{split}.csv") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list, header: list, path: Path) -> int:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    panel_index = json.loads((SVG_ROOT / "panel_index.json").read_text())
    # panel -> set(sample_name)
    panel_secs: dict[str, set] = {}
    for sample, panel in panel_index.items():
        panel_secs.setdefault(panel, set()).add(sample)

    splits = {s: read_split(s) for s in ("train", "val", "test")}
    header = list(splits["train"][0].keys())

    summary = []
    for panel, secs in sorted(panel_secs.items(), key=lambda x: -len(x[1])):
        sub = {
            s: [r for r in splits[s] if r["sample_name"] in secs]
            for s in splits
        }
        n_tr, n_va, n_te = len(sub["train"]), len(sub["val"]), len(sub["test"])
        if not (n_tr >= MIN_TRAIN and n_te >= MIN_TEST):
            summary.append((panel, len(secs), n_tr, n_va, n_te, "SKIP"))
            continue

        panel_dir = OUT_ROOT / f"svg0529_{panel}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        # --- gene_panel.json -------------------------------------------------
        meta = json.loads((SVG_ROOT / panel / "svg_top100.meta.json").read_text())
        genes = np.load(SVG_ROOT / panel / "svg_top100.npy", allow_pickle=True).tolist()
        gene_panel = {
            "k": int(meta.get("K", len(genes))),
            "hvg_method": "panel_aware_svg_morans_i",
            "panel_id": meta.get("panel_id", panel),
            "n_sections_in_aggregate": meta.get("n_sections_in_aggregate"),
            "n_sections_in_panel": meta.get("n_sections_in_panel"),
            "merged_var_index_size": meta.get("merged_var_index_size"),
            "weight_by": meta.get("weight_by"),
            "min_detection_frac": meta.get("min_detection_frac"),
            "min_morans_i_floor": meta.get("min_morans_i_floor"),
            "min_morans_i_floor_frac": meta.get("min_morans_i_floor_frac"),
            "min_mean_log1p_expr": meta.get("min_mean_log1p_expr"),
            "source_npy": str(SVG_ROOT / panel / "svg_top100.npy"),
            "source_meta": str(SVG_ROOT / panel / "svg_top100.meta.json"),
            "panel": genes,
        }
        (panel_dir / "gene_panel.json").write_text(json.dumps(gene_panel, indent=2))

        # --- manifests -------------------------------------------------------
        all_rows = sub["train"] + sub["val"] + sub["test"]
        write_manifest(all_rows, header, panel_dir / f"MANIFEST_{panel}.csv")
        for s in ("train", "val", "test"):
            write_manifest(sub[s], header, panel_dir / f"MANIFEST_{panel}_{s}.csv")

        summary.append((panel, len(secs), n_tr, n_va, n_te, f"OK n_genes={len(genes)}"))

    print(f"{'panel':32s} {'tot':>4} {'train':>5} {'val':>4} {'test':>4}  status")
    for panel, tot, tr, va, te, status in summary:
        print(f"{panel:32s} {tot:>4} {tr:>5} {va:>4} {te:>4}  {status}")
    built = [s for s in summary if s[-1].startswith("OK")]
    print(f"\nBuilt {len(built)} panel_dirs under {OUT_ROOT}/svg0529_*")


if __name__ == "__main__":
    main()
