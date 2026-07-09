"""Re-derive 2D-GEX metrics on the Phoenix∩SVG-100 gene overlap (16 genes), so
the zero-shot Phoenix generative baseline can be compared head-to-head with
SIGMMA / regressor / contrastive-FT methods on a common, fairly-evaluable gene
set and on scale-invariant pattern metrics only.

No model re-runs: every method's per-gene values are read from the JSONs it
already produced (probe-style ``results*/uni.json`` and predict-style
``*_bin.json`` both come from the shared ``summarize()``), then subset to the 16
overlap genes:

  - **Pearson**  : ``per_gene_pearson`` is pooled over FOVs → a 16-value
    distribution per method (paired *by gene*, n=16).
  - **SSIM**     : ``ssim_per_slide_per_gene`` → per-slide median over the 16
    genes → a 14-value distribution per method (paired *by slide*, n=14).
    SSIM is computed on min-max-standardized maps, so it is scale-invariant and
    valid despite Phoenix's log-normalised (vs log1p-count) units.

MSE / Moran's I are intentionally excluded: MSE is not unit-comparable to
Phoenix, and Moran's I is not stored per-gene (cannot be subset).

Writes:
  outputs/gex_2d/svg_panel_2d22ba31_step1000_test/_aggregate/
      comparison_phoenix_overlap.csv     # scalar summary per method
      phoenix_overlap_distributions.json # per-gene + per-slide arrays (for the figure)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES_ROOT = ROOT / "outputs/gex_2d/svg_panel_2d22ba31_step1000_test"
AGG_DIR = RES_ROOT / "_aggregate"
OVERLAP_JSON = ROOT / "outputs/gex_2d/svg_panel_2d22ba31/phoenix_overlap_genes.json"

# (label, relative json path). Mirrors aggregate_gex_2d_svg_panel_2d22ba31_test.py
# plus Phoenix. SIGMMA shown at alpha=0.1 / repr=uni (representative).
CONDITIONS = [
    ("Phoenix (zero-shot)", "phoenix/phoenix_bin.json"),
    ("UNI2-h no FT",        "uni2h_no_ft/results_linear_sweep/uni.json"),
    ("BLEEP FT",            "bleep_uni_step1000/results_linear_sweep/uni.json"),
    ("CLIP FT",             "clip_uni_step1000/results_linear_sweep/uni.json"),
    ("PLIP FT",             "plip_uni_step1000/results_linear_sweep/uni.json"),
    ("OmiCLIP FT",          "omiclip_uni_step1000/results_linear_sweep/uni.json"),
    ("ST-Net",              "stnet_svg_cache/stnet_bin.json"),
    ("HisToGene",           "histogene_svg_cache/histogene_bin.json"),
    ("DeepPT",              "deeppt_svg_cache/deeppt_bin.json"),
    ("SIGMMA",              "sigmma_a01_step1000/results/uni.json"),
]


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def pearson_over_overlap(s: dict, overlap: list[str]) -> list[float]:
    """Per-gene pooled Pearson for the overlap genes that are present + finite."""
    pg = s.get("per_gene_pearson", {})
    return [float(pg[g]) for g in overlap if g in pg and _finite(pg.get(g))]


def ssim_per_slide_over_overlap(s: dict, overlap: list[str]) -> dict[str, float]:
    """Per-slide median SSIM over the overlap genes (present + finite)."""
    sps = s.get("ssim_per_slide_per_gene", {})
    out = {}
    for sid, genes in sps.items():
        vals = [float(genes[g]) for g in overlap if g in genes and _finite(genes.get(g))]
        if vals:
            out[sid] = float(np.median(vals))
    return out


def main():
    meta = json.load(open(OVERLAP_JSON))
    overlap = meta["overlap_genes"]
    print(f"[overlap] {len(overlap)} genes: {overlap}", flush=True)

    dists: dict[str, dict] = {}
    rows: list[dict] = []
    for label, rel in CONDITIONS:
        p = RES_ROOT / rel
        if not p.exists():
            print(f"  [skip] {label}: missing {rel}", flush=True)
            continue
        s = json.load(open(p))
        pearson = pearson_over_overlap(s, overlap)
        ssim_slide = ssim_per_slide_over_overlap(s, overlap)
        dists[label] = {
            "pearson_per_gene": pearson,
            "ssim_per_slide": ssim_slide,
        }
        rows.append({
            "row": label,
            "Pearson_med16":   (float(np.median(pearson)) if pearson else float("nan")),
            "Pearson_n_genes": len(pearson),
            "SSIM_med_slide":  (float(np.median(list(ssim_slide.values())))
                                if ssim_slide else float("nan")),
            "SSIM_n_slides":   len(ssim_slide),
        })

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    header = ["row", "Pearson_med16", "Pearson_n_genes", "SSIM_med_slide", "SSIM_n_slides"]
    with open(AGG_DIR / "comparison_phoenix_overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(AGG_DIR / "phoenix_overlap_distributions.json", "w") as f:
        json.dump({"overlap_genes": overlap, "distributions": dists}, f, indent=2)

    # markdown preview
    print(f"\n[aggregate] {len(rows)} methods → {AGG_DIR/'comparison_phoenix_overlap.csv'}\n")
    print(" | ".join(header))
    print("-|-".join("-" * len(h) for h in header))
    for r in rows:
        print(" | ".join(
            f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in header
        ))


if __name__ == "__main__":
    main()
