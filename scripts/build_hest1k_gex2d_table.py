"""Assemble the HEST-1k zero-shot GEX-2D benchmark table.

Reads the strict shared-gene zero-shot result JSONs (frozen main-94 heads applied
to external HEST-1k sections) for the tissues with >=10 shared genes — SKCM and
IDC — and emits one tidy CSV + a markdown table per tissue across all conditions
(SIGMMA scales + UNI2-h + 4 FT baselines).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RES = ROOT / "outputs/gex_2d/public_zeroshot_strict_step2000"
OUT_CSV = ROOT / "outputs/gex_2d/hest1k_zeroshot_benchmark_table.csv"

# tissue -> panel hash  (>=10 shared genes only)
TISSUES = {
    "SKCM": "80b4f72e3da8",
    "IDC":  "2bbc576266c5",
}

# display label -> result-file tag
# Regression baselines (ST-Net/HisToGene/DeepPT) are reported in their headline
# bin mode (target_grid=14); their JSONs carry no n_shared_genes/sections field
# (back-filled from the panel meta below) but are evaluated on the SAME shared
# genes / sections / FOVs as the contrastive + SIGMMA methods.
METHODS = [
    ("UNI2-h (no-FT)",  "uni2h_no_ft"),
    ("ST-Net",          "stnet_bin"),
    ("HisToGene",       "histogene_bin"),
    ("DeepPT",          "deeppt_bin"),
    ("BLEEP",           "bleep_uni_step2000"),
    ("CLIP",            "clip_uni_step2000"),
    ("PLIP",            "plip_uni_step2000"),
    ("OmiCLIP",         "omiclip_uni_step2000"),
    ("SIGMMA uni",      "sigmma_a01_step2000__uni"),
    ("SIGMMA micro",    "sigmma_a01_step2000__micro"),
    ("SIGMMA meso",     "sigmma_a01_step2000__meso"),
    ("SIGMMA macro",    "sigmma_a01_step2000__macro"),
    ("SIGMMA concat",   "sigmma_a01_step2000__concat"),
]

METRICS = [
    ("Pearson",  "median_per_gene_pearson"),
    ("MSE_std",  "median_mse_per_slide_standardized"),
    ("Moran_d",  "median_morans_i_diff"),
    ("SSIM_fov", "median_ssim_per_fov"),
]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def main() -> None:
    rows = []
    for tissue, h in TISSUES.items():
        pdir = RES / h
        meta = None
        for label, tag in METHODS:
            f = pdir / f"{tag}.json"
            if not f.exists():
                print(f"  [warn] missing {tissue}/{tag}")
                continue
            d = json.load(open(f))
            meta = meta or d
            rows.append({
                "tissue": tissue, "panel": h, "method": label,
                # regression baselines store no gene/section metadata -> back-fill
                # from the panel meta (same shared genes / sections by construction)
                "n_shared_genes": d.get("n_shared_genes")
                or (meta or {}).get("n_shared_genes"),
                "n_sections": len(d.get("sections") or (meta or {}).get("sections") or []),
                "num_val_fovs": d.get("num_val_fovs"),
                **{mk: d.get(jk) for mk, jk in METRICS},
            })
        if meta:
            print(f"\n### {tissue}  (panel {h}, {meta.get('n_shared_genes')} genes, "
                  f"{len(meta.get('sections', []))} sections, "
                  f"{meta.get('num_val_fovs')} FOVs) ###")
            hdr = f"{'method':16s} | " + " | ".join(f"{m:>8s}" for m, _ in METRICS)
            print(hdr); print("-" * len(hdr))
            for r in rows:
                if r["tissue"] != tissue:
                    continue
                print(f"{r['method']:16s} | " +
                      " | ".join(f"{fmt(r[m]):>8s}" for m, _ in METRICS))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        fields = ["tissue", "panel", "method", "n_shared_genes", "n_sections",
                  "num_val_fovs"] + [m for m, _ in METRICS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
