"""Aggregate test-cohort 2D GEX metrics across all conditions on
panel_2d22ba31 SVG-100 (14 test sections × 28,407 FOVs at K=14).

Reads:
  - probe-style JSONs (SIGMMA α-sweep + UNI2-h-noFT + 4 contrastive FT baselines)
    at outputs/gex_2d/svg_panel_2d22ba31_step1000_test/<cond>/results{,_linear_sweep}/<repr>.json
  - predict-style JSONs (HisToGene, ST-Net) at the same out_root with
    <baseline>_{bin,fov}.json

Writes:
  outputs/gex_2d/svg_panel_2d22ba31_step1000_test/_aggregate/comparison.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

OUT_ROOT = Path("outputs/gex_2d/svg_panel_2d22ba31_step1000_test")
AGG_DIR = OUT_ROOT / "_aggregate"

# Column mapping: (CSV column, JSON key)
COLS = [
    ("Pearson",     "median_per_gene_pearson"),
    ("R2_pool",     "per_bin_r2_pooled"),
    ("MoranI_d",    "median_morans_i_diff"),
    ("MSE_std",     "median_mse_per_gene_standardized"),
    ("SSIM_slide",  "median_ssim_per_slide"),
    ("MSEslideStd", "median_mse_per_slide_standardized"),
]

CSV_HEADER = ["row"] + [c for c, _ in COLS]


def _scalar(v: Any) -> float:
    """Unwrap probe-style {mean, std, values} OR pass through flat float."""
    if isinstance(v, dict) and "mean" in v:
        return float(v["mean"])
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def row_for(label: str, json_path: Path) -> dict[str, Any]:
    s = json.load(open(json_path))
    out = {"row": label}
    for col, key in COLS:
        out[col] = _scalar(s.get(key))
    return out


def main():
    rows: list[dict[str, Any]] = []

    # --- UNI2-h no FT -------------------------------------------------------
    rows.append(row_for(
        "UNI2-h no FT",
        OUT_ROOT / "uni2h_no_ft" / "results_linear_sweep" / "uni.json",
    ))

    # --- Contrastive FT baselines (linear_sweep) -----------------------------
    for name, sub in [
        ("BLEEP FT",    "bleep_uni_step1000"),
        ("CLIP FT",     "clip_uni_step1000"),
        ("PLIP FT",     "plip_uni_step1000"),
        ("OmiCLIP FT",  "omiclip_uni_step1000"),
    ]:
        p = OUT_ROOT / sub / "results_linear_sweep" / "uni.json"
        if p.exists():
            rows.append(row_for(name, p))

    # --- HisToGene + ST-Net (predict_baseline_gex_2d, bin mode = K=14) ------
    for name, sub, base in [
        ("HisToGene (K=14)", "histogene_svg_cache", "histogene"),
        ("ST-Net (K=14)",    "stnet_svg_cache",    "stnet"),
        ("DeepPT (K=14)",    "deeppt_svg_cache",   "deeppt"),
    ]:
        bin_path = OUT_ROOT / sub / f"{base}_bin.json"
        if bin_path.exists():
            rows.append(row_for(name, bin_path))

    # Path A (FOV-level, ~Visium spot scale) — different schema (no spatial
    # metrics). Report Pearson + R²_pool only; other columns NaN.
    for name, sub, base in [
        ("HisToGene (Path A, FOV)", "histogene_svg_cache", "histogene"),
        ("ST-Net (Path A, FOV)",    "stnet_svg_cache",    "stnet"),
        ("DeepPT (Path A, FOV)",    "deeppt_svg_cache",   "deeppt"),
    ]:
        fov_path = OUT_ROOT / sub / f"{base}_fov.json"
        if fov_path.exists():
            s = json.load(open(fov_path))
            r = {"row": name}
            r["Pearson"] = _scalar(s.get("median_per_gene_pearson"))
            r["R2_pool"] = _scalar(s.get("per_bin_r2_pooled"))
            for col, _ in COLS:
                r.setdefault(col, float("nan"))
            rows.append(r)

    # --- SIGMMA α-sweep × 5 reprs -------------------------------------------
    for alpha in ["a00", "a01", "a02", "a04"]:
        for repr_name in ["uni", "micro", "meso", "macro", "concat"]:
            p = OUT_ROOT / f"sigmma_{alpha}_step1000" / "results" / f"{repr_name}.json"
            if p.exists():
                rows.append(row_for(f"SIGMMA {alpha}/{repr_name}", p))

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = AGG_DIR / "comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_HEADER})

    print(f"[aggregate] wrote {len(rows)} rows to {out_csv}", flush=True)
    # Echo a markdown-style preview to stdout for quick eyeballing.
    print()
    print(" | ".join(CSV_HEADER))
    print("-|-".join(["-" * len(c) for c in CSV_HEADER]))
    for r in rows:
        vals = []
        for c in CSV_HEADER:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append("nan" if math.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        print(" | ".join(vals))


if __name__ == "__main__":
    main()
