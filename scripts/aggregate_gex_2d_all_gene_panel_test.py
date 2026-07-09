"""Aggregate test-cohort 2D GEX metrics across all conditions on the
all-gene (svg0529, panel_35a7bd78801c) cohort at step2000 (n=33 test sections,
g14 dense bin).

Reads:
  - probe-style JSONs (UNI2-h no-FT, 4 contrastive FT baselines, SIGMMA a01)
    at outputs/gex_2d/svg0529_step2000_test/<cond>/results{,_linear_sweep}/<repr>.json
  - predict-style regressor JSONs (HisToGene, ST-Net, DeepPT) at
    outputs/gex_2d/svg0529_panel_35a7bd78801c_test/<base>_svg0529_cache/<base>_{bin,fov}.json

Writes:
  outputs/gex_2d/svg0529_step2000_test/_aggregate/comparison.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
PROBE_ROOT = ROOT / "outputs/gex_2d/svg0529_step2000_test"
REGRESS_ROOT = ROOT / "outputs/gex_2d/svg0529_panel_35a7bd78801c_test"
AGG_DIR = PROBE_ROOT / "_aggregate"

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
    """Unwrap probe-style {mean, std, ...} OR pass through flat float."""
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

    # --- Regressors (predict_baseline_gex_2d, K=14 dense bin) ----------------
    for name, sub, base in [
        ("ST-Net (K=14)",    "stnet_svg0529_cache",    "stnet"),
        ("HisToGene (K=14)", "histogene_svg0529_cache", "histogene"),
        ("DeepPT (K=14)",    "deeppt_svg0529_cache",   "deeppt"),
    ]:
        p = REGRESS_ROOT / sub / f"{base}_bin.json"
        if p.exists():
            rows.append(row_for(name, p))

    # Path A (FOV-level) — different schema (no spatial metrics). Pearson +
    # R2_pool only; other columns NaN.
    for name, sub, base in [
        ("ST-Net (Path A, FOV)",    "stnet_svg0529_cache",    "stnet"),
        ("HisToGene (Path A, FOV)", "histogene_svg0529_cache", "histogene"),
        ("DeepPT (Path A, FOV)",    "deeppt_svg0529_cache",   "deeppt"),
    ]:
        p = REGRESS_ROOT / sub / f"{base}_fov.json"
        if p.exists():
            s = json.load(open(p))
            r = {"row": name}
            r["Pearson"] = _scalar(s.get("median_per_gene_pearson"))
            r["R2_pool"] = _scalar(s.get("per_bin_r2_pooled"))
            for col, _ in COLS:
                r.setdefault(col, float("nan"))
            rows.append(r)

    # --- UNI2-h no FT --------------------------------------------------------
    rows.append(row_for(
        "UNI2-h no FT",
        PROBE_ROOT / "uni2h_no_ft" / "results_linear_sweep" / "uni.json",
    ))

    # --- Contrastive FT baselines (linear_sweep) -----------------------------
    for name, sub in [
        ("BLEEP FT",    "bleep_uni_step2000"),
        ("CLIP FT",     "clip_uni_step2000"),
        ("PLIP FT",     "plip_uni_step2000"),
        ("OmiCLIP FT",  "omiclip_uni_step2000"),
    ]:
        p = PROBE_ROOT / sub / "results_linear_sweep" / "uni.json"
        if p.exists():
            rows.append(row_for(name, p))

    # --- SIGMMA a01 (MLP head) × reprs ---------------------------------------
    for repr_name in ["uni", "concat", "micro", "meso", "macro"]:
        p = PROBE_ROOT / "sigmma_a01_step2000" / "results" / f"{repr_name}.json"
        if p.exists():
            rows.append(row_for(f"SIGMMA a01 MLP/{repr_name}", p))

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = AGG_DIR / "comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_HEADER})

    print(f"[aggregate] wrote {len(rows)} rows to {out_csv}", flush=True)
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
