"""Plot Pearson(bin_size) for H&E→GEX methods at multiple bin granularities.

Reads per-grid result JSONs written by ``probe_gex_2d.py phase=eval`` (with
``--eval_target_grids`` set) and ``predict_baseline_gex_2d.py`` (one bin-mode
run per ``--target_grid``). Produces one figure showing how median per-gene
Pearson degrades as bin size shrinks (Visium-scale ~64-128 µm → sub-cellular
~9 µm).

Hypothesis being illustrated: H&E→GEX *regressor* baselines (STnet, DeepPT,
HisToGene) emit one (G,) vector per tile and never see per-bin supervision, so
they degrade catastrophically at fine grids. Spatial-aware probes (SIGMMA
multi-scale + UNI single-scale + fine-tuned UNI variants) are scale-flexible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# H&E FOV physical size: fov_size_px * um_per_px = 256 * 0.5 = 128 µm.
FOV_UM = 128.0


# ---------------------------------------------------------------------------
# Method registry: where to find results + how to render
# ---------------------------------------------------------------------------
# Each entry: (display_name, family, color, marker, linestyle, results_loader)
# family ∈ {"probe-uni", "probe-sigmma", "probe-ft-uni", "baseline"}.
def probe_path(out_root: str, repr_name: str, tg: int) -> Path:
    """Probe per-grid JSON path. Native grid filename is '{repr}_g{tg}.json'."""
    return Path(out_root) / "results_linear_sweep" / f"{repr_name}_g{tg}.json"


def baseline_bin_path(out_root_base: str, tg: int, baseline: str) -> Path:
    """Baseline bin-mode JSON at tg ∈ {2, 7, 14} lives in suffixed dirs."""
    if tg == 14:
        return Path(out_root_base) / f"{baseline}_bin.json"
    return Path(out_root_base.rstrip("/") + f"_g{tg}") / f"{baseline}_bin.json"


def baseline_fov_path(out_root_base: str, baseline: str) -> Path:
    return Path(out_root_base) / f"{baseline}_fov.json"


# Palette from [[project_figure_palette]] + extensions.
PALETTE = {
    "UNI":          "#888888",
    "SIGMMA micro": "#d62728",
    "SIGMMA meso":  "#2ca02c",
    "SIGMMA macro": "#1f77b4",
    "SIGMMA concat": "#000000",
    "BLEEP-UNI":    "#ff7f0e",
    "CLIP-UNI":     "#8c564b",
    "OmiCLIP-UNI":  "#e377c2",
    "PLIP-UNI":     "#17becf",
    "STnet":        "#bbbbbb",
    "DeepPT":       "#999999",
    "HisToGene":    "#777777",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def collect_probe(out_root: str, repr_name: str, target_grids: list[int]) -> dict:
    """Return {tg: {metric: value, ...}} for a probe."""
    points = {}
    for tg in target_grids:
        d = _load_json(probe_path(out_root, repr_name, tg))
        if d is not None:
            points[tg] = {
                "median_per_gene_pearson": d["median_per_gene_pearson"],
                "per_bin_r2_pooled": d.get("per_bin_r2_pooled"),
                "num_val_fovs": d.get("num_val_fovs"),
            }
    return points


def collect_baseline(out_root_base: str, baseline: str, target_grids: list[int]) -> dict:
    """For baselines: tg=1 reads from {baseline}_fov.json (FOV-aggregate eval).
    Other tg reads from {out_root_base}{_g{tg} if tg!=14}/{baseline}_bin.json."""
    points = {}
    for tg in target_grids:
        if tg == 1:
            d = _load_json(baseline_fov_path(out_root_base, baseline))
        else:
            d = _load_json(baseline_bin_path(out_root_base, tg, baseline))
        if d is not None:
            points[tg] = {
                "median_per_gene_pearson": d["median_per_gene_pearson"],
                "per_bin_r2_pooled": d.get("per_bin_r2_pooled"),
                "num_val_fovs": d.get("num_val_fovs"),
            }
    return points


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curve(rows: list[dict], out_path: Path, metric: str = "median_per_gene_pearson",
               title: str = "H&E → GEX prediction vs bin granularity"):
    """rows: [{name, family, color, points: {tg: {metric: val}}}, ...]"""
    fig, ax = plt.subplots(figsize=(7, 5))

    family_style = {
        "probe-uni":      {"linestyle": "-",  "marker": "o", "linewidth": 2.0},
        "probe-sigmma":   {"linestyle": "-",  "marker": "o", "linewidth": 2.0},
        "probe-ft-uni":   {"linestyle": "--", "marker": "s", "linewidth": 1.5},
        "baseline":       {"linestyle": ":",  "marker": "v", "linewidth": 1.5},
    }

    for row in rows:
        pts = row["points"]
        if not pts:
            continue
        tgs = sorted(pts.keys())
        bin_um = [FOV_UM / tg for tg in tgs]
        ys = [pts[tg][metric] for tg in tgs]
        style = family_style.get(row["family"], {})
        ax.plot(bin_um, ys, color=row["color"], label=row["name"], **style)

    # Visium scale anchor: ~55 µm spot diameter, ~100 µm pitch.
    ax.axvspan(55, 100, color="gold", alpha=0.15, label="Visium spot/pitch")

    ax.set_xscale("log")
    ax.set_xlabel("Bin size (µm)")
    ax.set_ylabel("Median per-gene Pearson")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] saved {out_path.with_suffix('.png')}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_root", default="outputs/gex_2d/disp_g14_cohort",
                    help="Base cohort dir for vanilla UNI + SIGMMA probes")
    ap.add_argument("--ft_cohort_pattern",
                    default="outputs/gex_2d/disp_g14_cohort_{ft}",
                    help="Pattern with {ft} placeholder for FT-UNI cohort dirs")
    ap.add_argument("--baseline_root", default="outputs/gex_2d/results/baselines_step950",
                    help="Base dir for baseline tg=14 + tg=1 (fov) outputs; "
                         "_g{tg} suffix used for tg ∈ {2, 7}.")
    ap.add_argument("--target_grids", default="1,2,7,14",
                    help="Comma-separated grids to plot. Probes need _g{tg}.json files; "
                         "baselines need either _fov.json (tg=1) or _bin.json under suffixed dirs.")
    ap.add_argument("--out", default="outputs/figures/gex_2d/resolution_curve",
                    help="Output figure path (no extension; both .png and .pdf written)")
    ap.add_argument("--summary_csv", default=None,
                    help="If set, write a CSV with all collected points")
    args = ap.parse_args()

    target_grids = [int(x) for x in args.target_grids.split(",") if x.strip()]

    rows = []

    # Probes — vanilla UNI + 4 SIGMMA scales (in cohort_root)
    for repr_name, display in [
        ("uni",    "UNI"),
        ("micro",  "SIGMMA micro"),
        ("meso",   "SIGMMA meso"),
        ("macro",  "SIGMMA macro"),
        ("concat", "SIGMMA concat"),
    ]:
        points = collect_probe(args.cohort_root, repr_name, target_grids)
        rows.append({
            "name": display, "family": "probe-uni" if repr_name == "uni" else "probe-sigmma",
            "color": PALETTE[display], "points": points,
        })

    # Probes — 4 FT-UNI variants (one cohort dir each, input_repr=uni)
    for ft, display in [
        ("bleep_uni",   "BLEEP-UNI"),
        ("clip_uni",    "CLIP-UNI"),
        ("omiclip_uni", "OmiCLIP-UNI"),
        ("plip_uni",    "PLIP-UNI"),
    ]:
        ft_root = args.ft_cohort_pattern.format(ft=ft)
        points = collect_probe(ft_root, "uni", target_grids)
        rows.append({
            "name": display, "family": "probe-ft-uni",
            "color": PALETTE[display], "points": points,
        })

    # Baselines
    for bb, display in [
        ("stnet",     "STnet"),
        ("deeppt",    "DeepPT"),
        ("histogene", "HisToGene"),
    ]:
        points = collect_baseline(args.baseline_root, bb, target_grids)
        rows.append({
            "name": display, "family": "baseline",
            "color": PALETTE[display], "points": points,
        })

    if args.summary_csv:
        with open(args.summary_csv, "w") as f:
            f.write("method,family,target_grid,bin_size_um,median_per_gene_pearson,per_bin_r2_pooled,num_val_fovs\n")
            for row in rows:
                for tg, vals in sorted(row["points"].items()):
                    f.write(f"{row['name']},{row['family']},{tg},{FOV_UM/tg:.3f},"
                            f"{vals['median_per_gene_pearson']:.4f},"
                            f"{vals.get('per_bin_r2_pooled') if vals.get('per_bin_r2_pooled') is not None else ''},"
                            f"{vals.get('num_val_fovs', '')}\n")
        print(f"[csv]  wrote {args.summary_csv}", flush=True)

    # Report missing
    for row in rows:
        missing = [tg for tg in target_grids if tg not in row["points"]]
        if missing:
            print(f"  [warn] {row['name']:<14} missing grids: {missing}", flush=True)

    plot_curve(rows, Path(args.out))


if __name__ == "__main__":
    main()
