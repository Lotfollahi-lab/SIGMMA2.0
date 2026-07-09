"""H1 specialization analysis — does SIGMMA's per-FOV gain depend on cell density?

Computes per-FOV R1/R2/R3 area shares from MoNuSAC test tiles, joins with
per-FOV alternative_metrics.csv for each finetuned model, and regresses
Δ_image = metric(model) − metric(uni) on R1_share.

Outputs (in --out_dir):
  per_fov_regions.csv             image_id, R1_share, R2_share, R3_share, n_R1
  per_fov_deltas.csv              tag, image_id, R1_share, R2_share, R3_share,
                                  ΔAJI, ΔPQ, ΔDetF1@0.5
  h1_slopes.csv                   tag, metric, intercept, slope, slope_CI_lo,
                                  slope_CI_hi, n
  figures/h1_density_vs_delta.png|pdf
  figures/h1_quartile_bars.png|pdf
"""

from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.eva_critique.data import iter_monusac_test

# ----- palette from project_figure_palette.md -----------------------------------
PALETTE = {
    "uni":               "#888888",
    "sigmma_multiscale": "#2ca02c",   # meso green (multiscale)
    "sigmma_no_lora":    "#1f77b4",   # macro blue
    "per_scale_0":       "#d62728",   # micro red
    "per_scale_1":       "#2ca02c",
    "per_scale_2":       "#1f77b4",
    "bleep_uni_ft":      "#ff7f0e",
    "clip_uni_ft":       "#9467bd",
    "omiclip_uni_ft":    "#8c564b",
    "plip_uni_ft":       "#e377c2",
}

# Models considered the "primary" ones for the H1 figure
PRIMARY_MODELS = ["bleep_uni_ft", "omiclip_uni_ft", "clip_uni_ft", "plip_uni_ft",
                  "sigmma_multiscale", "sigmma_no_lora"]


def build_per_fov_regions(data_root: str) -> pd.DataFrame:
    rows = []
    for tile in iter_monusac_test(data_root, split="test"):
        rows.append({
            "image_id":  tile.image_id,
            "R1_share":  float(tile.R1.mean()),
            "R2_share":  float(tile.R2.mean()),
            "R3_share":  float(tile.R3.mean()),
            "R4_share":  float(tile.R4.mean()),
            "tissue":    float(tile.tissue.mean()),
            "n_R1_px":   int(tile.R1.sum()),
        })
    return pd.DataFrame(rows)


def load_per_fov_metrics(csv_root: str) -> dict[str, pd.DataFrame]:
    files = {
        "uni":              f"{csv_root}/_diagnostics_eval/sigmma_multiscale/alternative_metrics.csv",
        "sigmma_multiscale":f"{csv_root}/_diagnostics_eval/sigmma_multiscale/alternative_metrics.csv",
        "sigmma_no_lora":   f"{csv_root}/sigmma_no_lora_eval/sigmma_no_lora/alternative_metrics.csv",
        "per_scale_0":      f"{csv_root}/per_scale/scale0/alternative_metrics.csv",
        "per_scale_1":      f"{csv_root}/per_scale/scale1/alternative_metrics.csv",
        "per_scale_2":      f"{csv_root}/per_scale/scale2/alternative_metrics.csv",
        "bleep_uni_ft":     f"{csv_root}/baseline_uni_eval/bleep_uni_ft/alternative_metrics.csv",
        "clip_uni_ft":      f"{csv_root}/baseline_uni_eval/clip_uni_ft/alternative_metrics.csv",
        "plip_uni_ft":      f"{csv_root}/baseline_uni_eval/plip_uni_ft/alternative_metrics.csv",
        "omiclip_uni_ft":   f"{csv_root}/baseline_uni_eval/omiclip_uni_ft/alternative_metrics.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for tag, p in files.items():
        df = pd.read_csv(p)
        wanted = "uni" if tag == "uni" else df["model"].unique()
        if tag == "uni":
            d = df[df["model"] == "uni"].copy()
        else:
            d = df[df["model"] != "uni"].copy()
        # Some sigmma_multiscale CSVs have 3 scales × 85 rows; collapse by image_id mean.
        d = d.groupby("image_id", as_index=False)[["AJI", "PQ", "DQ", "SQ", "DetF1@0.5",
                                                    "DetF1@0.7", "HD95"]].mean()
        out[tag] = d
    return out


def compute_deltas(metrics: dict[str, pd.DataFrame], regions: pd.DataFrame) -> pd.DataFrame:
    uni = metrics["uni"]
    rows = []
    for tag, d in metrics.items():
        if tag == "uni":
            continue
        m = d.merge(uni[["image_id", "AJI", "PQ", "DetF1@0.5"]].rename(
            columns={"AJI": "AJI_uni", "PQ": "PQ_uni", "DetF1@0.5": "DetF1_uni"}
        ), on="image_id", how="inner")
        m = m.merge(regions, on="image_id", how="inner")
        for _, r in m.iterrows():
            rows.append({
                "tag":          tag,
                "image_id":     r["image_id"],
                "R1_share":     r["R1_share"],
                "R2_share":     r["R2_share"],
                "R3_share":     r["R3_share"],
                "AJI_model":    r["AJI"],
                "PQ_model":     r["PQ"],
                "DetF1_model":  r["DetF1@0.5"],
                "AJI_uni":      r["AJI_uni"],
                "PQ_uni":       r["PQ_uni"],
                "DetF1_uni":    r["DetF1_uni"],
                "ΔAJI":         r["AJI"]    - r["AJI_uni"],
                "ΔPQ":          r["PQ"]     - r["PQ_uni"],
                "ΔDetF1":       r["DetF1@0.5"] - r["DetF1_uni"],
            })
    return pd.DataFrame(rows)


def bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 42):
    """OLS slope + percentile bootstrap CI on (slope, intercept)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xx, yy = x[idx], y[idx]
        if np.ptp(xx) == 0:
            slopes[b] = np.nan
            continue
        slopes[b] = np.polyfit(xx, yy, 1)[0]
    slope, intercept = np.polyfit(x, y, 1)
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return float(intercept), float(slope), float(lo), float(hi)


def regression_table(deltas: pd.DataFrame, metrics=("ΔAJI", "ΔPQ", "ΔDetF1")):
    rows = []
    for tag, d in deltas.groupby("tag"):
        for m in metrics:
            intercept, slope, lo, hi = bootstrap_slope(d["R1_share"].to_numpy(), d[m].to_numpy())
            rows.append({"tag": tag, "metric": m,
                         "intercept": round(intercept, 4),
                         "slope":     round(slope, 4),
                         "slope_lo":  round(lo, 4),
                         "slope_hi":  round(hi, 4),
                         "n":         len(d)})
    return pd.DataFrame(rows)


# ----- plotting ----------------------------------------------------------------

def plot_scatter_grid(deltas: pd.DataFrame, models: list[str], metric: str,
                      out_png: str):
    """One row of scatter+regression panels for the selected models."""
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 3.0),
                              sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    y_all = deltas[metric].to_numpy()
    y_lo, y_hi = np.nanpercentile(y_all, [1, 99])
    pad = 0.05 * (y_hi - y_lo)
    for ax, m in zip(axes, models):
        d = deltas[deltas["tag"] == m]
        x = d["R1_share"].to_numpy()
        y = d[metric].to_numpy()
        ax.scatter(x, y, s=14, alpha=0.6, color=PALETTE.get(m, "#444"),
                   edgecolor="none")
        intercept, slope, lo, hi = bootstrap_slope(x, y)
        xs = np.linspace(0, x.max() * 1.02, 50)
        ax.plot(xs, intercept + slope * xs, color="black", lw=1.4)
        # CI band: refit at each bootstrap; here approximate with slope CI x mean x
        ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.7)
        ax.set_title(f"{m}\nslope={slope:+.3f} [{lo:+.3f},{hi:+.3f}]", fontsize=9)
        ax.set_xlabel("R1 share (cell-dense)")
        ax.set_ylim(y_lo - pad, y_hi + pad)
    axes[0].set_ylabel(f"{metric}  (model − uni, per FOV)")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def plot_quartile_bars(deltas: pd.DataFrame, models: list[str], metric: str,
                        out_png: str):
    """Bars: ΔMetric per R1-share quartile, grouped by model."""
    q = pd.qcut(deltas["R1_share"], q=4, labels=["Q1 (lowest R1)", "Q2", "Q3", "Q4 (highest R1)"])
    deltas = deltas.assign(R1_q=q)
    agg = (deltas[deltas["tag"].isin(models)]
           .groupby(["tag", "R1_q"])[metric]
           .agg(["mean", "sem"]).reset_index())

    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    quartiles = ["Q1 (lowest R1)", "Q2", "Q3", "Q4 (highest R1)"]
    width = 0.13
    xpos = np.arange(len(quartiles))
    for i, m in enumerate(models):
        sub = agg[agg["tag"] == m].set_index("R1_q").reindex(quartiles)
        ax.bar(xpos + (i - (len(models) - 1) / 2) * width,
               sub["mean"], width, yerr=sub["sem"],
               color=PALETTE.get(m, "#444"), label=m, capsize=2, lw=0)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(quartiles)
    ax.set_ylabel(f"{metric} (model − uni)")
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/monusac")
    ap.add_argument("--csv_root",  default="outputs/eva_critique")
    ap.add_argument("--out_dir",   default="outputs/eva_critique/h1_specialization")
    ap.add_argument("--figures_dir", default="figures/diagnostics")
    args = ap.parse_args()

    os.makedirs(args.out_dir,     exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("[h1] building per-FOV regions…")
    regions = build_per_fov_regions(args.data_root)
    regions.to_csv(os.path.join(args.out_dir, "per_fov_regions.csv"), index=False)
    print(f"     n_fov={len(regions)}  R1_share: median={regions['R1_share'].median():.3f}, "
          f"q25={regions['R1_share'].quantile(.25):.3f}, q75={regions['R1_share'].quantile(.75):.3f}")

    print("[h1] loading per-model metrics…")
    metrics = load_per_fov_metrics(args.csv_root)

    print("[h1] computing deltas…")
    deltas = compute_deltas(metrics, regions)
    deltas.to_csv(os.path.join(args.out_dir, "per_fov_deltas.csv"), index=False)

    print("[h1] running regressions…")
    reg = regression_table(deltas)
    reg.to_csv(os.path.join(args.out_dir, "h1_slopes.csv"), index=False)
    print(reg.pivot(index="tag", columns="metric",
                     values=["slope", "slope_lo", "slope_hi"]).round(4).to_string())

    print("[h1] plotting…")
    plot_scatter_grid(deltas, PRIMARY_MODELS, "ΔAJI",
                       os.path.join(args.figures_dir, "h1_density_vs_dAJI.png"))
    plot_scatter_grid(deltas, PRIMARY_MODELS, "ΔPQ",
                       os.path.join(args.figures_dir, "h1_density_vs_dPQ.png"))
    plot_quartile_bars(deltas, PRIMARY_MODELS, "ΔAJI",
                        os.path.join(args.figures_dir, "h1_quartile_dAJI.png"))
    plot_quartile_bars(deltas, PRIMARY_MODELS, "ΔPQ",
                        os.path.join(args.figures_dir, "h1_quartile_dPQ.png"))

    print(f"[h1] done. CSVs in {args.out_dir}, figures in {args.figures_dir}.")


if __name__ == "__main__":
    main()
