"""Aggregate per-tile CSVs into deltas + paired tests + figures + 1-page summary.

Outputs (under --out_dir):
  decoder_ladder.pdf
  region_metric_forest_plot.pdf
  summary.md   (one-page summary mapping evidence to H1/H2/H3)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Stats helpers
# ============================================================
def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x); x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(x.mean()), lo, hi


def paired_wilcoxon(deltas: np.ndarray) -> float:
    d = np.asarray(deltas); d = d[~np.isnan(d)]
    if d.size < 5 or np.all(d == 0):
        return float("nan")
    try:
        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def holm_bonferroni(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        v = (n - rank) * p[idx]
        running = max(running, v)
        adj[idx] = min(1.0, running)
    return list(adj)


# ============================================================
# Region forest plot (Component 1)
# ============================================================
def region_forest(c1_csv: str, out_pdf: str):
    df = pd.read_csv(c1_csv)
    metrics = [("R1_dice", "+"), ("R1_iou", "+"), ("R1_recall", "+"),
               ("R2_fp_rate", "-"), ("R2_meanp", "-"),
               ("R3_fp_rate", "-"), ("R3_meanp", "-"),
               ("R4_dice", "+")]
    rows = []
    for metric, _ in metrics:
        d_u = df.query("model == 'uni'")[["image_id", metric]].set_index("image_id")
        d_s = df.query("model == 'sigmma'")[["image_id", metric]].set_index("image_id")
        joined = d_u.join(d_s, lsuffix="_uni", rsuffix="_sig").dropna()
        delta = (joined[f"{metric}_sig"] - joined[f"{metric}_uni"]).values
        m, lo, hi = bootstrap_ci(delta)
        p = paired_wilcoxon(delta)
        rows.append({"metric": metric, "delta_mean": m, "ci_lo": lo, "ci_hi": hi,
                     "p_raw": p, "n": joined.shape[0]})
    pvals = [r["p_raw"] for r in rows]
    holm = holm_bonferroni(pvals)
    for r, q in zip(rows, holm):
        r["p_holm"] = q

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        ax.errorbar(r["delta_mean"], y, xerr=[[r["delta_mean"] - r["ci_lo"]],
                                              [r["ci_hi"] - r["delta_mean"]]],
                    fmt="o", color="black", capsize=3)
        sig = "*" if r["p_holm"] < 0.05 else ""
        ax.text(r["ci_hi"] + 0.005, y, f"p={r['p_holm']:.2g}{sig}",
                va="center", fontsize=8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([r["metric"] for r in rows], fontsize=8)
    ax.set_xlabel("SIGMMA - UNI (paired delta, mean ± 95% CI)")
    ax.set_title("Region-stratified metrics")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return rows


# ============================================================
# Decoder ladder (Component 4)
# ============================================================
def decoder_ladder(decoder_csv: str, out_pdf: str):
    df = pd.read_csv(decoder_csv)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, metric in zip(axes, ("dice", "aji", "pq")):
        for model in df["model"].unique():
            sub = df[df.model == model].sort_values("decoder_capacity")
            ax.plot(sub["decoder_capacity"], sub[metric], "o-", label=model)
        ax.set_xscale("log")
        ax.set_xlabel("decoder params")
        ax.set_ylabel(metric.upper())
        ax.legend(fontsize=8)
    fig.suptitle("Decoder sensitivity")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def collect_decoder_csv(out_root: str, out_csv: str):
    """Walk {out_root}/{model_dir}/{decoder}/summary.json files and assemble a CSV.

    `model_dir` is `uni` or `sigmma_scale{N}`. The CSV's `model` column carries
    the same string so per-scale comparisons (sigmma_scale0 vs uni, etc.) are
    direct and pairwise.
    """
    import glob, json
    rows = []
    for pat in ("uni", "sigmma_scale*", "sigmma_no_lora",
                "bleep_uni_ft", "clip_uni_ft", "plip_uni_ft", "omiclip_uni_ft"):
        for d in sorted(glob.glob(os.path.join(out_root, pat))):
            model_dir = os.path.basename(d.rstrip("/"))
            for sj in sorted(glob.glob(os.path.join(d, "*", "summary.json"))):
                with open(sj) as f:
                    s = json.load(f)
                # Coarse capacity bucket (decoder param count); real measurement
                # is recomputed offline if needed.
                cap = {"linear": 1e3, "mlp": 5e5, "light_unet": 2e6,
                       "unet_skip": 4e6}.get(s.get("decoder", ""), float("nan"))
                rows.append({"model": model_dir,
                             "decoder": s.get("decoder"),
                             "scale": s.get("scale", -1),
                             "decoder_capacity": cap,
                             "dice": s.get("best_val_dice"),
                             "aji": s.get("aji", float("nan")),
                             "pq":  s.get("pq", float("nan"))})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ============================================================
# One-page summary
# ============================================================
TEMPLATE = """# EVA-critique summary — MoNuSAC, SIGMMA vs pretrained UNI

**Hypotheses** under test:

- **H1 — decoder bottleneck.** EVA's linear decoder (D0) is too shallow to
  exploit SIGMMA's representation. Evidence: widening UNI vs SIGMMA gap as
  decoder capacity grows (D0 → D3) on Dice / AJI / PQ.
- **H2 — average dilution.** SIGMMA improves cell-dense regions (R1) but
  introduces lipid/OOT FPs that wash out in the global Dice. Evidence:
  positive R1 deltas + negative R2/R3 FP deltas in the region-stratified
  analysis. If both move in SIGMMA's favour, average dilution alone cannot
  account for the EVA gap.
- **H3 — Dice insensitivity.** Detection F1, AJI, PQ, Boundary F1 reflect
  per-instance correctness that pixel Dice misses. Evidence: SIGMMA wins on
  AJI / PQ / Det F1@0.7+ even when Dice ties.

## Component 1 — region-stratified metrics

{c1_table}

## Component 2 — alternative metrics

{c2_table}

## Component 3 — instance pipeline

τ\\* per model (max F1@0.5):

{c3_table}

## Component 4 — decoder sensitivity + linear probe

{c4_table}

Linear probe (CLS+mean ridge, 5-fold image-grouped):

{probe_table}

## Component 5 — attention–prediction consistency (head-mean)

{c5_table}

## Verdict

- {h1_verdict}
- {h2_verdict}
- {h3_verdict}
"""


def _summary_table(df: pd.DataFrame, metrics: List[Tuple[str, str]]) -> str:
    lines = ["| metric | UNI | SIGMMA | Δ (SIG−UNI) | 95% CI | p (Holm) |",
             "|---|---|---|---|---|---|"]
    pvals_raw = []
    rows = []
    for m, _ in metrics:
        if m not in df.columns:
            rows.append((m, "nan", "nan", "nan", "(nan,nan)", "nan"))
            pvals_raw.append(1.0)
            continue
        u = df.query("model=='uni'")[["image_id", m]].set_index("image_id")
        s = df.query("model=='sigmma'")[["image_id", m]].set_index("image_id")
        j = u.join(s, lsuffix="_u", rsuffix="_s").dropna()
        delta = (j[f"{m}_s"] - j[f"{m}_u"]).values
        d_mean, lo, hi = bootstrap_ci(delta)
        p = paired_wilcoxon(delta)
        pvals_raw.append(p if not np.isnan(p) else 1.0)
        rows.append((m, f"{j[f'{m}_u'].mean():.4f}", f"{j[f'{m}_s'].mean():.4f}",
                     f"{d_mean:+.4f}", f"({lo:+.4f}, {hi:+.4f})", f"{p:.2g}"))
    holm = holm_bonferroni(pvals_raw)
    for r, q in zip(rows, holm):
        lines.append("| " + " | ".join(list(r[:-1]) + [f"{q:.2g}"]) + " |")
    return "\n".join(lines)


def _stack_per_scale(per_scale_root: str, scales: List[int], filename: str) -> pd.DataFrame:
    """Concat the same CSV from each per-scale folder, tag with `scale`."""
    parts = []
    for s in scales:
        p = os.path.join(per_scale_root, f"scale{s}", filename)
        if not os.path.exists(p):
            print(f"[summarize] WARNING: missing {p}")
            continue
        df = pd.read_csv(p)
        df["scale"] = int(s)
        # Re-tag SIGMMA rows so model field carries scale info too — keeps
        # downstream group-bys honest if the user wants per-scale comparisons.
        if "model" in df.columns:
            df["model_scale"] = df.apply(
                lambda r: f"sigmma_s{int(s)}" if r["model"] == "sigmma" else r["model"],
                axis=1,
            )
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _stack_probe(probe_root: str, scales: List[int]) -> pd.DataFrame:
    parts = []
    for s in scales:
        p = os.path.join(probe_root, f"probe_scale{s}", "linear_probe_results.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p); df["scale"] = int(s)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="Top-level output directory; receives stacked CSVs + summary.md.")
    ap.add_argument("--per_scale_root", required=True,
                    help="Parent of scale{0,1,2}/ folders containing pipeline_eval CSVs.")
    ap.add_argument("--scales", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--decoder_root", required=True,
                    help="Parent of {model}/{decoder}/summary.json (component 4).")
    ap.add_argument("--probe_root", required=True,
                    help="Parent of probe_scale{0,1,2}/ folders.")
    args = ap.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    # ----- Stack per-scale pipeline_eval CSVs -----
    df_c1 = _stack_per_scale(args.per_scale_root, args.scales, "region_stratified_metrics.csv")
    df_c2 = _stack_per_scale(args.per_scale_root, args.scales, "alternative_metrics.csv")
    df_c3 = _stack_per_scale(args.per_scale_root, args.scales, "instance_metrics.csv")
    df_c5 = _stack_per_scale(args.per_scale_root, args.scales, "attention_consistency.csv")
    df_c1.to_csv(os.path.join(args.out_dir, "region_stratified_metrics.csv"), index=False)
    df_c2.to_csv(os.path.join(args.out_dir, "alternative_metrics.csv"), index=False)
    df_c3.to_csv(os.path.join(args.out_dir, "instance_metrics.csv"), index=False)
    df_c5.to_csv(os.path.join(args.out_dir, "attention_consistency.csv"), index=False)

    # ----- Component 1: per-scale forest plots + tables -----
    c1_tables = []
    for s in args.scales:
        sub = df_c1[df_c1["scale"] == s].copy()
        if sub.empty: continue
        tmp = os.path.join(fig_dir, f"_c1_scale{s}.csv"); sub.to_csv(tmp, index=False)
        region_forest(tmp, os.path.join(fig_dir, f"region_metric_forest_plot_scale{s}.pdf"))
        c1_tables.append(f"### scale {s}\n" + _summary_table(sub,
            [("R1_dice", "+"), ("R1_iou", "+"), ("R1_recall", "+"),
             ("R2_fp_rate", "-"), ("R3_fp_rate", "-"), ("R4_dice", "+")]))
    c1_table = "\n\n".join(c1_tables) if c1_tables else "(no rows)"

    # ----- Component 2: per-scale tables -----
    c2_tables = []
    for s in args.scales:
        sub = df_c2[df_c2["scale"] == s].copy()
        if sub.empty: continue
        c2_tables.append(f"### scale {s}\n" + _summary_table(sub,
            [("AJI", "+"), ("PQ", "+"), ("DQ", "+"), ("SQ", "+"),
             ("HD95", "-"), ("ASD", "-"),
             ("BF1_d2", "+"), ("BF1_d3", "+"), ("BF1_d5", "+"),
             ("DetF1@0.5", "+"), ("DetF1@0.7", "+"), ("DetF1@0.9", "+"),
             ("AUPRC_R1", "+"), ("AUPRC_R2", "-"),
             ("ECE_R1", "-"), ("ECE_R3", "-")]))
    c2_table = "\n\n".join(c2_tables) if c2_tables else "(no rows)"

    # ----- Component 3: pick tau* per (model, scale) -----
    tau_star: Dict[str, float] = {}
    c3_lines = ["| scale | model | τ\\* | DetF1@0.5 | DetF1@0.7 | DetF1@0.9 | mIoU(TP) | n_split | n_merge |",
                "|---|---|---|---|---|---|---|---|---|"]
    for s in args.scales:
        sub_s = df_c3[df_c3["scale"] == s]
        for model in sub_s["model"].unique():
            grp = sub_s[sub_s.model == model].groupby("tau").agg(
                f1=("DetF1@0.5", "mean")).reset_index()
            if grp.empty: continue
            tau = float(grp.loc[grp.f1.idxmax(), "tau"])
            tau_star[f"{model}_s{s}"] = tau
            sel = sub_s[(sub_s.model == model) & (sub_s.tau == tau)]
            c3_lines.append(f"| {s} | {model} | {tau} | {sel['DetF1@0.5'].mean():.4f} "
                            f"| {sel['DetF1@0.7'].mean():.4f} "
                            f"| {sel['DetF1@0.9'].mean():.4f} "
                            f"| {sel['mIoU_TPs@0.5'].mean():.4f} "
                            f"| {sel['n_split'].sum():.0f} "
                            f"| {sel['n_merge'].sum():.0f} |")
    c3_table = "\n".join(c3_lines)

    # Component 4 — decoder ladder
    decoder_csv = os.path.join(args.out_dir, "decoder_sensitivity.csv")
    df_dec = collect_decoder_csv(args.decoder_root, decoder_csv)
    if not df_dec.empty:
        decoder_ladder(decoder_csv, os.path.join(fig_dir, "decoder_ladder.pdf"))
        c4_table = df_dec.to_markdown(index=False)
    else:
        c4_table = "(no decoder summaries found at " + args.decoder_root + ")"

    # Linear probe — stack per-scale results.
    df_probe = _stack_probe(args.probe_root, args.scales)
    if not df_probe.empty:
        df_probe.to_csv(os.path.join(args.out_dir, "linear_probe_results.csv"), index=False)
        probe_table = (df_probe.groupby(["model", "scale", "split"])["auroc"]
                       .agg(["mean", "std", "count"])
                       .round(4).to_markdown())
    else:
        probe_table = "(no linear_probe_results.csv found in probe_scale*/)"

    # Component 5 — head-mean, per scale.
    c5_tables = []
    for s in args.scales:
        sub = df_c5[(df_c5["scale"] == s) & (df_c5.head == "head_mean")]
        if sub.empty: continue
        c5_tables.append(f"### scale {s}\n" + _summary_table(sub,
            [("AlignGT", "+"), ("Waste_R2", "-"), ("Waste_R3", "-"),
             ("Pearson_r", "+"), ("Spearman_rho", "+"),
             ("Delta_R1", "+"), ("Delta_R2", "-"), ("Delta_R3", "-")]))
    c5_table = "\n\n".join(c5_tables) if c5_tables else "(no rows)"

    # ---- Verdicts: emit one line per scale; users read tables for nuance.
    def _verdict_h2():
        lines = []
        for s in args.scales:
            sub = df_c1[df_c1["scale"] == s]
            u = sub.query("model=='uni'").set_index("image_id")
            sg = sub.query("model=='sigmma'").set_index("image_id")
            j = u.join(sg, lsuffix="_u", rsuffix="_s")
            if j.empty: continue
            d_r1 = (j["R1_dice_s"] - j["R1_dice_u"]).mean()
            d_r2 = (j["R2_fp_rate_s"] - j["R2_fp_rate_u"]).mean()
            d_r3 = (j["R3_fp_rate_s"] - j["R3_fp_rate_u"]).mean()
            verdict = "supported" if (d_r1 > 0 and (d_r2 < 0 or d_r3 < 0)) else "inconclusive"
            lines.append(f"  - scale {s}: ΔR1_dice={d_r1:+.3f}, ΔR2_fpr={d_r2:+.3f}, "
                         f"ΔR3_fpr={d_r3:+.3f} → **{verdict}**")
        return "**H2 — average dilution.** Per scale (paired):\n" + "\n".join(lines)

    def _verdict_h3():
        lines = []
        for s in args.scales:
            sub = df_c2[df_c2["scale"] == s]
            u = sub.query("model=='uni'").set_index("image_id")
            sg = sub.query("model=='sigmma'").set_index("image_id")
            j = u.join(sg, lsuffix="_u", rsuffix="_s").dropna(subset=["AJI_u", "AJI_s"])
            if j.empty: continue
            d_aji = (j["AJI_s"] - j["AJI_u"]).mean()
            d_pq  = (j["PQ_s"]  - j["PQ_u"]).mean()
            d_det = (j["DetF1@0.7_s"] - j["DetF1@0.7_u"]).mean()
            verdict = "supported" if (d_aji > 0 or d_pq > 0 or d_det > 0) else "not supported"
            lines.append(f"  - scale {s}: ΔAJI={d_aji:+.3f}, ΔPQ={d_pq:+.3f}, "
                         f"ΔDetF1@0.7={d_det:+.3f} → **{verdict}**")
        return "**H3 — Dice insensitivity.** Per scale (paired):\n" + "\n".join(lines)

    def _verdict_h1():
        if df_dec.empty:
            return ("**H1 — decoder bottleneck. Untested.** Decoder ladder is empty "
                    "— train the four decoders to populate it.")
        # df_dec rows tagged with model ∈ {uni, sigmma_scale0, sigmma_scale1, sigmma_scale2}
        lines = []
        for sigmma_model in sorted(m for m in df_dec["model"].unique() if m.startswith("sigmma")):
            sub = df_dec[df_dec["model"].isin(["uni", sigmma_model])]
            gap = sub.pivot(index="decoder", columns="model", values="dice").reindex(
                ["linear", "mlp", "light_unet", "unet_skip"])
            if gap.isna().all().all() or "uni" not in gap.columns or sigmma_model not in gap.columns:
                continue
            diff = gap[sigmma_model] - gap["uni"]
            d_lin = float(diff.get("linear", float("nan")))
            d_skp = float(diff.get("unet_skip", float("nan")))
            verdict = "supported" if (not np.isnan(d_lin) and not np.isnan(d_skp) and d_skp > d_lin) else "not supported"
            lines.append(f"  - {sigmma_model}: ΔDice(linear)={d_lin:+.3f}, "
                         f"ΔDice(unet_skip)={d_skp:+.3f} → **{verdict}**")
        return "**H1 — decoder bottleneck.** Per scale (Dice gap, SIGMMA−UNI):\n" + "\n".join(lines)

    summary = TEMPLATE.format(
        c1_table=c1_table, c2_table=c2_table, c3_table=c3_table,
        c4_table=c4_table, probe_table=probe_table, c5_table=c5_table,
        h1_verdict=_verdict_h1(),
        h2_verdict=_verdict_h2(),
        h3_verdict=_verdict_h3(),
    )
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(summary)
    print(f"[done] wrote {os.path.join(args.out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
