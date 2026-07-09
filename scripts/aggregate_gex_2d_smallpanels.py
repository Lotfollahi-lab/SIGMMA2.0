"""Phase 4 — aggregate the small-panel GEX-2D analyses (A + B) into tables.

A (panel-specific supervised): outputs/gex_2d/smallpanels_step2000_test/<HASH>/<cond>/
   results[_linear_sweep]/<repr>.json  ->  per-panel x method x metric table +
   pooled section-level paired Wilcoxon (SIGMMA-best vs each baseline, Holm).
B (strict zero-shot): outputs/gex_2d/public_zeroshot_strict_step2000/<HASH>/<tag>.json
   vs the old leaked outputs/gex_2d/public_zeroshot_step2000/<HASH>/<tag>.json  ->
   strict numbers + Delta_leak.

Writes CSVs under outputs/gex_2d/smallpanels_aggregate/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
A_ROOT = ROOT / "outputs/gex_2d/smallpanels_step2000_test"
B_STRICT = ROOT / "outputs/gex_2d/public_zeroshot_strict_step2000"
B_LEAKED = ROOT / "outputs/gex_2d/public_zeroshot_step2000"
OUT = ROOT / "outputs/gex_2d/smallpanels_aggregate"

PANELS = {"1ca7fe57390a": "ovary", "a87c22a4bcb9": "lung", "2df22e67cd31": "brain"}
HEADLINE = "median_mse_per_slide_standardized"   # lower better
METRICS = [HEADLINE, "median_per_gene_pearson", "per_bin_r2_pooled",
           "median_morans_i_diff", "median_ssim_per_slide"]
BASELINES = ["uni2h_no_ft", "bleep_uni_step2000", "clip_uni_step2000",
             "plip_uni_step2000", "omiclip_uni_step2000"]
SIGMMA_REPRS = ["uni", "concat", "micro", "meso", "macro"]


def load_per_slide(cond_dir: Path, repr_: str) -> dict | None:
    for sub in ("results", "results_linear_sweep"):
        p = cond_dir / sub / f"{repr_}.json"
        if p.exists():
            return json.load(open(p))
    return None


def per_section(d: dict, metric: str) -> dict:
    return dict(d["per_slide"][metric]["values"])


# ---------------- Analysis A ----------------
def analysis_A():
    rows = []
    # section-level values per (panel, method-key) for the headline metric (for pooled test)
    headline_vals: dict[tuple[str, str], dict] = {}
    for h, tissue in PANELS.items():
        # methods: baselines (repr uni) + sigmma 5 reprs
        method_specs = [(b, "uni") for b in BASELINES] + \
                       [("sigmma_a01_step2000", r) for r in SIGMMA_REPRS]
        for cond, repr_ in method_specs:
            d = load_per_slide(A_ROOT / h / cond, repr_)
            if d is None:
                continue
            mkey = cond if cond != "sigmma_a01_step2000" else f"sigmma/{repr_}"
            row = {"panel": h, "tissue": tissue, "method": mkey,
                   "n_sections": len(per_section(d, HEADLINE))}
            for m in METRICS:
                vals = np.array(list(per_section(d, m).values()), float)
                row[f"{m}__mean"] = float(np.nanmean(vals))
                row[f"{m}__sd"] = float(np.nanstd(vals))
            rows.append(row)
            headline_vals[(h, mkey)] = per_section(d, HEADLINE)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "A_per_panel.csv", index=False)

    # SIGMMA-best repr per panel by headline (lower MSE_slide_std)
    sigmma_best = {}
    for h in PANELS:
        cand = {r: headline_vals.get((h, f"sigmma/{r}")) for r in SIGMMA_REPRS}
        cand = {r: v for r, v in cand.items() if v}
        if not cand:
            continue
        best = min(cand, key=lambda r: np.nanmean(list(cand[r].values())))
        sigmma_best[h] = f"sigmma/{best}"

    # Pooled section-level paired Wilcoxon: SIGMMA-best vs each baseline.
    pooled = []
    for b in BASELINES:
        sig_v, base_v = [], []
        for h in PANELS:
            sk = sigmma_best.get(h)
            if not sk:
                continue
            sdict, bdict = headline_vals.get((h, sk)), headline_vals.get((h, b))
            if not sdict or not bdict:
                continue
            for sec in sdict.keys() & bdict.keys():
                sig_v.append(sdict[sec]); base_v.append(bdict[sec])
        sig_v, base_v = np.array(sig_v), np.array(base_v)
        n = len(sig_v)
        rec = {"baseline": b, "n_sections": n,
               "sigmma_best_mean": float(np.nanmean(sig_v)) if n else np.nan,
               "baseline_mean": float(np.nanmean(base_v)) if n else np.nan,
               "median_delta(sig-base)": float(np.nanmedian(sig_v - base_v)) if n else np.nan}
        if n >= 2 and np.any(sig_v != base_v):
            try:
                rec["wilcoxon_p"] = float(stats.wilcoxon(sig_v, base_v).pvalue)
            except ValueError:
                rec["wilcoxon_p"] = np.nan
        else:
            rec["wilcoxon_p"] = np.nan
        pooled.append(rec)
    pooled_df = pd.DataFrame(pooled)
    # Holm correction over the baseline family
    pv = pooled_df["wilcoxon_p"].values
    order = np.argsort(np.where(np.isnan(pv), np.inf, pv))
    holm = np.full(len(pv), np.nan)
    m = np.sum(~np.isnan(pv))
    rank = 0
    for i in order:
        if np.isnan(pv[i]):
            continue
        holm[i] = min(1.0, (m - rank) * pv[i])
        rank += 1
    pooled_df["holm_p"] = holm
    pooled_df.to_csv(OUT / "A_pooled_stats.csv", index=False)
    print(f"[A] sigmma_best per panel: {sigmma_best}")
    print(pooled_df.to_string(index=False))
    return df, pooled_df


# ---------------- Analysis B ----------------
def analysis_B():
    rows = []
    for h in ["1ca7fe57390a", "a87c22a4bcb9", "2df22e67cd31", "11bbd69265ed", "80b4f72e3da8"]:
        strict_dir = B_STRICT / h
        if not strict_dir.is_dir():
            continue
        for jf in sorted(strict_dir.glob("*.json")):
            s = json.load(open(jf))
            tag = jf.stem
            leaked = B_LEAKED / h / f"{tag}.json"
            lk = json.load(open(leaked)) if leaked.exists() else {}
            rows.append({
                "panel": h, "method": tag,
                "n_eval_sections": len(s.get("sections", [])),
                "n_shared_strict": s.get("n_shared_genes"),
                "median_pearson_strict": s.get("median_per_gene_pearson"),
                "mse_slide_std_strict": s.get("median_mse_per_slide_standardized"),
                "n_shared_leaked": lk.get("n_shared_genes"),
                "median_pearson_leaked": lk.get("median_per_gene_pearson"),
                "delta_pearson(strict-leaked)":
                    (s.get("median_per_gene_pearson") - lk["median_per_gene_pearson"])
                    if lk.get("median_per_gene_pearson") is not None else None,
            })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "B_strict_vs_leaked.csv", index=False)
    print("\n[B] strict zero-shot vs leaked:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    analysis_A()
    analysis_B()
    print(f"\n[done] tables under {OUT}")
