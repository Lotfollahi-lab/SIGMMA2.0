"""FOV-paired Wilcoxon for EVA-MoNuSAC: SIGMMA α=0.1 vs each baseline.

Natural unit is FOV (every model × seed combo evaluated on the same 85 test FOVs).
Earlier seed-paired (n=5) test had no FOV-level dependence and hit Wilcoxon's
W-min floor at p=0.0625 — wrong unit. This script computes FOV-paired (n=85)
after averaging out seed noise per FOV.

Usage:
  uv run python scripts/eva_critique/_compute_fov_paired_wilcoxon.py
"""
from __future__ import annotations

import csv
import glob
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")

CONDS = {
    "UNI2-h":       ROOT / "outputs/eva_critique_uni2h/seed*/instance_metrics_{v}.csv",
    "BLEEP-ft":     ROOT / "outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft/seed*/instance_metrics_{v}.csv",
    "CLIP-ft":      ROOT / "outputs/eva_critique_baselines_uni2h_step1000/clip_uni_ft/seed*/instance_metrics_{v}.csv",
    "PLIP-ft":      ROOT / "outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed*/instance_metrics_{v}.csv",
    "OmiCLIP-ft":   ROOT / "outputs/eva_critique_baselines_uni2h_step1000/omiclip_uni_ft/seed*/instance_metrics_{v}.csv",
    "SIGMMA α=0.1": ROOT / "outputs/eva_critique_gex_sweep_step1000/a01/seed*/instance_metrics_{v}.csv",
}


def per_fov_seed_avg(pat_template: Path, metric: str, view: str) -> dict[str, float]:
    """Returns dict[image_id] -> mean across all seeds for one metric."""
    pat = str(pat_template).replace("{v}", view)
    by_fov = defaultdict(list)
    for f in sorted(glob.glob(pat)):
        with open(f) as fp:
            for r in csv.DictReader(fp):
                if r[metric] not in ("", "nan"):
                    by_fov[r["image_id"]].append(float(r[metric]))
    return {fov: statistics.mean(vs) for fov, vs in by_fov.items()}


def paired_wilcoxon_fov(a: dict[str, float], b: dict[str, float]):
    common = sorted(set(a) & set(b))
    av = np.array([a[k] for k in common])
    bv = np.array([b[k] for k in common])
    diff = av - bv
    n_nz = int(np.sum(diff != 0))
    if n_nz < 3:
        return n_nz, float(diff.mean()), float("nan")
    _, p = wilcoxon(av, bv, zero_method="wilcox", alternative="two-sided")
    return n_nz, float(diff.mean()), float(p)


def sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    for view in ["global"]:
        print(f"\n=== Per-FOV paired Wilcoxon ({view.upper()}, seed-averaged, n=85 FOVs) ===")
        print("SIGMMA α=0.1 vs each baseline:")
        header = f"  {'Baseline':12s}  {'AJI Δ':>10s}  {'AJI p':>10s}  {'PQ Δ':>10s}  {'PQ p':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        sig_aji = per_fov_seed_avg(CONDS["SIGMMA α=0.1"], "AJI", view)
        sig_pq  = per_fov_seed_avg(CONDS["SIGMMA α=0.1"], "PQ",  view)
        for name in ["UNI2-h", "BLEEP-ft", "CLIP-ft", "PLIP-ft", "OmiCLIP-ft"]:
            b_aji = per_fov_seed_avg(CONDS[name], "AJI", view)
            b_pq  = per_fov_seed_avg(CONDS[name], "PQ",  view)
            _, d_a, p_a = paired_wilcoxon_fov(sig_aji, b_aji)
            _, d_p, p_p = paired_wilcoxon_fov(sig_pq,  b_pq)
            print(f"  {name:12s}  {d_a:>+10.4f}  {p_a:>7.4f} {sig_marker(p_a):>3s}  {d_p:>+10.4f}  {p_p:>7.4f} {sig_marker(p_p):>3s}")


if __name__ == "__main__":
    main()
