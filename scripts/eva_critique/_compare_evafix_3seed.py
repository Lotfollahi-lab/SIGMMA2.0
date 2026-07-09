"""3-seed trend analysis: SIGMMA α=0.1 EVA-fix (bs=64, wd=0.01) vs existing (bs=32, wd=0).

Run after all 3 seeds (42, 2, 6) complete:
  python scripts/eva_critique/_compare_evafix_3seed.py
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
SEEDS = [42, 2, 6]

EXISTING_PAT = ROOT / "outputs/eva_critique_gex_sweep_step1000/a01/seed{s}/instance_metrics_{view}.csv"
EVAFIX_PAT   = ROOT / "outputs/eva_critique_gex_sweep_evafix/a01/seed{s}/instance_metrics_{view}.csv"

BASELINES_N5 = {
    "UNI2-h":     ROOT / "outputs/eva_critique_uni2h/seed*/instance_metrics_global.csv",
    "BLEEP-ft":   ROOT / "outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft/seed*/instance_metrics_global.csv",
    "CLIP-ft":    ROOT / "outputs/eva_critique_baselines_uni2h_step1000/clip_uni_ft/seed*/instance_metrics_global.csv",
    "PLIP-ft":    ROOT / "outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed*/instance_metrics_global.csv",
    "OmiCLIP-ft": ROOT / "outputs/eva_critique_baselines_uni2h_step1000/omiclip_uni_ft/seed*/instance_metrics_global.csv",
}


def fov_mean(f: Path, metric: str) -> float:
    with open(f) as fp:
        vals = [float(r[metric]) for r in csv.DictReader(fp) if r[metric] not in ("", "nan")]
    return statistics.mean(vals)


def per_seed(pat: str, seeds, metric: str, view: str = "global") -> list[float]:
    out = []
    for s in seeds:
        f = Path(str(pat).format(s=s, view=view))
        if f.exists():
            out.append(fov_mean(f, metric))
    return out


def baseline_mean_std(pat: Path, metric: str):
    import glob
    vals = [fov_mean(Path(f), metric) for f in sorted(glob.glob(str(pat)))]
    return np.mean(vals), np.std(vals, ddof=1), len(vals)


def main():
    print("=== SIGMMA α=0.1: existing (bs32/wd0) vs EVA-fix (bs64/wd0.01), 3 seeds ===\n")
    for view in ["global"]:
        print(f"--- {view.upper()} view ---")
        for metric in ["AJI", "PQ", "DetF1_07"]:
            old = per_seed(EXISTING_PAT, SEEDS, metric, view)
            new = per_seed(EVAFIX_PAT,   SEEDS, metric, view)
            print(f"  {metric:10s} seeds={SEEDS}")
            print(f"    existing : {[f'{v:.4f}' for v in old]}  mean={np.mean(old):.4f}")
            print(f"    EVA-fix  : {[f'{v:.4f}' for v in new]}  mean={np.mean(new):.4f}")
            deltas = [n - o for n, o in zip(new, old)]
            n_pos = sum(d > 0 for d in deltas)
            print(f"    Δ        : {[f'{d:+.4f}' for d in deltas]}  mean Δ={np.mean(deltas):+.4f}  ({n_pos}/3 favor EVA-fix)")
            print()

    # GLOBAL vs N=5 baselines
    print("=== Global view: EVA-fix 3-seed mean vs baselines N=5 ===\n")
    print(f"{'Model':25s}  {'AJI':>16s}  {'PQ':>16s}")
    print("-" * 65)
    for name, pat in BASELINES_N5.items():
        am, asd, n = baseline_mean_std(pat, "AJI")
        pm, psd, _ = baseline_mean_std(pat, "PQ")
        print(f"{name:25s}  {am:.4f} ± {asd:.4f}  {pm:.4f} ± {psd:.4f}")

    # SIGMMA existing N=5
    am_exist, asd_exist, n_exist = baseline_mean_std(
        ROOT / "outputs/eva_critique_gex_sweep_step1000/a01/seed*/instance_metrics_global.csv", "AJI")
    pm_exist, psd_exist, _ = baseline_mean_std(
        ROOT / "outputs/eva_critique_gex_sweep_step1000/a01/seed*/instance_metrics_global.csv", "PQ")
    print(f"{'SIGMMA existing (N=5)':25s}  {am_exist:.4f} ± {asd_exist:.4f}  {pm_exist:.4f} ± {psd_exist:.4f}")

    # SIGMMA EVA-fix 3-seed
    new_aji = per_seed(EVAFIX_PAT, SEEDS, "AJI", "global")
    new_pq  = per_seed(EVAFIX_PAT, SEEDS, "PQ",  "global")
    if len(new_aji) == 3:
        print(f"{'SIGMMA EVA-fix (n=3)':25s}  {np.mean(new_aji):.4f} ± {np.std(new_aji,ddof=1):.4f}  {np.mean(new_pq):.4f} ± {np.std(new_pq,ddof=1):.4f}")


if __name__ == "__main__":
    main()
