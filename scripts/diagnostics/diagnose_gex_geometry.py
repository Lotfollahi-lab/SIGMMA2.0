"""Diagnostic: is the H&E–spatial InfoNCE objective learning GEX-predictive geometry
or merely position-discriminative geometry?

Inputs (npz with these arrays):
    h_off         (N, D)    LoRA-off / UNI H&E embeddings (required)
    h_on          (N, D)    LoRA-on H&E embeddings (required)
    gex           (N, G)    gene expression matrix (required)
    s_emb         (N, D_s)  spatial embeddings (optional)
    domain_label  (N,)      string/int tissue domain labels (optional)
    fov_id        (N,)      identifier (optional, propagated to outputs)

Assumptions:
- positives are same index i; all j != i are negatives.
- the script never materializes a full NxN matrix unless N is small. Pair sampling
  defaults to ``n_pairs = 200_000`` random (i, j) draws with i != j.

Outputs (under --out_dir, default outputs/diagnostics/gex_geometry/):
    summary.json                       top-line interpretation flags
    1_negative_gex.csv                 hard-negative GEX summary
    2_geometry_corr.csv                Spearman/Pearson H&E↔GEX
    3_same_domain.csv                  if domain_label provided
    4_hard_negatives.csv               if s_emb provided
    figs/neg_gex_hist.png              histogram of off-diagonal GEX similarity
    figs/h_off_vs_gex_hexbin.png       hexbin H&E-off vs GEX (sampled)
    figs/h_on_vs_gex_hexbin.png        hexbin H&E-on vs GEX (sampled)
    figs/same_domain_bar.png           same/diff domain similarity bars

Interpretation rules (printed + written to summary.json):
    (a) alignment improves but Spearman unchanged -> position matching only
    (b) many high-GEX negatives -> biological positives treated as negatives
    (c) same-domain similarity decreases after LoRA -> instance disc may hurt GEX pred
    (d) Spearman increases & same-domain similarity preserved -> contrastive helps
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_pairs (i, j) with i != j (off-diagonal). Approximate sampling with
    rejection — acceptable since collision prob is 1/n."""
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    eq = i == j
    while eq.any():
        j[eq] = rng.integers(0, n, size=int(eq.sum()))
        eq = i == j
    return i, j


def cos_pairs(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return (x[i] * x[j]).sum(axis=-1)


def measure_negative_gex(gex_n: np.ndarray, rng, n_pairs: int) -> dict:
    """(1) Off-diagonal GEX similarity distribution + hard-negative tail."""
    N = gex_n.shape[0]
    i, j = sample_pairs(N, n_pairs, rng)
    sim = cos_pairs(gex_n, i, j)
    finite = sim[np.isfinite(sim)]
    summary = {
        "n_pairs":  int(finite.size),
        "mean":     float(finite.mean()),
        "median":   float(np.median(finite)),
        "std":      float(finite.std()),
        "p90":      float(np.percentile(finite, 90)),
        "p95":      float(np.percentile(finite, 95)),
        "p99":      float(np.percentile(finite, 99)),
        "frac_gt_0_5": float((finite > 0.5).mean()),
        "frac_gt_0_7": float((finite > 0.7).mean()),
        "frac_gt_0_8": float((finite > 0.8).mean()),
    }
    # Hard-negative tails (top-1/5/10% by GEX similarity)
    for pct in (1, 5, 10):
        thr = np.percentile(finite, 100 - pct)
        tail = finite[finite >= thr]
        summary[f"top{pct}pct_mean"] = float(tail.mean())
        summary[f"top{pct}pct_median"] = float(np.median(tail))
        summary[f"top{pct}pct_threshold"] = float(thr)
    return summary, sim


def measure_geometry_corr(h_off_n, h_on_n, gex_n, rng, n_pairs: int) -> dict:
    """(2) Spearman/Pearson H&E↔GEX similarity correlation; LoRA-induced shift."""
    N = h_off_n.shape[0]
    i, j = sample_pairs(N, n_pairs, rng)
    s_off = cos_pairs(h_off_n, i, j)
    s_on  = cos_pairs(h_on_n, i, j)
    s_gex = cos_pairs(gex_n, i, j)
    valid = np.isfinite(s_off) & np.isfinite(s_on) & np.isfinite(s_gex)
    s_off, s_on, s_gex = s_off[valid], s_on[valid], s_gex[valid]

    pearson_off, _ = pearsonr(s_off, s_gex)
    pearson_on,  _ = pearsonr(s_on,  s_gex)
    # Subsample for Spearman if very large (it is O(n log n) per call)
    sub = min(50_000, len(s_gex))
    idx = rng.choice(len(s_gex), size=sub, replace=False)
    spearman_off, _ = spearmanr(s_off[idx], s_gex[idx])
    spearman_on,  _ = spearmanr(s_on[idx],  s_gex[idx])

    # delta similarity = LoRA-induced movement; does LoRA push GEX-similar pairs
    # apart more than dissimilar ones? (correlation between delta and gex_sim)
    delta = s_on - s_off
    pearson_delta, _ = pearsonr(delta, s_gex)
    spearman_delta, _ = spearmanr(delta[idx], s_gex[idx])

    return {
        "n_pairs": int(len(s_gex)),
        "pearson_off": float(pearson_off),
        "pearson_on":  float(pearson_on),
        "delta_pearson":  float(pearson_on - pearson_off),
        "spearman_off": float(spearman_off),
        "spearman_on":  float(spearman_on),
        "delta_spearman":  float(spearman_on - spearman_off),
        "pearson_delta_gex": float(pearson_delta),
        "spearman_delta_gex": float(spearman_delta),
        # raw arrays for plotting (sub-sample to keep memory bounded)
        "_s_off": s_off, "_s_on": s_on, "_s_gex": s_gex,
    }


def measure_same_domain(h_off_n, h_on_n, domain_label, rng, n_pairs: int) -> Optional[pd.DataFrame]:
    """(3) Same-domain vs different-domain negative similarity, LoRA on vs off."""
    if domain_label is None:
        return None
    N = len(domain_label)
    i, j = sample_pairs(N, n_pairs, rng)
    s_off = cos_pairs(h_off_n, i, j)
    s_on  = cos_pairs(h_on_n, i, j)

    di = np.asarray(domain_label)[i]
    dj = np.asarray(domain_label)[j]
    same = di == dj

    rows = []
    rows.append({
        "stratum": "all_negatives",
        "n":             int(len(same)),
        "h_off_mean":    float(s_off.mean()),
        "h_off_median":  float(np.median(s_off)),
        "h_on_mean":     float(s_on.mean()),
        "h_on_median":   float(np.median(s_on)),
        "delta_mean":    float((s_on - s_off).mean()),
    })
    for label, mask, name in [
        (None, same,  "same_domain"),
        (None, ~same, "different_domain"),
    ]:
        if not mask.any(): continue
        rows.append({
            "stratum": name,
            "n":            int(mask.sum()),
            "h_off_mean":   float(s_off[mask].mean()),
            "h_off_median": float(np.median(s_off[mask])),
            "h_on_mean":    float(s_on[mask].mean()),
            "h_on_median":  float(np.median(s_on[mask])),
            "delta_mean":   float((s_on[mask] - s_off[mask]).mean()),
        })

    # Per-domain-pair (tumor-tumor, stroma-stroma, ...)
    uniq = np.unique(domain_label)
    for d in uniq:
        mask = (di == d) & (dj == d)
        if mask.sum() < 50: continue
        rows.append({
            "stratum": f"{d}-{d}",
            "n":            int(mask.sum()),
            "h_off_mean":   float(s_off[mask].mean()),
            "h_off_median": float(np.median(s_off[mask])),
            "h_on_mean":    float(s_on[mask].mean()),
            "h_on_median":  float(np.median(s_on[mask])),
            "delta_mean":   float((s_on[mask] - s_off[mask]).mean()),
        })
    return pd.DataFrame(rows)


def measure_hard_infonce_negatives(h_on_n, s_emb_n, gex_n,
                                    domain_label, ks=(1, 5, 10)) -> Optional[pd.DataFrame]:
    """(4) For each anchor i, find top-k j != i by H&E→spatial logit; report their
    GEX similarity stats + (optional) same-domain ratio."""
    if s_emb_n is None:
        return None
    N = h_on_n.shape[0]
    # Logit matrix is N×N — only feasible up to a few thousand.
    if N > 5000:
        # subsample anchors
        anchors = np.random.default_rng(0).choice(N, size=min(2000, N), replace=False)
    else:
        anchors = np.arange(N)
    sim_mat = h_on_n[anchors] @ s_emb_n.T               # (n_anchor, N)
    # Mask out self-pairs (for anchors j != i)
    for ai, a in enumerate(anchors):
        sim_mat[ai, a] = -np.inf

    rows = []
    for k in ks:
        # top-k indices per anchor (k_topk = k)
        topk_idx = np.argpartition(-sim_mat, k, axis=1)[:, :k]   # (n_anchor, k)
        gex_a = gex_n[anchors]                                    # (n_anchor, G)
        # pairwise GEX cosine
        gex_pair_sim = np.zeros((len(anchors), k))
        for ai in range(len(anchors)):
            ga = gex_a[ai]
            gj = gex_n[topk_idx[ai]]
            gex_pair_sim[ai] = (ga[None, :] * gj).sum(axis=1)
        flat = gex_pair_sim.flatten()
        row = {
            "k": int(k),
            "n_anchors": int(len(anchors)),
            "gex_sim_mean":     float(flat.mean()),
            "gex_sim_median":   float(np.median(flat)),
            "gex_sim_p90":      float(np.percentile(flat, 90)),
            "gex_sim_frac_gt_0_7": float((flat > 0.7).mean()),
            "gex_sim_frac_gt_0_8": float((flat > 0.8).mean()),
        }
        if domain_label is not None:
            anc_dom = np.asarray(domain_label)[anchors]
            neg_dom = np.asarray(domain_label)[topk_idx]   # (n_anchor, k)
            same = (anc_dom[:, None] == neg_dom).flatten()
            row["same_domain_ratio"] = float(same.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _hexbin(x, y, title, path, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    hb = ax.hexbin(x, y, gridsize=40, cmap="viridis", bins="log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.colorbar(hb, ax=ax, label="log10(N)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True, help="npz with h_off, h_on, gex (+optional s_emb, domain_label).")
    ap.add_argument("--out_dir", default="outputs/diagnostics/gex_geometry")
    ap.add_argument("--n_pairs", type=int, default=200_000,
                    help="random off-diagonal (i, j) pairs to sample.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    d = np.load(args.input_npz, allow_pickle=True)
    h_off = d["h_off"].astype(np.float32)
    h_on  = d["h_on"].astype(np.float32)
    gex   = d["gex"].astype(np.float32)
    s_emb = d["s_emb"].astype(np.float32) if "s_emb" in d.files else None
    domain_label = d["domain_label"].tolist() if "domain_label" in d.files else None
    N = h_off.shape[0]
    print(f"[load] N={N}  D_h={h_off.shape[1]}  G={gex.shape[1]}  "
          f"s_emb={'yes' if s_emb is not None else 'no'}  "
          f"domain={'yes' if domain_label is not None else 'no'}", flush=True)

    # L2-normalize embeddings (they may already be normalized, but be safe);
    # GEX is normalized too so cosine = directional similarity (not abs amount).
    h_off_n = _l2(h_off)
    h_on_n  = _l2(h_on)
    gex_n   = _l2(gex + 1e-8)
    s_emb_n = _l2(s_emb) if s_emb is not None else None

    # ---------- (1) Negative GEX similarity ----------
    print("[1] negative-GEX similarity ...", flush=True)
    s1, gex_sim_sample = measure_negative_gex(gex_n, rng, args.n_pairs)
    pd.DataFrame([s1]).to_csv(out_dir / "1_negative_gex.csv", index=False)
    print(f"    p99={s1['p99']:.3f}  frac_gt_0.7={s1['frac_gt_0_7']:.3f}  "
          f"frac_gt_0.8={s1['frac_gt_0_8']:.3f}", flush=True)

    # Histogram
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(gex_sim_sample, bins=80, color="#1f77b4", edgecolor="white")
    for q in (0.5, 0.7, 0.8):
        ax.axvline(q, ls="--", color="grey", lw=0.6)
    ax.set_xlabel("off-diagonal GEX cosine similarity")
    ax.set_ylabel("count")
    ax.set_title(f"N={N} FOVs · {len(gex_sim_sample)} sampled pairs")
    fig.tight_layout(); fig.savefig(out_dir / "figs/neg_gex_hist.png", dpi=120); plt.close(fig)

    # ---------- (2) Geometry correlation ----------
    print("[2] H&E↔GEX geometry correlation ...", flush=True)
    s2 = measure_geometry_corr(h_off_n, h_on_n, gex_n, rng, args.n_pairs)
    save2 = {k: v for k, v in s2.items() if not k.startswith("_")}
    pd.DataFrame([save2]).to_csv(out_dir / "2_geometry_corr.csv", index=False)
    print(f"    pearson_off={s2['pearson_off']:.4f}  pearson_on={s2['pearson_on']:.4f}  "
          f"delta={s2['delta_pearson']:+.4f}", flush=True)
    print(f"    spearman_off={s2['spearman_off']:.4f}  spearman_on={s2['spearman_on']:.4f}  "
          f"delta={s2['delta_spearman']:+.4f}", flush=True)
    print(f"    corr(LoRA-shift, GEX-sim): pearson={s2['pearson_delta_gex']:+.4f}  "
          f"spearman={s2['spearman_delta_gex']:+.4f}", flush=True)

    # Hexbin scatters (sub-sample to 50k for plotting speed)
    keep = rng.choice(len(s2["_s_gex"]), size=min(50_000, len(s2["_s_gex"])), replace=False)
    _hexbin(s2["_s_gex"][keep], s2["_s_off"][keep],
            f"LoRA-off · ρ_S={s2['spearman_off']:.3f}", out_dir / "figs/h_off_vs_gex_hexbin.png",
            "GEX cosine sim", "H&E_off cosine sim")
    _hexbin(s2["_s_gex"][keep], s2["_s_on"][keep],
            f"LoRA-on · ρ_S={s2['spearman_on']:.3f}", out_dir / "figs/h_on_vs_gex_hexbin.png",
            "GEX cosine sim", "H&E_on cosine sim")

    # ---------- (3) Same-domain negatives ----------
    if domain_label is not None:
        print("[3] same-domain negative analysis ...", flush=True)
        df3 = measure_same_domain(h_off_n, h_on_n, domain_label, rng, args.n_pairs)
        df3.to_csv(out_dir / "3_same_domain.csv", index=False)
        print(df3.to_string(index=False), flush=True)
        # Bar plot for top strata
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        top = df3.head(8)
        x = np.arange(len(top))
        w = 0.4
        ax.bar(x - w/2, top["h_off_mean"], width=w, label="LoRA-off", color="#888888")
        ax.bar(x + w/2, top["h_on_mean"],  width=w, label="LoRA-on",  color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(top["stratum"], rotation=30, ha="right")
        ax.set_ylabel("mean H&E cosine"); ax.legend()
        ax.set_title("Same-domain vs different-domain negative similarity")
        fig.tight_layout(); fig.savefig(out_dir / "figs/same_domain_bar.png", dpi=120); plt.close(fig)
    else:
        df3 = None
        print("[3] skipped (no domain_label)", flush=True)

    # ---------- (4) Hard InfoNCE negatives ----------
    if s_emb_n is not None:
        print("[4] hard InfoNCE negative analysis ...", flush=True)
        df4 = measure_hard_infonce_negatives(h_on_n, s_emb_n, gex_n, domain_label)
        df4.to_csv(out_dir / "4_hard_negatives.csv", index=False)
        print(df4.to_string(index=False), flush=True)
    else:
        df4 = None
        print("[4] skipped (no s_emb)", flush=True)

    # ---------- Interpretation rules ----------
    flags = {
        "noise_floor_used_for_unchanged": 0.005,  # tunable
        "(a) alignment_up_spearman_unchanged":
            bool(abs(s2["delta_spearman"]) < 0.005 and s2["delta_pearson"] > 0.01),
        "(b) many_high_gex_negatives":
            bool(s1["frac_gt_0_7"] > 0.05),
        "(c) same_domain_sim_drops":
            bool(df3 is not None and
                 (df3.loc[df3.stratum == "same_domain", "h_on_mean"].values
                  - df3.loc[df3.stratum == "same_domain", "h_off_mean"].values).item() < -0.01),
        "(d) spearman_up_and_same_domain_preserved":
            bool(s2["delta_spearman"] > 0.005 and
                 (df3 is None or
                  abs((df3.loc[df3.stratum == "same_domain", "h_on_mean"].values
                       - df3.loc[df3.stratum == "same_domain", "h_off_mean"].values).item()) < 0.01)),
    }
    summary = {
        "N": int(N),
        "n_pairs": int(args.n_pairs),
        "1_negative_gex": s1,
        "2_geometry_corr": save2,
        "3_same_domain": None if df3 is None else df3.to_dict(orient="records"),
        "4_hard_negatives": None if df4 is None else df4.to_dict(orient="records"),
        "flags": flags,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[flags]")
    for k, v in flags.items():
        print(f"  {k}: {v}")
    print(f"\n[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
