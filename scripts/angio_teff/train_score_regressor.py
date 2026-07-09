#!/usr/bin/env python
"""Linear-probe regression of angio/Teff signature scores from per-cell image
embeddings, comparing SIGMMA vs frozen UNI2-h (the SIGMMA-specific-value test).

Step 2 of clinical-outcome path 2 (memory project_angio_teff_pipeline_design_
2026_06_28). The signature scores (regression targets) were computed from real
Xenium by compute_signature_scores.py and live as per-cell parquet keyed by
cell_id. Here we join them to the per-cell SIGMMA / UNI feature HDF5s (same cells,
joined by cell_id) and fit a Ridge linear probe per (representation, target),
then report held-out correlation.

Representations compared (all built by concatenating per-scale keys):
  sigmma_concat : z_sigmma_{micro,meso,macro}                 (512*3 = 1536d, SIGMMA output)
  sigmma_noproj : z_sigmma_{micro,meso,macro}_dense_noproj    (1536*3 = 4608d, LoRA backbone)
  uni_concat    : z_uni_{micro,meso,macro}                    (1536*3 = 4608d, frozen; dim-matched to noproj)
  uni_macro     : z_uni_macro                                 (1536d, frozen single-scale)

Headline = sigmma_concat vs uni_*, and the dim-matched sigmma_noproj vs uni_concat
(isolates the ST-aware LoRA adaptation).

Metrics are SECTION-LEVEL held-out: per-test-section Pearson r (mean+/-std over
test sections) is the honest unit (cells within a section are correlated, so a
pooled r is anti-conservative). Pooled r reported alongside for reference.

Run in the sigmma poetry venv.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

# representation -> (which feature dir, list of per-scale dataset keys)
REPRS = {
    "sigmma_concat": ("sigmma", ["z_sigmma_micro", "z_sigmma_meso", "z_sigmma_macro"]),
    "sigmma_noproj": ("sigmma", ["z_sigmma_micro_dense_noproj",
                                 "z_sigmma_meso_dense_noproj",
                                 "z_sigmma_macro_dense_noproj"]),
    "uni_concat": ("uni", ["z_uni_micro", "z_uni_meso", "z_uni_macro"]),
    "uni_macro": ("uni", ["z_uni_macro"]),
}
TARGETS = ["angio_mean", "teff_mean"]


def _decode(arr):
    return np.array([c.decode() if isinstance(c, bytes) else str(c) for c in arr])


def _load_split(samples, dirs, keys, score_dir, max_cells, seed):
    """Return X (n,d), {target: y}, section_id per cell — joined by cell_id."""
    feat_dir = Path(dirs)
    Xs, ys = [], {t: [] for t in TARGETS}
    secs = []
    rng = np.random.default_rng(seed)
    for s in samples:
        fh = feat_dir / f"{s}.h5"
        pq = score_dir / "per_cell" / f"{s}.parquet"
        if not fh.exists() or not pq.exists():
            print(f"    [skip] {s}: missing feat/score")
            continue
        with h5py.File(fh, "r") as f:
            cid = _decode(f["cell_id"][:])
            X = np.concatenate([f[k][:] for k in keys], axis=1).astype(np.float32)
        sdf = pd.read_parquet(pq, columns=["cell_id"] + TARGETS)
        sdf["cell_id"] = sdf["cell_id"].astype(str)
        smap = sdf.set_index("cell_id")
        # keep feature cells that have a score and a finite target
        idx = pd.Index(cid)
        present = idx.isin(smap.index)
        if present.sum() == 0:
            print(f"    [warn] {s}: 0 cell_id join (id format mismatch?)")
            continue
        X = X[present]
        sub = smap.loc[idx[present]]
        yvals = {t: sub[t].to_numpy(np.float32) for t in TARGETS}
        finite = np.all([np.isfinite(yvals[t]) for t in TARGETS], axis=0)
        X = X[finite]
        for t in TARGETS:
            yvals[t] = yvals[t][finite]
        n = X.shape[0]
        if max_cells and n > max_cells:
            sel = rng.choice(n, max_cells, replace=False)
            X = X[sel]
            for t in TARGETS:
                yvals[t] = yvals[t][sel]
            n = max_cells
        Xs.append(X)
        for t in TARGETS:
            ys[t].append(yvals[t])
        secs.append(np.full(n, s, dtype=object))
        print(f"    [ok] {s}: n={n} (joined {present.sum()}/{len(cid)})")
    if not Xs:
        raise SystemExit("no cells loaded for split")
    return (np.concatenate(Xs),
            {t: np.concatenate(ys[t]) for t in TARGETS},
            np.concatenate(secs))


def _per_section_pearson(pred, y, secs):
    rs = []
    for s in pd.unique(secs):
        m = secs == s
        if m.sum() >= 10 and np.std(y[m]) > 0 and np.std(pred[m]) > 0:
            rs.append((s, float(pearsonr(pred[m], y[m])[0]), int(m.sum())))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmma_dir", required=True)
    ap.add_argument("--uni_dir", required=True)
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_cells", type=int, default=15000,
                    help="per-section cell cap (memory); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    score_dir = Path(args.score_dir)
    dir_map = {"sigmma": args.sigmma_dir, "uni": args.uni_dir}
    splits = {sp: man[man["split"] == sp]["sample_name"].tolist()
              for sp in ("train", "val", "test")}
    print("[splits] " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))

    alphas = np.logspace(-1, 4, 12)
    results = []
    for rep, (src, keys) in REPRS.items():
        fdir = dir_map[src]
        print(f"\n===== representation: {rep} ({src}, {len(keys)} keys) =====")
        print("  [load] train")
        Xtr, ytr, _ = _load_split(splits["train"], fdir, keys, score_dir,
                                   args.max_cells, args.seed)
        print("  [load] test")
        Xte, yte, secte = _load_split(splits["test"], fdir, keys, score_dir,
                                      args.max_cells, args.seed)
        print("  [load] val")
        Xva, yva, secva = _load_split(splits["val"], fdir, keys, score_dir,
                                      args.max_cells, args.seed)
        # Manual float32 standardisation (sklearn StandardScaler.transform upcasts
        # the whole matrix to float64 -> a transient copy that blew the ~8GB
        # oversubscribed cap). mean/std in float64 for stability, cast to float32;
        # (X - mu) / sd stays float32. Free raw X immediately.
        mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = Xtr.std(axis=0, dtype=np.float64).astype(np.float32)
        sd[sd < 1e-6] = 1e-6
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd
        Zva = (Xva - mu) / sd
        d_feat = int(Xtr.shape[1])
        del Xtr, Xte, Xva
        for t in TARGETS:
            model = RidgeCV(alphas=alphas).fit(Ztr, ytr[t])
            for split_name, Z, yy, sec in (("test", Zte, yte[t], secte),
                                           ("val", Zva, yva[t], secva)):
                pred = model.predict(Z)
                pooled_r = float(pearsonr(pred, yy)[0]) if np.std(pred) > 0 else float("nan")
                pooled_rho = float(spearmanr(pred, yy)[0])
                ps = _per_section_pearson(pred, yy, sec)
                sec_rs = [r for _, r, _ in ps]
                rec = {
                    "rep": rep, "dim": d_feat, "target": t,
                    "split": split_name, "alpha": float(model.alpha_),
                    "pooled_pearson": pooled_r, "pooled_spearman": pooled_rho,
                    "n_sections": len(sec_rs),
                    "section_pearson_mean": float(np.mean(sec_rs)) if sec_rs else float("nan"),
                    "section_pearson_std": float(np.std(sec_rs)) if sec_rs else float("nan"),
                    "section_pearson": {s: round(r, 4) for s, r, _ in ps},
                }
                results.append(rec)
                if split_name == "test":
                    print(f"  [{t:10s} {split_name}] dim={rec['dim']} "
                          f"pooled_r={pooled_r:+.3f} rho={pooled_rho:+.3f} | "
                          f"sec_r={rec['section_pearson_mean']:+.3f}"
                          f"+/-{rec['section_pearson_std']:.3f} (n={len(sec_rs)})")
        del Ztr, Zte, Zva

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "section_pearson"}
                           for r in results])
    res_df.to_csv(out / "score_regression_results.csv", index=False)
    with open(out / "score_regression_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n===== TEST summary (section-level Pearson mean+/-std) =====")
    piv = res_df[res_df["split"] == "test"].pivot_table(
        index="rep", columns="target",
        values=["section_pearson_mean", "pooled_pearson"])
    print(piv.to_string())
    print(f"\nwrote {out / 'score_regression_results.csv'}")


if __name__ == "__main__":
    main()
