#!/usr/bin/env python
"""Stage-separability: does ST-enriched H&E (z_sigmma) separate T-stage better
than H&E alone (z_uni)?

Hypothesis: pT1 vs pT3+ tumours look alike to a frozen pathology backbone
(UNI2-h, "H&E alone") but separate once the backbone is ST-aware (SIGMMA
LoRA, "ST-enriched H&E"). Both embeddings are computed on the SAME H&E pixels
(no ST at inference) and the SAME cells, so the only moving part is the LoRA.

Per slide we mean-pool a per-cell embedding (default the macro CLS token,
1536-d for both z_uni and z_sigmma -> matched dim, fairest comparison) to one
slide vector, then:

  * unsupervised  -- UMAP + silhouette(stage) + leave-one-out kNN purity.
                     The literal reading of "look alike vs separate".
  * supervised    -- patient-grouped (StratifiedGroupKFold) repeated logistic
                     probe; per-fold test AUROC, paired (sigmma - uni) delta,
                     Wilcoxon signed-rank + bootstrap CI on the paired delta.

Cohorts:
  --cohort tcga    join slide -> case (first 3 barcode tokens) -> clinical
                   ``ajcc_t``; groups = case_submitter_id.
  --cohort inhouse join slide -> donor (CVx prefix) -> metadata ``T.stage``;
                   groups = donor. NOTE: in-house has ~1 donor per stage, so
                   the supervised probe is NOT powered -- emitted for the
                   qualitative UMAP panel only (see report caveat).

Both feature stages must already be cached (one h5 per slide under --uni_dir
and --sigmma_dir, cell_id-aligned).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Feature loading: per-slide mean-pool of a per-cell embedding.
# ----------------------------------------------------------------------
def _slide_vector(h5_path: Path, key: str) -> np.ndarray | None:
    """Mean-pool ``key`` over all cells in one slide -> (D,) float64, or None
    if the key/file is missing or empty."""
    try:
        with h5py.File(h5_path, "r") as f:
            if key not in f:
                return None
            arr = f[key][:]
    except (OSError, KeyError):
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    return arr.mean(axis=0).astype(np.float64)


def build_table(uni_dir: Path, sigmma_dir: Path, key: str) -> pd.DataFrame:
    """One row per slide present in BOTH caches. Columns: sample_name,
    ``uni`` (object: ndarray), ``sigmma`` (object: ndarray)."""
    uni_files = {p.stem: p for p in uni_dir.glob("*.h5")}
    sig_files = {p.stem: p for p in sigmma_dir.glob("*.h5")}
    shared = sorted(set(uni_files) & set(sig_files))
    rows = []
    n_uni_only, n_sig_only = len(set(uni_files) - set(sig_files)), len(set(sig_files) - set(uni_files))
    for s in shared:
        vu = _slide_vector(uni_files[s], f"z_uni_{key}")
        vs = _slide_vector(sig_files[s], f"z_sigmma_{key}")
        if vu is None or vs is None:
            continue
        if vu.shape != vs.shape:
            print(f"[warn] {s}: dim mismatch uni{vu.shape} sigmma{vs.shape}; skip")
            continue
        rows.append({"sample_name": s, "uni": vu, "sigmma": vs})
    print(f"[table] {len(rows)} slides with both embeddings "
          f"(uni-only {n_uni_only}, sigmma-only {n_sig_only}, key=z_*_{key})")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Stage labels.
# ----------------------------------------------------------------------
def _t_to_binary(t: str) -> str | None:
    """pT1{a,b,c} -> 'pT1' (organ-confined small); pT3{a,b,c}/T4 -> 'pT3+'
    (locally advanced). pT2 and blanks -> None (excluded from the contrast)."""
    if not isinstance(t, str) or not t.strip():
        return None
    t = t.strip().upper().replace("PT", "T")
    if t.startswith("T1"):
        return "pT1"
    if t.startswith("T3") or t.startswith("T4"):
        return "pT3+"
    return None  # T2*, TX, etc.


def label_tcga(df: pd.DataFrame, clinical_tsv: Path) -> pd.DataFrame:
    clin = pd.read_csv(clinical_tsv, sep="\t", dtype=str)
    t_by_case = dict(zip(clin["case_submitter_id"], clin["ajcc_t"]))
    df = df.copy()
    df["group"] = df["sample_name"].map(lambda s: "-".join(s.split("-")[:3]))
    df["t_raw"] = df["group"].map(t_by_case)
    df["stage"] = df["t_raw"].map(_t_to_binary)
    return df


def label_inhouse(df: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, dtype=str)
    # metadata 'Patient' col holds CV1..CV9; T.stage holds 1a/1b/3a...
    pat_col = "Patient" if "Patient" in meta.columns else meta.columns[0]
    t_col = "T.stage" if "T.stage" in meta.columns else "T stage"
    t_by_donor = {str(p).strip().upper(): f"T{str(t).strip()}"
                  for p, t in zip(meta[pat_col], meta[t_col]) if pd.notna(t)}
    df = df.copy()
    df["group"] = df["sample_name"].map(lambda s: s.split("-")[0].upper())
    df["t_raw"] = df["group"].map(t_by_donor)
    df["stage"] = df["t_raw"].map(_t_to_binary)
    return df


# ----------------------------------------------------------------------
# Unsupervised separability.
# ----------------------------------------------------------------------
def unsupervised(X: np.ndarray, y: np.ndarray, seed: int):
    """Returns dict with silhouette(stage), loo-kNN purity, and 2-D UMAP coords."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    Xs = StandardScaler().fit_transform(X)
    sil = float(silhouette_score(Xs, y)) if len(set(y)) > 1 else float("nan")

    k = min(10, len(Xs) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    neigh = idx[:, 1:]  # drop self
    purity = float(np.mean([(y[neigh[i]] == y[i]).mean() for i in range(len(y))]))

    try:
        import umap
        coords = umap.UMAP(n_neighbors=min(15, len(Xs) - 1), min_dist=0.1,
                           random_state=seed).fit_transform(Xs)
    except Exception as e:  # noqa: BLE001
        from sklearn.decomposition import PCA
        print(f"[umap] unavailable ({e}); falling back to PCA-2D")
        coords = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    return {"silhouette": sil, "knn_purity": purity, "coords": coords}


# ----------------------------------------------------------------------
# Supervised probe: patient-grouped, repeated, paired by fold.
# ----------------------------------------------------------------------
def supervised(X_uni, X_sig, y, groups, n_splits, n_repeats, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    yb = (y == "pT3+").astype(int)  # positive class = locally advanced
    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        print(f"[probe] only {n_groups} groups -> supervised probe skipped")
        return None

    def _auc_folds(X):
        aucs = []
        for r in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed + r)
            for tr, te in cv.split(X, yb, groups):
                if len(set(yb[te])) < 2 or len(set(yb[tr])) < 2:
                    continue
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
                )
                clf.fit(X[tr], yb[tr])
                p = clf.predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(yb[te], p))
        return np.array(aucs)

    auc_uni = _auc_folds(X_uni)
    auc_sig = _auc_folds(X_sig)
    n = min(len(auc_uni), len(auc_sig))
    if n == 0:
        print("[probe] no evaluable folds (a class has <2 groups) -> "
              "supervised probe skipped (cohort not powered for grouped CV)")
        return None
    auc_uni, auc_sig = auc_uni[:n], auc_sig[:n]
    delta = auc_sig - auc_uni

    from scipy.stats import wilcoxon
    try:
        w_p = float(wilcoxon(auc_sig, auc_uni).pvalue)
    except ValueError:
        w_p = float("nan")
    rng = np.random.default_rng(seed)
    boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    return {
        "n_folds": int(n), "n_splits": int(n_splits), "n_repeats": int(n_repeats),
        "auc_uni_mean": float(auc_uni.mean()), "auc_uni_std": float(auc_uni.std()),
        "auc_sigmma_mean": float(auc_sig.mean()), "auc_sigmma_std": float(auc_sig.std()),
        "delta_mean": float(delta.mean()), "delta_ci95": ci,
        "wilcoxon_p": w_p,
        "_auc_uni": auc_uni, "_auc_sigmma": auc_sig,
    }


# ----------------------------------------------------------------------
# Figure.
# ----------------------------------------------------------------------
def make_figure(df, uns_uni, uns_sig, sup, fig_path, cohort):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    UNI_C, SIG_C = "#888888", "#d62728"
    stage_order = ["pT1", "pT3+"]
    stage_marker = {"pT1": "o", "pT3+": "^"}
    stage_col = {"pT1": "#4477AA", "pT3+": "#CC3311"}
    y = df["stage"].to_numpy()

    has_probe = sup is not None
    ncol = 3 if has_probe else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0))

    for ax, uns, title, ec in [
        (axes[0], uns_uni, "H&E alone (UNI2-h)", UNI_C),
        (axes[1], uns_sig, "ST-enriched H&E (SIGMMA)", SIG_C),
    ]:
        c = uns["coords"]
        for st in stage_order:
            m = y == st
            ax.scatter(c[m, 0], c[m, 1], s=18, marker=stage_marker[st],
                       c=stage_col[st], edgecolors="none", alpha=0.8, label=st)
        ax.set_title(f"{title}\nsilhouette={uns['silhouette']:.3f}  "
                     f"kNN-purity={uns['knn_purity']:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(ec); sp.set_linewidth(1.5)
        ax.legend(fontsize=7, frameon=False, loc="best")

    if has_probe:
        ax = axes[2]
        data = [sup["_auc_uni"], sup["_auc_sigmma"]]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                        showfliers=False)
        for patch, col in zip(bp["boxes"], [UNI_C, SIG_C]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        ax.scatter(np.zeros(len(data[0])), data[0], s=10, c=UNI_C, alpha=0.5)
        ax.scatter(np.ones(len(data[1])), data[1], s=10, c=SIG_C, alpha=0.5)
        ax.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["UNI2-h", "SIGMMA"], fontsize=8)
        ax.set_ylabel("pT1 vs pT3+  AUROC\n(patient-grouped CV)", fontsize=8)
        ax.set_title(f"$\\Delta$={sup['delta_mean']:+.3f} "
                     f"[{sup['delta_ci95'][0]:+.3f},{sup['delta_ci95'][1]:+.3f}]\n"
                     f"Wilcoxon p={sup['wilcoxon_p']:.1e}", fontsize=8)

    n1 = int((y == "pT1").sum()); n3 = int((y == "pT3+").sum())
    fig.suptitle(f"H&E -> T-stage separability ({cohort}): "
                 f"pT1 (n={n1}) vs pT3+ (n={n3})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[fig] wrote {fig_path}")


# ----------------------------------------------------------------------
# Interpreted verdict (self-documenting; no-overclaim guardrails baked in).
# ----------------------------------------------------------------------
def write_report(summary, fig_path, report_path):
    """Emit a human-readable RESULTS markdown with a guarded verdict so the run
    self-documents even if no LLM session is alive to interpret it.

    Guardrails (no-overclaim): a 'SIGMMA separates better' claim requires the
    paired delta CI to exclude 0 AND mean delta >= 0.03 AUROC. We also report
    whether UNI is near chance, since the hypothesis ('look alike to H&E alone,
    separate once ST-aware') needs UNI low AND SIGMMA meaningfully higher.
    """
    s = summary
    L = []
    L.append(f"# H&E -> T-stage separability ({s['cohort']}, z_*_{s['emb_key']})\n")
    L.append(f"- slides: {s['n_slides']}  (pT1={s['n_pT1']}, pT3+={s['n_pT3+']}), "
             f"groups(patients)={s['n_groups']}")
    u = s["unsupervised"]
    L.append(f"- silhouette(stage): UNI {u['uni']['silhouette']:+.3f} | "
             f"SIGMMA {u['sigmma']['silhouette']:+.3f}")
    L.append(f"- kNN-purity:        UNI {u['uni']['knn_purity']:.3f} | "
             f"SIGMMA {u['sigmma']['knn_purity']:.3f}")
    sup = s["supervised"]
    if sup is None:
        L.append("\n## Verdict: INCONCLUSIVE")
        L.append("Supervised probe skipped — a stage class has <2 patient groups, "
                 "so patient-grouped CV is not estimable (cohort not powered). "
                 "Unsupervised numbers are confounded by class imbalance + "
                 "multiple-sections-per-patient clustering; not a stage test.")
    else:
        au, as_ = sup["auc_uni_mean"], sup["auc_sigmma_mean"]
        d, (lo, hi), p = sup["delta_mean"], sup["delta_ci95"], sup["wilcoxon_p"]
        L.append(f"\n## Supervised (patient-grouped {sup['n_splits']}-fold x "
                 f"{sup['n_repeats']}-repeat, {sup['n_folds']} folds)")
        L.append(f"- AUROC: UNI {au:.3f}+-{sup['auc_uni_std']:.3f} | "
                 f"SIGMMA {as_:.3f}+-{sup['auc_sigmma_std']:.3f}")
        L.append(f"- paired Delta(SIGMMA-UNI) = {d:+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  "
                 f"Wilcoxon p={p:.2e}")
        ci_excludes_0 = (lo > 0) or (hi < 0)
        if ci_excludes_0 and d >= 0.03:
            verdict = ("SUPPORTED — ST-enriched H&E separates pT1 vs pT3+ "
                       "better than H&E alone (CI excludes 0, Delta>=0.03).")
        elif ci_excludes_0 and d <= -0.03:
            verdict = ("CONTRADICTED — ST-enriched H&E separates WORSE than "
                       "H&E alone (CI excludes 0, Delta<=-0.03).")
        elif ci_excludes_0:
            verdict = (f"DISTINGUISHABLE BUT SMALL — Delta significant but |Delta|<0.03 "
                       f"({d:+.3f}); not a practically meaningful gain.")
        else:
            verdict = ("WASH — paired CI includes 0; no reliable difference "
                       "between ST-enriched and H&E-alone for T-stage.")
        prem = ("premise holds (UNI near chance, AUROC<=0.6)" if au <= 0.60
                else f"NOTE premise weak: UNI already separates stage (AUROC={au:.3f}); "
                      "'looks alike to H&E alone' is only partly true")
        L.append(f"\n## Verdict: {verdict}\n- {prem}")
    L.append(f"\nFigure: {fig_path}")
    report_path.write_text("\n".join(L) + "\n")
    print(f"[report] wrote {report_path}")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["tcga", "inhouse"], required=True)
    ap.add_argument("--uni_dir", required=True)
    ap.add_argument("--sigmma_dir", required=True)
    ap.add_argument("--emb_key", default="macro_cls",
                    help="suffix after z_uni_/z_sigmma_ (default macro_cls, 1536-d).")
    ap.add_argument("--clinical", default=None, help="TCGA-KIRC_clinical.tsv (tcga)")
    ap.add_argument("--metadata", default=None, help="RCC metadata.csv (inhouse)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = build_table(Path(args.uni_dir), Path(args.sigmma_dir), args.emb_key)
    if df.empty:
        raise SystemExit("no slides with both embeddings; check caches")

    if args.cohort == "tcga":
        assert args.clinical, "--clinical required for tcga"
        df = label_tcga(df, Path(args.clinical))
    else:
        assert args.metadata, "--metadata required for inhouse"
        df = label_inhouse(df, Path(args.metadata))

    labelled = df[df["stage"].notna()].reset_index(drop=True)
    print(f"[label] {len(labelled)}/{len(df)} slides in pT1/pT3+ contrast; "
          f"{labelled['stage'].value_counts().to_dict()}; "
          f"{labelled['group'].nunique()} groups")
    if labelled.empty:
        raise SystemExit("no slides fell into pT1/pT3+; check the T-stage join")

    X_uni = np.vstack(labelled["uni"].to_numpy())
    X_sig = np.vstack(labelled["sigmma"].to_numpy())
    y = labelled["stage"].to_numpy()
    groups = labelled["group"].to_numpy()

    uns_uni = unsupervised(X_uni, y, args.seed)
    uns_sig = unsupervised(X_sig, y, args.seed)
    sup = supervised(X_uni, X_sig, y, groups, args.n_splits, args.n_repeats, args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labelled[["sample_name", "group", "t_raw", "stage"]].to_csv(
        out_dir / f"stage_table_{args.cohort}_{args.emb_key}.csv", index=False)
    summary = {
        "cohort": args.cohort, "emb_key": args.emb_key,
        "n_slides": int(len(labelled)),
        "n_pT1": int((y == "pT1").sum()), "n_pT3+": int((y == "pT3+").sum()),
        "n_groups": int(labelled["group"].nunique()),
        "unsupervised": {
            "uni": {"silhouette": uns_uni["silhouette"], "knn_purity": uns_uni["knn_purity"]},
            "sigmma": {"silhouette": uns_sig["silhouette"], "knn_purity": uns_sig["knn_purity"]},
        },
        "supervised": None if sup is None else {
            k: v for k, v in sup.items() if not k.startswith("_")},
    }
    with open(out_dir / f"stage_separability_{args.cohort}_{args.emb_key}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fig_path = Path(args.fig_dir) / f"stage_separability_{args.cohort}_{args.emb_key}.png"
    make_figure(labelled, uns_uni, uns_sig, sup, fig_path, args.cohort)

    report_path = out_dir / f"RESULTS_{args.cohort}_{args.emb_key}.md"
    write_report(summary, fig_path, report_path)


if __name__ == "__main__":
    main()
