#!/usr/bin/env python
"""Mine "morphological doppelganger" crops: H&E patches that look morphologically
near-identical (high SSIM) yet carry OPPOSITE molecular identity (different cell
type / gene-expression profile).

This is the *existence check* for SIGMMA's core value proposition (see memory
project_sigmma_core_value_prop): if such crops exist, then a molecularly-defined
compartment is genuinely morphologically subtle — a pathologist / pure-H&E model
cannot separate the two by morphology alone, and only ST-aware alignment can.
NO model is used here: this establishes the premise from raw H&E + ST, before
asking whether SIGMMA resolves it.

Method
------
For one cohort (via an extract-style config with a data block), for each section:
  1. per-cell pixel centroids  (extract._load_preproc_centroids)
  2. per-cell cell-type label   (extract._load_labels_for_sample, config schema)
  3. keep cells of the two contrasted classes (--class_a vs --class_b), cap per
     class per section (stratified subsample)
  4. read the centered H&E crop from the slide zarr (same bbox convention as the
     niche extractor: x0 = clamp(cx - S/2, 0, W-S))
  5. drop background/near-empty crops (tissue fraction < --tissue_frac_min)
  6. per-cell gene-expression vector from the section AnnData, recovered to
     CP(target_sum)+log1p ONCE (handles the log1p(counts) .X state) so cosine
     distance is cross-section comparable.

Then, over the pooled crops (class_a vs class_b, across sections):
  * fast GLOBAL SSIM (single-window, vectorised via a covariance matmul) scores
    every class_a x class_b pair; take the top --shortlist by global SSIM.
  * exact windowed mean-SSIM (skimage) is recomputed on the shortlist for the
    reported number.
  * pairs are filtered (same-section pairs must be >= --min_sep_um apart to avoid
    overlapping-tissue triviality) and greedily de-duplicated so each crop is
    shown once, then the top --top_k are rendered.

Outputs
-------
  outputs/morph_subtle/<tag>_pairs.csv     — all shortlisted pairs + metrics
  figures/morph_subtle/<tag>_doppelgangers.png/.pdf — top-k pair panel

CPU only. Run in the sigmma poetry venv (scanpy + skimage + zarr).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py  # noqa: F401  (imported by extract helpers indirectly; kept explicit)
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
import zarr
from ml_collections.config_dict import ConfigDict
from scipy import sparse
from skimage.metrics import structural_similarity as ssim

from sigmma.application.cell_niche_transfer.extract import (
    label_config_from_cfg,
    _load_labels_for_sample,
    _load_preproc_centroids,
)

_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # RGB -> luma

# Kidney / ccRCC marker panel for the figure's molecular bar (present-subset used).
MARKER_GENES = [
    # ccRCC tumour epithelium
    "CA9", "NDUFA4L2", "VEGFA", "ANGPTL4", "NNMT",
    # nephron epithelium (proximal / distal / podocyte)
    "SLC34A1", "LRP2", "CUBN", "UMOD", "SLC12A1", "NPHS1", "NPHS2",
    # micro-environment (context)
    "PECAM1", "KDR", "PTPRC", "CD8A", "PDGFRB", "ACTA2", "COL1A1",
]


# --------------------------------------------------------------------------
# expression normalisation (mirror scripts/angio_teff/compute_signature_scores)
# --------------------------------------------------------------------------
def _nnz_sample(X) -> np.ndarray:
    sub = X[:2000] if X.shape[0] > 2000 else X
    d = sub.data if sparse.issparse(sub) else sub[sub != 0]
    return np.asarray(d[:200000], dtype=np.float64)


def _is_counts(X) -> bool:
    d = _nnz_sample(X)
    return d.size > 0 and float(np.nanmax(d)) > 30.0 and bool(np.allclose(d, np.round(d)))


def _is_log1p_counts(X) -> bool:
    d = _nnz_sample(X)
    if d.size == 0 or bool(np.allclose(d, np.round(d))):
        return False
    rec = np.expm1(d)
    return bool(np.isclose(d.min(), np.log1p(1.0), atol=1e-3)) and \
        float(np.mean(np.isclose(rec, np.round(rec), atol=1e-2))) > 0.95


def _to_cp_log1p(adata, target_sum: float) -> str:
    if _is_counts(adata.X):
        src = "raw_counts"
    elif _is_log1p_counts(adata.X):
        if sparse.issparse(adata.X):
            adata.X.data = np.rint(np.expm1(adata.X.data))
        else:
            adata.X = np.rint(np.expm1(adata.X))
        src = "expm1(log1p_counts)"
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        return "unknown->cp+log1p"
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return f"{src}->cp{target_sum:g}+log1p"


def _cell_ids(adata) -> np.ndarray:
    for k in ("cell_id", "CellID", "cell_ID", "barcode"):
        if k in adata.obs.columns:
            return adata.obs[k].astype(str).to_numpy()
    return adata.obs_names.astype(str).to_numpy()


# --------------------------------------------------------------------------
# crop reading
# --------------------------------------------------------------------------
def _bbox(cx: float, cy: float, S: int, W: int, H: int) -> tuple[int, int]:
    x0 = max(0, min(int(round(cx - S / 2)), W - S))
    y0 = max(0, min(int(round(cy - S / 2)), H - S))
    return x0, y0


def _read_rgb(he_zarr, x0: int, y0: int, S: int) -> np.ndarray:
    """(S, S, 3) uint8 crop from a (3, H, W) zarr."""
    chw = np.asarray(he_zarr[:, y0:y0 + S, x0:x0 + S])  # (3, S, S)
    return np.transpose(chw, (1, 2, 0)).astype(np.uint8)


def _tissue_fraction(gray: np.ndarray) -> float:
    """Fraction of stained (non-background) pixels; H&E background is bright."""
    return float((gray < 220).mean())


# --------------------------------------------------------------------------
# vectorised global (single-window) SSIM over a class_a x class_b block
# --------------------------------------------------------------------------
def _global_ssim_block(gray_a: np.ndarray, gray_b: np.ndarray,
                       L: float = 255.0) -> np.ndarray:
    """SSIM with the whole crop as one window, for every (a, b) pair.

    Returns (Na, Nb). Vectorised: luminance/contrast from per-crop mean/var,
    structure term from the cross-covariance matmul. Used only to SHORTLIST;
    the reported number is exact windowed SSIM recomputed on the shortlist.
    """
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    # float32 keeps the 256px pool within the oversubscribed ~8GB cap; this only
    # shortlists candidates (exact windowed SSIM is recomputed downstream).
    Xa = gray_a.reshape(len(gray_a), -1).astype(np.float32)
    Xb = gray_b.reshape(len(gray_b), -1).astype(np.float32)
    P = Xa.shape[1]
    mu_a = Xa.mean(1)          # (Na,)
    mu_b = Xb.mean(1)          # (Nb,)
    Xac = Xa - mu_a[:, None]
    Xbc = Xb - mu_b[:, None]
    var_a = (Xac ** 2).mean(1)  # (Na,)
    var_b = (Xbc ** 2).mean(1)  # (Nb,)
    cov = (Xac @ Xbc.T) / P     # (Na, Nb)
    lum = (2 * np.outer(mu_a, mu_b) + C1) / (mu_a[:, None] ** 2 + mu_b[None, :] ** 2 + C1)
    cs = (2 * cov + C2) / (var_a[:, None] + var_b[None, :] + C2)
    return (lum * cs).astype(np.float32)


# --------------------------------------------------------------------------
def _stratified_cap(idx: np.ndarray, rng: np.random.Generator, cap: int) -> np.ndarray:
    if cap <= 0 or len(idx) <= cap:
        return idx
    return np.sort(rng.choice(idx, size=cap, replace=False))


def process_section(row, *, label_name, class_a, class_b, label_cfg, crop_px,
                    cap, tissue_frac_min, target_sum, rng, gene_ref):
    """Return dict of pooled per-cell arrays for one section (both classes)."""
    sample = row["sample_name"]
    norm = row.get("norm", sample)
    he_path = row["hematoxylin_eosin_image"]
    sp_path = row["spatial_preprocessed"]
    if not (isinstance(he_path, str) and Path(he_path).exists()):
        print(f"[skip] {sample}: H&E zarr missing", flush=True)
        return None
    if not (isinstance(sp_path, str) and Path(sp_path).exists()):
        print(f"[skip] {sample}: spatial_preprocessed missing", flush=True)
        return None

    centroids = _load_preproc_centroids(sp_path, label_cfg["um_per_px"])
    labels = _load_labels_for_sample(
        row.get("labels_h5ad", label_cfg["labels_h5ad"]), norm,
        sample_col=label_cfg["sample_col"], label_specs=label_cfg["label_specs"],
        normalize_sample=label_cfg["normalize_sample"],
    )
    if labels.empty:
        print(f"[skip] {sample}: no labels matched (norm={norm})", flush=True)
        return None
    if label_name not in labels.columns:
        raise SystemExit(f"label '{label_name}' not in {list(labels.columns)}")

    common = sorted(set(centroids.index) & set(labels.index))
    lab = labels.loc[common, label_name].astype(str)
    cls = lab.to_numpy()
    if class_a not in set(cls) or class_b not in set(cls):
        uniq, cnt = np.unique(cls, return_counts=True)
        print(f"[info] {sample}: classes present -> "
              f"{dict(zip(uniq.tolist(), cnt.tolist()))}", flush=True)
        # still continue if at least one of the two present in other sections
    common = np.array(common)

    keep_ids = []
    keep_cls = []
    for cname in (class_a, class_b):
        sel = common[cls == cname]
        sel = _stratified_cap(sel, rng, cap)
        keep_ids.extend(sel.tolist())
        keep_cls.extend([cname] * len(sel))
    if not keep_ids:
        return None
    keep_ids = np.array(keep_ids)
    keep_cls = np.array(keep_cls)

    # crops
    he = zarr.open(he_path, mode="r")
    H, W = he.shape[1], he.shape[2]
    S = crop_px
    if H < S or W < S:
        print(f"[skip] {sample}: slide smaller than crop ({H}x{W} < {S})", flush=True)
        return None

    rgb_list, gray_list, x0s, ys0, cx_um, cy_um, ok_ids, ok_cls = [], [], [], [], [], [], [], []
    for cid, cname in zip(keep_ids, keep_cls):
        cx = float(centroids.loc[cid, "x_pixel"])
        cy = float(centroids.loc[cid, "y_pixel"])
        x0, y0 = _bbox(cx, cy, S, W, H)
        rgb = _read_rgb(he, x0, y0, S)
        gray = rgb.astype(np.float32) @ _GRAY
        if _tissue_fraction(gray) < tissue_frac_min:
            continue
        rgb_list.append(rgb)
        gray_list.append(gray)
        x0s.append(x0); ys0.append(y0)
        cx_um.append(cx * label_cfg["um_per_px"])
        cy_um.append(cy * label_cfg["um_per_px"])
        ok_ids.append(cid); ok_cls.append(cname)
    if not ok_ids:
        return None

    # expression (full read; subset pool cells; align to gene_ref)
    adata = sc.read_h5ad(sp_path)
    _to_cp_log1p(adata, target_sum)
    var_up = {}
    for j, g in enumerate(adata.var_names):
        var_up.setdefault(str(g).upper(), j)  # first occurrence wins
    if gene_ref["names"] is None:
        gene_ref["names"] = [str(g).upper() for g in adata.var_names]
        gene_ref["set"] = set(gene_ref["names"])
    cids_all = _cell_ids(adata)
    row_of = {c: i for i, c in enumerate(cids_all)}
    ref_names = gene_ref["names"]
    ref_cols = np.array([var_up.get(g, -1) for g in ref_names])
    present = ref_cols >= 0
    Xall = adata.X
    expr = np.full((len(ok_ids), len(ref_names)), np.nan, dtype=np.float32)
    marker_cols = {g: var_up.get(g.upper(), -1) for g in MARKER_GENES}
    marker = np.full((len(ok_ids), len(MARKER_GENES)), np.nan, dtype=np.float32)
    for i, cid in enumerate(ok_ids):
        r = row_of.get(cid)
        if r is None:
            continue
        xrow = Xall[r]
        xrow = xrow.toarray().ravel() if sparse.issparse(xrow) else np.asarray(xrow).ravel()
        expr[i, present] = xrow[ref_cols[present]]
        for k, g in enumerate(MARKER_GENES):
            c = marker_cols[g]
            if c >= 0:
                marker[i, k] = xrow[c]
    del adata

    return {
        "sample": np.array([sample] * len(ok_ids)),
        "cell_id": np.array(ok_ids),
        "cls": np.array(ok_cls),
        "rgb": np.stack(rgb_list),          # (n, S, S, 3) uint8
        "gray": np.stack(gray_list),        # (n, S, S) float32
        "cx_um": np.array(cx_um), "cy_um": np.array(cy_um),
        "expr": expr,                       # (n, G) float32
        "marker": marker,                   # (n, M) float32
    }


def _sample_pair_ssim(ii, jj, *, gray, sample, cx_um, cy_um, same_group,
                      n_pairs, min_sep_um, win, rng):
    """Sample up to n_pairs (i in ii, j in jj) windowed SSIM values. Same-section
    pairs must be >= min_sep_um apart (fair to the mining constraint), so within-
    type pairs are NOT inflated by overlapping-neighbourhood crops."""
    out, seen = [], set()
    max_tries = n_pairs * 40
    tries = 0
    while len(out) < n_pairs and tries < max_tries:
        tries += 1
        ia = int(ii[rng.integers(len(ii))])
        ib = int(jj[rng.integers(len(jj))])
        if ia == ib:
            continue
        key = (min(ia, ib), max(ia, ib))
        if key in seen:
            continue
        if sample[ia] == sample[ib]:
            if np.hypot(cx_um[ia] - cx_um[ib], cy_um[ia] - cy_um[ib]) < min_sep_um:
                continue
        seen.add(key)
        out.append(float(ssim(gray[ia], gray[ib], data_range=255.0,
                              win_size=win, gaussian_weights=True)))
    return np.asarray(out)


def _q(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: float("nan") for k in ("n", "p50", "p90", "p99", "max")}
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def _cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 5:
        return np.nan
    a, b = u[m], v[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - (a @ b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="extract-style YAML (data block)")
    ap.add_argument("--mode", choices=["mine", "dist"], default="mine",
                    help="mine=render doppelganger pairs; dist=cross-vs-within SSIM "
                         "distributions (is 0.3 'similar'? answered relative to within-type)")
    ap.add_argument("--n_pairs_per_group", type=int, default=8000)
    ap.add_argument("--label_name", default="level_2_cell_type")
    ap.add_argument("--class_a", default="Tumour epithelium")
    ap.add_argument("--class_b", default="Nephron epithelium")
    ap.add_argument("--crop_px", type=int, default=128, help="crop side in px (0.5um/px)")
    ap.add_argument("--max_per_class_per_section", type=int, default=150)
    ap.add_argument("--tissue_frac_min", type=float, default=0.55)
    ap.add_argument("--min_sep_um", type=float, default=150.0,
                    help="same-section pairs must be at least this far apart")
    ap.add_argument("--shortlist", type=int, default=4000)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--target_sum", type=float, default=1e4)
    ap.add_argument("--sections", default=None, help="comma list to restrict (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="rcc_tumour_vs_nephron")
    ap.add_argument("--out_dir", default="outputs/morph_subtle")
    ap.add_argument("--fig_dir", default="figures/morph_subtle")
    args = ap.parse_args()

    cfg = ConfigDict(yaml.safe_load(open(args.config)))
    label_cfg = label_config_from_cfg(cfg.data)
    label_cfg["labels_h5ad"] = cfg.data.labels_h5ad
    manifest = pd.read_csv(cfg.data.evaluable_slides_csv)
    if "norm" not in manifest.columns:
        manifest["norm"] = manifest["sample_name"]
    if args.sections:
        want = set(args.sections.split(","))
        manifest = manifest[manifest["sample_name"].isin(want)]
    rng = np.random.default_rng(args.seed)
    gene_ref = {"names": None, "set": None}

    pooled = []
    for _, row in manifest.iterrows():
        res = process_section(
            row, label_name=args.label_name, class_a=args.class_a,
            class_b=args.class_b, label_cfg=label_cfg, crop_px=args.crop_px,
            cap=args.max_per_class_per_section, tissue_frac_min=args.tissue_frac_min,
            target_sum=args.target_sum, rng=rng, gene_ref=gene_ref,
        )
        if res is not None:
            pooled.append(res)
            print(f"[ok] {res['sample'][0]}: pooled {len(res['cell_id'])} crops "
                  f"({(res['cls']==args.class_a).sum()} A / "
                  f"{(res['cls']==args.class_b).sum()} B)", flush=True)

    if not pooled:
        raise SystemExit("no crops pooled — check config / class names")

    # concat pool
    def cat(key):
        return np.concatenate([p[key] for p in pooled], axis=0)
    sample = cat("sample"); cell_id = cat("cell_id"); cls = cat("cls")
    rgb = cat("rgb"); gray = cat("gray")
    cx_um = cat("cx_um"); cy_um = cat("cy_um")
    expr = cat("expr"); marker = cat("marker")

    a_idx = np.flatnonzero(cls == args.class_a)
    b_idx = np.flatnonzero(cls == args.class_b)
    print(f"\n[pool] class_a={len(a_idx)}  class_b={len(b_idx)}  "
          f"total={len(cls)}", flush=True)
    if len(a_idx) == 0 or len(b_idx) == 0:
        raise SystemExit("one class empty after pooling")

    win = min(7, args.crop_px // 4 * 2 + 1)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dist":
        _run_dist(a_idx, b_idx, gray=gray, sample=sample, cx_um=cx_um, cy_um=cy_um,
                  win=win, rng=rng, args=args, out_dir=out_dir, fig_dir=fig_dir)
        return

    # global SSIM shortlist over the a x b block
    gssim = _global_ssim_block(gray[a_idx], gray[b_idx])  # (Na, Nb)
    flat = gssim.ravel()
    k = min(args.shortlist, flat.size)
    top = np.argpartition(flat, -k)[-k:]
    top = top[np.argsort(flat[top])[::-1]]
    ai = a_idx[top // gssim.shape[1]]
    bi = b_idx[top % gssim.shape[1]]

    # exact windowed SSIM + molecular distance on the shortlist, with filters
    rows = []
    win = min(7, args.crop_px // 4 * 2 + 1)
    for gi, (ia, ib) in enumerate(zip(ai, bi)):
        same = sample[ia] == sample[ib]
        if same:
            d = float(np.hypot(cx_um[ia] - cx_um[ib], cy_um[ia] - cy_um[ib]))
            if d < args.min_sep_um:
                continue
        s = float(ssim(gray[ia], gray[ib], data_range=255.0,
                       win_size=win, gaussian_weights=True))
        mdist = _cosine_dist(expr[ia], expr[ib])
        rows.append({
            "rank_global": gi, "ssim_win": s, "ssim_global": float(gssim.ravel()[top[gi]]),
            "expr_cos_dist": mdist,
            "idx_a": int(ia), "idx_b": int(ib),
            "sample_a": sample[ia], "sample_b": sample[ib],
            "cell_a": cell_id[ia], "cell_b": cell_id[ib],
            "class_a": cls[ia], "class_b": cls[ib],
            "same_section": bool(same),
        })
    pairs = pd.DataFrame(rows).sort_values("ssim_win", ascending=False).reset_index(drop=True)

    csv_path = out_dir / f"{args.tag}_pairs.csv"
    pairs.drop(columns=["idx_a", "idx_b"]).to_csv(csv_path, index=False)
    print(f"\n[write] {len(pairs)} shortlisted pairs -> {csv_path}", flush=True)

    # greedy de-dup so each crop shown once, then render top_k
    used = set()
    chosen = []
    for _, r in pairs.iterrows():
        if r["idx_a"] in used or r["idx_b"] in used:
            continue
        used.add(r["idx_a"]); used.add(r["idx_b"])
        chosen.append(r)
        if len(chosen) >= args.top_k:
            break
    print(f"[fig] rendering {len(chosen)} de-duplicated top pairs", flush=True)
    _render(chosen, rgb, marker, args, fig_dir)


def _run_dist(a_idx, b_idx, *, gray, sample, cx_um, cy_um, win, rng, args,
              out_dir, fig_dir):
    """Answer 'is 0.3 similar?' by comparing the cross-type SSIM distribution to
    the within-type distributions. If cross overlaps within, morphology cannot
    separate the two molecular classes (= morphologically subtle)."""
    n = args.n_pairs_per_group
    kw = dict(gray=gray, sample=sample, cx_um=cx_um, cy_um=cy_um,
              n_pairs=n, min_sep_um=args.min_sep_um, win=win, rng=rng)
    cross = _sample_pair_ssim(a_idx, b_idx, same_group=False, **kw)
    within_a = _sample_pair_ssim(a_idx, a_idx, same_group=True, **kw)
    within_b = _sample_pair_ssim(b_idx, b_idx, same_group=True, **kw)

    stats = {"cross_A_vs_B": _q(cross),
             f"within_{args.class_a}": _q(within_a),
             f"within_{args.class_b}": _q(within_b)}
    df = pd.DataFrame(stats).T
    print("\n===== windowed SSIM distributions =====", flush=True)
    print(df.to_string(), flush=True)
    # Key read: fraction of cross pairs that exceed the within-type medians.
    med_a, med_b = np.nanmedian(within_a), np.nanmedian(within_b)
    frac_over = float((cross > min(med_a, med_b)).mean())
    print(f"\n[interp] cross median={np.nanmedian(cross):.3f}  "
          f"within medians A={med_a:.3f} B={med_b:.3f}", flush=True)
    print(f"[interp] fraction of cross pairs above the smaller within-median "
          f"= {frac_over:.2%}  (high overlap => morphologically subtle)", flush=True)
    df.to_csv(out_dir / f"{args.tag}_ssim_distributions.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(cross.min(), within_a.min(), within_b.min()),
                       max(cross.max(), within_a.max(), within_b.max()), 60)
    ax.hist(within_a, bins=bins, density=True, alpha=0.5, color="#d62728",
            label=f"within {args.class_a} (n={within_a.size})")
    ax.hist(within_b, bins=bins, density=True, alpha=0.5, color="#1f77b4",
            label=f"within {args.class_b} (n={within_b.size})")
    ax.hist(cross, bins=bins, density=True, histtype="step", lw=2, color="k",
            label=f"cross A vs B (n={cross.size})")
    ax.set_xlabel("windowed SSIM (grayscale H&E crops)")
    ax.set_ylabel("density")
    ax.set_title(f"Cross- vs within-type H&E SSIM ({args.crop_px}px)\n"
                 "overlap => tumour/nephron morphologically inseparable")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fig_dir / f"{args.tag}_ssim_distributions.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[fig] wrote {p}", flush=True)
    plt.close(fig)


def _render(chosen, rgb, marker, args, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(chosen)
    if n == 0:
        print("[fig] nothing to render", flush=True)
        return
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n),
                             gridspec_kw={"width_ratios": [1, 1, 1.4]})
    if n == 1:
        axes = axes[None, :]
    mk = np.array(MARKER_GENES)
    for i, r in enumerate(chosen):
        ia, ib = int(r["idx_a"]), int(r["idx_b"])
        axes[i, 0].imshow(rgb[ia]); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{r['class_a']}\n{r['sample_a']}", fontsize=7)
        axes[i, 1].imshow(rgb[ib]); axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{r['class_b']}\n{r['sample_b']}", fontsize=7)
        ma, mb = marker[ia], marker[ib]
        x = np.arange(len(mk))
        axes[i, 2].bar(x - 0.2, np.nan_to_num(ma), width=0.4, label="A", color="#d62728")
        axes[i, 2].bar(x + 0.2, np.nan_to_num(mb), width=0.4, label="B", color="#1f77b4")
        axes[i, 2].set_xticks(x)
        axes[i, 2].set_xticklabels(mk, rotation=90, fontsize=5)
        axes[i, 2].tick_params(axis="y", labelsize=5)
        axes[i, 2].set_title(
            f"SSIM={r['ssim_win']:.3f}  exprCos={r['expr_cos_dist']:.2f}"
            f"  {'same' if r['same_section'] else 'cross'}-sec", fontsize=7)
        if i == 0:
            axes[i, 2].legend(fontsize=6, loc="upper right")
    fig.suptitle(
        f"Morphological doppelgangers: {args.class_a} vs {args.class_b}\n"
        f"high H&E SSIM, opposite molecular identity ({args.crop_px}px crops)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    for ext in ("png", "pdf"):
        p = fig_dir / f"{args.tag}_doppelgangers.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[fig] wrote {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
