#!/usr/bin/env python
"""EMT axis scoring + spatial EMT+/- region visualization for a Xenium section.

EMT is treated as a continuous epithelial<->mesenchymal AXIS (not on/off). The
score follows the Mak et al. 2016 definition: per-gene z-score across cells, then
  emt_axis = mean(z[M markers]) - mean(z[E markers])    (>0 = mesenchymal)
Signal cells (>80th pct of either pole) are split into EMT- (epithelial) / EMT+
(mesenchymal); the rest are low-signal parenchyma.

Two marker signatures are selectable via --signature (only panel-present genes are
scored); see SIGNATURES below for the gene lists and their literature sources.
"""
import argparse, sys, csv, re
from pathlib import Path
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap

from sigmma.utils.tissue_map import tissue_of

# --- EMT marker signatures (only panel-present genes are scored at runtime) ---
# Each entry holds M (mesenchymal pole), E (epithelial pole) and the source paper.
SIGNATURES = {
    # Core 16-gene EMT marker set (M 13 / E 3).
    # Source: Dev Dyn — https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/dvdy.24485
    "core16": {
        "label": "core 16-gene",
        "source": "https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/dvdy.24485",
        "M": ["VIM", "CDH2", "FOXC2", "SNAI1", "SNAI2", "TWIST1", "FN1", "ITGB6",
              "MMP2", "MMP3", "MMP9", "SOX10", "UGCG"],
        "E": ["CDH1", "DSP", "OCLN"],
    },
    # Mak et al. 2016 patient-derived pan-cancer EMT signature (M 52 / E 25).
    # Source: Clin Cancer Res 22(3):609 —
    #   https://aacrjournals.org/clincancerres/article/22/3/609/254665/A-Patient-Derived-Pan-Cancer-EMT-Signature
    "mak2016": {
        "label": "Mak 2016 pan-cancer 77-gene",
        "source": ("https://aacrjournals.org/clincancerres/article/22/3/609/"
                   "254665/A-Patient-Derived-Pan-Cancer-EMT-Signature"),
        "E": ["AP1G1", "ATP8B1", "CDH1", "CDS1", "CGN", "CLDN4", "CNOT1", "CTNND1",
              "DYNC1LI2", "ERBB3", "ESRP1", "ESRP2", "F11R", "GALNT3", "GPR56",
              "GRHL2", "HOOK1", "IRF6", "MAP7", "MARVELD2", "MARVELD3", "MYO5B",
              "OCLN", "PRSS8", "SPINT1"],
        "M": ["ADAM12", "ADAMTS12", "ADAMTS2", "AEBP1", "ANGPTL2", "ANTXR1", "AXL",
              "BNC2", "CALD1", "CDH2", "CMTM3", "CNRIP1", "COL10A1", "COL1A1",
              "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
              "COL8A1", "DACT1", "EMP3", "FAP", "FBN1", "FN1", "FSTL1", "GPC6",
              "GYPC", "HTRA1", "INHBA", "ITGA11", "LOXL2", "LRRC15", "MMP2", "MSRB3",
              "NAP1L3", "NID2", "OLFML2B", "PCOLCE", "PDGFRB", "PMP22", "POSTN",
              "SPARC", "SPOCK1", "SULF1", "SYT11", "THBS2", "VCAN", "VIM", "ZEB2"],
    },
}


def _union_pole(key):
    """Dedup union of a pole (key='M' or 'E') across core16 + Mak2016."""
    seen, out = set(), []
    for s in ("core16", "mak2016"):
        for g in SIGNATURES[s][key]:
            if g not in seen:
                seen.add(g); out.append(g)
    return out


# union signature: combine both gene lists per pole (CDH2/VIM/FN1/MMP2 [M] and
# CDH1/OCLN [E] overlap, so they are deduplicated). M 61 / E 26.
SIGNATURES["union"] = {
    "label": "core16 + Mak2016 union",
    "source": SIGNATURES["core16"]["source"] + " + " + SIGNATURES["mak2016"]["source"],
    "M": _union_pole("M"),
    "E": _union_pole("E"),
}

MANIFESTS = [
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv",
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_val.csv",
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_test.csv",
]

# cell-type labels (RCC cohort): tumour epithelium == cancer cells. Keyed by
# `batch` (clean section name) + `cell_id` so we can join onto the section adata.
LABELS_H5 = ("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
             "sanger_renal_cell_carcinoma/labels.h5ad")
CANCER_LEVEL = "level_3_cell_type"  # annotation column to read the cancer label from
CANCER_CT = "Tumour epithelium"     # value in CANCER_LEVEL flagged as cancer cell
                                    # (identical between level_2 and level_3)


def resolve_paths(section):
    for mf in MANIFESTS:
        if not Path(mf).exists():
            continue
        for row in csv.DictReader(open(mf)):
            if row["sample_name"] == section:
                return row["spatial_preprocessed"], row.get("hematoxylin_eosin_image", "")
    raise SystemExit(f"section {section} not found in manifests")


def zscore_mean(a, genes):
    """Mean of per-gene z-scores across cells (Mak et al. 2016 EMT-score definition).

    `genes` must already be filtered to those present in a.var_names.
    """
    if not genes:
        return np.zeros(a.n_obs)
    X = a[:, genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return np.asarray(Z.mean(1)).ravel()


def load_he(he_zarr, ds):
    """Load H&E zarr (C,H,W uint8), downsample by stride `ds`, return (H',W',3)."""
    import zarr
    z = zarr.open(he_zarr, mode="r")
    img = np.asarray(z[:, ::ds, ::ds])      # (3, H', W')
    img = np.transpose(img, (1, 2, 0))      # (H', W', 3)
    return img


def load_cancer_mask(section, obs_names, ct_level=CANCER_LEVEL):
    """Join the RCC cell-type atlas onto this section's cells.

    Reads only index arrays from the multi-GB labels.h5ad (never X), subsets to the
    section's batch, and joins on `cell_id`. Returns (cancer, celltype, labeled):
      - cancer:   bool array, True where the cell is CANCER_CT (Tumour epithelium)
      - celltype: object array of `ct_level` labels (None for unlabeled cells)
      - labeled:  bool array, True where the cell was found in the atlas
    Returns (None, None, None) if the labels file or this section is unavailable.
    """
    if not Path(LABELS_H5).exists():
        return None, None, None
    import h5py
    f = h5py.File(LABELS_H5, "r"); obs = f["obs"]
    dec = lambda v: v.decode() if isinstance(v, bytes) else v
    bcats = [dec(x) for x in obs["batch"]["categories"][:]]
    # labels.h5ad batch names are serial-suffix-free (e.g. CV9-KID-0-FT-1), while the
    # manifest carries -s3/-S3 serial suffixes; strip them to match the same section.
    strip = lambda s: re.sub(r"-[sS]\d+$", "", s)
    bidx = {b: i for i, b in enumerate(bcats)}
    if section in bidx:
        sidx = bidx[section]
    else:
        bnorm = {strip(b): i for i, b in enumerate(bcats)}
        sidx = bnorm.get(strip(section))
        if sidx is None:
            print(f"[cancer] {section} not in {Path(LABELS_H5).name}, skipping cancer overlay")
            return None, None, None
        print(f"[cancer] {section} matched labels batch '{bcats[sidx]}' (serial-suffix normalized)")
    sel = np.where(obs["batch"]["codes"][:] == sidx)[0]
    cid_cats = obs["cell_id"]["categories"][:]
    cid = [dec(cid_cats[c]) for c in obs["cell_id"]["codes"][sel]]
    # full cell-type label (for EMT+/- composition) and the derived cancer flag
    ctcats = [dec(x) for x in obs[ct_level]["categories"][:]]
    ct_codes = obs[ct_level]["codes"][sel]
    ct_map = {c: (ctcats[k] if k >= 0 else None) for c, k in zip(cid, ct_codes)}
    celltype = np.array([ct_map.get(c, None) for c in obs_names], dtype=object)
    labeled = np.array([c in ct_map for c in obs_names])
    cancer = celltype == CANCER_CT
    # a matched batch name does NOT guarantee the same physical slice: some serial
    # sections (-s5/-s6) share a name but have disjoint cell_ids. If essentially no
    # cells overlap, the join is meaningless -> treat as unlabeled (placeholder panel).
    if labeled.mean() < 0.05:
        print(f"[cancer] only {labeled.sum()}/{len(labeled)} cells overlap the labels "
              f"batch (serial-slice mismatch) -> treating as no cancer labels")
        return None, None, None
    return cancer, celltype, labeled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="DF29-LIV-0-FFPE-1-S1-i")
    ap.add_argument("--outdir", default="figures/cross_tissue_retrieval/emt")
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--he_downsample", type=int, default=8,
                    help="stride downsample factor for the H&E image")
    ap.add_argument("--signature", default="union", choices=list(SIGNATURES),
                    help="EMT marker signature to score (see SIGNATURES)")
    ap.add_argument("--require_cancer_labels", action="store_true",
                    help="skip the section entirely if it has no usable cancer labels")
    ap.add_argument("--celltype_level", default=CANCER_LEVEL,
                    choices=["level_1_cell_type", "level_2_cell_type", "level_3_cell_type"],
                    help="cell-type annotation level for the EMT+/- composition panel")
    args = ap.parse_args()

    h5, he_zarr = resolve_paths(args.section)
    print(f"[load] {args.section}\n       {h5}")
    a = ad.read_h5ad(h5)
    # var index is gene_name (fallback to a gene_name column if present)
    if "gene_name" in a.var.columns:
        a.var_names = [str(g) for g in a.var["gene_name"].values]
    a.var_names = [str(g) for g in a.var_names]
    a.var_names_make_unique()
    print(f"[load] {a.n_obs:,} cells x {a.n_vars:,} genes (X already log-normalized)")

    sig = SIGNATURES[args.signature]
    M = [g for g in sig["M"] if g in a.var_names]
    E = [g for g in sig["E"] if g in a.var_names]
    print(f"[signature] {args.signature}: {sig['label']}  ({sig['source']})")
    print(f"[markers] M {len(M)}/{len(sig['M'])}: {M}")
    print(f"[markers] E {len(E)}/{len(sig['E'])}: {E}")

    # EMT score (Mak et al. 2016 definition): per-gene z-score across cells, then
    # average within each pole; axis = mean(z[M]) - mean(z[E])  (>0 = mesenchymal).
    # Mv/Ev are already in comparable z-units, so the axis is symmetric by construction.
    Mv = zscore_mean(a, M)
    Ev = zscore_mean(a, E)
    axis = Mv - Ev
    a.obs["emt_axis"] = axis

    # "signal" = cell enriched in at least one pole (>80th pct of that pole's score);
    # everything else is low-signal parenchyma.
    m_thr, e_thr = np.quantile(Mv, 0.80), np.quantile(Ev, 0.80)
    is_signal = (Mv > m_thr) | (Ev > e_thr)
    lab = np.full(a.n_obs, "parenchyma / low-signal", dtype=object)
    lab[is_signal & (Mv >= Ev)] = "EMT+ (mesenchymal)"
    lab[is_signal & (Mv < Ev)] = "EMT- (epithelial)"
    a.obs["emt_region"] = lab
    print(f"[thresholds] M80={m_thr:.3f} E80={e_thr:.3f}")
    for k in ["EMT- (epithelial)", "parenchyma / low-signal", "EMT+ (mesenchymal)"]:
        print(f"   {k}: {(lab==k).sum():,}  ({(lab==k).mean():.1%})")

    # cancer cells + full cell type from the RCC label atlas, joined by cell_id
    cancer, celltype, labeled = load_cancer_mask(
        args.section, [str(c) for c in a.obs_names], ct_level=args.celltype_level)
    if cancer is not None:
        print(f"[cancer] {CANCER_CT}: {cancer.sum():,} cells ({cancer.mean():.1%}); "
              f"{labeled.sum():,}/{a.n_obs:,} cells labeled")
    elif args.require_cancer_labels:
        print(f"[skip] {args.section} has no usable cancer labels; "
              f"--require_cancer_labels set -> not plotting")
        return

    # --- EMT+ vs EMT- cell-type composition (only where atlas labels exist) ---
    # For each EMT pole, what fraction of its (labeled) cells is each cell type?
    # comp_types: cell types ordered by overall abundance; pos/neg_frac: within-pole
    # proportions; enrich: log2(EMT+ frac / EMT- frac) per type (diverging readout).
    comp_types = pos_frac = neg_frac = enrich = None
    pos_n = neg_n = 0
    if celltype is not None:
        ep, mn = "EMT- (epithelial)", "EMT+ (mesenchymal)"
        pos_sel = (lab == mn) & labeled        # EMT+ cells that carry an atlas label
        neg_sel = (lab == ep) & labeled        # EMT- cells that carry an atlas label
        pos_n, neg_n = int(pos_sel.sum()), int(neg_sel.sum())
        ct_pos = celltype[pos_sel]
        ct_neg = celltype[neg_sel]
        # rank cell types by combined abundance across the two poles (most common first)
        all_ct = np.concatenate([ct_pos, ct_neg]) if (pos_n + neg_n) else np.array([], dtype=object)
        uniq, cnt = np.unique(all_ct.astype(str), return_counts=True)
        comp_types = list(uniq[np.argsort(-cnt)])

        def _frac(ct_arr, n):
            if n == 0:
                return np.zeros(len(comp_types))
            vc = {t: 0 for t in comp_types}
            u, c = np.unique(ct_arr.astype(str), return_counts=True)
            vc.update(dict(zip(u, c)))
            return np.array([vc[t] for t in comp_types]) / n

        pos_frac = _frac(ct_pos, pos_n)
        neg_frac = _frac(ct_neg, neg_n)
        eps = 1e-3  # pseudo-count on the proportion scale to bound the log-ratio
        enrich = np.log2((pos_frac + eps) / (neg_frac + eps))
        print(f"[composition] level={args.celltype_level}  "
              f"EMT+ labeled={pos_n:,}  EMT- labeled={neg_n:,}")
        print(f"   {'cell type':<32} {'EMT+%':>8} {'EMT-%':>8} {'log2(+/-)':>10}")
        for t, pf, nf, en in sorted(zip(comp_types, pos_frac, neg_frac, enrich),
                                    key=lambda r: -abs(r[3])):
            print(f"   {t:<32} {pf*100:>7.1f}% {nf*100:>7.1f}% {en:>+10.2f}")

    # --- per-gene expression by cell type (heatmaps under the EMT-score row) ---
    # mean log-norm expression of each E/M marker within each atlas cell type
    # (labeled cells only). ct_order = cell types by abundance; Mmat/Emat are
    # (n_genes x n_celltypes) and get z-scored per gene at plot time.
    ct_order = Mmat = Emat = None
    if celltype is not None:
        ctv = celltype[labeled].astype(str)
        u, c = np.unique(ctv, return_counts=True)
        ct_order = list(u[np.argsort(-c)])

        def ct_expr(genes):
            if not genes:
                return None
            Xg = a[:, genes].X
            Xg = Xg.toarray() if hasattr(Xg, "toarray") else np.asarray(Xg)
            mat = np.full((len(genes), len(ct_order)), np.nan)
            for j, ct in enumerate(ct_order):
                sel = celltype == ct
                if sel.any():
                    mat[:, j] = np.asarray(Xg[sel]).mean(0)
            return mat

        Mmat, Emat = ct_expr(M), ct_expr(E)

    xy = a.obsm["spatial"]
    # random draw order so no class is hidden by z-order overplotting
    order = np.random.default_rng(0).permutation(a.n_obs)
    x, y = xy[order, 0], xy[order, 1]
    axis_o = axis[order]; lab_o = lab[order]
    Mo, Eo = Mv[order], Ev[order]
    cancer_o = cancer[order] if cancer is not None else None
    # group outputs by authoritative tissue_type (sample_metadata.csv only).
    # Sections absent from the metadata fall into "unknown" rather than a
    # filename-parsed guess (split('-') is never used for tissue).
    tissue = tissue_of(args.section) or "unknown"
    outdir = Path(args.outdir) / tissue
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load H&E up front: the bottom row of the combined figure needs it ---
    he = None
    if he_zarr and Path(he_zarr).exists():
        ds = args.he_downsample
        print(f"[he] loading {he_zarr} (downsample {ds})")
        he = load_he(he_zarr, ds)
        H, W = he.shape[:2]
        he_extent = [0, W * ds, H * ds, 0]   # downsampled image -> full-res coords
    else:
        print(f"[he] no H&E zarr for {args.section}, bottom row left blank")

    def rlim(v):  # robust 2-98 pct limits
        return np.quantile(v, 0.02), np.quantile(v, 0.98)

    def scatter(ax, c, cmap, norm=None, title="", cbar=True):
        sm = ax.scatter(x, y, c=c, s=args.point_size, cmap=cmap, norm=norm,
                        linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        for s in ax.spines.values(): s.set_visible(False)
        if cbar:
            cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
        return sm

    # combined figure: stacked rows, each a subfigure so it can carry its own header.
    # row0 (H&E + cancer) is unheaded; the rest get an EMT-theme header. The gene x
    # cell-type heatmap row is given extra height (33 M genes + rotated x labels).
    fig = plt.figure(figsize=(12, 25))
    fig.suptitle(f"EMT axis [{sig['label']}] - {args.section}  ({a.n_obs:,} cells)",
                 fontsize=14, fontweight="bold", y=0.998)
    subfigs = fig.subfigures(7, 1, hspace=0.0,
                             height_ratios=[1, 1, 1.8, 1, 1, 1, 1])
    ROW_HEADERS = [None, "EMT score", "E/M gene expression by cell type",
                   "EMT axis & regions", "H&E overlay with EMT", "EMT distribution",
                   "EMT+/- cell-type composition"]
    axrows = []
    for sf, hdr in zip(subfigs, ROW_HEADERS):
        if hdr:
            sf.suptitle(hdr, fontsize=12, fontweight="bold", x=0.02, ha="left")
        axrows.append(sf.subplots(1, 2))
        sf.subplots_adjust(wspace=0.22)

    def draw_he(ax):  # H&E background in full-res spatial coords
        ax.imshow(he, extent=he_extent, interpolation="nearest")
        ax.set_xlim(0, W * ds); ax.set_ylim(H * ds, 0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    def blank(ax, msg=None):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if msg:
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    fontsize=11, transform=ax.transAxes)

    amax = np.quantile(np.abs(axis), 0.98)

    # ===== row 0 (no header): H&E, cancer cells on H&E =====
    ax_he0, ax_can = axrows[0]
    if he is not None:
        draw_he(ax_he0); ax_he0.set_title("H&E", fontsize=10)
        draw_he(ax_can); ax_can.set_title("Cancer cells on H&E", fontsize=10)
        if cancer_o is not None:
            ax_can.scatter(x[cancer_o], y[cancer_o], s=args.point_size, c="#39FF14",
                           linewidths=0, rasterized=True, alpha=0.8,
                           label=f"{CANCER_CT} (n={int(cancer.sum()):,})")
            ax_can.legend(loc="upper right", fontsize=6, markerscale=3, framealpha=0.9)
        else:
            ax_can.text(0.5, 0.5, "no cancer labels\nfor this section", ha="center",
                        va="center", fontsize=10, color="white", transform=ax_can.transAxes)
    else:
        blank(ax_he0, "no H&E available"); blank(ax_can)

    # ===== row 1 (EMT score): Mesenchymal score, Epithelial score =====
    mlo, mhi = rlim(Mv)
    scatter(axrows[1][0], Mo, "magma", None,
            f"Mesenchymal score ({len(M)} genes)").set_clim(mlo, mhi)
    elo, ehi = rlim(Ev)
    scatter(axrows[1][1], Eo, "viridis", None,
            f"Epithelial score ({len(E)} genes)").set_clim(elo, ehi)

    # ===== row 2 (E/M gene expression by cell type): M heatmap, E heatmap =====
    ax_mh, ax_eh = axrows[2]

    def draw_heatmap(ax, mat, genes, title):
        if mat is None or ct_order is None:
            blank(ax, "no cell-type labels\nfor this section"); return
        # z-score per gene (row) across cell types so each marker's expression
        # pattern is visible despite differing per-gene baselines
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True)
        Z = (mat - mu) / (sd + 1e-9)
        im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(np.arange(len(ct_order)))
        ax.set_xticklabels(ct_order, fontsize=6, rotation=90)
        ax.set_yticks(np.arange(len(genes)))
        ax.set_yticklabels(genes, fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(length=0)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=6); cb.set_label("row z-score", fontsize=7)

    draw_heatmap(ax_mh, Mmat, M, f"Mesenchymal markers ({len(M)} genes) x cell type")
    draw_heatmap(ax_eh, Emat, E, f"Epithelial markers ({len(E)} genes) x cell type")
    # room for rotated cell-type x labels + clear the row header above
    subfigs[2].subplots_adjust(bottom=0.26, top=0.84, wspace=0.40)

    # ===== row 3 (EMT axis & regions): EMT axis, EMT+/- regions =====
    scatter(axrows[3][0], axis_o, "RdBu_r",
            TwoSlopeNorm(vcenter=0, vmin=-amax, vmax=amax),
            "EMT axis = z(M) - z(E)  (red=mesenchymal)")

    # categorical region map: parenchyma first (bottom), signal cells on top
    axc = axrows[3][1]
    cmap3 = [("parenchyma / low-signal", "#DDDDDD"),
             ("EMT- (epithelial)", "#2166AC"),
             ("EMT+ (mesenchymal)", "#B2182B")]
    for k, col in cmap3:
        m = lab_o == k
        axc.scatter(x[m], y[m], s=args.point_size, c=col, linewidths=0,
                    rasterized=True, label=f"{k} (n={(lab==k).sum():,})")
    axc.set_aspect("equal"); axc.invert_yaxis()
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title("EMT+/- regions (>80th-pct signal cells)", fontsize=10)
    for s in axc.spines.values(): s.set_visible(False)
    axc.legend(loc="upper right", fontsize=6, markerscale=3, framealpha=0.9)

    # ===== row 4 (H&E overlay with EMT): the two H&E overlays =====
    ax_o1, ax_o2 = axrows[4]
    if he is not None:
        draw_he(ax_o1); draw_he(ax_o2)
        # left: continuous EMT axis over H&E (signal cells only, parenchyma shows tissue)
        sig_o = lab_o != "parenchyma / low-signal"
        sm = ax_o1.scatter(x[sig_o], y[sig_o], c=axis_o[sig_o], cmap="RdBu_r",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-amax, vmax=amax),
                           s=args.point_size, linewidths=0, rasterized=True, alpha=0.75)
        ax_o1.set_title("EMT axis on H&E  (red=mesenchymal)", fontsize=10)
        cb = ax_o1.figure.colorbar(sm, ax=ax_o1, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        # right: EMT+/- categorical over H&E (parenchyma left as bare tissue)
        for k, col in [("EMT- (epithelial)", "#2166AC"), ("EMT+ (mesenchymal)", "#B2182B")]:
            m = lab_o == k
            ax_o2.scatter(x[m], y[m], s=args.point_size, c=col, linewidths=0,
                          rasterized=True, alpha=0.75, label=f"{k} (n={(lab==k).sum():,})")
        ax_o2.set_title("EMT+/- regions on H&E", fontsize=10)
        ax_o2.legend(loc="upper right", fontsize=6, markerscale=3, framealpha=0.9)
    else:
        blank(ax_o1, "no H&E available"); blank(ax_o2)

    # ===== row 5 (EMT distribution): histogram, E-M state space =====
    # histogram (signal cells only — parenchyma zero-peak masked)
    axh = axrows[5][0]
    sig_cells = lab != "parenchyma / low-signal"
    axh.hist(axis[sig_cells], bins=80, color="#777777")
    axh.axvline(0, color="k", ls=":", lw=0.8)
    axh.set_xlabel("EMT axis  z(M) - z(E)", fontsize=9)
    axh.set_ylabel("signal cells", fontsize=9)
    axh.set_title(f"EMT axis distribution (signal cells, n={sig_cells.sum():,})", fontsize=10)
    axh.tick_params(labelsize=7)
    for s in ["top", "right"]: axh.spines[s].set_visible(False)

    # E vs M state space, colored by axis
    axs = axrows[5][1]
    sub = np.random.default_rng(1).choice(a.n_obs, min(40000, a.n_obs), replace=False)
    axs.scatter(Ev[sub], Mv[sub], c=axis[sub], cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=0, vmin=-amax, vmax=amax),
                s=2, linewidths=0, rasterized=True)
    axs.set_xlabel("Epithelial score", fontsize=9)
    axs.set_ylabel("Mesenchymal score", fontsize=9)
    axs.set_title("E - M state space", fontsize=10); axs.tick_params(labelsize=7)
    for s in ["top", "right"]: axs.spines[s].set_visible(False)

    # ===== row 6 (cell-type composition): grouped fractions, log2 enrichment =====
    ax_comp, ax_enr = axrows[6]
    if comp_types:
        yy = np.arange(len(comp_types))[::-1]  # most abundant at top
        h = 0.4
        # left: side-by-side proportions per cell type (EMT- blue, EMT+ red)
        ax_comp.barh(yy + h/2, neg_frac * 100, height=h, color="#2166AC",
                     label=f"EMT- epithelial (n={neg_n:,})")
        ax_comp.barh(yy - h/2, pos_frac * 100, height=h, color="#B2182B",
                     label=f"EMT+ mesenchymal (n={pos_n:,})")
        ax_comp.set_yticks(yy); ax_comp.set_yticklabels(comp_types, fontsize=7)
        ax_comp.set_xlabel("% of labeled cells in pole", fontsize=9)
        ax_comp.set_title(f"Cell-type composition ({args.celltype_level})", fontsize=10)
        ax_comp.tick_params(labelsize=7)
        ax_comp.legend(loc="lower right", fontsize=7, framealpha=0.9)
        for s in ["top", "right"]: ax_comp.spines[s].set_visible(False)

        # right: log2 enrichment EMT+ vs EMT- (red bar = enriched in EMT+ pole)
        cols = ["#B2182B" if e >= 0 else "#2166AC" for e in enrich]
        ax_enr.barh(yy, enrich, height=0.6, color=cols)
        ax_enr.axvline(0, color="k", lw=0.8)
        # y order matches the left panel; drop the duplicate cell-type labels here so
        # they don't collide with the left panel's bars
        ax_enr.set_yticks(yy); ax_enr.set_yticklabels([])
        ax_enr.set_xlabel("log2(EMT+ frac / EMT- frac)", fontsize=9)
        ax_enr.set_title("Enrichment: EMT+ (red) vs EMT- (blue)", fontsize=10)
        ax_enr.tick_params(labelsize=7)
        for s in ["top", "right"]: ax_enr.spines[s].set_visible(False)
    else:
        blank(ax_comp, "no cell-type labels\nfor this section"); blank(ax_enr)
    # reserve room under the bottom row for the footnote, drop the axes down so the
    # subplot titles clear the row header, and widen the gap between the two panels
    subfigs[6].subplots_adjust(bottom=0.22, top=0.80, wspace=0.30)

    fig.text(0.5, 0.012, f"EMT signature: {sig['label']}  -  source: {sig['source']}",
             ha="center", fontsize=7, color="#555555")
    stem = outdir / f"emt_combined_{args.section}_{args.signature}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
