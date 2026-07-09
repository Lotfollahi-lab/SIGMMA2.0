"""2D epithelial / mesenchymal scoring for RCC, restricted to the tumour-epithelium compartment.

Pivot away from the 1D (M - E) axis (scripts/emt_signature_spatial.py): a subtraction
collapses hybrid co-expressors (E-high AND M-high) onto ~0, indistinguishable from
low-signal parenchyma, and its z-score zero is the section's population mean
(composition-dependent). Here we keep E and M as *separate* axes and use a
control-gene baseline (scanpy score_genes / Seurat AddModuleScore style), so that
0 means "enriched above matched-expression background genes" rather than "above the
slide average". That zero is composition-robust and comparable across slides.

Scoring is computed on ALL cells (so the control-gene baseline is well estimated and
the normal-epithelium / fibroblast anchors live in the same space) but the EMT
region analysis is restricted to Tumour-epithelium cells.

  - E score: 8 panel-present epithelial genes
  - M score: 14 panel-present mesenchymal / EMT genes
  - quadrant (within tumour epi): E>0 & M<=0 -> epithelial; E<=0 & M>0 -> mesenchymal;
    E>0 & M>0 -> hybrid (partial EMT, the "ambiguous" region); else low-signal.

Anchors: epithelial basal = normal Nephron epithelium centroid; mesenchymal pole =
Fibroblast centroid (both from the RCC label atlas).
"""
import argparse
from pathlib import Path
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt_signature_spatial import (resolve_paths, load_cancer_mask, load_he,
                                   CANCER_CT, SIGNATURES)

# --- gene sets ---
# "curated": EMT-program-restricted panel-present set (RCC Xenium 5K, 2026-06-02),
#   deliberately excludes fibroblast/ECM-identity genes.
# "union"/"core16"/"mak2016": the full literature signatures from the 1D analysis
#   (emt_signature_spatial.SIGNATURES); only panel-present genes are scored.
CURATED_E = ["CDH1", "EPCAM", "CLDN7", "OCLN", "CDH16", "TJP1", "GRHL2", "MUC1"]
CURATED_M = ["CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "TWIST2",
             "MMP9", "PDGFRA", "PDGFRB", "THY1", "POSTN", "SERPINE1", "ITGB6"]

# "emt_program": clean EMT *program* set that EXCLUDES ECM / fibroblast-IDENTITY
#   genes (collagens, FAP, PDGFRA/B, THY1, POSTN, LRRC15, ...) to avoid the stromal-
#   contamination confound when scoring tumour-epithelium in imaging-based ST:
#     - bulk / single-cell EMT signatures are confounded by stromal content and must
#       be decoupled  (Tyler & Tirosh, Nat Commun 2021, PMID 33972543);
#     - Xenium segmentation / transcript spillover makes neighbouring fibroblast
#       transcripts appear as false E+M co-expression inside tumour cells
#       (Xenium signal-contamination, Nat Methods 2026; Baysor, Nat Biotechnol 2022).
#   M = canonical EMT transcription factors + cadherin switch (CDH2) + two tumour-
#   cell-autonomous effectors (AXL, ITGB6) added to stabilise the otherwise sparse
#   EMT-TF signal in Xenium. E = canonical epithelial junction / identity markers.
#   NOTE: user is independently verifying the literature for these gene choices.
EMT_PROGRAM_E = ["CDH1", "EPCAM", "CLDN7", "OCLN", "TJP1"]
EMT_PROGRAM_M = ["SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2",
                 "FOXC2", "PRRX1", "CDH2", "AXL", "ITGB6"]


def gene_sets(name):
    """Return (E_genes, M_genes) for a --gene_set choice (full, pre-panel-filter)."""
    if name == "curated":
        return CURATED_E, CURATED_M
    if name == "emt_program":
        return EMT_PROGRAM_E, EMT_PROGRAM_M
    return SIGNATURES[name]["E"], SIGNATURES[name]["M"]

# normal kidney epithelium (epithelial basal anchor) and fibroblast (M-pole anchor)
NEPHRON_EPI = {"Proximal tubule", "Distal tubule", "Mixed tubular", "Podocyte",
               "Juxtaglomerular cell"}
FIBROBLAST = {"Resident fibroblast", "Cancer associated fibroblast"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True)
    ap.add_argument("--gene_set", default="union",
                    choices=["union", "curated", "emt_program", "core16", "mak2016"],
                    help="E/M gene set: 'union' = full 1D literature union (default), "
                         "'emt_program' = clean EMT-TF set, no ECM/fibroblast genes, "
                         "'curated' = EMT-program-restricted set")
    ap.add_argument("--ctrl_size", type=int, default=50,
                    help="control gene-set size for score_genes baseline")
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--outdir", default="figures/cross_tissue_retrieval/emt_2d")
    ap.add_argument("--anchor_file", default=None,
                    help="JSON of cohort-pooled anchor centroids "
                         "(emt_2d_reduce_anchors.py output); when given, the E-anchor "
                         "(nephron-epi) and M-anchor (fibroblast) ★ markers use these "
                         "cross-section centroids instead of this section's own (which "
                         "are noisy where normal epithelium is scarce). Per-section "
                         "anchor clouds are still drawn from this section's cells.")
    ap.add_argument("--dump_anchor_stats", default=None,
                    help="directory to write this section's anchor-cell sums+counts "
                         "(<section>_<gene_set>.json) for cross-section pooling")
    ap.add_argument("--dump_cell_scores", default=None,
                    help="directory to write per-cell tumour-epi (E,M) scores + EMT-axis "
                         "projection + coords (<section>_<gene_set>_cells.parquet) so "
                         "dispersion / drift / spatial-front metrics can be recomputed "
                         "without re-scoring")
    ap.add_argument("--anchors_only", action="store_true",
                    help="compute scores + dump anchor stats, then exit before plotting "
                         "(pass 1 of the pooled-anchor workflow)")
    ap.add_argument("--min_tumour_cells", type=int, default=1000,
                    help="skip the figure when the tumour-epithelium compartment has "
                         "fewer than this many cells: the tumour-epi-restricted 2D EMT "
                         "analysis is uninformative for near-normal slices (e.g. "
                         "CV9-KID-0-FT-5-S4, 113 tumour cells, where the plot is just the "
                         "normal-epi anchor cloud). The anchor-pool dump (--anchors_only) "
                         "runs BEFORE this check, so excluded slices still contribute "
                         "their normal-epi / fibroblast cells to the pooled anchors.")
    args = ap.parse_args()

    h5, he_zarr = resolve_paths(args.section)
    print(f"[load] {args.section}\n       {h5}")
    a = ad.read_h5ad(h5)
    if "gene_name" in a.var.columns:
        a.var_names = [str(g) for g in a.var["gene_name"].values]
    a.var_names = [str(g) for g in a.var_names]
    a.var_names_make_unique()
    print(f"[load] {a.n_obs:,} cells x {a.n_vars:,} genes (X already log-normalized)")

    E_full, M_full = gene_sets(args.gene_set)
    E = [g for g in E_full if g in a.var_names]
    M = [g for g in M_full if g in a.var_names]
    print(f"[gene_set] {args.gene_set}")
    print(f"[markers] E {len(E)}/{len(E_full)}: {E}")
    print(f"[markers] M {len(M)}/{len(M_full)}: {M}")

    # control-gene baseline scores (computed on ALL cells); 0 = above matched bg
    sc.tl.score_genes(a, E, score_name="E_score", ctrl_size=args.ctrl_size, random_state=0)
    sc.tl.score_genes(a, M, score_name="M_score", ctrl_size=args.ctrl_size, random_state=0)
    Es = a.obs["E_score"].to_numpy()
    Ms = a.obs["M_score"].to_numpy()

    # compartments from the RCC label atlas
    cancer, celltype, labeled = load_cancer_mask(
        args.section, [str(c) for c in a.obs_names])
    if cancer is None or labeled.sum() == 0:
        raise SystemExit(f"[skip] {args.section}: no usable cancer/compartment labels")
    ct = np.array([str(c) for c in celltype], dtype=object)
    is_tum = cancer & labeled
    is_nep = np.array([c in NEPHRON_EPI for c in ct]) & labeled
    is_fib = np.array([c in FIBROBLAST for c in ct]) & labeled
    print(f"[compartments] tumour-epi {is_tum.sum():,}  nephron-epi {is_nep.sum():,}  "
          f"fibroblast {is_fib.sum():,}  (labeled {labeled.sum():,}/{a.n_obs:,})")

    # per-section anchor centroids in (E, M) space (noisy where a compartment is scarce)
    nep_c = (Es[is_nep].mean(), Ms[is_nep].mean()) if is_nep.any() else (np.nan, np.nan)
    fib_c = (Es[is_fib].mean(), Ms[is_fib].mean()) if is_fib.any() else (np.nan, np.nan)
    tum_c = (Es[is_tum].mean(), Ms[is_tum].mean())
    print(f"[anchors:section] nephron-epi(E,M)=({nep_c[0]:+.3f},{nep_c[1]:+.3f})  "
          f"fibroblast=({fib_c[0]:+.3f},{fib_c[1]:+.3f})  tumour-epi=({tum_c[0]:+.3f},{tum_c[1]:+.3f})")

    # pass 1: write this section's anchor-cell sums + counts for cross-section pooling.
    # store sums (not means) so the reducer can take the exact cell-weighted centroid.
    if args.dump_anchor_stats:
        import json
        dd = Path(args.dump_anchor_stats); dd.mkdir(parents=True, exist_ok=True)
        stats = {
            "section": args.section, "gene_set": args.gene_set,
            "nephron_epi": {"sum_E": float(Es[is_nep].sum()), "sum_M": float(Ms[is_nep].sum()),
                            "n": int(is_nep.sum())},
            "fibroblast": {"sum_E": float(Es[is_fib].sum()), "sum_M": float(Ms[is_fib].sum()),
                           "n": int(is_fib.sum())},
        }
        dpath = dd / f"{args.section}_{args.gene_set}.json"
        dpath.write_text(json.dumps(stats, indent=2))
        print(f"[dump] anchor stats -> {dpath}")

    # use cohort-pooled anchors (★ markers) when provided; the per-section anchor
    # CLOUDS still come from this section's cells (so absence is visible), but the
    # reference stars are the stable cross-section centroids.
    anchor_src = "section"
    if args.anchor_file and Path(args.anchor_file).exists():
        import json
        anc = json.loads(Path(args.anchor_file).read_text())
        nep_c = (anc["nephron_epi"]["E"], anc["nephron_epi"]["M"])
        fib_c = (anc["fibroblast"]["E"], anc["fibroblast"]["M"])
        anchor_src = "cohort-pooled"
        print(f"[anchors:{anchor_src}] file={args.anchor_file}  "
              f"nephron-epi(E,M)=({nep_c[0]:+.3f},{nep_c[1]:+.3f})  "
              f"fibroblast=({fib_c[0]:+.3f},{fib_c[1]:+.3f})  "
              f"(n_cells nep={anc['nephron_epi'].get('n_cells','?')}, "
              f"fib={anc['fibroblast'].get('n_cells','?')})")

    if args.anchors_only:
        print(f"[anchors_only] {args.section}: done (no figures)")
        return

    # auto-exclude near-normal slices: too few tumour-epi cells -> the tumour-epi
    # quadrant analysis is uninformative (the figure would show only the normal-epi
    # anchor cloud). Runs after the anchor dump so the pool still gets these cells.
    n_tum = int(is_tum.sum())
    if n_tum < args.min_tumour_cells:
        raise SystemExit(f"[skip] {args.section}: only {n_tum:,} tumour-epi cells "
                         f"(< --min_tumour_cells {args.min_tumour_cells}); 2D EMT "
                         f"analysis uninformative, no figure produced")

    # quadrant label WITHIN tumour epithelium (baseline-anchored threshold at 0)
    et, mt = Es[is_tum], Ms[is_tum]
    quad = np.full(et.shape, "low-signal", dtype=object)
    quad[(et > 0) & (mt <= 0)] = "epithelial"
    quad[(et <= 0) & (mt > 0)] = "mesenchymal"
    quad[(et > 0) & (mt > 0)] = "hybrid"
    print("[tumour-epi quadrants]")
    for k in ["epithelial", "hybrid", "mesenchymal", "low-signal"]:
        n = int((quad == k).sum())
        print(f"   {k:<12} {n:>8,}  ({n/len(quad):.1%})")

    # ===== intra-tumour EMT heterogeneity (dispersion / plasticity) =====
    # Pivot away from "EMT proportion" (fraction of tumour-epi cells in the
    # hybrid/mesenchymal quadrants): that count is unrelated to metastatic severity
    # AND is confounded by the per-section control-gene baseline (where 0 sits shifts
    # the quadrant boundaries). The hypothesis here is that metastatic competence
    # tracks how SPREAD the tumour-epithelium is across the EMT spectrum (plasticity),
    # not its mean state. Variance-based spreads are translation-invariant, so a
    # per-section baseline shift moves the whole cloud but leaves these metrics intact.
    EM = np.c_[et, mt]                                  # tumour-epi cells in (E, M)
    cov = np.cov(EM, rowvar=False)                      # 2x2 covariance
    total_var = float(cov[0, 0] + cov[1, 1])            # var_E + var_M = trace(cov)
    gen_var = float(np.sqrt(max(np.linalg.det(cov), 0.0)))  # sqrt|cov| ~ 2D spread area
    std_E, std_M = float(et.std()), float(mt.std())
    # spread along the EMT progression axis (E-anchor -> M-anchor): the biologically
    # relevant direction. With cohort-pooled anchors (--anchor_file) this axis is the
    # SAME across sections, so the projection std is directly comparable section-to-
    # section. Falls back to the raw M axis if an anchor is missing.
    if np.isfinite(nep_c[0]) and np.isfinite(fib_c[0]):
        axis = np.array([fib_c[0] - nep_c[0], fib_c[1] - nep_c[1]])
    else:
        axis = np.array([0.0, 1.0])
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    proj = EM @ axis                                    # EMT pseudo-coordinate per cell
    emt_axis_std = float(proj.std())
    emt_axis_iqr = float(np.subtract(*np.percentile(proj, [75, 25])))
    # quadrant-state entropy: multi-state co-occupancy (high = cells spread across
    # several EMT states = plasticity). NOT shift-invariant -> baseline-dependent,
    # reported for comparison with the continuous spreads above.
    fr = np.array([(quad == k).mean()
                   for k in ["epithelial", "hybrid", "mesenchymal", "low-signal"]])
    nz = fr[fr > 0]
    quad_entropy = float(-(nz * np.log2(nz)).sum())     # bits, 0..2
    print("[tumour-epi dispersion]")
    print(f"   std_E             {std_E:.4f}")
    print(f"   std_M             {std_M:.4f}")
    print(f"   total_var         {total_var:.4f}   (var_E+var_M, shift-invariant)")
    print(f"   gen_var           {gen_var:.5f}   (sqrt|cov|, 2D spread area)")
    print(f"   emt_axis_std      {emt_axis_std:.4f}   (spread along E->M anchor axis)")
    print(f"   emt_axis_iqr      {emt_axis_iqr:.4f}   (robust EMT-axis spread)")
    print(f"   quad_entropy      {quad_entropy:.4f}   (bits, baseline-dependent)")

    # spatial coords
    if "spatial" in a.obsm:
        xy = np.asarray(a.obsm["spatial"])
    else:
        xy = np.c_[a.obs["x_centroid"].to_numpy(), a.obs["y_centroid"].to_numpy()]
    xt = xy[is_tum]

    # optional: persist per-cell tumour-epi scores + coords so drift / spatial-front
    # metrics (the other EMT-severity candidates) can be computed later WITHOUT
    # re-scoring all 54 sections.
    if args.dump_cell_scores:
        import pandas as pd
        dc = Path(args.dump_cell_scores); dc.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "E_score": et, "M_score": mt, "emt_axis": proj, "quad": quad,
            "x": xt[:, 0], "y": xt[:, 1],
        }).to_parquet(dc / f"{args.section}_{args.gene_set}_cells.parquet", index=False)
        print(f"[dump] per-cell tumour-epi scores -> "
              f"{dc / f'{args.section}_{args.gene_set}_cells.parquet'}")

    # ===== figure: (left) 2D E/M state space, (right) hybrid spatial map =====
    from matplotlib.lines import Line2D
    QCOL = {"epithelial": "#1f77b4", "hybrid": "#FFB300",
            "mesenchymal": "#2ca02c", "low-signal": "#cccccc"}
    # EMT interpretation of each tumour-epi quadrant (shown in the legend)
    QLAB = {"epithelial": "epithelial (no EMT)",
            "hybrid": "hybrid (partial EMT)",
            "mesenchymal": "mesenchymal (full EMT)",
            "low-signal": "low-signal (E−/M−, ambiguous)"}
    # box aspect: match the scatter height to the equal-aspect spatial map
    # (kept — reused by the combined figure below).
    dxw = float(xy[:, 0].max() - xy[:, 0].min())
    dyh = float(xy[:, 1].max() - xy[:, 1].min())
    map_box = (dyh / dxw) if dxw else 1.0

    def dot(color, label, alpha=0.9):  # proxy legend handle (round marker)
        return Line2D([0], [0], marker="o", linestyle="", markersize=6, alpha=alpha,
                      markerfacecolor=color, markeredgecolor="none", label=label)

    # ===== [DISABLED 2026-06-03] standalone 2D E/M state-space figure =====
    # Only emt_2d_combined_<section> is produced now (user request); this block
    # (which saved emt_2d_<section>_<gene_set>.png/pdf) is commented out.
    # fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    # ax = axs[0]
    # # background = the OTHER cell compartments whose centroids define the anchors
    # # (normal nephron-epi -> E-anchor, fibroblast -> M-anchor); NOT tumour cells.
    # ax.scatter(Es[is_nep], Ms[is_nep], s=1, c="#88aacc", alpha=0.15)
    # ax.scatter(Es[is_fib], Ms[is_fib], s=1, c="#88cc88", alpha=0.15)
    # # tumour-epi cells colored by quadrant (z-order: hybrid drawn last, on top)
    # for k in ["low-signal", "epithelial", "mesenchymal", "hybrid"]:
    #     ax.scatter(et[quad == k], mt[quad == k], s=2, c=QCOL[k], alpha=0.4)
    # ax.axhline(0, color="k", lw=0.6, ls="--"); ax.axvline(0, color="k", lw=0.6, ls="--")
    # for (cx, cy), name, col in [(nep_c, "E-anchor", "#1f4e79"),
    #                             (fib_c, "M-anchor", "#1f7a1f")]:
    #     if np.isfinite(cx):
    #         ax.scatter([cx], [cy], s=180, marker="*", c=col, edgecolor="white",
    #                    linewidth=1.2, zorder=5)
    #         ax.annotate(name, (cx, cy), fontsize=8, fontweight="bold", color=col,
    #                     xytext=(4, 4), textcoords="offset points")
    # ax.set_xlabel("E score (control-gene baseline)", fontsize=10)
    # ax.set_ylabel("M score (control-gene baseline)", fontsize=10)
    # ax.set_title(f"2D E/M state space — tumour epithelium\n{args.section}  "
    #              f"[{args.gene_set}: E{len(E)}/M{len(M)}]", fontsize=10)
    # # legend: tumour-epi states ordered by EMT progression, then the anchor sources.
    # # ★ anchors are centroids of the OTHER (non-tumour) reference cells.
    # tum_entries = [dot(QCOL[k], f"tumour: {QLAB[k]} ({(quad == k).mean():.0%})")
    #                for k in ["epithelial", "hybrid", "mesenchymal", "low-signal"]]
    # anchor_entries = [
    #     dot("#88aacc", "E-anchor cells: normal nephron-epi", alpha=0.5),
    #     dot("#88cc88", "M-anchor cells: fibroblast", alpha=0.5),
    #     Line2D([0], [0], marker="*", linestyle="", markersize=11, markerfacecolor="#444444",
    #            markeredgecolor="white",
    #            label=("★ anchor = cohort-pooled centroid (all sections, not tumour)"
    #                   if anchor_src == "cohort-pooled"
    #                   else "★ anchor = centroid of those cells (not tumour)")),
    # ]
    # ax.legend(handles=tum_entries + anchor_entries, fontsize=6,
    #           loc="upper left", framealpha=0.9, title="tumour-epi EMT state  /  anchors",
    #           title_fontsize=6)
    # ax.set_box_aspect(map_box)  # same drawn height as the spatial map on the right
    #
    # ax = axs[1]
    # ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c="#eeeeee")  # all cells faint
    # # color every tumour cell by quadrant in ONE pass (cell order) so no class is
    # # drawn on top of the others — avoids hybrid looking artificially dominant.
    # ax.scatter(xt[:, 0], xt[:, 1], s=args.point_size, c=[QCOL[q] for q in quad],
    #            alpha=0.6, linewidths=0)
    # # box_aspect == data aspect gives uniform scaling (looks "equal") but fills the
    # # box, so it lines up with the left scatter instead of shrinking with whitespace.
    # ax.set_box_aspect(map_box); ax.invert_yaxis()
    # ax.set_title("Tumour-epi EMT region map (hybrid = partial EMT)", fontsize=10)
    # ax.legend(handles=[dot(QCOL[k], QLAB[k])
    #                    for k in ["epithelial", "hybrid", "mesenchymal", "low-signal"]],
    #           fontsize=7, loc="best")
    # ax.set_xticks([]); ax.set_yticks([])

    # `foot` (figure footnote) is kept — reused by the combined figure.
    foot = (
        "E / M score = scanpy score_genes (Seurat AddModuleScore style): mean log-normalised expression of the "
        f"E ({len(E)} genes) or M ({len(M)} genes) set minus a matched-expression control gene set "
        f"(ctrl_size={args.ctrl_size}); score > 0 = enriched above background, NOT above the slide mean.\n"
        "★ anchors are reference centroids in (E, M) score space, NOT tumour cells:  "
        "E-anchor = normal nephron epithelium  •  M-anchor = fibroblast (resident + cancer-associated).  "
        "Tumour-epithelium cells are scored relative to these epithelial / mesenchymal poles."
        + ("\n★ anchor stars are COHORT-POOLED across all labeled RCC sections (stable where this "
           "section's normal epithelium / fibroblast is scarce); the faint clouds are this section's own cells."
           if anchor_src == "cohort-pooled" else "")
    )
    # fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8, color="#444444")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # stem = outdir / f"emt_2d_{args.section}_{args.gene_set}"
    # fig.tight_layout(rect=[0, 0.05, 1, 1])
    # fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    # fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    # print(f"[saved] {stem}.png / .pdf")

    # `rlim` (robust color limits) is kept — reused by the combined figure.
    def rlim(v):  # robust 2-98 pct color limits
        lo, hi = np.percentile(v, [2, 98]); return lo, hi

    # ===== [DISABLED 2026-06-03] continuous EMT-score map figure =====
    # Only emt_2d_combined_<section> is produced now (user request); this block
    # (which saved emt_2d_scoremap_<section>_<gene_set>.png/pdf) is commented out.
    # fig2, axs2 = plt.subplots(1, 3, figsize=(19, 6.2))
    # xall = xy[:, 0]; yall = xy[:, 1]
    # order = np.argsort(np.abs(et) + np.abs(mt))  # draw strong-signal cells on top
    #
    # elo, ehi = rlim(et); mlo, mhi = rlim(mt)
    # for ax, val, lo, hi, cmap, title in [
    #     (axs2[0], et, elo, ehi, "Blues", f"E score ({len(E)} genes)"),
    #     (axs2[1], mt, mlo, mhi, "Reds", f"M score ({len(M)} genes)"),
    # ]:
    #     ax.scatter(xall, yall, s=0.4, c="#f0f0f0")
    #     sc_ = ax.scatter(xt[order, 0], xt[order, 1], s=args.point_size,
    #                      c=val[order], cmap=cmap, vmin=lo, vmax=hi)
    #     ax.figure.colorbar(sc_, ax=ax, fraction=0.046, pad=0.02)
    #     ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    #     ax.set_title(title, fontsize=11)
    #
    # # RGB blend: R = M, B = E (clipped at 0, scaled to robust max) -> hybrid = magenta
    # ax = axs2[2]
    # r = np.clip(mt, 0, None); b = np.clip(et, 0, None)
    # r = r / (np.percentile(r[r > 0], 98) + 1e-9) if (r > 0).any() else r
    # b = b / (np.percentile(b[b > 0], 98) + 1e-9) if (b > 0).any() else b
    # rgb = np.clip(np.c_[r, np.zeros_like(r), b], 0, 1)
    # ax.scatter(xall, yall, s=0.4, c="#101010")
    # ax.scatter(xt[:, 0], xt[:, 1], s=args.point_size, c=rgb)
    # ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    # ax.set_title("E+M blend  (blue=epithelial, red=mesenchymal, magenta=hybrid)", fontsize=11)
    # ax.set_facecolor("#101010")
    #
    # fig2.suptitle(f"EMT score (control-gene baseline) — tumour epithelium — {args.section}",
    #               fontsize=12)
    # fig2.tight_layout()
    # stem2 = outdir / f"emt_2d_scoremap_{args.section}_{args.gene_set}"
    # fig2.savefig(f"{stem2}.png", dpi=150, bbox_inches="tight")
    # fig2.savefig(f"{stem2}.pdf", bbox_inches="tight")
    # print(f"[saved] {stem2}.png / .pdf")

    # ===== combined figure (1D-style stacked rows, 2D-adapted) =====
    # rows: H&E + tumour location | E/M score (tumour-epi) | EMT score (scatter + map)
    #       | EMT distribution
    he = he_extent = None
    if he_zarr and Path(he_zarr).exists():
        ds = 8
        print(f"[he] loading {he_zarr} (downsample {ds})")
        he = load_he(he_zarr, ds)
        Hh, Ww = he.shape[:2]
        he_extent = [0, Ww * ds, Hh * ds, 0]
    else:
        print(f"[he] no H&E zarr for {args.section}, top row left blank")

    def draw_he(ax):
        ax.imshow(he, extent=he_extent, interpolation="nearest")
        ax.set_xlim(he_extent[0], he_extent[1]); ax.set_ylim(he_extent[2], he_extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    def spatial_blank(ax, title):  # faint all-cells outline, no H&E
        ax.scatter(xy[:, 0], xy[:, 1], s=0.4, c="#eeeeee")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    # compact height: each row's drawn map is ~(half-width * map_box) tall; size the
    # figure to that instead of a fixed 1/3 so wide-tissue rows don't leave big gaps.
    sub_w = 12 / 2.3                       # approx per-subplot width at figw=12
    row_h = sub_w * map_box + 1.1          # + header/title/colorbar padding
    figc = plt.figure(figsize=(12, max(8.0, row_h * 3 + 0.6)))
    figc.suptitle(f"EMT 2D [{args.gene_set}: E{len(E)}/M{len(M)}] — tumour epithelium — "
                  f"{args.section}  ({a.n_obs:,} cells)", fontsize=13, fontweight="bold")
    sfs = figc.subfigures(3, 1, hspace=0.03)
    HEADERS = [None, "E / M score (tumour epithelium)", "EMT score"]
    rax = []
    for sf, hdr in zip(sfs, HEADERS):
        if hdr:
            sf.suptitle(hdr, fontsize=12, fontweight="bold", x=0.02, ha="left")
        rax.append(sf.subplots(1, 2)); sf.subplots_adjust(wspace=0.22)

    # --- row 0: H&E | tumour-cell location ---
    ax0, ax1 = rax[0]
    if he is not None:
        draw_he(ax0); ax0.set_title("H&E", fontsize=10)
        draw_he(ax1)
        ax1.scatter(xt[:, 0], xt[:, 1], s=args.point_size, c="#39FF14", linewidths=0,
                    alpha=0.8, label=f"{CANCER_CT} (n={int(is_tum.sum()):,})")
        ax1.set_title("Tumour-cell location on H&E", fontsize=10)
        ax1.legend(loc="upper right", fontsize=6, markerscale=3, framealpha=0.9)
    else:
        for ax in (ax0, ax1):
            ax.set_xticks([]); ax.set_yticks([])
        ax0.text(0.5, 0.5, "no H&E", ha="center", va="center", transform=ax0.transAxes)
        spatial_blank(ax1, "Tumour-cell location")
        ax1.scatter(xt[:, 0], xt[:, 1], s=args.point_size, c="#d62728", linewidths=0, alpha=0.8)

    # --- row 1: E score map | M score map (tumour-epi only) ---
    eord = np.argsort(np.abs(et) + np.abs(mt))
    elo, ehi = rlim(et); mlo, mhi = rlim(mt)
    for ax, val, lo, hi, cmap, title in [
        (rax[1][0], et, elo, ehi, "Blues", f"E score ({len(E)} genes)"),
        (rax[1][1], mt, mlo, mhi, "Reds", f"M score ({len(M)} genes)"),
    ]:
        ax.scatter(xy[:, 0], xy[:, 1], s=0.4, c="#f0f0f0")
        sc_ = ax.scatter(xt[eord, 0], xt[eord, 1], s=args.point_size, c=val[eord],
                         cmap=cmap, vmin=lo, vmax=hi)
        ax.figure.colorbar(sc_, ax=ax, fraction=0.046, pad=0.02)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    # --- row 2: 2D E/M state-space scatter | EMT region (quadrant) spatial map ---
    axsc, axmap = rax[2]
    axsc.scatter(Es[is_nep], Ms[is_nep], s=1, c="#88aacc", alpha=0.12)
    axsc.scatter(Es[is_fib], Ms[is_fib], s=1, c="#88cc88", alpha=0.12)
    for k in ["low-signal", "epithelial", "mesenchymal", "hybrid"]:
        axsc.scatter(et[quad == k], mt[quad == k], s=2, c=QCOL[k], alpha=0.4)
    axsc.axhline(0, color="k", lw=0.6, ls="--"); axsc.axvline(0, color="k", lw=0.6, ls="--")
    for (cx, cy), name, col in [(nep_c, "E-anchor", "#1f4e79"), (fib_c, "M-anchor", "#1f7a1f")]:
        if np.isfinite(cx):
            axsc.scatter([cx], [cy], s=160, marker="*", c=col, edgecolor="white",
                         linewidth=1.2, zorder=5)
            axsc.annotate(name, (cx, cy), fontsize=8, fontweight="bold", color=col,
                          xytext=(4, 4), textcoords="offset points")
    axsc.set_xlabel("E score", fontsize=10); axsc.set_ylabel("M score", fontsize=10)
    axsc.set_title("E/M state space (★ anchors = "
                   + ("cohort-pooled ref cells)" if anchor_src == "cohort-pooled"
                      else "non-tumour ref cells)"), fontsize=10)
    axsc.legend(handles=[dot(QCOL[k], f"{QLAB[k]} ({(quad == k).mean():.0%})")
                         for k in ["epithelial", "hybrid", "mesenchymal", "low-signal"]],
                fontsize=6, loc="upper left", framealpha=0.9)
    axsc.set_box_aspect(map_box)  # match drawn height to the region map on the right

    axmap.scatter(xy[:, 0], xy[:, 1], s=0.4, c="#eeeeee")
    # one pass, cell order — no class drawn on top (hybrid not privileged as foreground)
    axmap.scatter(xt[:, 0], xt[:, 1], s=args.point_size, c=[QCOL[q] for q in quad],
                  alpha=0.6, linewidths=0)
    axmap.set_box_aspect(map_box); axmap.invert_yaxis(); axmap.set_xticks([]); axmap.set_yticks([])
    axmap.set_title("EMT region map (hybrid = partial EMT)", fontsize=10)
    axmap.legend(handles=[dot(QCOL[k], QLAB[k])
                          for k in ["epithelial", "hybrid", "mesenchymal", "low-signal"]],
                 fontsize=6, loc="best")

    figc.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8, color="#444444")
    stem3 = outdir / f"emt_2d_combined_{args.section}_{args.gene_set}"
    figc.savefig(f"{stem3}.png", dpi=160, bbox_inches="tight")
    figc.savefig(f"{stem3}.pdf", bbox_inches="tight")
    print(f"[saved] {stem3}.png / .pdf")


if __name__ == "__main__":
    main()
