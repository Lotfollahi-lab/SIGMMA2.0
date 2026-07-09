"""Fibrosis scoring + spatial fibrotic-region visualization for a LIVER Xenium section.

NOT an EMT analysis. Liver fibrosis is hepatic-stellate-cell (HSC) activation and ECM
deposition, which is a single fibrogenic gradient, not an epithelial<->mesenchymal 2D
state space (there is no partial-EMT "hybrid" biology to expose, so the 2D quadrant
framing used for RCC in scripts/emt_2d_score.py does not transfer). Here we score ONE
activated-myofibroblast / ECM program and map where it is high in tissue space.

  - fibrosis score = scanpy score_genes (control-gene baseline) of the activated-
    myofibroblast / ECM set. score > 0 = enriched above a matched-expression
    background, NOT above the slide mean -> composition-robust, comparable across
    slides, and the threshold "fibrotic := score > 0" needs no arbitrary percentile.
  - hepatocytes / endothelium / immune sit at or below 0 (they don't express the
    fibrogenic program), so the fibrotic regions stand out without a cell-type atlas
    (liver has none: no labels.h5ad, manifest `annotation` empty).

"2D" here means the 2D SPATIAL tissue map (x_centroid, y_centroid), not a 2D score
state space. Designed for the SGP302 fresh-frozen (FO) liver cohort, whose 5K panel
(hash d74da03b) carries the canonical fibrosis core (ACTA2/COL1A1/COL3A1/FN1/SPARC).
"""
import argparse
from pathlib import Path
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from emt_signature_spatial import resolve_paths, load_he

# Activated HSC / myofibroblast + ECM fibrosis program (panel-present subset scored).
#   contractile (ACTA2/TAGLN) + fibrillar/basement ECM (COL1/3/5/6, FN1) +
#   activated-fibroblast identity (FAP, LRRC15, POSTN, THY1, PDGFRA/B) + matricellular
#   / crosslinking (SPARC, FBLN2, DPT, ELN, VCAN, TNC, MFAP4, LOX/LOXL1/2) + fibrogenic
#   signalling (TGFB1, CCN2/CTGF, SERPINE1/PAI-1, TIMP1).
FIBROSIS = ["ACTA2", "TAGLN", "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
            "COL6A1", "COL6A2", "COL6A3", "FN1", "SPARC", "FAP", "LRRC15", "POSTN",
            "THY1", "PDGFRA", "PDGFRB", "TGFB1", "CCN2", "CTGF", "SERPINE1", "TIMP1",
            "LOX", "LOXL1", "LOXL2", "FBLN2", "DPT", "ELN", "TNC", "VCAN", "MFAP4"]

# Optional hepatocyte program: only used (--show_hepatocyte) to draw the parenchyma
#   for context; not required to find fibrosis.
HEPATOCYTE = ["ALB", "APOA1", "APOB", "TTR", "TF", "CYP3A4", "HNF4A", "ASGR1",
              "SERPINA1", "APOC3", "FGA", "FGB", "CPS1", "ARG1"]


def present(genes, var_names):
    vs = set(var_names)
    return [g for g in dict.fromkeys(genes) if g in vs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="DF38-LIV-0-FO-1-s1")
    ap.add_argument("--ctrl_size", type=int, default=50,
                    help="control gene-set size for the score_genes baseline")
    ap.add_argument("--top_pct", type=float, default=12.0,
                    help="fibrotic mask = top this %% of cells by fibrosis score "
                         "(per-section adaptive; isolates septa/portal tracts, fixes "
                         "the loose score>0 baseline that floods ~73%% of cells)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="absolute fibrosis-score cutoff; overrides --top_pct when set "
                         "(use for cross-section-comparable masks)")
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--he_downsample", type=int, default=8)
    ap.add_argument("--show_hepatocyte", action="store_true",
                    help="also draw the hepatocyte (parenchyma) score map for context")
    ap.add_argument("--dump_cell_scores", default=None,
                    help="directory to write per-cell fibrosis score + coords parquet")
    ap.add_argument("--outdir", default="figures/cross_tissue_retrieval/liver_fibrosis")
    args = ap.parse_args()

    h5, he_zarr = resolve_paths(args.section)
    print(f"[load] {args.section}\n       {h5}")
    a = ad.read_h5ad(h5)
    if "gene_name" in a.var.columns:
        a.var_names = [str(g) for g in a.var["gene_name"].values]
    a.var_names = [str(g) for g in a.var_names]
    a.var_names_make_unique()
    print(f"[load] {a.n_obs:,} cells x {a.n_vars:,} genes (X log-normalized)")

    Fg = present(FIBROSIS, a.var_names)
    print(f"[markers] fibrosis {len(Fg)}/{len(FIBROSIS)}: {Fg}")
    if len(Fg) < 5:
        raise SystemExit("[skip] too few panel-present fibrosis markers")
    sc.tl.score_genes(a, Fg, score_name="fib", ctrl_size=args.ctrl_size, random_state=0)
    fib = a.obs["fib"].to_numpy()

    if args.threshold is not None:
        thr = args.threshold
        thr_desc = f"score > {thr:g} (absolute)"
    else:
        thr = float(np.percentile(fib, 100 - args.top_pct))
        thr_desc = f"top {args.top_pct:g}% (score > {thr:.2f})"
    is_fibrotic = fib > thr
    n_fib = int(is_fibrotic.sum())
    print(f"[fibrotic] {thr_desc}: {n_fib:,}/{a.n_obs:,} ({n_fib/a.n_obs:.1%})")
    print(f"[score] min{fib.min():+.2f} q50{np.median(fib):+.2f} q90{np.percentile(fib,90):+.2f} "
          f"q99{np.percentile(fib,99):+.2f} max{fib.max():+.2f}")

    Hs = None
    if args.show_hepatocyte:
        Hg = present(HEPATOCYTE, a.var_names)
        if Hg:
            sc.tl.score_genes(a, Hg, score_name="hep", ctrl_size=args.ctrl_size, random_state=0)
            Hs = a.obs["hep"].to_numpy()
            print(f"[markers] hepatocyte {len(Hg)}: {Hg}")

    if "spatial" in a.obsm:
        xy = np.asarray(a.obsm["spatial"])
    else:
        xy = np.c_[a.obs["x_centroid"].to_numpy(), a.obs["y_centroid"].to_numpy()]

    if args.dump_cell_scores:
        import pandas as pd
        dc = Path(args.dump_cell_scores); dc.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"fibrosis": fib, "fibrotic": is_fibrotic,
                      "x": xy[:, 0], "y": xy[:, 1]}).to_parquet(
            dc / f"{args.section}_cells.parquet", index=False)
        print(f"[dump] {dc / f'{args.section}_cells.parquet'}")

    # ===== figure: 2x2 spatial panels =====
    dxw = float(xy[:, 0].max() - xy[:, 0].min())
    dyh = float(xy[:, 1].max() - xy[:, 1].min())
    map_box = (dyh / dxw) if dxw else 1.0

    he = he_extent = None
    if he_zarr and Path(he_zarr).exists():
        ds = args.he_downsample
        print(f"[he] loading {he_zarr} (downsample {ds})")
        he = load_he(he_zarr, ds)
        Hh, Ww = he.shape[:2]
        he_extent = [0, Ww * ds, Hh * ds, 0]
    else:
        print(f"[he] no H&E zarr for {args.section}")

    def draw_he(ax):
        ax.imshow(he, extent=he_extent, interpolation="nearest")
        ax.set_xlim(he_extent[0], he_extent[1]); ax.set_ylim(he_extent[2], he_extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    def blank_spatial(ax):
        ax.scatter(xy[:, 0], xy[:, 1], s=0.4, c="#eeeeee")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

    sub_w = 12 / 2.3
    fig = plt.figure(figsize=(12, max(9.0, sub_w * map_box * 2 + 2.0)))
    fig.suptitle(f"Liver fibrosis ({len(Fg)}-gene activated-myofibroblast/ECM score) — "
                 f"{args.section}  ({a.n_obs:,} cells)", fontsize=13, fontweight="bold")
    axs = fig.subplots(2, 2)
    fib_o = np.argsort(fib)  # draw high-score cells last (on top)
    flo, fhi = np.percentile(fib, [2, 98])

    # (0,0) H&E
    ax = axs[0, 0]
    if he is not None:
        draw_he(ax); ax.set_title("H&E", fontsize=10)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "no H&E", ha="center", va="center", transform=ax.transAxes)

    # (0,1) continuous fibrosis score, all cells
    ax = axs[0, 1]
    blank_spatial(ax)
    sm = ax.scatter(xy[fib_o, 0], xy[fib_o, 1], s=args.point_size, c=fib[fib_o],
                    cmap="inferno", vmin=flo, vmax=fhi, linewidths=0)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label("fibrosis score", fontsize=8)
    ax.set_title("Fibrosis score (all cells)", fontsize=10)

    # (1,0) fibrotic cells (score>thr) over H&E
    ax = axs[1, 0]
    if he is not None:
        draw_he(ax)
    else:
        blank_spatial(ax)
    fc = is_fibrotic
    sm = ax.scatter(xy[fc, 0][np.argsort(fib[fc])], xy[fc, 1][np.argsort(fib[fc])],
                    s=args.point_size, c=np.sort(fib[fc]), cmap="inferno",
                    vmin=thr, vmax=fhi, linewidths=0, alpha=0.85)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label("fibrosis score", fontsize=8)
    ax.set_title(f"Fibrotic cells ({thr_desc}, n={n_fib:,}) on H&E", fontsize=10)

    # (1,1) hepatocyte context map OR score histogram
    ax = axs[1, 1]
    if Hs is not None:
        blank_spatial(ax)
        ho = np.argsort(Hs); hlo, hhi = np.percentile(Hs, [2, 98])
        sm = ax.scatter(xy[ho, 0], xy[ho, 1], s=args.point_size, c=Hs[ho],
                        cmap="Greens", vmin=hlo, vmax=hhi, linewidths=0)
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label("hepatocyte score", fontsize=8)
        ax.set_title("Hepatocyte (parenchyma) score", fontsize=10)
    else:
        ax.hist(fib, bins=100, color="#888888")
        ax.axvline(thr, color="#B2182B", ls="--", lw=1.2, label=thr_desc)
        ax.set_xlabel("fibrosis score", fontsize=9); ax.set_ylabel("cells", fontsize=9)
        ax.set_title(f"Fibrosis score distribution ({n_fib/a.n_obs:.1%} > thr)", fontsize=10)
        ax.legend(fontsize=8); ax.tick_params(labelsize=7)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)

    foot = (
        f"Fibrosis score = scanpy score_genes (control-gene baseline) of {len(Fg)} activated-myofibroblast / ECM genes: "
        f"{', '.join(Fg)}.\n"
        f"score is the control-gene baseline (ctrl_size={args.ctrl_size}); fibrotic mask = {thr_desc}. "
        "Hepatocytes / endothelium / immune sit at/below 0. No cell-type atlas used."
    )
    fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=7.5, color="#444444", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"liver_fibrosis_{args.section}"
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
