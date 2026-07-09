"""Figure: SIGMMA Xenium cohort overview (15 datasets).

Three panels stitched into one figure sharing a common dataset y-axis:
  A. Bar plot   — # sections and # cells per dataset, as a twin-axis grouped bar
                  (two bars per dataset: bottom x-axis = sections, top x-axis =
                  cells; the two metrics differ by ~100x so each has its own
                  colour-matched x-axis)
  B. Bubble plot — gene panel per dataset. Two bands (x = {Xenium 5K /
                   <500-gene targeted}); WITHIN each band the bubble's
                   x-position maps the panel's gene count on a shared
                   genes-per-x scale, so the 5001/5101 5K variants sit closer
                   together than the wider 280..480 targeted spread. Bubble
                   AREA = # sections on that
                   panel. Provenance (in-house vs public) is shown by the left
                   strip, NOT this axis. One bubble per (dataset, panel) entry,
                   so datasets with two panels show two bubbles.
  C. Bubble plot — section-level train/val/test split, bubble = the *proportion*
                   of that split within each dataset row (fractions sum to 1 per
                   row). In-house datasets split ~train/val/test; all HEST-1k
                   sections are held out as test only (100% test).

Style follows the project eva_critique convention (Helvetica 8 pt, left+bottom
spines, horizontal grid, png+pdf @600 dpi). Data verified 2026-05-27 from the
per-dataset MANIFEST files + gene_panel_info/section_to_gene_panel.csv; see memo
[[project_dataset_cohort_stats]]. ACC/CTCL panels classified from h5 var n_vars
(ACC all 5K; CTCL 60 x 5K + 18 x small skin custom). HEST panel sizes = mean
retained genes per dataset.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
OUTDIR = ROOT / "figures/dataset_overview"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Verified cohort data (2026-05-27).  Each row: display label, cells, sections,
# source ('inhouse'|'public'), split (train,val,test unique sections),
# panels = list of (panel_size_genes, n_sections, panel_category).
# panel_category in {'5K', 'custom', 'public'}.
# tile_share = True when train & val share sections (HEST tile-level split).
# ----------------------------------------------------------------------------
DATA = [
    # in-house (sorted by cells desc)
    # Pancreas is one directory but two cohorts split by Sanger run folder:
    # SGP254 (DJ* samples) = diabetic, SGP252 (Hml/Sml*) = pediatric. Counts
    # verified from sanger_pancreas_MANIFEST{,_train,_val,_test}.csv (2026-06-16):
    # diabetic 32 sec / 17,029,301 cells split (22,5,5); pediatric 20 sec /
    # 3,027,473 cells all held in train (20,0,0). Both on the Xenium 5K panel.
    dict(label="Pancreas (Diabetes)", key="sanger_pancreas_diabetic", src="inhouse", cells=17_029_301, sec=32,
         split=(22, 5, 5), tile=False, panels=[(5101, 32, "5K")]),
    dict(label="Lung",           key="sanger_lung",      src="inhouse", cells=12_392_549, sec=16,
         split=(12, 2, 2), tile=False, panels=[(341, 14, "custom"), (5001, 2, "5K")]),
    dict(label="Liver",          key="sanger_liver",     src="inhouse", cells=10_077_413, sec=40,
         split=(32, 4, 4), tile=False, panels=[(5001, 32, "5K"), (5101, 8, "5K")]),
    dict(label="Whole embryo",   key="sanger_whole_embryo", src="inhouse", cells=7_682_438, sec=20,
         split=(16, 2, 2), tile=False, panels=[(5001, 20, "5K")]),
    dict(label="Pediatric ovary", key="sanger_pediatric_ovary", src="inhouse", cells=6_378_731, sec=55,
         split=(44, 5, 6), tile=False, panels=[(480, 55, "custom")]),
    dict(label="Brain (GBM)",    key="sanger_glioblastoma", src="inhouse", cells=6_375_063, sec=19,
         split=(15, 2, 2), tile=False, panels=[(5001, 4, "5K"), (460, 15, "custom")]),
    dict(label="Kidney (RCC)",   key="sanger_renal_cell_carcinoma", src="inhouse", cells=4_543_275, sec=54,
         split=(44, 5, 5), tile=False, panels=[(5001, 54, "5K")]),
    dict(label="Pediatric Pancreas", key="sanger_pancreas_pediatric", src="inhouse", cells=3_027_473, sec=20,
         split=(20, 0, 0), tile=False, panels=[(5101, 20, "5K")]),
    dict(label="Skin (CTCL)",    key="sanger_CTCL",      src="inhouse", cells=2_926_601, sec=78,
         split=(62, 8, 8), tile=False, panels=[(5001, 60, "5K"), (300, 18, "custom")]),
    dict(label="Adrenal (ACC)",  key="sanger_adrenocortical_carcinoma", src="inhouse", cells=2_020_107, sec=22,
         split=(18, 2, 2), tile=False, panels=[(5001, 22, "5K")]),
    dict(label="Skin (beacon)",  key="sanger_beacon",    src="inhouse", cells=1_170_102, sec=84,
         split=(68, 8, 8), tile=False, panels=[(5001, 84, "5K")]),
    # public HEST-1k (sorted by cells desc) — all sections held out as test only
    dict(label="Breast (IDC)",   key="hest_1k_IDC",      src="public",  cells=1_728_056, sec=4,
         split=(0, 0, 4), tile=False, panels=[(280, 4, "public")]),
    dict(label="Pancreas (PAAD)",key="hest_1k_PAAD",     src="public",  cells=513_240, sec=3,
         split=(0, 0, 3), tile=False, panels=[(404, 3, "public")]),
    dict(label="Lung (LUAD)",    key="hest_1k_LUAD",     src="public",  cells=305_496, sec=2,
         split=(0, 0, 2), tile=False, panels=[(420, 2, "public")]),
    dict(label="Skin (SKCM)",    key="hest_1k_SKCM",     src="public",  cells=192_628, sec=2,
         split=(0, 0, 2), tile=False, panels=[(327, 2, "public")]),
]

# Colours
COL_INHOUSE = "#1f77b4"
COL_PUBLIC = "#d62728"
PANEL_COL = {"5K": "#1f77b4", "custom": "#2ca02c", "public": "#d62728"}
PANEL_LBL = {"5K": "Xenium 5K Pan Tissue (±add-on)",
             "custom": "In-house tissue panel",
             "public": "10x public panel (HEST-1k)"}
SPLIT_COL = {"train": "#4477AA", "val": "#DDAA33", "test": "#BB5566"}

SEC_COL = "#44AA99"   # bar colour: # sections (bottom x-axis)
CELL_COL = "#AA4499"  # bar colour: # cells   (top x-axis)
DOT_COL = "#666666"   # neutral dot colour for panels B & C

SEC_MAX_AREA = 252.0  # panel B: bubble area at the largest single-panel section count
SEC_MAX_N = 84.0      # panel B: max # sections of any one panel (Skin beacon, 5K)
SIZE_FRAC = 360.0     # panel C: marker area at frac=1.0


def bubble_area(frac):
    """Panel C marker area with radius ~ proportional to the split ratio, so
    80% reads visibly smaller than 100% (area = frac**2). Floored so the small
    ~10% val/test bubbles stay visible."""
    return max(frac ** 2 * SIZE_FRAC, 20.0)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    n = len(DATA)
    # y positions: first row on top
    ypos = {d["key"]: (n - 1 - i) for i, d in enumerate(DATA)}
    ylabels = [d["label"] for d in DATA]
    ycolors = [COL_INHOUSE if d["src"] == "inhouse" else COL_PUBLIC for d in DATA]
    # separator between in-house and public blocks
    n_inhouse = sum(d["src"] == "inhouse" for d in DATA)
    sep_y = (n - n_inhouse) - 0.5  # between last in-house and first public

    fig = plt.figure(figsize=(10, 10))  # fixed 10x10 in
    gs = fig.add_gridspec(
        1, 3, width_ratios=[2.3, 1.25, 1.25], wspace=0.12,
        left=0.20, right=0.985, top=0.84, bottom=0.22,
    )
    ax_size = fig.add_subplot(gs[0, 0])
    ax_pan = fig.add_subplot(gs[0, 1], sharey=ax_size)
    ax_spl = fig.add_subplot(gs[0, 2], sharey=ax_size)

    def style(ax, keep_yticks=False):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-0.7, n - 0.3)
        if not keep_yticks:
            ax.tick_params(labelleft=False, left=False)
        ax.axhline(sep_y, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=0)

    # ---- Panel A: # sections (bottom x) + # cells (top x), twin-axis bars ----
    ax_cell = ax_size.twiny()  # shares y, independent top x-axis for cells
    OFF = 0.21
    for d in DATA:
        y = ypos[d["key"]]
        cm = d["cells"] / 1e6
        ax_size.barh(y + OFF, d["sec"], color=SEC_COL, height=0.38, zorder=3)
        ax_size.text(d["sec"] + 1.5, y + OFF, str(d["sec"]), va="center",
                     ha="left", fontsize=5.5, color=SEC_COL)
        ax_cell.barh(y - OFF, cm, color=CELL_COL, height=0.38, zorder=3)
        txt = f"{cm:.2f}" if cm < 1 else f"{cm:.1f}"
        ax_cell.text(cm + 0.35, y - OFF, txt, va="center", ha="left",
                     fontsize=5.5, color=CELL_COL)

    ax_size.set_xlim(0, 96)
    ax_size.set_xlabel("# sections", color=SEC_COL)
    ax_size.tick_params(axis="x", colors=SEC_COL)
    ax_size.spines["bottom"].set_color(SEC_COL)
    ax_size.set_yticks(range(n))
    ax_size.set_yticklabels(ylabels[::-1])
    style(ax_size, keep_yticks=True)

    ax_cell.set_xlim(0, 22.5)
    ax_cell.set_xlabel("# cells (millions)", color=CELL_COL)
    ax_cell.tick_params(axis="x", colors=CELL_COL)
    ax_cell.set_ylim(-0.7, n - 0.3)
    ax_cell.spines["top"].set_color(CELL_COL)
    ax_cell.spines["right"].set_visible(False)
    ax_cell.spines["bottom"].set_visible(False)
    ax_cell.spines["left"].set_visible(False)
    ax_cell.tick_params(left=False, labelleft=False)

    # ---- Panel B: gene panel scale + within-band gene-count spread ----
    # Two bands (5K vs <500-gene targeted). Provenance (in-house vs public) is
    # already carried by the left strip, so this axis is purely panel scale.
    # WITHIN each band the bubble's x-position maps the panel's gene count on a
    # shared genes-per-x scale: the widest-span band fills its band width and
    # narrower-span bands spread proportionally less, so the 5001/5101 5K
    # variants (~100-gene span) sit closer together than the 280..480 targeted
    # panels (~200-gene span) instead of being stretched edge-to-edge.
    # Bubble AREA encodes the panel's # sections. One bubble per (dataset,
    # panel) entry, so datasets with two panels show two bubbles.
    def panel_col(cat):
        return 0 if cat == "5K" else 1
    COL2 = {0: "#1f77b4", 1: "#2ca02c"}  # 5K = blue, targeted = green
    BAND_X = {0: 1.0, 1: 0.0}  # targeted (<500) on the left, Xenium 5K on the right
    BAND_HALFW = 0.34  # within-band x spread half-width

    def sec_area(nsec):
        return max(nsec / SEC_MAX_N * SEC_MAX_AREA, 14.0)

    # per-band gene-count range (drives the within-band x map + edge labels)
    band_g = {0: [], 1: []}
    for d in DATA:
        for (sz, _, cat) in d["panels"]:
            band_g[panel_col(cat)].append(sz)
    gmin = {b: min(v) for b, v in band_g.items()}
    gmax = {b: max(v) for b, v in band_g.items()}
    gmid = {b: (gmin[b] + gmax[b]) / 2 for b in band_g}
    GSPAN = max(gmax[b] - gmin[b] for b in band_g) or 1.0  # widest band fills the band

    def panel_x(col, g):
        return BAND_X[col] + (g - gmid[col]) / GSPAN * 2 * BAND_HALFW

    for d in DATA:
        y = ypos[d["key"]]
        for (sz, nsec, cat) in d["panels"]:
            col = panel_col(cat)
            ax_pan.scatter(panel_x(col, sz), y, s=sec_area(nsec), color=COL2[col],
                           alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    ax_pan.set_xlim(-0.5, 1.5)
    # per-band gene-count ruler so each bubble's gene count is readable off x.
    # 5K has only two variants (5001/5101); targeted gets a 50-gene grid.
    tick_genes = {0: [gmin[0], gmax[0]], 1: [280, 330, 380, 430, 480]}
    tick_x, tick_lab, tick_col = [], [], []
    for col in (0, 1):
        for g in tick_genes[col]:
            tick_x.append(panel_x(col, g))
            tick_lab.append(f"{g:,}")
            tick_col.append(COL2[col])
    ax_pan.set_xticks(tick_x)
    ax_pan.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=4.5)
    for t, c in zip(ax_pan.get_xticklabels(), tick_col):
        t.set_color(c)
    # band-name headers below the gene-count ticks (provenance lives in the strip)
    for col, name in ((0, "5K"), (1, "<500-gene targeted")):
        ax_pan.text(BAND_X[col], -0.08, name, transform=ax_pan.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6, color="black")
    ax_pan.set_xlabel("Gene panel")
    ax_pan.xaxis.set_label_coords(0.5, -0.125)
    ax_pan.grid(axis="x", lw=0.3, color="#dddddd", zorder=0)
    style(ax_pan)

    # ---- panel B bubble-size legend (# sections) ----
    sec_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#888888",
               markeredgecolor="white", markersize=np.sqrt(sec_area(nv)),
               label=str(nv))
        for nv in (80, 40, 10)
    ]
    fig.legend(handles=sec_handles, title="bubble size = # sections",
               loc="lower center", bbox_to_anchor=(0.55, 0.155), ncol=3, frameon=False,
               handlelength=2.4, handletextpad=0.6, columnspacing=1.3, labelspacing=0.5,
               borderaxespad=0.0)

    # ---- Panel C: train/val/test split as per-row proportion (bubble = fraction) ----
    split_x = {"train": 0, "val": 1, "test": 2}
    for d in DATA:
        y = ypos[d["key"]]
        total = sum(d["split"])
        if total == 0:
            continue
        for s_name, cnt in zip(("train", "val", "test"), d["split"]):
            if cnt == 0:
                continue
            frac = cnt / total
            ax_spl.scatter(split_x[s_name], y, s=bubble_area(frac),
                           color=SPLIT_COL[s_name], alpha=0.82,
                           edgecolor="white", linewidth=0.5, zorder=3)
    ax_spl.set_xlim(-0.5, 2.5)
    ax_spl.set_xticks([0, 1, 2])
    ax_spl.set_xticklabels(["Train", "Val", "Test"])
    ax_spl.set_xlabel("Data split")
    ax_spl.grid(axis="x", lw=0.3, color="#dddddd", zorder=0)
    style(ax_spl)

    # ---- panel C bubble-size legend (split ratio) ----
    frac_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#888888",
               markeredgecolor="white", markersize=np.sqrt(bubble_area(f)),
               label=f"{int(f*100)}%")
        for f in (1.0, 0.8, 0.1)
    ]
    fig.legend(handles=frac_handles, title="bubble size = split ratio",
               loc="lower center", bbox_to_anchor=(0.85, 0.155), ncol=3, frameon=False,
               handlelength=2.8, handletextpad=0.7, columnspacing=1.8, labelspacing=0.5,
               borderaxespad=0.0)

    # ---- source group strip on the far left (in-house vs public) ----
    pos = ax_size.get_position()
    strip = fig.add_axes([0.050, pos.y0, 0.020, pos.height])
    strip.set_xlim(0, 1)
    strip.set_ylim(-0.7, n - 0.3)
    strip.axhspan(sep_y, n - 0.3, color="#3f3f3f", lw=0)   # top block = in-house
    strip.axhspan(-0.7, sep_y, color="#808080", lw=0)      # bottom block = public
    strip.text(0.5, (sep_y + n - 0.3) / 2, "In-house", rotation=90,
               va="center", ha="center", color="white", fontsize=7, fontweight="bold")
    strip.text(0.5, (-0.7 + sep_y) / 2, "Public", rotation=90,
               va="center", ha="center", color="white", fontsize=7, fontweight="bold")
    strip.set_xticks([])
    strip.set_yticks([])
    for s in strip.spines.values():
        s.set_visible(False)

    fig.suptitle("SIGMMA Xenium cohort — 15 datasets · 451 sections · 76.4M cells",
                 x=0.56, y=0.97, fontsize=11, fontweight="bold")

    fig.savefig(OUTDIR / "cohort_overview.png", dpi=600)
    fig.savefig(OUTDIR / "cohort_overview.pdf")
    print("saved:", OUTDIR / "cohort_overview.png")


if __name__ == "__main__":
    main()
