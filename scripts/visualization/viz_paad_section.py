"""Visualize one PAAD section: H&E whole-slide image + spatial transcriptomics profile.

Section: TENX116 (pancreatic adenocarcinoma, HEST-1k public Xenium).
Outputs an overview figure (H&E + total-transcript density) and a marker-gene
spatial panel, both in the same WSI pixel coordinate frame so they co-register.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import zarr

SECTION = "TENX116"
DATA_DIR = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_xenium/hest_1k_PAAD"
) / SECTION
OUT_DIR = Path("figures/viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Marker genes (present in the 468-gene panel) spanning the major PAAD compartments.
MARKERS = {
    "EPCAM": "Epithelial / tumour",
    "ACTA2": "Myofibroblast / stroma",
    "CD68": "Macrophage",
    "CD3D": "T cell",
    "PECAM1": "Endothelial",
    "CXCL2": "Chemokine (immune exclusion)",
}


def load_he_thumbnail(downsample: int = 16) -> np.ndarray:
    """Load the H&E zarr (C, H, W) uint8 and return an RGB thumbnail (H', W', 3)."""
    arr = zarr.open(str(DATA_DIR / "hematoxylin_eosin.zarr"), mode="r")
    img = arr[:, ::downsample, ::downsample]  # (3, H', W')
    return np.transpose(img, (1, 2, 0))  # (H', W', 3)


def main() -> None:
    print(f"Loading ST profile for {SECTION} ...")
    ad = sc.read_h5ad(DATA_DIR / "spatial_data_processed.h5")
    xy = ad.obsm["spatial"]  # (n_cells, 2) in WSI pixel coords (x, y)
    x, y = xy[:, 0], xy[:, 1]
    total = np.asarray(ad.obs["transcript_counts"]) if "transcript_counts" in ad.obs \
        else np.asarray(ad.X.sum(1)).ravel()
    print(f"  {ad.n_obs:,} cells x {ad.n_vars} genes")

    downsample = 4
    print("Loading H&E thumbnail ...")
    thumb = load_he_thumbnail(downsample)
    # extent in full-res pixel coords so scatter overlays align with the image
    extent = [0, thumb.shape[1] * downsample, thumb.shape[0] * downsample, 0]

    # ---- Figure 1: overview (H&E + transcript density) ----
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), constrained_layout=True)
    axes[0].imshow(thumb, extent=extent)
    axes[0].set_title(f"{SECTION} — H&E (pancreatic adenocarcinoma, Xenium)", fontsize=11)
    axes[0].axis("off")

    sc_ax = axes[1].scatter(
        x, y, c=np.log1p(total), s=0.4, cmap="magma", linewidths=0,
        rasterized=True,
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])
    axes[1].set_title(f"ST profile — log1p(total transcript counts), {ad.n_obs:,} cells", fontsize=11)
    axes[1].axis("off")
    cb = fig.colorbar(sc_ax, ax=axes[1], fraction=0.025, pad=0.01)
    cb.set_label("log1p counts", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    f1 = OUT_DIR / f"paad_{SECTION}_he_st_overview"
    fig.savefig(f1.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(f1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f1}.png/.pdf")

    # ---- Figure 2: marker-gene spatial panel ----
    markers = {g: lab for g, lab in MARKERS.items() if g in ad.var_names}
    n = len(markers)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.6 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (gene, label) in zip(axes, markers.items()):
        expr = np.asarray(ad[:, gene].X.todense()).ravel()
        order = np.argsort(expr)  # plot high-expressing cells on top
        sca = ax.scatter(
            x[order], y[order], c=expr[order], s=0.4, cmap="viridis",
            linewidths=0, rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{gene} — {label}", fontsize=10)
        ax.axis("off")
        cb = fig.colorbar(sca, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("log1p expr", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{SECTION} — PAAD marker-gene spatial expression", fontsize=12)

    f2 = OUT_DIR / f"paad_{SECTION}_marker_genes"
    fig.savefig(f2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(f2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f2}.png/.pdf")

    # ---- Figure 3: 3-gene RGB overlay (each gene = one additive colour) ----
    # Map three compartment markers to additive primaries so the merge reads like a
    # multi-channel immunofluorescence composite. Red + green + blue mix cleanly
    # (e.g. red+green->yellow, all three->white) without channel collisions.
    OVERLAY = [
        ("EPCAM", (1.0, 0.0, 0.0), "Epithelial / tumour"),   # red
        ("ACTA2", (0.0, 1.0, 0.0), "Myofibroblast / stroma"),  # green
        ("CD3D",  (0.0, 0.0, 1.0), "T cell"),                  # blue
    ]
    overlay = [(g, c, lab) for g, c, lab in OVERLAY if g in ad.var_names]

    def norm_expr(gene: str, pct: float = 99.0) -> np.ndarray:
        e = np.asarray(ad[:, gene].X.todense()).ravel()
        hi = np.percentile(e, pct)
        return np.clip(e / hi, 0, 1) if hi > 0 else e

    # Start from a light-gray base and alpha-composite each gene's colour on top
    # (weight = normalised expression). Cells expressing none of the three genes
    # stay light gray and blend into the background instead of going black.
    BG_GRAY = np.array([0.92, 0.92, 0.92], dtype=np.float32)
    norms = {gene: norm_expr(gene) for gene, _, _ in overlay}
    rgb = np.tile(BG_GRAY, (ad.n_obs, 1))
    for gene, colour, _ in overlay:
        a = norms[gene][:, None]
        rgb = rgb * (1 - a) + np.asarray(colour, dtype=np.float32)[None, :] * a
    rgb = np.clip(rgb, 0, 1)
    # plot most-expressing cells last so signal sits on top of the gray background
    order = np.argsort(np.sum([norms[g] for g, _, _ in overlay], axis=0))

    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    ax.set_facecolor(BG_GRAY)
    ax.scatter(x[order], y[order], c=rgb[order], s=0.6, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(
        f"{SECTION} — 3-gene RGB overlay "
        + " / ".join(f"{g} ({lab})" for g, _, lab in overlay),
        fontsize=11,
    )
    ax.axis("off")
    # legend swatches for each gene colour
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", label=g) for g, c, _ in overlay]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8,
              facecolor="white", labelcolor="black")

    f3 = OUT_DIR / f"paad_{SECTION}_gene_overlay_rgb"
    fig.savefig(f3.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(f3.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f3}.png/.pdf")

    # ---- Figure 4: H&E + 3-gene RGB overlay side by side (co-registered) ----
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), constrained_layout=True)
    axes[0].imshow(thumb, extent=extent)
    axes[0].set_title(f"{SECTION} — H&E (pancreatic adenocarcinoma, Xenium)", fontsize=11)
    axes[0].axis("off")

    axes[1].set_facecolor(BG_GRAY)
    axes[1].scatter(x[order], y[order], c=rgb[order], s=0.6, linewidths=0, rasterized=True)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])
    axes[1].set_title(
        "ST profile — 3-gene RGB overlay ("
        + ", ".join(f"{g}={lab}" for g, _, lab in overlay) + ")",
        fontsize=11,
    )
    axes[1].axis("off")
    axes[1].legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8,
                   facecolor="white", labelcolor="black")

    f4 = OUT_DIR / f"paad_{SECTION}_he_st_overlay"
    fig.savefig(f4.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(f4.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f4}.png/.pdf")

    # ---- Figure 5: cropped ROI (tumour/stroma/immune boundary) ----
    # Window chosen by scanning the slide for a region where all three markers
    # co-occur, i.e. compartment boundaries meet (full-res H&E for crispness).
    x0, y0, w = 700, 1000, 2500  # full-res pixel coords
    x1, y1 = x0 + w, y0 + w
    crop_arr = zarr.open(str(DATA_DIR / "hematoxylin_eosin.zarr"), mode="r")
    crop_img = np.transpose(crop_arr[:, y0:y1, x0:x1], (1, 2, 0))  # (h, w, 3)
    crop_extent = [x0, x1, y1, y0]
    inwin = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    co = order[np.isin(order, np.nonzero(inwin)[0])]  # keep top-of-stack ordering

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(crop_img, extent=crop_extent)
    axes[0].set_title(f"{SECTION} — H&E (crop)", fontsize=11)
    axes[1].set_facecolor(BG_GRAY)
    axes[1].scatter(x[co], y[co], c=rgb[co], s=6, linewidths=0, rasterized=True)
    axes[1].set_title(
        "ST 3-gene RGB overlay (crop) — "
        + ", ".join(f"{g}={lab}" for g, _, lab in overlay),
        fontsize=11,
    )
    axes[1].legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8,
                   facecolor="white", labelcolor="black")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("off")

    f5 = OUT_DIR / f"paad_{SECTION}_he_st_overlay_crop"
    fig.savefig(f5.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(f5.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f5}.png/.pdf")

    # ---- Figure 6: small crop, ST overlay on a faint grayscale H&E backdrop ----
    cx0, cy0, cw = 1600, 1400, 1300  # smaller full-res window on the boundary
    cx1, cy1 = cx0 + cw, cy0 + cw
    sub = np.transpose(crop_arr[:, cy0:cy1, cx0:cx1], (1, 2, 0))  # (h, w, 3)
    gray = sub.mean(2)  # luminance, grayscale H&E
    sub_extent = [cx0, cx1, cy1, cy0]
    smin = (x >= cx0) & (x < cx1) & (y >= cy0) & (y < cy1)
    so = order[np.isin(order, np.nonzero(smin)[0])]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    # left: original colour H&E crop
    axes[0].imshow(sub, extent=sub_extent)
    axes[0].set_title(f"{SECTION} — H&E (crop)", fontsize=11)
    # right: ST overlay on faint grayscale H&E backdrop (~70% transparent)
    axes[1].set_facecolor("white")
    axes[1].imshow(gray, cmap="gray", alpha=0.3, extent=sub_extent)
    axes[1].scatter(x[so], y[so], c=rgb[so], s=14, linewidths=0, alpha=0.95, rasterized=True)
    axes[1].set_title("ST 3-gene overlay on grayscale H&E", fontsize=11)
    axes[1].legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8,
                   facecolor="white", labelcolor="black")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(cx0, cx1)
        ax.set_ylim(cy1, cy0)
        ax.axis("off")

    f6 = OUT_DIR / f"paad_{SECTION}_he_st_overlay_crop_small"
    fig.savefig(f6.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(f6.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f6}.png/.pdf")


if __name__ == "__main__":
    main()
