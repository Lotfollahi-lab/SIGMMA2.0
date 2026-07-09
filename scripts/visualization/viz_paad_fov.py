"""PAAD TENX116: recolor whole-slide ST (turbo) + find a high-contrast tumour/stroma FOV.

The FOV is chosen automatically as the window where EPCAM+ (epithelial/tumour) and
ACTA2+ (myofibroblast/stroma) cells co-occur most strongly — i.e. a tumour-stroma
boundary, the most morphologically heterogeneous kind of field. The chosen FOV is
then shown as a full-resolution H&E crop alongside its cell-level ST overlays.
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

CMAP = "turbo"
WIN = 1200            # FOV window size in full-res pixels (0.5 um/px -> 600 um)
STEP = 400           # sliding-window step
MIN_CELLS = 2700     # require a DENSE window (excludes tissue/empty edges)


def expr(ad, gene):
    return np.asarray(ad[:, gene].X.todense()).ravel()


def main() -> None:
    print(f"Loading ST profile for {SECTION} ...")
    ad = sc.read_h5ad(DATA_DIR / "spatial_data_processed.h5")
    xy = ad.obsm["spatial"]
    x, y = xy[:, 0], xy[:, 1]
    total = np.asarray(ad.obs["transcript_counts"]) if "transcript_counts" in ad.obs \
        else np.asarray(ad.X.sum(1)).ravel()
    epcam, acta2 = expr(ad, "EPCAM"), expr(ad, "ACTA2")
    # "positive" = strongly expressing cell (upper quartile of non-zero) so each
    # marker tags a distinct compartment rather than ambient signal
    epcam_pos = epcam > np.quantile(epcam[epcam > 0], 0.75)
    acta2_pos = acta2 > np.quantile(acta2[acta2 > 0], 0.75)
    print(f"  {ad.n_obs:,} cells x {ad.n_vars} genes")

    img = zarr.open(str(DATA_DIR / "hematoxylin_eosin.zarr"), mode="r")
    _, H, W = img.shape

    # ---- Figure 1: whole-slide ST recolored in turbo ----
    ds = 16
    thumb = np.transpose(img[:, ::ds, ::ds], (1, 2, 0))
    extent = [0, thumb.shape[1] * ds, thumb.shape[0] * ds, 0]
    fig, ax = plt.subplots(figsize=(13, 4.4), constrained_layout=True)
    sca = ax.scatter(x, y, c=np.log1p(total), s=0.4, cmap=CMAP, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"{SECTION} — ST profile (log1p total counts, '{CMAP}')", fontsize=11)
    ax.axis("off")
    cb = fig.colorbar(sca, ax=ax, fraction=0.022, pad=0.01); cb.set_label("log1p counts", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    f1 = OUT_DIR / f"paad_{SECTION}_st_{CMAP}"
    fig.savefig(f1.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(f1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f1}.png/.pdf")

    # ---- Scan windows for the strongest tumour/stroma boundary ----
    best = None  # (score, x0, y0, n, f_ep, f_ac)
    x0s = range(int(x.min()), int(x.max()) - WIN, STEP)
    y0s = range(int(y.min()), int(y.max()) - WIN, STEP)
    for x0 in x0s:
        for y0 in y0s:
            m = (x >= x0) & (x < x0 + WIN) & (y >= y0) & (y < y0 + WIN)
            n = int(m.sum())
            if n < MIN_CELLS:
                continue
            f_ep = epcam_pos[m].mean()
            f_ac = acta2_pos[m].mean()
            # reward strong co-occurrence of BOTH compartments in a packed window
            score = f_ep * f_ac
            if best is None or score > best[0]:
                best = (score, x0, y0, n, f_ep, f_ac)
    _, x0, y0, n, f_ep, f_ac = best
    print(f"  FOV @ x[{x0}:{x0+WIN}] y[{y0}:{y0+WIN}]  cells={n}  EPCAM+={f_ep:.2f} ACTA2+={f_ac:.2f}")

    # full-res H&E crop (clip to bounds)
    yc0, yc1 = max(0, y0), min(H, y0 + WIN)
    xc0, xc1 = max(0, x0), min(W, x0 + WIN)
    crop = np.transpose(img[:, yc0:yc1, xc0:xc1], (1, 2, 0))
    crop_extent = [xc0, xc1, yc1, yc0]
    m = (x >= xc0) & (x < xc1) & (y >= yc0) & (y < yc1)

    # ---- Figure 2: FOV — H&E + cell overlays ----
    panels = [
        ("H&E", None, None),
        ("Cells (log1p total counts)", np.log1p(total), CMAP),
        ("EPCAM — epithelial / tumour", epcam, CMAP),
        ("ACTA2 — myofibroblast / stroma", acta2, CMAP),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
    axes = axes.ravel()
    for ax, (title, val, cmap) in zip(axes, panels):
        ax.imshow(crop, extent=crop_extent)
        if val is not None:
            order = np.argsort(val[m])
            sca = ax.scatter(
                x[m][order], y[m][order], c=val[m][order], s=8, cmap=cmap,
                linewidths=0.2, edgecolors="k", alpha=0.85, rasterized=True,
            )
            cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
        ax.set_xlim(xc0, xc1); ax.set_ylim(yc1, yc0)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"{SECTION} — tumour/stroma boundary FOV ({WIN}px ≈ {WIN//2} µm, {n} cells)",
        fontsize=13,
    )
    f2 = OUT_DIR / f"paad_{SECTION}_fov_tumour_stroma"
    fig.savefig(f2.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(f2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {f2}.png/.pdf")


if __name__ == "__main__":
    main()
