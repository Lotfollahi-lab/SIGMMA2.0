"""Overlay Xenium cell centroids on the H&E image for ACC test (and a val ref)
sections. Clearer version: stronger overlay + a zoomed inset to inspect
fine-grained image<->spatial registration."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy
import zarr

BASE = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_adrenocortical_carcinoma"
S = 16
SECS = [
    ("PX69156a-s3 (TEST)", "SGP270_run1/PX69156a-s3"),
    ("PD55831a (TEST)", "KR_PAX_HM_ACC_Hs_paed_adren_5k_1/PD55831a"),
    ("PX69159a-s3 (VAL)", "SGP270_run1/PX69159a-s3"),
]

# cols: H&E | H&E+cells (full) | zoom H&E+cells
fig, axes = plt.subplots(len(SECS), 3, figsize=(20, 6.5 * len(SECS)))
for i, (name, p) in enumerate(SECS):
    he = zarr.open(f"{BASE}/{p}/hematoxylin_eosin.zarr", mode="r")[:, ::S, ::S]
    img = np.transpose(he, (1, 2, 0))
    h, w = img.shape[:2]
    a = scanpy.read_h5ad(f"{BASE}/{p}/spatial_data_processed.h5")
    sp = np.asarray(a.obsm["spatial"]) / S
    x, y = sp[:, 0], sp[:, 1]

    # full H&E
    axes[i, 0].imshow(img); axes[i, 0].set_title(f"{name}  H&E ({w*S}x{h*S})", fontsize=11)
    # full overlay
    axes[i, 1].imshow(img)
    axes[i, 1].scatter(x, y, s=0.3, c="lime", alpha=0.25, linewidths=0)
    axes[i, 1].set_title(f"{name}  + cells (n={a.n_obs})", fontsize=11)
    # zoom: central 1/4 window
    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
    rw, rh = (x.max()-x.min())/4, (y.max()-y.min())/4
    axes[i, 2].imshow(img)
    axes[i, 2].scatter(x, y, s=2.0, c="lime", alpha=0.5, linewidths=0)
    axes[i, 2].set_xlim(cx-rw, cx+rw); axes[i, 2].set_ylim(cy+rh, cy-rh)
    axes[i, 2].set_title(f"{name}  ZOOM (center)", fontsize=11)

    for j in range(3):
        if j != 2:
            axes[i, j].set_xlim(0, w); axes[i, j].set_ylim(h, 0)
        axes[i, j].axis("off")

plt.tight_layout()
out = "/nfs/team361/dj16/projects/SIGMMA/figures/acc_test_he_cell_overlay.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved:", out)
