"""Full-resolution zoom: do Xenium cell centroids land on H&E nuclei?
If centroids sit on nuclei -> registered; if offset/shifted -> misaligned.
Compares an ACC test section vs a val section in the same tissue-center window."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy
import zarr

BASE = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_adrenocortical_carcinoma"
WIN = 800  # full-res window size (px)
SECS = [
    ("PX69156a-s3 (TEST)", "SGP270_run1/PX69156a-s3"),
    ("PD55831a (TEST)", "KR_PAX_HM_ACC_Hs_paed_adren_5k_1/PD55831a"),
    ("PX69159a-s3 (VAL)", "SGP270_run1/PX69159a-s3"),
]

fig, axes = plt.subplots(1, len(SECS), figsize=(7 * len(SECS), 7))
for k, (name, p) in enumerate(SECS):
    he = zarr.open(f"{BASE}/{p}/hematoxylin_eosin.zarr", mode="r")
    C, H, W = he.shape
    a = scanpy.read_h5ad(f"{BASE}/{p}/spatial_data_processed.h5")
    sp = np.asarray(a.obsm["spatial"])  # full-res px (x,y)
    # window centered on median cell position
    cx, cy = int(np.median(sp[:, 0])), int(np.median(sp[:, 1]))
    x0, y0 = max(0, cx - WIN // 2), max(0, cy - WIN // 2)
    x1, y1 = min(W, x0 + WIN), min(H, y0 + WIN)
    crop = np.transpose(he[:, y0:y1, x0:x1], (1, 2, 0))
    m = (sp[:, 0] >= x0) & (sp[:, 0] < x1) & (sp[:, 1] >= y0) & (sp[:, 1] < y1)
    ax = axes[k]
    ax.imshow(crop, extent=[x0, x1, y1, y0])
    ax.scatter(sp[m, 0], sp[m, 1], s=8, facecolors="none", edgecolors="lime", linewidths=0.6)
    ax.set_title(f"{name}\nfull-res window [{x0}:{x1}, {y0}:{y1}]  cells={int(m.sum())}", fontsize=11)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.axis("off")

plt.tight_layout()
out = "/nfs/team361/dj16/projects/SIGMMA/figures/acc_he_cell_zoom_fullres.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
