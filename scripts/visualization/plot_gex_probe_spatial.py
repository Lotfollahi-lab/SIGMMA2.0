"""Spatial 2D GEX probing visualisation.

For each chosen val FOV, render a row:

  H&E tile | GT log1p map (gene A) | UNI pred | SIGMMA-concat pred | GT (gene B) | UNI | concat | ...

so the eye can compare GT vs predictions of UNI vs SIGMMA spatially, the same
way the niche overlay showed labels on H&E.

Reads:
  - /lustre/.../sigmma_outputs/cache/gex_2d/{uni,sigmma}/<sample>.h5  (features)
  - /lustre/.../sigmma_outputs/cache/data/gex_2d/<tag>/<sample>.h5    (GT + bbox)
  - outputs/gex_2d/heads/{uni,concat}/best.pt
  - outputs/gex_2d/results/{uni,concat}.json              (per-gene Pearson)
  - outputs/gex_2d/gene_panel.json
  - H&E zarr for `BK24-SKI-27-FO-01-S3`

Writes figures/gex_2d/spatial_pred_vs_gt.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import zarr

from sigmma.evaluation.gex_2d_probing import ProbingHead2D
from sigmma.evaluation.gex_cache_io import iter_section_fovs, read_fov_ids

SAMPLE = "BK24-SKI-27-FO-01-S3"
CACHE_ROOT = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/gex_2d")
# GT targets + FOV bboxes live in the data cache (one <sample>.h5 per section);
# point this at the data-cache dir for the SAME panel the heads were trained on.
DATA_ROOT = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/data/gex_2d/dispersion_g56_fov256"
)
HEADS = Path("outputs/gex_2d/heads")
PANEL_FILE = Path("outputs/gex_2d/gene_panel.json")
RES_DIR = Path("outputs/gex_2d/results")
OUT = Path("figures/gex_2d/spatial_pred_vs_gt")
HE_ZARR = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
    "sanger_beacon/20241010__132615__SGP194_SKI_RUN1/BK24-SKI-27-FO-01-S3/"
    "hematoxylin_eosin.zarr"
)
TARGET_GRID = 56


def assemble_concat(sigmma_npz: dict) -> torch.Tensor:
    """3-scale concat: bilinear up each to (TARGET_GRID, TARGET_GRID) and stack."""
    def up(arr):
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        if t.shape[-1] != TARGET_GRID:
            t = F.interpolate(t, size=(TARGET_GRID, TARGET_GRID),
                              mode="bilinear", align_corners=False)
        return t

    return torch.cat([up(sigmma_npz["micro"]),
                      up(sigmma_npz["meso"]),
                      up(sigmma_npz["macro"])], dim=1)


def assemble_uni(uni_npz: dict) -> torch.Tensor:
    t = torch.from_numpy(uni_npz["features"].astype(np.float32)).unsqueeze(0)
    return F.interpolate(t, size=(TARGET_GRID, TARGET_GRID),
                         mode="bilinear", align_corners=False)


def get_val_ids(sample: str, seed: int = 42) -> list[int]:
    """Reproduce the smoke-0b intra-slide 80/20 split used in training."""
    ids = sorted(read_fov_ids(CACHE_ROOT / "uni", sample))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    n_val = max(1, int(0.2 * len(ids)))
    return [ids[i] for i in idx[:n_val]]


def pick_top_genes(k: int = 3) -> list[tuple[str, int, float, float]]:
    panel = json.load(open(PANEL_FILE))["panel"]
    uni_pearson = json.load(open(RES_DIR / "uni.json"))["per_gene_pearson"]
    concat_pearson = json.load(open(RES_DIR / "concat.json"))["per_gene_pearson"]
    ranked = sorted(panel, key=lambda g: -(uni_pearson[g] or -1))
    out = []
    for g in ranked[:k]:
        out.append((g, panel.index(g), uni_pearson[g], concat_pearson[g]))
    return out


def read_he_tile(x_start: int, y_start: int, fov_size: int) -> np.ndarray:
    z = zarr.open(str(HE_ZARR), mode="r")
    tile = np.asarray(z[:, y_start:y_start + fov_size, x_start:x_start + fov_size])
    return np.transpose(tile, (1, 2, 0))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel = json.load(open(PANEL_FILE))["panel"]
    G = len(panel)
    device = torch.device("cpu")

    head_uni = ProbingHead2D(in_channels=1024, num_genes=G).to(device).eval()
    head_uni.load_state_dict(torch.load(HEADS / "uni" / "best.pt", map_location=device))
    head_concat = ProbingHead2D(in_channels=3072, num_genes=G).to(device).eval()
    head_concat.load_state_dict(torch.load(HEADS / "concat" / "best.pt", map_location=device))

    val_ids = get_val_ids(SAMPLE)
    print(f"[paths] |val|={len(val_ids)}")

    # Pick 4 FOVs spread across the val set.
    chosen_ids = [val_ids[i] for i in np.linspace(0, len(val_ids) - 1, 4).astype(int)]
    print(f"[fovs] chosen ids: {chosen_ids}")

    # Join the per-section caches by fov_id: features (uni), scales (sigmma),
    # GT + bbox (data).
    uni_by_id = {d["fov_id"]: d for d in iter_section_fovs(CACHE_ROOT / "uni", SAMPLE, fields=["features"])}
    sig_by_id = {d["fov_id"]: d for d in iter_section_fovs(CACHE_ROOT / "sigmma", SAMPLE, fields=["micro", "meso", "macro"])}
    data_by_id = {d["fov_id"]: d for d in iter_section_fovs(DATA_ROOT, SAMPLE, fields=["target"])}

    top_genes = pick_top_genes(k=3)
    gene_idx = [gi for _, gi, _, _ in top_genes]
    print(f"[genes] top by uni Pearson: {top_genes}")

    # Pre-compute per-FOV: he_tile, gt[g], uni_pred[g], concat_pred[g]
    rows = []
    for fid in chosen_ids:
        uni_fov = uni_by_id[fid]
        sig_fov = sig_by_id[fid]
        data_fov = data_by_id[fid]
        x_start = data_fov["x_start"]; y_start = data_fov["y_start"]
        fov_size = data_fov["fov_size"]
        he = read_he_tile(x_start, y_start, fov_size)
        with torch.no_grad():
            feats_uni = assemble_uni(uni_fov)
            feats_cat = assemble_concat(sig_fov)
            pred_uni = head_uni(feats_uni)[0].cpu().numpy()       # (G, 56, 56)
            pred_cat = head_concat(feats_cat)[0].cpu().numpy()
        gt = np.log1p(data_fov["target"].astype(np.float32))      # (G, 56, 56)
        rows.append({
            "he": he, "gt": gt, "pred_uni": pred_uni, "pred_cat": pred_cat,
            "fov_id": int(fid),
        })

    # Layout: 4 rows × (1 H&E + 3 genes × 3 panels) = 4 × 10
    n_genes = len(gene_idx)
    n_cols = 1 + 3 * n_genes
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.0 * n_cols, 2.3 * len(rows)),
                             constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]

    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["he"])
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_title(f"H&E\nFOV {row['fov_id']}", fontsize=9)

        for gi, (gname, gidx, ur, cr) in enumerate(top_genes):
            gt_map = row["gt"][gidx]
            up_map = row["pred_uni"][gidx]
            cp_map = row["pred_cat"][gidx]
            vmin = min(gt_map.min(), up_map.min(), cp_map.min())
            vmax = max(gt_map.max(), up_map.max(), cp_map.max())
            col0 = 1 + 3 * gi
            for ax, m, ttl in zip(
                axes[r, col0:col0 + 3],
                (gt_map, up_map, cp_map),
                (f"{gname} GT", f"UNI pred", f"SIGMMA-concat pred"),
            ):
                im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    if "GT" in ttl:
                        ax.set_title(f"{gname}\nGT  (uni r={ur:.2f}, concat r={cr:.2f})", fontsize=8)
                    else:
                        ax.set_title(ttl, fontsize=8)
            # Colorbar attached to the rightmost panel of this gene group.
            cb = fig.colorbar(im, ax=axes[r, col0 + 2], fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"2D GEX probing — spatial pred vs GT  ({SAMPLE}, 4 of 54 val FOVs, "
        "log1p counts on a 56×56 grid)",
        fontsize=12, weight="bold"
    )
    fig.savefig(str(OUT) + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    print(f"saved {OUT}.png")
    print(f"saved {OUT}.pdf")


if __name__ == "__main__":
    main()
