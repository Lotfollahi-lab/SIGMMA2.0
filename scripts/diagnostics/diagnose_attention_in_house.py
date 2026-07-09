"""Visualize CLS->patch attention for base UNI2-h vs SIGMMA-finetuned UNI2-h
on in-house Xenium H&E 256x256 macro-scale FOVs.

Output: a 3 (FOV) x 4 (input/UNI/SIGMMA/Δ) grid PNG + PDF + selected_fovs.csv.

Reuses:
  - extract_cls_attention + load_uni/load_sigmma from scripts.eva_critique.models
  - _read_data_h5 + _read_crops_from_zarr from sigmma.application.cell_niche_transfer.extract
  - _upsample_attn from scripts.eva_critique.pipeline_eval

Caveat (encoded in caption): the figure shows ViT-CLS attention, which is
adjacent to but not identical to the SIGMMA contrastive signal (dense
mask-pool). The shift is consistent with the weight-level LoRA delta
analysis (project_lora_delta_a01_step1000) — qualitative only.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import zarr
from matplotlib import pyplot as plt
from skimage.measure import find_contours

from ml_collections.config_dict import ConfigDict

from scripts.eva_critique.models import load_sigmma, load_uni, _ViTBackbone
from scripts.eva_critique.pipeline_eval import _upsample_attn
from sigmma.application.cell_niche_transfer.extract import (
    _IMAGENET_MEAN, _IMAGENET_STD, _read_crops_from_zarr, _read_data_h5,
)
from sigmma.utils import load_model


# ---------------------------------------------------------------------------
# FOV selection
# ---------------------------------------------------------------------------
def _empty_fraction(crop_3hw: np.ndarray, threshold: int = 220) -> float:
    """Fraction of pixels that are 'background-white' across all 3 channels."""
    return float(((crop_3hw > threshold).all(axis=0)).mean())


@torch.no_grad()
def _scan_delta(
    h5_dir: Path,
    slides_df: pd.DataFrame,
    enc_uni: _ViTBackbone,
    enc_sig: _ViTBackbone,
    device: torch.device,
    *,
    split: str,
    n_total: int,
    seed: int,
    min_neighbors_in_fov: int = 5,
) -> pd.DataFrame:
    """Sample N anchor cells across cached slides in `split` (+val fallback)
    and compute per-FOV attention Δ stats vs base UNI.

    Returns a DataFrame with one row per scanned FOV, including:
      sample, anchor_cell_idx, anchor_cell_id, bbox_x, bbox_y, niche_category,
      n_cells_in_fov, empty_fraction, mask_patch_frac, mean_rgb,
      delta_l1_mean, delta_l1_max, delta_l2
    """
    rng = np.random.default_rng(seed)

    def _slides_for(sv: str) -> list[str]:
        names = slides_df.loc[slides_df["split"] == sv, "sample_name"].tolist()
        return [s for s in names if (h5_dir / f"{s}.h5").exists()]

    slides = _slides_for(split)
    if split == "test":
        slides = slides + [s for s in _slides_for("val") if s not in slides]
    if not slides:
        raise SystemExit("no cached slides available")
    per_slide = max(1, n_total // len(slides))

    rows = []
    for sample in slides:
        data = _read_data_h5(h5_dir / f"{sample}.h5")
        nc_all = np.array([
            (v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v))
            for v in data["niche_category"]
        ])
        bbox = data["bbox_macro"]
        with h5py.File(h5_dir / f"{sample}.h5", "r") as f:
            xp = f["x_pixel"][:]; yp = f["y_pixel"][:]
        size = int(data["scale_sizes"][-1])

        he_path = slides_df.loc[slides_df["sample_name"] == sample,
                                "hematoxylin_eosin_image"].iloc[0]
        he_zarr = zarr.open(he_path, mode="r")

        order = rng.permutation(len(bbox))
        taken = 0
        for idx in order:
            if taken >= per_slide:
                break
            x0, y0 = int(bbox[idx, 0]), int(bbox[idx, 1])
            in_fov = ((xp >= x0) & (xp < x0 + size)
                      & (yp >= y0) & (yp < y0 + size))
            n_in_fov = int(in_fov.sum())
            if n_in_fov < min_neighbors_in_fov:
                continue
            # Fetch the crop.
            crop = _read_crops_from_zarr(
                he_zarr, np.array([[x0, y0]], dtype=np.int32), size,
            )[0]  # (3, S, S) uint8
            rgb = crop.transpose(1, 2, 0)
            ef = _empty_fraction(crop)
            mean_rgb = float(rgb.astype(np.float32).mean())

            # Mask-patch coverage: fraction of 16x16 patches whose 14x14 block
            # contains any cell mask pixel from cells in this FOV.
            # Cheap proxy: aggregate masks of cells in this FOV onto crop coords.
            in_idx = np.where(in_fov)[0]
            mask_union = np.zeros((size, size), dtype=bool)
            for j in in_idx:
                m = data["mask_macro"][j]
                x_off = int(bbox[j, 0] - x0); y_off = int(bbox[j, 1] - y0)
                dy0 = max(0, y_off); dy1 = min(size, y_off + size)
                dx0 = max(0, x_off); dx1 = min(size, x_off + size)
                sy0 = max(0, -y_off); sx0 = max(0, -x_off)
                h_ = dy1 - dy0; w_ = dx1 - dx0
                if h_ <= 0 or w_ <= 0:
                    continue
                mask_union[dy0:dy0 + h_, dx0:dx0 + w_] |= m[sy0:sy0 + h_, sx0:sx0 + w_]
            mask_patch_frac = float(((mask_union.reshape(16, size // 16, 16, size // 16)
                                      .any(axis=(1, 3)))).mean())

            # Forward both encoders.
            attn_u = _attn_for_fov(enc_uni, rgb, device)
            attn_s = _attn_for_fov(enc_sig, rgb, device)
            d = attn_s.mean(0) - attn_u.mean(0)
            delta_l1_mean = float(np.abs(d).mean())
            delta_l1_max = float(np.abs(d).max())
            delta_l2 = float(np.sqrt((d * d).sum()))

            nc_str = nc_all[idx]
            rows.append(dict(
                sample_name=sample,
                anchor_cell_idx=int(idx),
                anchor_cell_id=str(data["cell_id"][idx]),
                bbox_x=int(x0), bbox_y=int(y0),
                niche_category=str(nc_str),
                n_cells_in_fov=n_in_fov,
                empty_fraction=round(ef, 4),
                mask_patch_frac=round(mask_patch_frac, 4),
                mean_rgb=round(mean_rgb, 2),
                delta_l1_mean=delta_l1_mean,
                delta_l1_max=delta_l1_max,
                delta_l2=delta_l2,
            ))
            taken += 1
        print(f"[scan:{sample}] {taken} FOVs scanned")
    return pd.DataFrame(rows)


def _analyze_top_features(df: pd.DataFrame, out_dir: Path,
                          *, top_frac: float = 0.10) -> str:
    """Compare top-decile vs bottom-decile of ||Δ̄||_1 on key features.
    Returns a multi-line markdown summary; also writes csv + txt."""
    df = df.sort_values("delta_l1_mean", ascending=False).reset_index(drop=True)
    n = len(df)
    k = max(1, int(n * top_frac))
    top = df.head(k); bot = df.tail(k); rest = df.iloc[k:n - k]

    lines = []
    lines.append(f"## Δ-attention feature analysis  (N scanned = {n}, top/bottom decile size = {k})\n")
    lines.append(f"### Headline Δ stats")
    for label, sub in (("Top decile", top), ("Bottom decile", bot)):
        lines.append(f"- {label}: ||Δ̄||_1 mean = {sub.delta_l1_mean.mean():.5f}, "
                     f"max |Δ| = {sub.delta_l1_max.mean():.5f}")

    lines.append("\n### Numeric features (mean ± std)")
    feats = ["n_cells_in_fov", "empty_fraction", "mask_patch_frac", "mean_rgb"]
    for f in feats:
        lines.append(
            f"- {f}: top={top[f].mean():.3f}±{top[f].std():.3f}  "
            f"vs bottom={bot[f].mean():.3f}±{bot[f].std():.3f}  "
            f"vs all-rest={rest[f].mean():.3f}±{rest[f].std():.3f}"
        )

    lines.append("\n### Niche distribution (top vs bottom)")
    top_nc = top.niche_category.value_counts(normalize=True)
    bot_nc = bot.niche_category.value_counts(normalize=True)
    all_nc = sorted(set(top_nc.index) | set(bot_nc.index))
    for nc in all_nc:
        t = float(top_nc.get(nc, 0)); b = float(bot_nc.get(nc, 0))
        delta = t - b
        lines.append(f"- {nc:30s}  top={t:.2%}  bot={b:.2%}  Δ={delta:+.2%}")

    summary = "\n".join(lines)
    (out_dir / "delta_feature_analysis.txt").write_text(summary)
    df.to_csv(out_dir / "scan_all_stats.csv", index=False)
    return summary


def _select_fovs(
    h5_dir: Path,
    slides_df: pd.DataFrame,
    *,
    n_fovs: int,
    split: str,
    seed: int,
    min_neighbors_in_fov: int = 5,
    niches: list[str] | None = None,
    boundary_pairs: list[tuple[str, str]] | None = None,
    boundary_min_frac: float = 0.20,
    tissue_edge: bool = False,
    edge_min_empty_frac: float = 0.15,
    edge_max_empty_frac: float = 0.70,
) -> pd.DataFrame:
    """Pick n_fovs anchor cells from distinct slides in the requested split.

    If ``niches`` is given, each FOV's anchor cell is drawn from one of the
    requested niche_category labels (rotating, one per niche). Falls back to
    ``split=='val'`` slides if the requested split has too few cached h5s.
    """
    rng = np.random.default_rng(seed)

    def _slides_for(split_value: str) -> list[str]:
        names = slides_df.loc[slides_df["split"] == split_value, "sample_name"].tolist()
        return [s for s in names if (h5_dir / f"{s}.h5").exists()]

    slides = _slides_for(split)
    if len(slides) < n_fovs:
        extra = _slides_for("val") if split == "test" else _slides_for("test")
        print(f"[warn] only {len(slides)} {split}-split slides cached; "
              f"backfilling from {'val' if split=='test' else 'test'}")
        for s in extra:
            if s not in slides:
                slides.append(s)
            if len(slides) >= n_fovs:
                break
    if len(slides) < n_fovs:
        raise SystemExit(f"need {n_fovs} cached slides, only {len(slides)} available")

    rng.shuffle(slides)

    # If tissue-edge mode, find FOVs with significant empty-tissue area
    # (white background pixels) — e.g., follicle lumen, slide edge, vessel
    # lumen, adipose voids. Each picked FOV must still contain at least
    # ``min_neighbors_in_fov`` cells (so it's a partial-tissue crop, not
    # entirely off-tissue).
    if tissue_edge:
        used_slides: set[str] = set()
        picked = []
        for sample in slides:
            if len(picked) >= n_fovs:
                break
            if sample in used_slides:
                continue
            data = _read_data_h5(h5_dir / f"{sample}.h5")
            with h5py.File(h5_dir / f"{sample}.h5", "r") as f:
                xp = f["x_pixel"][:]; yp = f["y_pixel"][:]
            bbox = data["bbox_macro"]
            size = data["scale_sizes"][-1]
            he_path = slides_df.loc[slides_df["sample_name"] == sample,
                                    "hematoxylin_eosin_image"].iloc[0]
            he_zarr = zarr.open(he_path, mode="r")

            # Shuffle and scan; check empty-fraction on a budget of candidates
            # to keep this fast (zarr reads are the bottleneck).
            order = rng.permutation(len(bbox))
            scanned = 0
            chosen = None
            for idx in order:
                scanned += 1
                if scanned > 300:
                    break
                x0, y0 = int(bbox[idx, 0]), int(bbox[idx, 1])
                in_fov = ((xp >= x0) & (xp < x0 + size)
                          & (yp >= y0) & (yp < y0 + size))
                n_in_fov = int(in_fov.sum())
                if not (min_neighbors_in_fov <= n_in_fov <= 80):
                    # need some cells, but tissue-edge means partial coverage
                    continue
                # Read the crop; cheap because it's a single slice.
                crop = _read_crops_from_zarr(he_zarr,
                                             np.array([[x0, y0]], dtype=np.int32),
                                             size)[0]  # (3, S, S) uint8
                ef = _empty_fraction(crop)
                if edge_min_empty_frac <= ef <= edge_max_empty_frac:
                    nc = data["niche_category"][idx]
                    nc_str = nc.decode() if isinstance(nc, (bytes, np.bytes_)) else str(nc)
                    chosen = dict(
                        sample_name=sample, anchor_cell_idx=int(idx),
                        anchor_cell_id=str(data["cell_id"][idx]),
                        bbox_x=int(x0), bbox_y=int(y0),
                        niche_category=nc_str,
                        n_cells_in_fov=n_in_fov,
                        empty_fraction=round(ef, 3),
                    )
                    break
            if chosen is None:
                print(f"[warn] {sample}: no tissue-edge FOV in "
                      f"{[edge_min_empty_frac, edge_max_empty_frac]} found "
                      f"in {scanned} candidates; skip")
                continue
            used_slides.add(sample)
            picked.append(chosen)
        return pd.DataFrame(picked)

    # If boundary pairs were specified, find FOVs straddling two niches.
    if boundary_pairs:
        if len(boundary_pairs) < n_fovs:
            boundary_pairs = (boundary_pairs * ((n_fovs + len(boundary_pairs) - 1)
                                                // len(boundary_pairs)))[:n_fovs]
        used_slides: set[str] = set()
        picked = []
        for (niche_A, niche_B) in boundary_pairs[:n_fovs]:
            chosen = None
            for sample in slides:
                if sample in used_slides:
                    continue
                data = _read_data_h5(h5_dir / f"{sample}.h5")
                nc_all = np.array([
                    (v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v))
                    for v in data["niche_category"]
                ])
                # Anchor candidates are cells of niche_A whose FOV contains
                # ≥boundary_min_frac of niche_B cells (relative to all cells
                # within the FOV).
                cand = np.where(nc_all == niche_A)[0]
                if len(cand) == 0:
                    continue
                with h5py.File(h5_dir / f"{sample}.h5", "r") as f:
                    xp = f["x_pixel"][:]; yp = f["y_pixel"][:]
                bbox = data["bbox_macro"]
                size = data["scale_sizes"][-1]
                cand_shuf = rng.permutation(cand)
                for idx in cand_shuf:
                    x0, y0 = int(bbox[idx, 0]), int(bbox[idx, 1])
                    in_fov = ((xp >= x0) & (xp < x0 + size)
                              & (yp >= y0) & (yp < y0 + size))
                    n_in_fov = int(in_fov.sum())
                    if n_in_fov < min_neighbors_in_fov:
                        continue
                    nc_in_fov = nc_all[in_fov]
                    n_A = int((nc_in_fov == niche_A).sum())
                    n_B = int((nc_in_fov == niche_B).sum())
                    if n_A / n_in_fov >= boundary_min_frac \
                            and n_B / n_in_fov >= boundary_min_frac:
                        nc_str = f"{niche_A} ↔ {niche_B}"
                        chosen = (sample, int(idx), n_in_fov, nc_str,
                                  int(bbox[idx, 0]), int(bbox[idx, 1]),
                                  str(data["cell_id"][idx]),
                                  n_A, n_B)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                print(f"[warn] no boundary FOV for {niche_A}↔{niche_B}, skip")
                continue
            sample, idx, n_in_fov, nc_str, bx, by, cid, n_A, n_B = chosen
            used_slides.add(sample)
            picked.append(dict(
                sample_name=sample, anchor_cell_idx=idx, anchor_cell_id=cid,
                bbox_x=bx, bbox_y=by, niche_category=nc_str,
                n_cells_in_fov=n_in_fov, n_A=n_A, n_B=n_B,
            ))
        return pd.DataFrame(picked)

    # If niches were specified, rotate through them; otherwise no filter.
    if niches:
        if len(niches) < n_fovs:
            # Cycle if user gave fewer niches than FOVs
            niches = (niches * ((n_fovs + len(niches) - 1) // len(niches)))[:n_fovs]
        # Find the best (sample, anchor) for each requested niche, trying
        # distinct slides where possible.
        used_slides: set[str] = set()
        picked = []
        for want_niche in niches[:n_fovs]:
            chosen = None
            for sample in slides:
                if sample in used_slides:
                    continue
                data = _read_data_h5(h5_dir / f"{sample}.h5")
                nc_all = np.array([
                    (v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v))
                    for v in data["niche_category"]
                ])
                cand = np.where(nc_all == want_niche)[0]
                if len(cand) == 0:
                    continue
                # Centroids for density check.
                with h5py.File(h5_dir / f"{sample}.h5", "r") as f:
                    xp = f["x_pixel"][:]; yp = f["y_pixel"][:]
                bbox = data["bbox_macro"]
                size = data["scale_sizes"][-1]
                cand_shuf = rng.permutation(cand)
                for idx in cand_shuf:
                    x0, y0 = int(bbox[idx, 0]), int(bbox[idx, 1])
                    in_fov = ((xp >= x0) & (xp < x0 + size)
                              & (yp >= y0) & (yp < y0 + size))
                    if int(in_fov.sum()) >= min_neighbors_in_fov:
                        chosen = (sample, int(idx), int(in_fov.sum()), want_niche,
                                  int(bbox[idx, 0]), int(bbox[idx, 1]),
                                  str(data["cell_id"][idx]))
                        break
                if chosen is not None:
                    break
            if chosen is None:
                print(f"[warn] niche {want_niche!r}: no qualifying anchor found, skip")
                continue
            sample, idx, n_in_fov, nc_str, bx, by, cid = chosen
            used_slides.add(sample)
            picked.append(dict(
                sample_name=sample, anchor_cell_idx=idx, anchor_cell_id=cid,
                bbox_x=bx, bbox_y=by, niche_category=nc_str,
                n_cells_in_fov=n_in_fov,
            ))
        return pd.DataFrame(picked)

    # Original niche-agnostic selection.
    picked = []
    for sample in slides:
        if len(picked) >= n_fovs:
            break
        data = _read_data_h5(h5_dir / f"{sample}.h5")
        N = len(data["cell_id"])
        bbox = data["bbox_macro"]
        with h5py.File(h5_dir / f"{sample}.h5", "r") as f:
            xp = f["x_pixel"][:]; yp = f["y_pixel"][:]
        size = data["scale_sizes"][-1]
        order = rng.permutation(N)
        chosen = None
        for idx in order:
            x0, y0 = int(bbox[idx, 0]), int(bbox[idx, 1])
            in_fov = ((xp >= x0) & (xp < x0 + size)
                      & (yp >= y0) & (yp < y0 + size))
            if int(in_fov.sum()) >= min_neighbors_in_fov:
                chosen = (idx, int(in_fov.sum()))
                break
        if chosen is None:
            print(f"[warn] {sample}: no FOV with >={min_neighbors_in_fov} cells, "
                  f"using random anchor")
            chosen = (int(order[0]), 0)
        idx, n_in_fov = chosen
        nc = data["niche_category"][idx]
        nc_str = nc.decode() if isinstance(nc, (bytes, np.bytes_)) else str(nc)
        picked.append(dict(
            sample_name=sample, anchor_cell_idx=int(idx),
            anchor_cell_id=str(data["cell_id"][idx]),
            bbox_x=int(bbox[idx, 0]), bbox_y=int(bbox[idx, 1]),
            niche_category=nc_str, n_cells_in_fov=int(n_in_fov),
        ))
    return pd.DataFrame(picked)


def _fov_data(
    fov_row: pd.Series,
    h5_dir: Path,
    slides_df: pd.DataFrame,
) -> dict:
    """Load (rgb, all cell masks within FOV, niche labels) for one FOV."""
    sample = fov_row["sample_name"]
    bbox = np.array([[fov_row["bbox_x"], fov_row["bbox_y"]]], dtype=np.int32)
    data = _read_data_h5(h5_dir / f"{sample}.h5")
    size = int(data["scale_sizes"][-1])  # macro = 256

    he_path = slides_df.loc[slides_df["sample_name"] == sample,
                            "hematoxylin_eosin_image"].iloc[0]
    he_zarr = zarr.open(he_path, mode="r")
    crops = _read_crops_from_zarr(he_zarr, bbox, size)  # (1, 3, 256, 256) uint8
    rgb = crops[0].transpose(1, 2, 0)  # (256, 256, 3)

    # Find all cells whose macro bbox overlaps with the anchor FOV.
    all_bbox = data["bbox_macro"]
    overlap_mask = (
        (all_bbox[:, 0] + size > bbox[0, 0])
        & (all_bbox[:, 0] < bbox[0, 0] + size)
        & (all_bbox[:, 1] + size > bbox[0, 1])
        & (all_bbox[:, 1] < bbox[0, 1] + size)
    )
    overlap_idx = np.where(overlap_mask)[0]

    masks_in_fov = []
    for i in overlap_idx:
        # mask_macro[i] is at this cell's own bbox coordinate system. Shift to anchor FOV.
        mask = data["mask_macro"][i]  # (256, 256) bool
        x_off = int(all_bbox[i, 0] - bbox[0, 0])
        y_off = int(all_bbox[i, 1] - bbox[0, 1])
        # Place mask onto a 256x256 canvas aligned to anchor FOV.
        canvas = np.zeros((size, size), dtype=bool)
        # Source region in mask: pixels that, when shifted, fall within canvas
        sy0 = max(0, -y_off); sy1 = size - max(0, y_off + size - size)
        sx0 = max(0, -x_off); sx1 = size - max(0, x_off + size - size)
        # Dst region in canvas
        dy0 = max(0, y_off); dy1 = min(size, y_off + size)
        dx0 = max(0, x_off); dx1 = min(size, x_off + size)
        if dy1 <= dy0 or dx1 <= dx0 or sy1 <= sy0 or sx1 <= sx0:
            continue
        h = min(dy1 - dy0, sy1 - sy0)
        w = min(dx1 - dx0, sx1 - sx0)
        canvas[dy0:dy0 + h, dx0:dx0 + w] = mask[sy0:sy0 + h, sx0:sx0 + w]
        if canvas.any():
            masks_in_fov.append(canvas)

    return dict(rgb=rgb, masks=masks_in_fov, anchor_idx=int(fov_row["anchor_cell_idx"]))


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _attn_for_fov(
    encoder: _ViTBackbone,
    rgb256: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """rgb256 (256, 256, 3) uint8 -> (16, 14, 14) float32 softmax."""
    x = torch.from_numpy(rgb256.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, 256, 256)
    x = x.to(device).float() / 255.0
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = _IMAGENET_MEAN.to(x)
    std = _IMAGENET_STD.to(x)
    x = (x - mean) / std
    attn = encoder.extract_cls_attention(x)  # (1, 16, 14, 14)
    return attn.squeeze(0).float().cpu().numpy()


def _sanity_checks(enc_uni: _ViTBackbone, enc_sig: _ViTBackbone, device: torch.device) -> None:
    """Cheap startup invariants: shape, non-negativity, attention < 1 (softmax sanity).

    Note: per-head patch-row sums vary in [0, 1] depending on how much each head
    attends to prefix tokens (CLS + 8 register tokens in UNI2-h); we don't
    require a tight expected sum.
    """
    rng = np.random.default_rng(0)
    fake = rng.uniform(0, 255, size=(2, 256, 256, 3)).astype(np.uint8)
    for name, enc in (("UNI", enc_uni), ("SIGMMA", enc_sig)):
        for i in range(fake.shape[0]):
            a = _attn_for_fov(enc, fake[i], device)
            assert a.ndim == 3 and a.shape[0] == enc.num_heads, \
                f"{name}: unexpected shape {a.shape} (heads={enc.num_heads})"
            g = a.shape[1]
            assert a.shape[2] == g, f"{name}: non-square grid {a.shape}"
            assert (a >= 0).all(), f"{name}: negative attention values"
            # Each entry is a softmax probability, so must be in [0, 1].
            assert a.max() <= 1.0 + 1e-4, f"{name}: attn > 1.0 ({a.max()})"
            # Partial row sum (patch positions only) is at most 1.
            s = a.sum(axis=(-2, -1))
            assert s.max() <= 1.0 + 1e-4, f"{name}: patch row sum > 1 ({s.max():.4f})"
    print(f"[sanity] passed. grid={g}x{g}, heads={enc_uni.num_heads}/{enc_sig.num_heads}, "
          f"num_prefix={enc_uni.num_prefix}/{enc_sig.num_prefix}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _attn_overlay_native(a14: np.ndarray, size: int) -> np.ndarray:
    """Upsample (14,14) -> (size,size) bilinear (size usually 256)."""
    t = torch.from_numpy(a14.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out.squeeze().numpy()


def _plot_grid(
    fov_rows: pd.DataFrame,
    fov_data_list: list[dict],
    attn_uni: list[np.ndarray],
    attn_sig: list[np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    """3 (FOV) x 4 (input/UNI/SIGMMA/Δ) grid; returns per-FOV stats."""
    n = len(fov_rows)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 2.5, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    # Cross-FOV symmetric vmax for Δ (RdBu_r). Δ defined as SIGMMA − UNI.
    delta_mean_list = [as_.mean(0) - au.mean(0) for au, as_ in zip(attn_uni, attn_sig)]
    delta_vmax = max(float(np.abs(d).max()) for d in delta_mean_list)

    stats = []
    for r, (_, fov_row) in enumerate(fov_rows.iterrows()):
        rgb = fov_data_list[r]["rgb"]
        masks = fov_data_list[r]["masks"]
        au = attn_uni[r].mean(0)   # head-mean (14, 14)
        as_ = attn_sig[r].mean(0)
        d14 = as_ - au

        # col 0: input + cell mask contours
        ax = axes[r, 0]
        ax.imshow(rgb)
        for m in masks:
            for c in find_contours(m.astype(np.float32), level=0.5):
                ax.plot(c[:, 1], c[:, 0], color="#00d7ff", lw=0.6)
        ax.set_ylabel(f"{fov_row['sample_name']}\n{fov_row['niche_category']}",
                      fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("H&E + cell masks", fontsize=10)

        # cols 1, 2: UNI / SIGMMA attention overlay
        for ci, (name, a14) in enumerate((("UNI2-h", au), ("SIGMMA α=0.1", as_))):
            ax = axes[r, 1 + ci]
            A = _attn_overlay_native(a14, size=rgb.shape[0])
            ax.imshow(rgb)
            ax.imshow(A, cmap="inferno", alpha=0.55,
                      vmin=A.min(), vmax=A.max())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)

        # col 3: Δ map (RdBu_r, symmetric vmax)
        ax = axes[r, 3]
        D = _attn_overlay_native(d14, size=rgb.shape[0])
        im = ax.imshow(D, cmap="RdBu_r", vmin=-delta_vmax, vmax=delta_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("Δ = SIGMMA − UNI", fontsize=10)
        l1 = float(np.abs(d14).mean())
        ax.text(0.02, 0.98, f"|Δ|₁={l1:.3f}", transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        stats.append(dict(sample_name=fov_row["sample_name"],
                          anchor_cell_id=fov_row["anchor_cell_id"],
                          niche_category=fov_row["niche_category"],
                          n_cells_in_fov=int(fov_row["n_cells_in_fov"]),
                          delta_l1_mean=l1,
                          delta_l1_max=float(np.abs(d14).max())))

    # One shared colorbar for the Δ column.
    cbar = fig.colorbar(im, ax=axes[:, 3].tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("Δ attention", fontsize=8)
    fig.suptitle(
        "CLS→patch attention (last ViT block, head-mean): base UNI2-h vs "
        "SIGMMA α=0.1 step1000 on Xenium H&E\n"
        "Caveat: ViT-CLS attention; SIGMMA contrastive signal flows through "
        "dense mask-pool, not CLS.",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    out_png = out_dir / "fov_grid.png"
    out_pdf = out_dir / "fov_grid.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_png} + {out_pdf}")
    return pd.DataFrame(stats)


@torch.no_grad()
def _scale_query_attn_for_fov(
    sigmma_model,
    rgb256: np.ndarray,
    device: torch.device,
    *,
    scale_idx: int = 0,
) -> np.ndarray:
    """SIGMMA scale-query cross-attention: how scale_queries[scale_idx] attends
    over the adaptively-pooled region tokens at that scale.

    Returns (num_heads, grid, grid) float32, where grid = 2^(num_resolutions-1-s).
    For num_resolutions=3: micro→4×4, meso→2×2, macro→1×1.
    """
    image_encoder = sigmma_model.image_encoder
    inner = image_encoder.model

    # ImageNet-normalize a 224 crop.
    x = torch.from_numpy(rgb256.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255.0
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = _IMAGENET_MEAN.to(x); std = _IMAGENET_STD.to(x)
    x = (x - mean) / std

    # 1) ViT forward.
    tokens_all = inner.forward_features(x)  # (1, num_prefix + 256, D)
    num_prefix = int(getattr(inner, "num_prefix_tokens", 1))
    patches = tokens_all[:, num_prefix:, :].float()
    pe = inner.patch_embed.patch_size
    patch_size = int(pe[0] if isinstance(pe, (tuple, list)) else pe)
    P = 224 // patch_size
    D = patches.shape[-1]
    B = 1
    patches_2d = patches.transpose(1, 2).contiguous().view(B, D, P, P)

    # 2) Adaptive pool to per-scale grid.
    num_resolutions = sigmma_model.num_resolutions
    grid = 2 ** (num_resolutions - 1 - scale_idx)
    region_tokens = F.adaptive_avg_pool2d(patches_2d, grid)              # (1, D, g, g)
    region_tokens = region_tokens.flatten(2).transpose(1, 2)              # (1, g², D)

    # 3) Cross-attention with scale_queries[s] as Q; expose per-head weights.
    q = sigmma_model.scale_queries[scale_idx].expand(B, -1, -1).to(region_tokens)
    _, attn_weights = sigmma_model.scale_pools[scale_idx](
        q, region_tokens, region_tokens,
        need_weights=True, average_attn_weights=False,
    )
    # attn_weights: (B, num_heads, 1, g²) → (num_heads, g, g)
    attn = attn_weights[0, :, 0, :].reshape(-1, grid, grid).float().cpu().numpy()
    return attn


def _plot_scale_query_six_heads(
    fov_rows: pd.DataFrame,
    fov_data_list: list[dict],
    attn_per_fov_per_scale: list[dict[int, np.ndarray]],
    out_dir: Path,
    *,
    n_heads_show: int = 6,
    scale_idx: int = 0,
) -> list[int]:
    """SIGMMA scale-query (Option B) attention: per-FOV row of input + 6 heads
    at the requested scale (default micro). No UNI baseline (UNI lacks scale_pools)."""
    n_fovs = len(fov_rows)
    sample_attn = attn_per_fov_per_scale[0][scale_idx]    # (num_heads, g, g)
    num_heads = sample_attn.shape[0]
    head_ids = list(np.linspace(0, num_heads - 1, n_heads_show, dtype=int))
    cols = 1 + n_heads_show
    panel_in = 2.2
    fig, axes = plt.subplots(n_fovs, cols,
                             figsize=(cols * panel_in, n_fovs * panel_in))
    if n_fovs == 1:
        axes = axes[np.newaxis, :]

    scale_name = {0: "micro (4×4)", 1: "meso (2×2)", 2: "macro (1×1)"}[scale_idx]

    for fi, (_, fov_row) in enumerate(fov_rows.iterrows()):
        rgb = fov_data_list[fi]["rgb"]
        masks = fov_data_list[fi]["masks"]
        size = rgb.shape[0]
        attn = attn_per_fov_per_scale[fi][scale_idx]

        # col 0: H&E + cyan cell-mask contour
        ax = axes[fi, 0]
        ax.imshow(rgb)
        for m in masks:
            for c in find_contours(m.astype(np.float32), level=0.5):
                ax.plot(c[:, 1], c[:, 0], color="#1f77ff", lw=0.7)
        ax.set_ylabel(f"{fov_row['sample_name'][:18]}\n{fov_row['niche_category']}",
                      fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if fi == 0:
            ax.set_title("H&E + cell mask", fontsize=10)

        # cols 1..n_heads_show: per-head scale-query attention heatmap
        for ci, h_idx in enumerate(head_ids):
            ax = axes[fi, 1 + ci]
            A = _attn_overlay_native(attn[h_idx], size=size)
            ax.imshow(A, cmap="viridis", vmin=A.min(), vmax=A.max())
            ax.set_xticks([]); ax.set_yticks([])
            if fi == 0:
                ax.set_title(f"head {h_idx + 1}", fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.text(
        0.5, 0.01,
        f"SIGMMA scale-query cross-attention at {scale_name} scale "
        f"(Option B: scale_pools[{scale_idx}] with learned scale_queries[{scale_idx}]).\n"
        f"Heatmap shows how the learnable query attends to "
        f"adaptively-pooled region tokens — the actual signal SIGMMA's "
        f"contrastive head receives. Blue contour: cell-segmentation mask.",
        ha="center", va="bottom", fontsize=9,
    )
    tag = ["micro", "meso", "macro"][scale_idx]
    out_png = out_dir / f"fov_grid_scale_query_{tag}.png"
    out_pdf = out_dir / f"fov_grid_scale_query_{tag}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[plot] saved {out_png} + {out_pdf}  (heads={head_ids}, scale={tag})")
    return [int(h) for h in head_ids]


def _plot_top_shifted_patches(
    fov_rows: pd.DataFrame,
    fov_data_list: list[dict],
    attn_uni: list[np.ndarray],
    attn_sig: list[np.ndarray],
    out_dir: Path,
    *,
    top_k_patches: int = 5,
) -> pd.DataFrame:
    """For each FOV: find the top-K grid positions with largest |Δ̄|, outline them
    on H&E + Δ map, and report the per-patch Δ.

    Layout: n_fovs rows × 4 cols:
      [H&E + cell mask + top-K patch boxes]
      [UNI2-h head-mean attn]
      [SIGMMA head-mean attn]
      [|Δ| heatmap with top-K patch boxes]
    """
    n = len(fov_rows)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 2.5, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    g = attn_uni[0].shape[1]                  # patch grid dim (e.g., 16)
    abs_delta_list = [np.abs(s.mean(0) - u.mean(0)) for u, s in zip(attn_uni, attn_sig)]
    delta_vmax = max(float(d.max()) for d in abs_delta_list)

    box_rows = []
    for r, (_, fov_row) in enumerate(fov_rows.iterrows()):
        rgb = fov_data_list[r]["rgb"]
        masks = fov_data_list[r]["masks"]
        size = rgb.shape[0]
        u14 = attn_uni[r].mean(0)
        s14 = attn_sig[r].mean(0)
        d14 = s14 - u14
        abs_d14 = np.abs(d14)

        # Top-K patch positions in the 16x16 grid.
        flat_idx = np.argsort(abs_d14.ravel())[::-1][:top_k_patches]
        ys, xs = np.unravel_index(flat_idx, abs_d14.shape)
        patch_px = size // g  # patch side in source-image pixels

        # col 0: H&E + cell masks + top-K boxes
        ax = axes[r, 0]
        ax.imshow(rgb)
        for m in masks:
            for c in find_contours(m.astype(np.float32), level=0.5):
                ax.plot(c[:, 1], c[:, 0], color="#00d7ff", lw=0.5)
        for rank, (yi, xi) in enumerate(zip(ys, xs)):
            sign = d14[yi, xi]
            color = "#d62728" if sign > 0 else "#1f77b4"
            ax.add_patch(plt.Rectangle((xi * patch_px, yi * patch_px),
                                       patch_px, patch_px,
                                       fill=False, edgecolor=color, lw=1.5))
            ax.text(xi * patch_px + 2, yi * patch_px + patch_px - 2,
                    str(rank + 1), color=color, fontsize=8, fontweight="bold")
            box_rows.append(dict(
                sample_name=fov_row["sample_name"],
                anchor_cell_id=fov_row["anchor_cell_id"],
                rank=rank + 1,
                grid_x=int(xi), grid_y=int(yi),
                px_x=int(xi * patch_px), px_y=int(yi * patch_px),
                delta_value=float(d14[yi, xi]),
                abs_delta=float(abs_d14[yi, xi]),
                sign=("SIGMMA>UNI" if sign > 0 else "SIGMMA<UNI"),
            ))
        ax.set_ylabel(f"{fov_row['sample_name'][:18]}\n{fov_row['niche_category']}",
                      fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"H&E + top-{top_k_patches} shifted patches", fontsize=10)

        # col 1: UNI head-mean
        ax = axes[r, 1]
        A = _attn_overlay_native(u14, size=size)
        ax.imshow(A, cmap="viridis", vmin=A.min(), vmax=A.max())
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("UNI2-h (head-mean)", fontsize=10)

        # col 2: SIGMMA head-mean
        ax = axes[r, 2]
        A = _attn_overlay_native(s14, size=size)
        ax.imshow(A, cmap="viridis", vmin=A.min(), vmax=A.max())
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("SIGMMA α=0.1 (head-mean)", fontsize=10)

        # col 3: |Δ| heatmap (raw 16x16, no upsample → integer patch grid visible)
        ax = axes[r, 3]
        im = ax.imshow(abs_d14, cmap="magma", vmin=0, vmax=delta_vmax,
                       interpolation="nearest")
        for rank, (yi, xi) in enumerate(zip(ys, xs)):
            sign = d14[yi, xi]
            color = "#ff6b6b" if sign > 0 else "#4dabf7"
            ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1,
                                       fill=False, edgecolor=color, lw=1.5))
            ax.text(xi, yi, str(rank + 1), color=color, fontsize=8,
                    ha="center", va="center", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("|Δ| (16×16 grid)", fontsize=10)

    fig.colorbar(im, ax=axes[:, 3].tolist(), shrink=0.7, pad=0.02,
                 label="|SIGMMA − UNI|")
    fig.suptitle(
        f"Top-{top_k_patches} patches with largest |Δ| per FOV. Red box = "
        f"SIGMMA attends MORE than UNI; blue box = LESS.",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out_png = out_dir / "fov_top_shifted_patches.png"
    out_pdf = out_dir / "fov_top_shifted_patches.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    df_box = pd.DataFrame(box_rows)
    df_box.to_csv(out_dir / "top_shifted_patches.csv", index=False)
    print(f"[plot] saved {out_png} + {out_pdf} + top_shifted_patches.csv")
    return df_box


def _plot_six_heads(
    fov_rows: pd.DataFrame,
    fov_data_list: list[dict],
    attn_uni: list[np.ndarray],   # each (num_heads, g, g) base UNI2-h
    attn_sig: list[np.ndarray],   # each (num_heads, g, g) SIGMMA fine-tuned
    out_dir: Path,
    *,
    n_heads_show: int = 6,
    seed: int = 0,
) -> list[int]:
    """Manuscript-style: per-FOV TWO rows (UNI2-h baseline + SIGMMA) of
    input + 6 individual head attention heatmaps (viridis, no overlay)."""
    n_fovs = len(fov_rows)
    num_heads = attn_sig[0].shape[0]
    head_ids = list(np.linspace(0, num_heads - 1, n_heads_show, dtype=int))
    cols = 1 + n_heads_show
    rows = n_fovs * 2  # UNI + SIGMMA per FOV
    panel_in = 2.0
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * panel_in, rows * panel_in))
    if rows == 1:
        axes = axes[np.newaxis, :]

    cmap = "viridis"

    for fi, (_, fov_row) in enumerate(fov_rows.iterrows()):
        rgb = fov_data_list[fi]["rgb"]
        masks = fov_data_list[fi]["masks"]
        size = rgb.shape[0]

        for mi, (model_label, attn_per_fov) in enumerate(
            (("UNI2-h", attn_uni[fi]), ("SIGMMA α=0.1", attn_sig[fi]))
        ):
            r = fi * 2 + mi  # absolute row index

            # col 0: H&E with cyan cell-mask contour
            ax = axes[r, 0]
            ax.imshow(rgb)
            for m in masks:
                for c in find_contours(m.astype(np.float32), level=0.5):
                    ax.plot(c[:, 1], c[:, 0], color="#1f77ff", lw=0.7)
            ax.set_ylabel(
                f"{fov_row['sample_name']}\n{fov_row['niche_category']}\n{model_label}",
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("H&E + cell mask", fontsize=10)

            # cols 1..n_heads_show: per-head attention heatmap (pure cmap).
            for ci, h_idx in enumerate(head_ids):
                ax = axes[r, 1 + ci]
                A = _attn_overlay_native(attn_per_fov[h_idx], size=size)
                ax.imshow(A, cmap=cmap, vmin=A.min(), vmax=A.max())
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"head {h_idx + 1}", fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    fig.text(
        0.5, 0.01,
        f"Attention maps from {n_heads_show} attention heads in the last encoder "
        f"layer (L = 24) of the UNI image encoder backbone (baseline UNI2-h vs "
        f"SIGMMA fine-tuned),\n"
        f"illustrating class-token–to-patch attention distributions. Blue contour "
        f"overlaid on the input images indicates the cell-segmentation mask, "
        f"marking the boundaries of cell-rich regions.",
        ha="center", va="bottom", fontsize=9,
    )
    out_png = out_dir / "fov_grid_six_heads.png"
    out_pdf = out_dir / "fov_grid_six_heads.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[plot] saved {out_png} + {out_pdf}  (heads={head_ids})")
    return [int(h) for h in head_ids]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmma_config", required=True)
    ap.add_argument("--manifest", required=True,
                    help="cell_niche_evaluable_slides.csv")
    ap.add_argument("--h5_dir", required=True,
                    help="cache/data/cell_niche/scales_64_128_256")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_fovs", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_heads_show", type=int, default=6,
                    help="Heads displayed in the six-heads manuscript-style figure")
    ap.add_argument("--niches", default=None,
                    help="Comma-separated list of niche_category labels to draw "
                         "FOV anchors from, e.g. "
                         "'Pilosebaceous unit,Neurovascular,Epidermis'. If unset, "
                         "FOVs are chosen niche-agnostic.")
    ap.add_argument("--boundary_pairs", default=None,
                    help="Comma-separated pairs of niches (joined by ':') for "
                         "niche-boundary FOV selection, e.g. "
                         "'Epidermis:Dermis layers,Pilosebaceous unit:Dermis layers'. "
                         "Each FOV will straddle the two niches (≥20% each).")
    ap.add_argument("--tissue_edge", action="store_true",
                    help="Pick FOVs where the H&E crop has a substantial "
                         "fraction of empty/background pixels (e.g., follicle "
                         "lumen, slide edge, vessel lumen, adipose voids). "
                         "Tunable via --edge_min_empty / --edge_max_empty.")
    ap.add_argument("--edge_min_empty", type=float, default=0.15,
                    help="Min fraction of background pixels for tissue-edge "
                         "FOV (default 0.15).")
    ap.add_argument("--edge_max_empty", type=float, default=0.70,
                    help="Max fraction of background pixels (default 0.70).")
    ap.add_argument("--scan_n", type=int, default=0,
                    help="If >0, scan this many random FOVs across slides, "
                         "rank by ||Δ̄||_1 and pick top --n_fovs for plotting + "
                         "feature analysis. Overrides --niches/--boundary/--tissue_edge.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slides_df = pd.read_csv(args.manifest)
    h5_dir = Path(args.h5_dir)

    # Load encoders first (needed for scan mode).
    enc_uni = load_uni(args.sigmma_config, device)
    enc_sig = load_sigmma(args.sigmma_config, device, scale_idx=0)
    print(f"[load] enc_uni={enc_uni.name} enc_sig={enc_sig.name}")
    _sanity_checks(enc_uni, enc_sig, device)

    # 1) Pick FOVs.
    if args.scan_n > 0:
        print(f"[scan] sweeping {args.scan_n} FOVs...")
        scan_df = _scan_delta(h5_dir, slides_df, enc_uni, enc_sig, device,
                              split=args.split, n_total=args.scan_n, seed=args.seed)
        print(f"[scan] {len(scan_df)} FOVs scanned. Top-{args.n_fovs} "
              f"by ||Δ̄||_1:")
        scan_df = scan_df.sort_values("delta_l1_mean", ascending=False).reset_index(drop=True)
        print(scan_df.head(args.n_fovs).to_string(index=False))
        fov_rows = scan_df.head(args.n_fovs)[
            ["sample_name", "anchor_cell_idx", "anchor_cell_id",
             "bbox_x", "bbox_y", "niche_category", "n_cells_in_fov"]
        ].copy()
        summary = _analyze_top_features(scan_df, out_dir)
        print(summary)
    else:
        niches_list = [s.strip() for s in args.niches.split(",")] if args.niches else None
        boundary_pairs = None
        if args.boundary_pairs:
            boundary_pairs = [
                tuple(p.strip() for p in pair.split(":"))
                for pair in args.boundary_pairs.split(",")
            ]
            for p in boundary_pairs:
                if len(p) != 2:
                    raise SystemExit(f"--boundary_pairs entry must be 'A:B', got {p}")
        fov_rows = _select_fovs(h5_dir, slides_df,
                                n_fovs=args.n_fovs, split=args.split, seed=args.seed,
                                niches=niches_list, boundary_pairs=boundary_pairs,
                                tissue_edge=args.tissue_edge,
                                edge_min_empty_frac=args.edge_min_empty,
                                edge_max_empty_frac=args.edge_max_empty)
        print(f"[select] {len(fov_rows)} FOVs")
        print(fov_rows.to_string(index=False))
    fov_rows.to_csv(out_dir / "selected_fovs.csv", index=False)

    # 3) For each FOV, gather + extract attention
    fov_data_list, attn_uni, attn_sig = [], [], []
    for _, row in fov_rows.iterrows():
        d = _fov_data(row, h5_dir, slides_df)
        fov_data_list.append(d)
        attn_uni.append(_attn_for_fov(enc_uni, d["rgb"], device))
        attn_sig.append(_attn_for_fov(enc_sig, d["rgb"], device))
        print(f"[attn] {row['sample_name']}: "
              f"||Δ̄||₁={float(np.abs(attn_sig[-1].mean(0) - attn_uni[-1].mean(0)).mean()):.4f}")

    # 4) Plot grid (UNI vs SIGMMA + Δ)
    stats = _plot_grid(fov_rows, fov_data_list, attn_uni, attn_sig, out_dir)
    stats.to_csv(out_dir / "per_fov_stats.csv", index=False)
    print(stats.to_string(index=False))

    # 5) Manuscript-style: 6 heads, UNI2-h baseline above SIGMMA per FOV
    _plot_six_heads(fov_rows, fov_data_list, attn_uni, attn_sig, out_dir,
                    n_heads_show=args.n_heads_show, seed=args.seed)

    # 6) Top-shifted patches per FOV (highlight where attention moved most)
    _plot_top_shifted_patches(fov_rows, fov_data_list, attn_uni, attn_sig,
                              out_dir, top_k_patches=5)

    # 7) Option B — SIGMMA scale-query (scale_pools[s]) attention. Load full
    #    SIGMMA model for access to scale_pools/scale_queries; skip if it can't be
    #    loaded (e.g., ablation configs without scale heads).
    print("[scale-query] loading full SIGMMA model for scale_pools attention...")
    cfg = ConfigDict(yaml.safe_load(open(args.sigmma_config, "r")))
    sigmma_full = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=cfg.model.checkpoint_path,
        train=False,
    )
    sigmma_full = sigmma_full.to(device).eval()
    attn_per_fov_per_scale = []
    for fi, fov in enumerate(fov_data_list):
        per_scale = {}
        for s in (0, 1):   # skip macro (1x1, not visualizable)
            per_scale[s] = _scale_query_attn_for_fov(
                sigmma_full, fov["rgb"], device, scale_idx=s,
            )
        attn_per_fov_per_scale.append(per_scale)
    _plot_scale_query_six_heads(fov_rows, fov_data_list, attn_per_fov_per_scale,
                                out_dir, n_heads_show=args.n_heads_show,
                                scale_idx=0)
    _plot_scale_query_six_heads(fov_rows, fov_data_list, attn_per_fov_per_scale,
                                out_dir, n_heads_show=args.n_heads_show,
                                scale_idx=1)

    print(f"[done] outputs under {out_dir}")


if __name__ == "__main__":
    main()
