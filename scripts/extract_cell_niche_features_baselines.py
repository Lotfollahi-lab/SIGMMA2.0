"""
Per-cell H&E feature extraction for fine-tuned UNI baselines
(BLEEP / openAI_CLIP / PLIP / loki_OmiCLIP) — niche label-transfer task.

Mirrors ``scripts/extract_cell_niche_features.py`` but uses one of the four
baseline-trained UNI backbones instead of SIGMMA. The cell-centred crop +
polygon-mask token-pool pipeline is identical, so results are directly
comparable to the existing ``z_sigmma_*`` / ``z_uni_*`` columns.

Per slide it writes ``<out_dir>/<sample>.h5`` with these datasets (3 scales,
{short} ∈ {bleep,clip,plip,omiclip}):
  z_{short}_{micro,meso,macro}                 projected dense pool (proj dim)
  z_{short}_{micro,meso,macro}_dense_noproj    raw 1024-d dense pool
  z_{short}_{micro,meso,macro}_cls             CLS token (1024-d)
  cell_id / niche_category / niches_terra      labels (string + int code)

The cell ordering is ``sorted(set(polys) & set(centroids) & set(labels))`` —
identical to the SIGMMA extractor, so embeddings can be stacked column-wise
across feature dirs without re-aligning by cell_id.

Usage:
    python scripts/extract_cell_niche_features_baselines.py \
        --niche_config configs/cell_niche_transfer.yaml \
        --baseline_config configs/in-house_eva_bleep_uni.yaml \
        --out_dir /lustre/.../cell_niche/features_baselines/{JOBID}/bleep \
        [--slide BK25-SKI-27-FO-1-S3]
"""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from ml_collections.config_dict import ConfigDict

from sigmma.application.cell_niche_transfer.crops import (
    extract_centered_crop,
    rasterize_polygon,
)
from sigmma.application.cell_niche_transfer.extract import (
    _DEFAULT_LABEL_SPECS,
    _DEFAULT_SAMPLE_COL,
    _DEFAULT_UM_PER_PX,
    _load_labels_for_sample,
    _load_polygons,
    _load_preproc_centroids,
    encode_with_mask_pool,
    label_config_from_cfg,
    subsample_valid_ids,
)


_BASELINE_CLASS_BY_NAME = {
    "openAI_CLIP": ("sigmma.model.baselines", "openAI_CLIP"),
    "PLIP": ("sigmma.model.baselines", "PLIP"),
    "loki_OmiCLIP": ("sigmma.model.baselines", "loki_OmiCLIP"),
    "BLEEP": ("sigmma.model.baselines", "BLEEP"),
}
_BASELINE_SHORT = {
    "BLEEP": "bleep",
    "openAI_CLIP": "clip",
    "PLIP": "plip",
    "loki_OmiCLIP": "omiclip",
}


def _build_baseline(baseline_config_path: str, device: torch.device):
    """Build a UNI-FT baseline model + return (model, uni_wrapper, projection,
    short_name). Mirrors ``scripts/eva_critique/models.load_baseline_uni`` but
    keeps the projection head for downstream probing.
    """
    cfg = ConfigDict(yaml.safe_load(open(baseline_config_path)))
    if cfg.model.get("image_backbone", "default") != "uni":
        raise ValueError(
            f"{baseline_config_path}: image_backbone must be 'uni', got "
            f"{cfg.model.get('image_backbone')!r}"
        )

    mod_name, cls_name = _BASELINE_CLASS_BY_NAME[cfg.model.name]
    BaselineCls = getattr(importlib.import_module(mod_name), cls_name)

    kwargs = dict(
        image_backbone="uni",
        uni_ckpt_path=str(cfg.model.uni_ckpt_path),
    )
    if cfg.model.name == "BLEEP":
        kwargs.update(dict(
            spot_input_dim=int(cfg.model.spot_input_dim),
            image_embedding_dim=int(cfg.model.get("image_embedding_dim", 1024)),
            projection_dim=int(cfg.model.get("projection_dim", 256)),
            spot_hidden_dim=int(cfg.model.get("spot_hidden_dim", 512)),
            temperature=float(cfg.model.get("contrastive_temperature", 1.0)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            image_size=int(cfg.model.get("image_size", 224)),
            pretrained_image_encoder=False,
        ))
    elif cfg.model.name == "loki_OmiCLIP":
        kwargs.update(dict(
            pretrained_ckpt_path=str(cfg.model.pretrained_ckpt_path),
            freeze_encoders=False,
            loss_temperature=None,
        ))
    else:  # openAI_CLIP / PLIP
        kwargs.update(dict(
            hf_model_id=cfg.model.get(
                "hf_model_id",
                "openai/clip-vit-base-patch32" if cfg.model.name == "openAI_CLIP"
                else "vinid/plip",
            ),
            freeze_encoders=False,
            loss_temperature=None,
        ))

    model = BaselineCls(**kwargs)

    ckpt_path = cfg.model.get("checkpoint_path", None)
    if ckpt_path is None:
        raise ValueError(f"{baseline_config_path}: model.checkpoint_path required.")
    state = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load/{cfg.model.name}] missing {len(missing)} keys "
              f"(first: {missing[:3]})", flush=True)
    if unexpected:
        print(f"[load/{cfg.model.name}] unexpected {len(unexpected)} keys "
              f"(first: {unexpected[:3]})", flush=True)

    # All four baselines wrap the UNI backbone in a ``_UNIImageBackbone``; the
    # inner timm ViT is at ``.uni.model`` and the projection head is exposed
    # as ``image_projection`` on the parent baseline class.
    if cfg.model.name == "BLEEP":
        uni_wrapper = model.image_encoder.uni  # UNI(...) instance
    else:
        uni_wrapper = model._uni_image_encoder.uni
    projection_head = model.image_projection

    model = model.to(device).eval()
    uni_wrapper = uni_wrapper.to(device).eval()
    projection_head = projection_head.to(device).eval()

    return model, uni_wrapper, projection_head, _BASELINE_SHORT[cfg.model.name]


@torch.no_grad()
def _apply_projection(projection: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a baseline's projection head, casting input to the head's dtype.

    Linear heads (CLIP/PLIP/OmiCLIP) expose ``.weight``; BLEEP's residual
    ProjectionHead does not, so we walk its first Linear submodule instead.
    """
    weight = getattr(projection, "weight", None)
    if weight is None:
        for m in projection.modules():
            if isinstance(m, torch.nn.Linear):
                weight = m.weight
                break
    if weight is not None:
        x = x.to(weight.dtype)
    return projection(x)


def process_slide_baseline(
    *,
    slide_row: pd.Series,
    uni_wrapper: torch.nn.Module,
    projection_head: torch.nn.Module,
    short_name: str,
    labels_h5ad: str,
    scale_sizes: Tuple[int, int, int] = (64, 128, 256),
    encoder_input: int = 224,
    patch_size: int = 16,
    batch_size: int = 32,
    out_path: Path,
    sample_col: str = _DEFAULT_SAMPLE_COL,
    label_specs: Tuple[Tuple[str, str], ...] = _DEFAULT_LABEL_SPECS,
    normalize_sample: bool = True,
    um_per_px: float = _DEFAULT_UM_PER_PX,
    max_cells_per_slide: int | None = None,
    subsample_seed: int = 42,
) -> int:
    sample_name = slide_row["sample_name"]
    norm = slide_row.get("norm", sample_name)
    print(f"[{sample_name}] loading slide assets...", flush=True)

    px_per_um = 1.0 / um_per_px
    he_zarr = zarr.open(slide_row["hematoxylin_eosin_image"], mode="r")
    polys = _load_polygons(Path(slide_row["raw_dir"]))
    centroids = _load_preproc_centroids(slide_row["spatial_preprocessed"], um_per_px)
    labels = _load_labels_for_sample(
        labels_h5ad, norm,
        sample_col=sample_col, label_specs=label_specs,
        normalize_sample=normalize_sample,
    )

    valid_ids = sorted(set(polys) & set(centroids.index) & set(labels.index))
    print(f"[{sample_name}] cells with polygon ∧ centroid ∧ label: "
          f"{len(valid_ids)}", flush=True)
    if not valid_ids:
        return 0
    if max_cells_per_slide and len(valid_ids) > max_cells_per_slide:
        valid_ids = subsample_valid_ids(
            valid_ids, labels, max_cells_per_slide,
            seed=subsample_seed, strat_col=label_specs[0][0],
        )
        print(f"[{sample_name}] subsampled to {len(valid_ids)} cells "
              f"(stratified by {label_specs[0][0]}, seed={subsample_seed})", flush=True)

    scale_names = ("micro", "meso", "macro")
    label_names = [name for name, _ in label_specs]
    out: Dict[str, list] = {"cell_id": []}
    for name in label_names:
        out[name] = []
        out[f"{name}_code"] = []
    for s in scale_names:
        out[f"z_{short_name}_{s}"] = []
        out[f"z_{short_name}_{s}_dense_noproj"] = []
        out[f"z_{short_name}_{s}_cls"] = []

    n_done = 0
    for batch_start in range(0, len(valid_ids), batch_size):
        batch_ids = valid_ids[batch_start: batch_start + batch_size]

        per_scale_crops = {s: [] for s in scale_sizes}
        per_scale_masks = {s: [] for s in scale_sizes}
        for cid in batch_ids:
            row = centroids.loc[cid]
            cx_px, cy_px = float(row["x_pixel"]), float(row["y_pixel"])
            verts_px = polys[cid] * px_per_um  # µm → px at um_per_px µm/px
            for size in scale_sizes:
                crop, (x0, y0) = extract_centered_crop(he_zarr, (cx_px, cy_px), size)
                cmask = rasterize_polygon(verts_px, (x0, y0), (size, size))
                per_scale_crops[size].append(crop)
                per_scale_masks[size].append(cmask)

        for scale_idx, size in enumerate(scale_sizes):
            crops_t = torch.from_numpy(np.stack(per_scale_crops[size]))
            masks_t = torch.from_numpy(np.stack(per_scale_masks[size]))

            # Single forward: no LoRA switching (baselines are single-state
            # full fine-tuned UNI). Apply projection head externally so we
            # can support BLEEP's residual ProjectionHead without modifying
            # ``encode_with_mask_pool``.
            _, pooled_raw, cls = encode_with_mask_pool(
                uni_wrapper,
                crops_t,
                masks_t,
                encoder_input=encoder_input,
                patch_size=patch_size,
                projection_head=None,
            )
            proj = _apply_projection(projection_head, pooled_raw)

            sname = scale_names[scale_idx]
            out[f"z_{short_name}_{sname}"].append(proj.cpu().numpy())
            out[f"z_{short_name}_{sname}_dense_noproj"].append(pooled_raw.cpu().numpy())
            out[f"z_{short_name}_{sname}_cls"].append(cls.cpu().numpy())

        for cid in batch_ids:
            r = labels.loc[cid]
            out["cell_id"].append(cid)
            for name in label_names:
                v = r[name]
                out[name].append("" if pd.isna(v) else str(v))
                out[f"{name}_code"].append(int(r[f"{name}_code"]))

        n_done += len(batch_ids)
        if n_done % (batch_size * 10) == 0 or n_done == len(valid_ids):
            print(f"[{sample_name}] {n_done}/{len(valid_ids)} cells", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["sample_name"] = sample_name
        f.attrs["normalized_sample"] = norm
        f.attrs["split"] = slide_row.get("split", "")
        f.attrs["baseline"] = short_name
        f.attrs["scale_sizes"] = list(scale_sizes)
        f.attrs["encoder_input"] = encoder_input
        f.attrs["patch_size"] = patch_size
        f.attrs["label_names"] = label_names

        f.create_dataset("cell_id", data=np.array(out["cell_id"], dtype="S"))
        for name in label_names:
            f.create_dataset(name, data=np.array(out[name], dtype="S"))
            f.create_dataset(f"{name}_code",
                             data=np.array(out[f"{name}_code"], dtype=np.int16))
        for s in scale_names:
            f.create_dataset(f"z_{short_name}_{s}",
                             data=np.concatenate(out[f"z_{short_name}_{s}"], axis=0))
            f.create_dataset(f"z_{short_name}_{s}_dense_noproj",
                             data=np.concatenate(
                                 out[f"z_{short_name}_{s}_dense_noproj"], axis=0))
            f.create_dataset(f"z_{short_name}_{s}_cls",
                             data=np.concatenate(out[f"z_{short_name}_{s}_cls"], axis=0))
    print(f"[{sample_name}] wrote {out_path} ({n_done} cells)", flush=True)
    return n_done


def _normalize(s: str) -> str:
    import re
    if not isinstance(s, str):
        return s
    s = s.upper()
    s = re.sub(r"-FO-0+(\d)", r"-FO-\1", s)
    s = re.sub(r"-S\d+(-[A-Z]\d+)?$", "", s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--niche_config", required=True,
                        help="configs/cell_niche_transfer.yaml — crop sizes, "
                             "labels h5ad, evaluable-slides manifest.")
    parser.add_argument("--baseline_config", required=True,
                        help="configs/in-house_eva_{bleep,clip,plip,omiclip}_uni.yaml "
                             "— pinned baseline ckpt path.")
    parser.add_argument("--out_dir", required=True,
                        help="Directory for per-slide *.h5 outputs.")
    parser.add_argument("--slide", default=None,
                        help="Optional sample_name to process a single slide.")
    args = parser.parse_args()

    niche_cfg = ConfigDict(yaml.safe_load(open(args.niche_config)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, uni_wrapper, projection_head, short_name = _build_baseline(
        args.baseline_config, device,
    )

    manifest = pd.read_csv(niche_cfg.data.evaluable_slides_csv)
    if "norm" not in manifest.columns:
        manifest["norm"] = manifest["sample_name"].map(_normalize)
    if args.slide is not None:
        manifest = manifest[manifest["sample_name"] == args.slide]
        if manifest.empty:
            raise SystemExit(f"slide '{args.slide}' not in evaluable manifest")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_cfg = label_config_from_cfg(niche_cfg.data)
    max_cells = (int(niche_cfg.extract["max_cells_per_slide"])
                 if "max_cells_per_slide" in niche_cfg.extract else None)
    sub_seed = (int(niche_cfg.extract["subsample_seed"])
                if "subsample_seed" in niche_cfg.extract else 42)

    total_cells = 0
    failed = []
    for _, row in manifest.iterrows():
        sample = row["sample_name"]
        out_path = out_dir / f"{sample}.h5"
        if out_path.exists():
            print(f"[skip] {out_path} exists")
            continue
        try:
            n = process_slide_baseline(
                slide_row=row,
                uni_wrapper=uni_wrapper,
                projection_head=projection_head,
                short_name=short_name,
                labels_h5ad=niche_cfg.data.labels_h5ad,
                scale_sizes=tuple(niche_cfg.extract.scale_sizes),
                encoder_input=int(niche_cfg.extract.encoder_input),
                patch_size=int(niche_cfg.extract.patch_size),
                batch_size=int(niche_cfg.extract.batch_size),
                out_path=out_path,
                max_cells_per_slide=max_cells,
                subsample_seed=sub_seed,
                **label_cfg,
            )
            total_cells += n
        except (FileNotFoundError, KeyError, OSError) as e:
            # Same slides that the SIGMMA extractor skipped (missing zarr /
            # parquet / preprocessed h5) will trip here. Log and continue so a
            # single bad slide does not abort all four baselines.
            print(f"[err] {sample}: {type(e).__name__}: {e}", flush=True)
            failed.append(sample)
    print(f"\n[done] baseline={short_name}: {total_cells:,} cells under {out_dir}")
    if failed:
        print(f"[done] skipped {len(failed)} slides: {failed}")


if __name__ == "__main__":
    main()
