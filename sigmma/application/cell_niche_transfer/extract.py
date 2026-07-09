"""
Per-cell SIGMMA + UNI feature extraction split into three stages so a
multi-α SIGMMA sweep reuses ckpt-independent work.

Stages (one HDF5 per slide per stage):

  1. ``process_slide_data``     — writes ``<sample>.h5`` to the **data**
     cache: ``cell_id`` + niche labels + per-scale polygon-mask grids and
     bbox origins. GPU not required. Cheap.

  2. ``process_slide_features`` — reads the **data** HDF5, fetches pixel
     crops on demand from the slide zarr, encodes them with one backbone
     (SIGMMA or raw UNI), mask-mean-pools the patch tokens, and writes the
     corresponding ``z_*`` features under the **uni** or **sigmma** cache.

The downstream probe (probe_niche_cross_slide.py) joins the three HDF5s
per slide on ``cell_id`` (order is guaranteed identical because each stage
iterates the same sorted set of valid cells).

Projection head selection (SIGMMA only):
  * dense-attnpool (post-2026-05-14): ``sigmma_model.scale_mlps[scale_idx]``
  * legacy per-scale-LoRA: ``image_projection_heads[scale_idx]`` with
    per-scale LoRA adapter ``scale_{scale_idx}`` routed in.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr

from sigmma.model.models_image_backbones import set_lora_adapter
from .crops import extract_centered_crop, rasterize_polygon


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Default (skin / beacon) label schema. A different tissue overrides these via the
# config (data.sample_match_column + data.niche_labels) so the same extractor serves
# whole_embryo (sample_col="sample", labels [("niche40", "niche40")]) etc.
_DEFAULT_SAMPLE_COL = "Sample ID"
_DEFAULT_LABEL_SPECS = (("niche_category", "niche_category"),
                        ("niches_terra", "Niches_TERRA"))
_DEFAULT_UM_PER_PX = 0.5  # H&E grid resolution; px = µm / um_per_px (skin verified, WEM verified)


def label_config_from_cfg(data_cfg) -> Dict[str, object]:
    """Parse the optional niche-label schema from a config ``data`` block into the
    kwargs shared by ``process_slide_data`` / ``process_slide_baseline`` /
    ``_load_labels_for_sample``. Missing keys fall back to the skin defaults, so
    every existing config keeps today's behavior verbatim.

    Recognized keys (all optional):
      ``sample_match_column`` (str)   — categorical obs column to match the slide.
      ``niche_labels``        (list)  — ``[{name, column}, ...]`` label specs.
      ``normalize_sample``    (bool)  — apply skin's ``_norm`` to the match column.
      ``um_per_px``           (float) — H&E grid resolution for µm→px.
    """
    def _get(key, default):
        try:
            return data_cfg[key]
        except (KeyError, AttributeError, TypeError):
            return default

    nl = _get("niche_labels", None)
    if nl:
        label_specs = tuple((e["name"], e["column"]) for e in nl)
    else:
        label_specs = _DEFAULT_LABEL_SPECS
    return {
        "sample_col": _get("sample_match_column", _DEFAULT_SAMPLE_COL),
        "label_specs": label_specs,
        "normalize_sample": bool(_get("normalize_sample", True)),
        "um_per_px": float(_get("um_per_px", _DEFAULT_UM_PER_PX)),
    }


# =====================================================================
# Token-pooling primitive (shared by both feature stages)
# =====================================================================

def _normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - _IMAGENET_MEAN.to(x)) / _IMAGENET_STD.to(x)


def _patch_mask_from_pixel_mask(mask_hw: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reduce a (B, H, W) bool mask to a (B, H/p, W/p) bool patch grid (any-overlap)."""
    B, H, W = mask_hw.shape
    g_h, g_w = H // patch_size, W // patch_size
    return (
        mask_hw[:, : g_h * patch_size, : g_w * patch_size]
        .reshape(B, g_h, patch_size, g_w, patch_size)
        .any(dim=2)
        .any(dim=3)
    )


@torch.no_grad()
def encode_with_mask_pool(
    backbone: torch.nn.Module,
    crops_uint8: torch.Tensor,
    cell_mask: torch.Tensor,
    *,
    encoder_input: int = 224,
    patch_size: int = 16,
    projection_head: torch.nn.Module | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode crops with mask-restricted token pooling AND extract the CLS token.

    Returns three tensors so downstream code can ablate dense pooling vs CLS:
      - ``pooled``: mask-mean-pool of patch tokens, optionally projected.
      - ``pooled_raw``: same mean-pool but without ``projection_head`` applied
        (D-dim for the backbone). Equal to ``pooled`` when no projection given.
      - ``cls``: the CLS token at index 0 of ``forward_features`` output. None
        if the backbone output has no CLS row.
    """
    device = next(backbone.parameters()).device
    crops = crops_uint8.to(device).float() / 255.0
    crops = F.interpolate(
        crops, size=encoder_input, mode="bilinear", align_corners=False
    )
    crops = _normalize_imagenet(crops)

    # Auto-detect patch size from the backbone (UNI v1: 16, UNI2-h: 14).
    try:
        pe = backbone.model.patch_embed.patch_size
        detected = int(pe[0] if isinstance(pe, (tuple, list)) else pe)
        if detected != patch_size:
            if not getattr(backbone, "_patch_size_override_logged", False):
                print(
                    f"  [encode] patch_size override: caller={patch_size} → backbone={detected}",
                    flush=True,
                )
                backbone._patch_size_override_logged = True
            patch_size = detected
    except AttributeError:
        pass

    mask = cell_mask.to(device).float().unsqueeze(1)
    mask = F.interpolate(mask, size=encoder_input, mode="nearest").squeeze(1).bool()
    patch_mask = _patch_mask_from_pixel_mask(mask, patch_size)

    feats = backbone.model.forward_features(crops)
    num_prefix = int(getattr(backbone.model, "num_prefix_tokens", 1))
    if feats.dim() == 3 and feats.shape[1] == num_prefix + patch_mask[0].numel():
        cls = feats[:, 0, :]
        tokens = feats[:, num_prefix:, :]
    else:
        cls = None
        tokens = feats

    B = tokens.shape[0]
    mask_flat = patch_mask.view(B, -1)
    counts = mask_flat.sum(dim=-1)

    empty = counts == 0
    if empty.any():
        g = patch_mask.shape[-1]
        centre_idx = (g // 2) * g + (g // 2)
        mask_flat = mask_flat.clone()
        mask_flat[empty, centre_idx] = True
        counts = mask_flat.sum(dim=-1)

    weights = mask_flat.float() / counts.clamp(min=1).unsqueeze(-1).float()
    pooled_raw = (tokens * weights.unsqueeze(-1)).sum(dim=1)

    if projection_head is not None:
        proj_dtype = next(projection_head.parameters()).dtype
        pooled = projection_head(pooled_raw.to(proj_dtype))
    else:
        pooled = pooled_raw
    return pooled, pooled_raw, cls


# =====================================================================
# Slide-side helpers
# =====================================================================

def _resolve_xenium_output_dir(raw_dir: Path) -> Path:
    raw_dir = Path(raw_dir)
    if (raw_dir / "cell_boundaries.parquet").exists():
        return raw_dir
    candidates = sorted(raw_dir.glob("output-XETG*"))
    if not candidates:
        raise FileNotFoundError(
            f"No output-XETG* subdir under {raw_dir} (and no parquets at this level)"
        )
    return candidates[0]


def _load_polygons(raw_dir: Path) -> Dict[str, np.ndarray]:
    out_dir = _resolve_xenium_output_dir(raw_dir)
    bnd = pd.read_parquet(out_dir / "cell_boundaries.parquet")
    polys: Dict[str, np.ndarray] = {}
    for cid, group in bnd.groupby("cell_id", sort=False):
        polys[cid] = group[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float32)
    return polys


def _load_preproc_centroids(preproc_path: str,
                            um_per_px: float = _DEFAULT_UM_PER_PX) -> pd.DataFrame:
    """Per-cell pixel centroids. Prefers ``obs/x_pixel``/``y_pixel`` if present;
    otherwise derives them from the µm centroids (``obs/x_centroid``/``y_centroid``)
    at ``um_per_px`` µm/px — i.e. ``px = µm / um_per_px``. The 05-29 re-preprocessing
    dropped the pixel columns and now stores µm centroids only, so this fallback is
    what the pipeline actually uses; it matches the polygon scaling in
    ``process_slide_data`` (verts_px = verts_µm / um_per_px)."""
    with h5py.File(preproc_path, "r") as f:
        obs = f["obs"]
        cell_ids = [c.decode() if isinstance(c, bytes) else c for c in obs["cell_id"][:]]
        if "x_pixel" in obs and "y_pixel" in obs:
            x_px = obs["x_pixel"][:]
            y_px = obs["y_pixel"][:]
        else:
            x_px = obs["x_centroid"][:] / um_per_px
            y_px = obs["y_centroid"][:] / um_per_px
    return pd.DataFrame(
        {"cell_id": cell_ids, "x_pixel": x_px, "y_pixel": y_px}
    ).set_index("cell_id")


def _load_labels_for_sample(
    labels_path: str,
    normalized_sample: str,
    *,
    sample_col: str = _DEFAULT_SAMPLE_COL,
    label_specs: Tuple[Tuple[str, str], ...] = _DEFAULT_LABEL_SPECS,
    normalize_sample: bool = True,
) -> pd.DataFrame:
    """Read per-cell niche labels for one slide from a curated ``labels.h5ad``.

    Schema is config-driven so the same loader serves multiple tissues:
      * ``sample_col``     — categorical ``obs`` column used to match the slide.
                             skin = ``"Sample ID"`` (normalized via ``_norm``);
                             whole_embryo = ``"sample"`` (raw, e.g. ``"S203"``).
      * ``label_specs``    — ordered ``(out_name, h5ad_column)`` pairs. The returned
                             DataFrame has ``<out_name>`` (str) + ``<out_name>_code``
                             (int) columns per spec.
      * ``normalize_sample`` — apply skin's FOV/section ``_norm`` to the match column
                             (True) or compare raw, case-insensitively (False).

    Matching is on the slide's ``normalized_sample`` value (the manifest ``norm``
    column). Returns an empty DataFrame if the slide is absent from the labels file.
    """
    import re

    def _norm(s: str) -> str:
        s = s.upper()
        s = re.sub(r"-FO-0+(\d)", r"-FO-\1", s)
        s = re.sub(r"-S\d+(-[A-Z]\d+)?$", "", s)
        return s

    target = normalized_sample if normalize_sample else str(normalized_sample).upper()
    with h5py.File(labels_path, "r") as f:
        sample_cats = [c.decode() if isinstance(c, bytes) else c
                       for c in f[f"obs/{sample_col}/categories"][:]]
        sample_codes = f[f"obs/{sample_col}/codes"][:]
        if normalize_sample:
            match_cats = [_norm(c) if c else "" for c in sample_cats]
        else:
            match_cats = [c.upper() if c else "" for c in sample_cats]
        target_codes = [i for i, c in enumerate(match_cats) if c == target]
        if not target_codes:
            return pd.DataFrame()
        in_sample = np.isin(sample_codes, target_codes)
        cid_cats = [c.decode() if isinstance(c, bytes) else c
                    for c in f["obs/cell_id/categories"][:]]
        cid_codes = f["obs/cell_id/codes"][:][in_sample]
        cell_ids = np.array(
            [cid_cats[c] if 0 <= c < len(cid_cats) else "" for c in cid_codes]
        )
        data: Dict[str, object] = {"cell_id": cell_ids}
        for out_name, col in label_specs:
            cats = [c.decode() if isinstance(c, bytes) else c
                    for c in f[f"obs/{col}/categories"][:]]
            codes = f[f"obs/{col}/codes"][:][in_sample]
            data[f"{out_name}_code"] = codes
            data[out_name] = [cats[c] if c >= 0 else None for c in codes]

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="cell_id", keep="first").set_index("cell_id")
    return df


def subsample_valid_ids(
    valid_ids: List[str],
    labels: pd.DataFrame,
    max_cells: int | None,
    *,
    seed: int,
    strat_col: str,
) -> List[str]:
    """Deterministically cap a slide to ``max_cells`` cells, stratified by
    ``strat_col`` so the niche-label distribution is preserved.

    MUST be called identically (same ``valid_ids`` order, ``seed``, ``strat_col``)
    by every stage that derives its own cell set — the data stage and the baseline
    extractor both do — so the probe's cell_id alignment across feature HDF5s holds.
    Returns the kept ids in canonical sorted order. A no-op when ``max_cells`` is
    falsy or the slide already has ``<= max_cells`` cells (so skin is unaffected)."""
    n = len(valid_ids)
    if not max_cells or n <= max_cells:
        return valid_ids
    vid = np.asarray(valid_ids)
    strat_series = labels.loc[vid, strat_col]
    strat = np.array(["" if pd.isna(v) else str(v) for v in strat_series])
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(strat, return_counts=True)
    chosen = []
    for cls, cnt in zip(classes, counts):
        idx = np.flatnonzero(strat == cls)
        k = min(int(np.ceil(max_cells * cnt / n)), int(cnt))
        chosen.append(rng.choice(idx, size=k, replace=False) if k < cnt else idx)
    chosen = np.concatenate(chosen)
    if len(chosen) > max_cells:
        chosen = rng.permutation(chosen)[:max_cells]
    return np.sort(vid[chosen]).tolist()


# =====================================================================
# Stage 1: data (ckpt + backbone independent)
# =====================================================================

def process_slide_data(
    *,
    slide_row: pd.Series,
    labels_h5ad: str,
    scale_sizes: Tuple[int, ...] = (64, 128, 256),
    out_path: Path,
    sample_col: str = _DEFAULT_SAMPLE_COL,
    label_specs: Tuple[Tuple[str, str], ...] = _DEFAULT_LABEL_SPECS,
    normalize_sample: bool = True,
    um_per_px: float = _DEFAULT_UM_PER_PX,
    max_cells_per_slide: int | None = None,
    subsample_seed: int = 42,
) -> int:
    """Write the data HDF5 for one slide: per-cell labels + per-scale bbox
    origins + per-scale rasterized polygon masks. No GPU. Reused across α
    sweeps and across backbones (UNI v1 ↔ UNI2-h).

    HDF5 layout (single group, root datasets):
      ``cell_id``                 : (N,) bytes
      ``niches_terra``            : (N,) bytes
      ``niches_terra_code``       : (N,) int16
      ``niche_category``          : (N,) bytes
      ``niche_category_code``     : (N,) int16
      ``x_pixel`` / ``y_pixel``   : (N,) int32  (raster centroid in 0.5 µm/px)
      ``bbox_<scale>``            : (N, 2) int32  (x_start, y_start at scale S)
      ``mask_<scale>``            : (N, S, S) bool (rasterized polygon)

    where ``<scale>`` ∈ {"micro", "meso", "macro"} maps to scale_sizes
    (64, 128, 256) by index. Returns the number of cells written.
    """
    sample_name = slide_row["sample_name"]
    norm = slide_row.get("norm", sample_name)
    print(f"[data:{sample_name}] loading slide assets…", flush=True)

    he_zarr = zarr.open(slide_row["hematoxylin_eosin_image"], mode="r")  # shape only
    H, W = he_zarr.shape[1], he_zarr.shape[2]
    px_per_um = 1.0 / um_per_px
    polys = _load_polygons(Path(slide_row["raw_dir"]))
    centroids = _load_preproc_centroids(slide_row["spatial_preprocessed"], um_per_px)
    labels = _load_labels_for_sample(
        labels_h5ad, norm,
        sample_col=sample_col, label_specs=label_specs,
        normalize_sample=normalize_sample,
    )

    valid_ids = sorted(set(polys) & set(centroids.index) & set(labels.index))
    print(f"[data:{sample_name}] cells with polygon ∧ centroid ∧ label: {len(valid_ids)}",
          flush=True)
    if not valid_ids:
        return 0
    if max_cells_per_slide and len(valid_ids) > max_cells_per_slide:
        valid_ids = subsample_valid_ids(
            valid_ids, labels, max_cells_per_slide,
            seed=subsample_seed, strat_col=label_specs[0][0],
        )
        print(f"[data:{sample_name}] subsampled to {len(valid_ids)} cells "
              f"(stratified by {label_specs[0][0]}, seed={subsample_seed})", flush=True)

    scale_names = ("micro", "meso", "macro")
    N = len(valid_ids)
    cell_ids = np.array(valid_ids, dtype="S")
    x_pixel = np.empty(N, dtype=np.int32)
    y_pixel = np.empty(N, dtype=np.int32)
    bboxes = {s: np.empty((N, 2), dtype=np.int32) for s in scale_names}
    masks = {s: np.empty((N, sz, sz), dtype=bool)
             for s, sz in zip(scale_names, scale_sizes)}

    label_names = [name for name, _ in label_specs]
    label_str = {name: np.empty(N, dtype=object) for name in label_names}
    label_code = {name: np.empty(N, dtype=np.int16) for name in label_names}

    for i, cid in enumerate(valid_ids):
        row = centroids.loc[cid]
        cx_px, cy_px = float(row["x_pixel"]), float(row["y_pixel"])
        x_pixel[i] = int(cx_px)
        y_pixel[i] = int(cy_px)
        verts_px = polys[cid] * px_per_um  # µm → px at um_per_px µm/px

        for scale_idx, (sname, ssize) in enumerate(zip(scale_names, scale_sizes)):
            x0 = int(round(cx_px - ssize / 2))
            y0 = int(round(cy_px - ssize / 2))
            # Clamp like extract_centered_crop does internally — we want the
            # SAME bbox the feature stages will read.
            x0 = max(0, min(x0, W - ssize))
            y0 = max(0, min(y0, H - ssize))
            bboxes[sname][i] = (x0, y0)
            masks[sname][i] = rasterize_polygon(verts_px, (x0, y0), (ssize, ssize))

        r = labels.loc[cid]
        for name in label_names:
            v = r[name]
            label_str[name][i] = "" if pd.isna(v) else str(v)
            label_code[name][i] = int(r[f"{name}_code"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["sample_name"] = sample_name
        f.attrs["normalized_sample"] = norm
        f.attrs["split"] = slide_row.get("split", "")
        f.attrs["scale_sizes"] = list(scale_sizes)
        f.attrs["label_names"] = label_names

        f.create_dataset("cell_id", data=cell_ids)
        f.create_dataset("x_pixel", data=x_pixel)
        f.create_dataset("y_pixel", data=y_pixel)
        for name in label_names:
            f.create_dataset(name,
                             data=np.array([s.encode() for s in label_str[name]]))
            f.create_dataset(f"{name}_code", data=label_code[name])
        for s in scale_names:
            f.create_dataset(f"bbox_{s}", data=bboxes[s])
            f.create_dataset(f"mask_{s}", data=masks[s], compression="gzip",
                             compression_opts=4, chunks=True)
    print(f"[data:{sample_name}] wrote {out_path} ({N} cells)", flush=True)
    return N


# =====================================================================
# Stage 2: per-backbone features
# =====================================================================

def _read_data_h5(data_h5_path: Path) -> dict:
    out: dict = {}
    with h5py.File(data_h5_path, "r") as f:
        out["sample_name"] = f.attrs["sample_name"]
        out["normalized_sample"] = f.attrs["normalized_sample"]
        out["split"] = f.attrs["split"]
        out["scale_sizes"] = tuple(int(s) for s in f.attrs["scale_sizes"])
        out["cell_id"] = np.array([c.decode() for c in f["cell_id"][:]])
        for s in ("micro", "meso", "macro"):
            out[f"bbox_{s}"] = f[f"bbox_{s}"][:]
            out[f"mask_{s}"] = f[f"mask_{s}"][:]
        # Label datasets are discovered from the attr written by process_slide_data;
        # fall back to the historical skin pair for caches written before that attr.
        label_names = [n for n in f.attrs.get(
            "label_names", ["niche_category", "niches_terra"])]
        out["label_names"] = label_names
        for name in label_names:
            out[name] = f[name][:]
            out[f"{name}_code"] = f[f"{name}_code"][:]
    return out


def _read_crops_from_zarr(
    he_zarr,
    bboxes: np.ndarray,
    size: int,
) -> np.ndarray:
    """Fetch a batch of (3, S, S) uint8 crops from a 3-channel zarr image."""
    out = np.empty((len(bboxes), 3, size, size), dtype=np.uint8)
    for i, (x0, y0) in enumerate(bboxes):
        out[i] = np.asarray(he_zarr[:, y0:y0 + size, x0:x0 + size])
    return out


def process_slide_features(
    *,
    slide_row: pd.Series,
    data_h5_path: Path,
    backbone_model: torch.nn.Module,
    kind: str,                      # "sigmma" | "uni"
    sigmma_model: torch.nn.Module | None = None,
    encoder_input: int = 224,
    patch_size: int = 16,
    batch_size: int = 32,
    out_path: Path,
) -> int:
    """Run one backbone over all cells in ``data_h5_path`` and write the
    matching feature HDF5.

    When ``kind == 'sigmma'``: ``backbone_model`` is the SIGMMA wrapper
    (its ``.image_encoder`` is the LoRA-adapted UNI). ``sigmma_model`` may
    be passed explicitly or default to ``backbone_model``; the routine
    derives the per-scale projection head:
      * dense-attnpool: ``sigmma_model.scale_mlps[scale_idx]``
      * legacy per-scale-LoRA: ``image_projection_heads[scale_idx]``

    When ``kind == 'uni'``: ``backbone_model`` is a raw UNI module (no LoRA,
    no projection). Writes ``z_uni_<scale>`` + ``z_uni_<scale>_cls``.
    """
    if kind not in {"sigmma", "uni"}:
        raise ValueError(f"unknown kind={kind!r}")
    data = _read_data_h5(data_h5_path)
    sample_name = data["sample_name"]
    cell_ids = data["cell_id"]
    N = len(cell_ids)
    scale_names = ("micro", "meso", "macro")
    scale_sizes = data["scale_sizes"]

    he_zarr = zarr.open(slide_row["hematoxylin_eosin_image"], mode="r")

    if kind == "sigmma":
        sm = sigmma_model if sigmma_model is not None else backbone_model
        # Dense-attnpool: single shared LoRA adapter — set once.
        adapter_names = getattr(backbone_model.image_encoder, "lora_adapter_names", None) \
            if hasattr(backbone_model, "image_encoder") \
            else getattr(backbone_model, "lora_adapter_names", None)
        if adapter_names and len(adapter_names) == 1:
            inner = backbone_model.image_encoder \
                if hasattr(backbone_model, "image_encoder") else backbone_model
            set_lora_adapter(inner, adapter_names[0])
        sigmma_image_encoder = sm.image_encoder if hasattr(sm, "image_encoder") else sm
    else:
        sigmma_image_encoder = None

    # Per-scale output buffers.
    out: Dict[str, list] = {}
    if kind == "sigmma":
        for s in scale_names:
            out[f"z_sigmma_{s}"] = []
            out[f"z_sigmma_{s}_dense_noproj"] = []
            out[f"z_sigmma_{s}_cls"] = []
    else:
        for s in scale_names:
            out[f"z_uni_{s}"] = []
            out[f"z_uni_{s}_cls"] = []

    n_done = 0
    for batch_start in range(0, N, batch_size):
        end = min(batch_start + batch_size, N)
        idx = slice(batch_start, end)

        for scale_idx, (sname, ssize) in enumerate(zip(scale_names, scale_sizes)):
            bbox = data[f"bbox_{sname}"][idx]
            masks_np = data[f"mask_{sname}"][idx]
            crops_np = _read_crops_from_zarr(he_zarr, bbox, ssize)

            crops_t = torch.from_numpy(crops_np)
            masks_t = torch.from_numpy(masks_np)

            if kind == "sigmma":
                if hasattr(sm, "scale_mlps"):
                    proj_head = sm.scale_mlps[scale_idx]
                else:
                    set_lora_adapter(sigmma_image_encoder, f"scale_{scale_idx}")
                    proj_head = sm.image_projection_heads[scale_idx]
                z, z_raw, z_cls = encode_with_mask_pool(
                    sigmma_image_encoder, crops_t, masks_t,
                    encoder_input=encoder_input, patch_size=patch_size,
                    projection_head=proj_head,
                )
                out[f"z_sigmma_{sname}"].append(z.cpu().numpy())
                out[f"z_sigmma_{sname}_dense_noproj"].append(z_raw.cpu().numpy())
                if z_cls is not None:
                    out[f"z_sigmma_{sname}_cls"].append(z_cls.cpu().numpy())
                else:
                    out[f"z_sigmma_{sname}_cls"].append(np.zeros((end - batch_start, 0), dtype=np.float32))
            else:
                z, _, z_cls = encode_with_mask_pool(
                    backbone_model, crops_t, masks_t,
                    encoder_input=encoder_input, patch_size=patch_size,
                    projection_head=None,
                )
                out[f"z_uni_{sname}"].append(z.cpu().numpy())
                if z_cls is not None:
                    out[f"z_uni_{sname}_cls"].append(z_cls.cpu().numpy())
                else:
                    out[f"z_uni_{sname}_cls"].append(np.zeros((end - batch_start, 0), dtype=np.float32))

        n_done = end
        if n_done % (batch_size * 10) == 0 or n_done == N:
            print(f"[{kind}:{sample_name}] {n_done}/{N} cells", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["sample_name"] = sample_name
        f.attrs["normalized_sample"] = data["normalized_sample"]
        f.attrs["split"] = data["split"]
        f.attrs["scale_sizes"] = list(scale_sizes)
        f.attrs["backbone_kind"] = kind
        f.create_dataset("cell_id", data=np.array([c.encode() for c in cell_ids]))
        for k, chunks in out.items():
            arr = np.concatenate(chunks, axis=0) if chunks else np.zeros((N, 0))
            # Drop the dummy 0-width CLS placeholder if the backbone has none.
            if arr.size == 0 and arr.shape[1] == 0:
                continue
            f.create_dataset(k, data=arr)
    print(f"[{kind}:{sample_name}] wrote {out_path}", flush=True)
    return N
