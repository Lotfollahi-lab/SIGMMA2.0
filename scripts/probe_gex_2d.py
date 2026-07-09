"""2D H&E → ST GEX probing driver (Framing A Task 1).

Phase A — cache per-FOV features + targets to disk (heavy: encoder forward +
transcript parquet IO; run once per backbone).
Phase B — train a 1×1-conv probing head from cached features.
Phase C — predict on held-out slides + dump per-gene Pearson / per-bin R² /
Moran's I match.

Sub-commands (selected by ``--phase``):
  - ``hvg``   : compute the top-K HVG panel from training-set parquets
  - ``cache`` : Phase A — write per-FOV .npz under ``--cache_root``
  - ``train`` : Phase B — train head for one ``--input_repr``
  - ``eval``  : Phase C — evaluate trained head on val/held-out slides
  - ``all``   : hvg + cache (backbone) + train + eval (input_repr)

Smoke 0b mode: ``--smoke_slide <sample_name>`` restricts cache + train + eval
to a single slide with an intra-slide FOV split. Pass criterion is in the
plan (median per-gene Pearson ≥ 0.10).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import zarr
from ml_collections.config_dict import ConfigDict

from sigmma.evaluation.gex_2d_probing import (
    LinearProbingHead2D,
    LinearSweepHead2D,
    LogNormalHead2D,
    ProbingHead2D,
    pool_predictions,
    predict_all,
    save_history,
    summarize,
    train_linear_sweep,
    train_probing_head,
)
from sigmma.evaluation.gex_cache_io import (
    SectionCacheWriter,
    iter_section_bboxes,
    read_fov_ids,
    section_h5_exists,
    section_h5_path,
)
from sigmma.evaluation.transcript_binning import (
    bin_fov,
    compute_top_k_hvg_by_dispersion,
    compute_top_k_hvg_by_mean,
    iter_fov_bboxes,
    load_slide_transcripts,
    manifest_path_to_transcripts_parquet,
)


HVG_SELECTORS = {
    "mean": compute_top_k_hvg_by_mean,
    "dispersion": compute_top_k_hvg_by_dispersion,
}


# ============================================================
# Encoder loaders
# ============================================================
def load_sigmma_for_probing(config_path: str, device: torch.device):
    from sigmma.utils import load_model

    cfg = ConfigDict(yaml.safe_load(open(config_path, "r")))
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=cfg.model.checkpoint_path,
        train=False,
    ).to(device).eval()
    return model, cfg


def load_uni_for_probing(config_path: str, device: torch.device):
    from sigmma.model.models_image_backbones import UNI

    cfg = ConfigDict(yaml.safe_load(open(config_path, "r")))
    uni = UNI(checkpoint_path=cfg.model.pretrained_image_encoder_ckpt_path)
    inner = uni.model.to(device).eval()
    return inner, cfg


# Backbones that wrap vanilla UNI and only differ by fine-tune weights.
# All four use _UNIImageBackbone in sigmma/model/baselines.py, but the
# attribute name on the parent model differs by baseline (BLEEP uses
# `image_encoder`, the others use `_uni_image_encoder`), so we probe
# both prefixes when loading.
_UNI_FT_BACKBONES = {"bleep_uni", "clip_uni", "omiclip_uni", "plip_uni"}
_UNI_FT_SUBMODULE_PREFIX_CANDIDATES = (
    "_uni_image_encoder.uni.model.",  # CLIP / PLIP / OmiCLIP variants
    "image_encoder.uni.model.",       # BLEEP variant
)


def load_finetuned_uni_for_probing(ft_ckpt_path: str, config_path: str,
                                    device: torch.device):
    """Load a fine-tuned UNI variant (BLEEP/CLIP/OmiCLIP/PLIP) for probing.

    The fine-tune checkpoints wrap UNI inside ``_UNIImageBackbone`` (see
    ``sigmma/model/baselines.py``), so the ViT-L/16 weights live under
    ``model_state_dict['_uni_image_encoder.uni.*']``. We instantiate the
    vanilla UNI architecture (so we can call ``forward_features`` exactly
    like the baseline UNI probe) and then overwrite its weights with the
    fine-tuned ones.
    """
    inner, cfg = load_uni_for_probing(config_path, device)
    print(f"  [ft-uni] loading fine-tune weights from {ft_ckpt_path}", flush=True)
    ck = torch.load(ft_ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    if "model_state_dict" not in ck:
        raise ValueError(
            f"fine-tune ckpt missing 'model_state_dict' key: got keys={list(ck.keys())}"
        )
    src = ck["model_state_dict"]
    extracted = {}
    matched_prefix = None
    for pfx in _UNI_FT_SUBMODULE_PREFIX_CANDIDATES:
        cand = {k[len(pfx):]: v for k, v in src.items() if k.startswith(pfx)}
        if cand:
            extracted = cand
            matched_prefix = pfx
            break
    if not extracted:
        raise ValueError(
            f"no keys with any of prefixes {_UNI_FT_SUBMODULE_PREFIX_CANDIDATES} "
            f"in fine-tune ckpt; first src keys: {list(src.keys())[:5]}"
        )
    print(f"  [ft-uni] matched prefix {matched_prefix!r}", flush=True)
    missing, unexpected = inner.load_state_dict(extracted, strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys when loading FT UNI: {unexpected[:5]}")
    if missing:
        # buffers (pos_embed etc) sometimes registered separately — warn but allow.
        print(f"  [ft-uni] missing {len(missing)} keys (first 3: {missing[:3]})", flush=True)
    print(f"  [ft-uni] loaded {len(extracted)} ViT weights into UNI", flush=True)
    return inner, cfg


@torch.no_grad()
def uni_patch_tokens(inner, x_resized: torch.Tensor) -> torch.Tensor:
    """Run UNI on a 224×224 batch, return (B, D, side, side) patch tokens.

    Strips all prefix tokens (CLS + register tokens). UNI v1 has 1 prefix
    (CLS) and 196 = 14×14 patches; UNI2-h has 9 prefixes (1 CLS + 8 reg)
    and 256 = 16×16 patches. ``inner.num_prefix_tokens`` (timm ViT) gives
    the right count for both.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x_resized.is_cuda):
        tokens = inner.forward_features(x_resized)  # (B, num_prefix + H*W, D)
    num_prefix = int(getattr(inner, "num_prefix_tokens", 1))
    patches = tokens[:, num_prefix:, :].float()
    B, N, D = patches.shape
    side = int(round(N ** 0.5))
    assert side * side == N, f"unexpected token count {N} after stripping {num_prefix} prefix tokens"
    return patches.transpose(1, 2).reshape(B, D, side, side).contiguous()


# ============================================================
# H&E pre-processing
# ============================================================
def _to_uni_input(tile_uint8: np.ndarray, encoder_input_size: int = 224) -> torch.Tensor:
    """(3, H, W) uint8 → (1, 3, encoder_input_size, encoder_input_size) float32 in [0, 1]."""
    t = torch.from_numpy(tile_uint8).float().unsqueeze(0) / 255.0
    if t.shape[-1] != encoder_input_size:
        t = F.interpolate(
            t,
            size=(encoder_input_size, encoder_input_size),
            mode="bilinear",
            align_corners=False,
        )
    return t


# ============================================================
# Phase A — caching (split into 3 backbone-independent stages)
# ============================================================
def cache_data(
    *,
    sample_name: str,
    he_zarr_path: str,
    index_zarr_path: str,
    parquet_path: str,
    gene_panel: list[str],
    fov_size: int,
    target_grid: int,
    tissue_threshold: float,
    qv_min: float,
    um_per_px: float,
    data_cache_dir: Path,
    fov_subset: Optional[Iterable[int]] = None,
) -> int:
    """Stage 1: encoder + ckpt-independent. Writes ``target`` + ``mask`` per FOV.

    The FOV bbox metadata (x_start, y_start, fov_size) is captured here so
    later stages can re-derive crops without re-scanning index zarr.
    """
    if section_h5_exists(data_cache_dir, sample_name):
        n = len(read_fov_ids(data_cache_dir, sample_name))
        print(f"  [skip] data: {n} FOVs already cached", flush=True)
        return n

    image_array = zarr.open(he_zarr_path, mode="r")
    H, W = image_array.shape[1], image_array.shape[2]

    print(f"  [load transcripts] {parquet_path}", flush=True)
    t0 = time.time()
    loaded = load_slide_transcripts(parquet_path, gene_panel, qv_min=qv_min, um_per_px=um_per_px)
    print(f"  [ok] {loaded['x_px'].size} panel transcripts in {time.time() - t0:.1f}s", flush=True)

    fov_subset_set = None if fov_subset is None else set(int(i) for i in fov_subset)

    written = 0
    skipped_oob = 0
    bbox_iter = list(iter_fov_bboxes(index_zarr_path, tissue_filtering_threshold=tissue_threshold))
    print(f"  [fovs] {len(bbox_iter)} valid FOVs (tissue >= {tissue_threshold})", flush=True)

    # Fast spatial bucketing: the FOV grid is a non-overlapping tiling aligned to
    # fov_size (stride==tile==fov_size), so each transcript falls in exactly one
    # FOV cell. Pre-bucket once (O(n log n)) so each FOV bins only its own cell's
    # transcripts instead of re-masking all N transcripts — critical for huge
    # sections (e.g. 60M-transcript ovary/brain slides, where the naive per-FOV
    # rescan is ~1 FOV/s). Falls back to the exact per-FOV path if unaligned.
    _ROWMUL = 1 << 32
    aligned = all((xs % fov_size == 0 and ys % fov_size == 0) for _, xs, ys, _ in bbox_iter)
    buckets = None
    if aligned and loaded["x_px"].size:
        col = (loaded["x_px"].astype(np.int64) // fov_size)
        row = (loaded["y_px"].astype(np.int64) // fov_size)
        key = col * _ROWMUL + row
        order = np.argsort(key, kind="stable")
        skey = key[order]
        uniq, first = np.unique(skey, return_index=True)
        ends = np.append(first[1:], len(order))
        buckets = {
            "x": loaded["x_px"][order], "y": loaded["y_px"][order],
            "g": loaded["gene_idx"][order],
            "bounds": {int(k): (int(s), int(e)) for k, s, e in zip(uniq, first, ends)},
        }

    # First pass: select in-bounds FOVs (and honor fov_subset) so the section
    # .h5 can be created with an exact row count; binning happens in pass two.
    valid = []
    for fov_id, x_start, y_start, _tissue in bbox_iter:
        if fov_subset_set is not None and fov_id not in fov_subset_set:
            continue
        if (x_start + fov_size > W) or (y_start + fov_size > H):
            skipped_oob += 1
            continue
        valid.append((int(fov_id), int(x_start), int(y_start)))

    num_genes = int(loaded["num_genes"])
    mask = np.ones((target_grid, target_grid), dtype=np.float32)
    with SectionCacheWriter(
        data_cache_dir, sample_name, len(valid),
        datasets={
            "target": ((num_genes, target_grid, target_grid), np.float32),
            "mask": ((target_grid, target_grid), np.float32),
            "fov_id": ((), np.int32),
            "x_start": ((), np.int32),
            "y_start": ((), np.int32),
        },
        attrs={"fov_size": int(fov_size), "num_genes": num_genes},
    ) as w:
        for i, (fov_id, x_start, y_start) in enumerate(valid):
            if buckets is not None:
                ck = (x_start // fov_size) * _ROWMUL + (y_start // fov_size)
                se = buckets["bounds"].get(int(ck))
                if se is None:
                    target = np.zeros((num_genes, target_grid, target_grid), dtype=np.float32)
                else:
                    s, e = se
                    sub = {"x_px": buckets["x"][s:e], "y_px": buckets["y"][s:e],
                           "gene_idx": buckets["g"][s:e], "num_genes": num_genes}
                    target = bin_fov(sub, x_start, y_start, fov_size, target_grid)
            else:
                target = bin_fov(loaded, x_start, y_start, fov_size, target_grid)
            w.write_row(i, target=target, mask=mask,
                        fov_id=fov_id, x_start=x_start, y_start=y_start)
            written += 1
    print(f"  [done] data: wrote {written} FOVs (skipped {skipped_oob} OOB)", flush=True)
    return written


def _iter_data_fovs(data_cache_dir: Path, sample_name: str):
    yield from iter_section_bboxes(data_cache_dir, sample_name)


def cache_uni_features(
    *,
    sample_name: str,
    he_zarr_path: str,
    encoder_input_size: int,
    data_cache_dir: Path,
    uni_cache_dir: Path,
    uni_inner,
    device: torch.device,
) -> int:
    """Stage 2a: encoder UNI patch tokens per FOV. Reads FOV bboxes from
    the shared ``data_cache_dir`` (stage 1) so the geometry matches exactly.
    """
    if section_h5_exists(uni_cache_dir, sample_name):
        n = len(read_fov_ids(uni_cache_dir, sample_name))
        print(f"  [skip] uni: {n} FOVs already cached", flush=True)
        return n
    fovs = list(_iter_data_fovs(data_cache_dir, sample_name))
    image_array = zarr.open(he_zarr_path, mode="r")
    written = 0
    writer = None
    try:
        for i, (fov_id, x_start, y_start, fov_size) in enumerate(fovs):
            tile = np.asarray(image_array[:, y_start:y_start + fov_size,
                                         x_start:x_start + fov_size])
            x = _to_uni_input(tile, encoder_input_size=encoder_input_size).to(device)
            feats = uni_patch_tokens(uni_inner, x)[0].cpu().numpy().astype(np.float16)
            if writer is None:
                writer = SectionCacheWriter(
                    uni_cache_dir, sample_name, len(fovs),
                    datasets={"features": (feats.shape, np.float16),
                              "fov_id": ((), np.int32)},
                )
            writer.write_row(i, features=feats, fov_id=int(fov_id))
            written += 1
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    if writer is not None:
        writer.close()
    print(f"  [done] uni: wrote {written} FOVs", flush=True)
    return written


def cache_sigmma_features(
    *,
    sample_name: str,
    he_zarr_path: str,
    data_cache_dir: Path,
    sigmma_cache_dir: Path,
    sigmma_model,
    device: torch.device,
) -> int:
    """Stage 2b: SIGMMA per-scale patch tokens per FOV. Reads FOV bboxes from
    the shared ``data_cache_dir``. Forward uses ``fov_size`` as encoder input
    (matches training-time behavior of ``get_image_patch_tokens_per_scale``).
    """
    if section_h5_exists(sigmma_cache_dir, sample_name):
        n = len(read_fov_ids(sigmma_cache_dir, sample_name))
        print(f"  [skip] sigmma: {n} FOVs already cached", flush=True)
        return n
    fovs = list(_iter_data_fovs(data_cache_dir, sample_name))
    image_array = zarr.open(he_zarr_path, mode="r")
    written = 0
    writer = None
    try:
        for i, (fov_id, x_start, y_start, fov_size) in enumerate(fovs):
            tile = np.asarray(image_array[:, y_start:y_start + fov_size,
                                         x_start:x_start + fov_size])
            x = _to_uni_input(tile, encoder_input_size=fov_size).to(device)
            scale_tokens = sigmma_model.get_image_patch_tokens_per_scale(x)
            micro = scale_tokens[0][0].float().cpu().numpy().astype(np.float16)
            meso = scale_tokens[1][0].float().cpu().numpy().astype(np.float16)
            macro = scale_tokens[2][0].float().cpu().numpy().astype(np.float16)
            if writer is None:
                writer = SectionCacheWriter(
                    sigmma_cache_dir, sample_name, len(fovs),
                    datasets={"micro": (micro.shape, np.float16),
                              "meso": (meso.shape, np.float16),
                              "macro": (macro.shape, np.float16),
                              "fov_id": ((), np.int32)},
                )
            writer.write_row(i, micro=micro, meso=meso, macro=macro, fov_id=int(fov_id))
            written += 1
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    if writer is not None:
        writer.close()
    print(f"  [done] sigmma: wrote {written} FOVs", flush=True)
    return written


def cache_sigmma_lora_features(
    *,
    sample_name: str,
    he_zarr_path: str,
    encoder_input_size: int,
    data_cache_dir: Path,
    sigmma_lora_cache_dir: Path,
    sigmma_model,
    device: torch.device,
) -> int:
    """Stage 2c: LoRA-adapted UNI patch tokens per FOV — single-scale, full
    spatial resolution. Skips SIGMMA's scale_pools / scale_mlps so the head
    sees the encoder's spatial layout after only the LoRA finetuning of the
    UNI backbone (no scale-specific aggregation).

    Cache layout mirrors ``cache_uni_features`` (key ``features``, shape
    ``(D, P, P)`` per FOV) so the dataset can read it via the same path as
    the plain ``uni`` representation, just from a different directory.
    """
    if section_h5_exists(sigmma_lora_cache_dir, sample_name):
        n = len(read_fov_ids(sigmma_lora_cache_dir, sample_name))
        print(f"  [skip] sigmma_lora: {n} FOVs already cached", flush=True)
        return n
    fovs = list(_iter_data_fovs(data_cache_dir, sample_name))
    image_array = zarr.open(he_zarr_path, mode="r")
    written = 0
    writer = None
    try:
        for i, (fov_id, x_start, y_start, fov_size) in enumerate(fovs):
            tile = np.asarray(image_array[:, y_start:y_start + fov_size,
                                         x_start:x_start + fov_size])
            x = _to_uni_input(tile, encoder_input_size=encoder_input_size).to(device)
            patches = sigmma_model.get_lora_image_patch_tokens(x)[0].float().cpu().numpy().astype(np.float16)
            if writer is None:
                writer = SectionCacheWriter(
                    sigmma_lora_cache_dir, sample_name, len(fovs),
                    datasets={"features": (patches.shape, np.float16),
                              "fov_id": ((), np.int32)},
                )
            writer.write_row(i, features=patches, fov_id=int(fov_id))
            written += 1
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    if writer is not None:
        writer.close()
    print(f"  [done] sigmma_lora: wrote {written} FOVs", flush=True)
    return written


def iter_manifest(manifest_path: str):
    df = pd.read_csv(manifest_path)
    for _, row in df.iterrows():
        yield row["sample_name"], row["hematoxylin_eosin_image"], row["index"]


# ============================================================
# Phase B / C — fov key listing
# ============================================================
def list_fov_keys(data_cache_dir: str | Path, sample_names: list[str]) -> list[tuple[str, int]]:
    """Enumerate (sample, fov_id) keys present in the data cache. Other caches
    (uni / sigmma) must be aligned by stage-1-driven creation order — readers
    fail fast if a referenced fov_id is missing."""
    keys: list[tuple[str, int]] = []
    for s in sample_names:
        if not section_h5_exists(data_cache_dir, s):
            continue
        for fov_id in read_fov_ids(data_cache_dir, s):
            keys.append((s, int(fov_id)))
    return keys


def manifest_sample_names(manifest_path: str) -> list[str]:
    return pd.read_csv(manifest_path)["sample_name"].astype(str).tolist()


def in_channels_for(
    input_repr: str,
    *,
    uni_cache_dir: str | Path | None = None,
    sigmma_cache_dir: str | Path | None = None,
    sigmma_lora_cache_dir: str | Path | None = None,
    sample_key: tuple[str, int] | None = None,
) -> int:
    """Channel dim of the probe-head input for ``input_repr``.

    Inspects the relevant cache (uni / sigmma / sigmma_lora) using
    ``sample_key`` so the head's in_channels matches the cached backbone
    (UNI v1 → 1024, UNI2-h → 1536).
    """
    if sample_key is None:
        return 3072 if input_repr == "concat" else 1024
    sample, _fov_id = sample_key
    # Per-section .h5 stacks FOVs along axis 0, so the channel dim of any FOV is
    # dataset.shape[1] (independent of which fov_id from sample_key).
    if input_repr == "uni":
        if uni_cache_dir is None:
            raise ValueError("uni_cache_dir required to infer dim for input_repr=uni")
        with h5py.File(section_h5_path(uni_cache_dir, sample), "r") as f:
            return int(f["features"].shape[1])
    if input_repr == "lora_uni":
        if sigmma_lora_cache_dir is None:
            raise ValueError("sigmma_lora_cache_dir required to infer dim for input_repr=lora_uni")
        with h5py.File(section_h5_path(sigmma_lora_cache_dir, sample), "r") as f:
            return int(f["features"].shape[1])
    if input_repr in {"micro", "meso", "macro", "concat"}:
        if sigmma_cache_dir is None:
            raise ValueError(f"sigmma_cache_dir required to infer dim for input_repr={input_repr}")
        with h5py.File(section_h5_path(sigmma_cache_dir, sample), "r") as f:
            if input_repr == "concat":
                return int(f["micro"].shape[1] + f["meso"].shape[1] + f["macro"].shape[1])
            return int(f[input_repr].shape[1])
    raise ValueError(f"unknown input_repr={input_repr!r}")


def _heads_subdir(head_type: str, loss: str = "mse") -> str:
    base = "heads" if head_type == "mlp" else f"heads_{head_type}"
    return base if loss == "mse" else f"{base}_{loss}"


def _results_subdir(head_type: str, loss: str = "mse") -> str:
    base = "results" if head_type == "mlp" else f"results_{head_type}"
    return base if loss == "mse" else f"{base}_{loss}"


def _parse_sweep_lrs(spec: str) -> list[float]:
    """Parse a LR-sweep specification.

    Forms:
      - "logspace:lo:hi:n" → ``np.logspace(log10(lo), log10(hi), n)``
      - comma-separated floats: "0.01,0.03,0.1"
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty --sweep_lrs")
    if spec.startswith("logspace:"):
        _, lo, hi, n = spec.split(":")
        arr = np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n))
        return [float(x) for x in arr]
    return [float(x) for x in spec.split(",")]


def build_head(head_type: str, in_channels: int, num_genes: int, hidden_dim: int,
               sweep_lrs: Optional[list[float]] = None, loss: str = "mse"):
    if loss in {"lognormal", "zilognormal"}:
        if head_type != "mlp":
            raise ValueError(
                f"loss={loss!r} is only supported with --head_type=mlp "
                f"(got head_type={head_type!r}); the (zero-inflated) lognormal "
                "head reuses the MLP trunk."
            )
        return LogNormalHead2D(in_channels=in_channels, num_genes=num_genes,
                               loss=loss, hidden_dim=hidden_dim)
    if loss != "mse":
        raise ValueError(f"unknown loss={loss!r}")
    if head_type == "mlp":
        return ProbingHead2D(in_channels=in_channels, num_genes=num_genes, hidden_dim=hidden_dim)
    if head_type == "linear":
        return LinearProbingHead2D(in_channels=in_channels, num_genes=num_genes)
    if head_type == "linear_sweep":
        if not sweep_lrs:
            raise ValueError("head_type=linear_sweep requires sweep_lrs to be non-empty")
        return LinearSweepHead2D(in_channels=in_channels, num_genes=num_genes, lrs=sweep_lrs)
    raise ValueError(f"unknown head_type={head_type!r}")


# ============================================================
# Phase entry points
# ============================================================
def phase_hvg(args):
    out_path = Path(args.panel_dir) / "gene_panel.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        if (existing.get("k") == args.top_k_genes
                and existing.get("qv_min") == args.qv_min
                and existing.get("hvg_method", "mean") == args.hvg_method):
            print(f"[hvg] reusing {out_path} (k={existing['k']}, qv_min={existing['qv_min']}, "
                  f"method={existing.get('hvg_method','mean')})", flush=True)
            return
        print(f"[hvg] existing panel uses different (k|qv_min|method) — recomputing", flush=True)
    print(f"[hvg] computing top-{args.top_k_genes} genes via method={args.hvg_method!r} "
          f"from train manifest", flush=True)
    parquet_paths = []
    missing = []
    for sample, he_zarr, _idx in iter_manifest(args.train_manifest):
        p = manifest_path_to_transcripts_parquet(he_zarr)
        if p is None:
            missing.append(sample)
            continue
        parquet_paths.append(p)
    print(f"  [info] {len(parquet_paths)} parquets resolved, {len(missing)} samples missing on NFS", flush=True)

    selector = HVG_SELECTORS[args.hvg_method]
    kwargs = {"k": args.top_k_genes, "qv_min": args.qv_min}
    if args.hvg_method == "dispersion":
        kwargs["abundance_floor_top"] = args.hvg_abundance_floor_top
    panel = selector(parquet_paths, **kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "k": args.top_k_genes, "qv_min": args.qv_min,
            "hvg_method": args.hvg_method,
            "hvg_abundance_floor_top": (
                args.hvg_abundance_floor_top if args.hvg_method == "dispersion" else None
            ),
            "num_train_parquets": len(parquet_paths),
            "missing_samples": missing,
            "panel": panel,
        }, f, indent=2)
    print(f"[hvg] wrote {out_path} ({len(panel)} genes); first 10: {panel[:10]}", flush=True)


def phase_cache(args):
    """Dispatch to one of three sub-stages: data | uni | sigmma.

    ``--backbone`` selects sub-stage:
      * ``data``                            → stage 1, writes ``--data_cache_dir`` only
      * ``uni`` | ``bleep_uni`` | ...       → stage 2a, writes ``--uni_cache_dir``
      * ``sigmma``                          → stage 2b, writes ``--sigmma_cache_dir``

    Stages 2a/2b require stage 1 to have run already (they read FOV bboxes
    from the data cache). All three write a ``cache_meta.json`` on success.
    """
    panel_path = Path(args.panel_dir) / "gene_panel.json"
    if not panel_path.exists():
        sys.exit(f"[cache] missing gene panel at {panel_path}; run phase=hvg first")
    panel = json.load(open(panel_path))["panel"]
    print(f"[cache] backbone={args.backbone}, |panel|={len(panel)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_grid = args.cache_target_grid or args.target_grid

    if args.backbone == "data":
        if args.data_cache_dir is None:
            sys.exit("--data_cache_dir required for --backbone=data")
        data_dir = Path(args.data_cache_dir)
    elif args.backbone == "sigmma":
        if args.data_cache_dir is None or args.sigmma_cache_dir is None:
            sys.exit("--data_cache_dir and --sigmma_cache_dir required for --backbone=sigmma")
        sigmma_model, cfg = load_sigmma_for_probing(args.sigmma_config, device)
        data_dir = Path(args.data_cache_dir)
        sigmma_dir = Path(args.sigmma_cache_dir)
    elif args.backbone == "sigmma_lora":
        if args.data_cache_dir is None or args.sigmma_lora_cache_dir is None:
            sys.exit("--data_cache_dir and --sigmma_lora_cache_dir required for --backbone=sigmma_lora")
        sigmma_model, cfg = load_sigmma_for_probing(args.sigmma_config, device)
        data_dir = Path(args.data_cache_dir)
        sigmma_lora_dir = Path(args.sigmma_lora_cache_dir)
    elif args.backbone == "uni":
        if args.data_cache_dir is None or args.uni_cache_dir is None:
            sys.exit("--data_cache_dir and --uni_cache_dir required for --backbone=uni")
        uni_inner, cfg = load_uni_for_probing(args.sigmma_config, device)
        data_dir = Path(args.data_cache_dir)
        uni_dir = Path(args.uni_cache_dir)
    elif args.backbone in _UNI_FT_BACKBONES:
        if not args.ft_ckpt_path:
            sys.exit(f"--ft_ckpt_path required for --backbone={args.backbone}")
        if args.data_cache_dir is None or args.uni_cache_dir is None:
            sys.exit("--data_cache_dir and --uni_cache_dir required for --backbone=*_uni")
        uni_inner, cfg = load_finetuned_uni_for_probing(
            args.ft_ckpt_path, args.sigmma_config, device
        )
        data_dir = Path(args.data_cache_dir)
        uni_dir = Path(args.uni_cache_dir)
    else:
        sys.exit(f"unknown backbone {args.backbone!r}")

    encoder_input_size = (
        int(getattr(cfg.model, "encoder_input_size", 224))
        if args.backbone != "data" else 224
    )

    manifests = []
    if args.smoke_slide is None:
        manifests = [args.train_manifest, args.val_manifest]
    else:
        for mpath in [args.train_manifest, args.val_manifest]:
            df = pd.read_csv(mpath)
            if (df["sample_name"] == args.smoke_slide).any():
                manifests = [mpath]
                break
        if not manifests:
            sys.exit(f"[cache] smoke slide {args.smoke_slide} not found in any manifest")

    total = 0
    skipped_no_parquet = 0
    for mpath in manifests:
        for sample, he_zarr, idx_zarr in iter_manifest(mpath):
            if args.smoke_slide is not None and sample != args.smoke_slide:
                continue
            print(f"[cache:{args.backbone}] {sample}", flush=True)
            if args.backbone == "data":
                parquet_path = manifest_path_to_transcripts_parquet(he_zarr)
                if parquet_path is None:
                    skipped_no_parquet += 1
                    continue
                n = cache_data(
                    sample_name=sample,
                    he_zarr_path=he_zarr,
                    index_zarr_path=idx_zarr,
                    parquet_path=parquet_path,
                    gene_panel=panel,
                    fov_size=args.fov_size,
                    target_grid=cache_grid,
                    tissue_threshold=args.tissue_threshold,
                    qv_min=args.qv_min,
                    um_per_px=args.um_per_px,
                    data_cache_dir=data_dir,
                )
            elif args.backbone == "sigmma":
                n = cache_sigmma_features(
                    sample_name=sample,
                    he_zarr_path=he_zarr,
                    data_cache_dir=data_dir,
                    sigmma_cache_dir=sigmma_dir,
                    sigmma_model=sigmma_model,
                    device=device,
                )
            elif args.backbone == "sigmma_lora":
                n = cache_sigmma_lora_features(
                    sample_name=sample,
                    he_zarr_path=he_zarr,
                    encoder_input_size=encoder_input_size,
                    data_cache_dir=data_dir,
                    sigmma_lora_cache_dir=sigmma_lora_dir,
                    sigmma_model=sigmma_model,
                    device=device,
                )
            else:  # uni / *_uni
                n = cache_uni_features(
                    sample_name=sample,
                    he_zarr_path=he_zarr,
                    encoder_input_size=encoder_input_size,
                    data_cache_dir=data_dir,
                    uni_cache_dir=uni_dir,
                    uni_inner=uni_inner,
                    device=device,
                )
            total += n

    # Write cache_meta.json after a successful sweep.
    from sigmma.caches import write_cache_meta, panel_hash
    # NB: ``hvg_method`` is intentionally NOT recorded here. The gene panel is
    # read verbatim from gene_panel.json (which may be a dispersion-HVG panel
    # OR a pre-baked panel-aware SVG panel), so the selector label is irrelevant
    # to the cached targets and only caused confusion when SVG panels were
    # cached under an "--hvg_method dispersion" flag. ``panel_hash`` already
    # pins the exact gene set. The selector still applies in phase=hvg only.
    preproc = {
        "fov_size": int(args.fov_size),
        "target_grid": int(cache_grid),
        "panel_hash": panel_hash(panel),
    }
    if args.backbone == "data":
        write_cache_meta(
            data_dir, kind="shared_data", task="gex_2d", preprocessing=preproc,
        )
    elif args.backbone == "sigmma":
        write_cache_meta(
            sigmma_dir, kind="sigmma_features", task="gex_2d",
            backbone_variant="uni2h",
            backbone_input_dim=int(cfg.model.image_encoder_input_dim),
            patch_size=int(args.patch_size_hint or 14),
            preprocessing=preproc,
            ckpt_path=cfg.model.checkpoint_path,
        )
    elif args.backbone == "sigmma_lora":
        write_cache_meta(
            sigmma_lora_dir, kind="sigmma_lora_features", task="gex_2d",
            backbone_variant="uni2h",
            backbone_input_dim=int(cfg.model.image_encoder_input_dim),
            patch_size=int(args.patch_size_hint or 14),
            preprocessing={**preproc, "encoder_input": int(encoder_input_size)},
            ckpt_path=cfg.model.checkpoint_path,
        )
    else:
        write_cache_meta(
            uni_dir, kind="uni_features", task="gex_2d",
            backbone_variant="uni2h",
            backbone_input_dim=int(cfg.model.image_encoder_input_dim),
            patch_size=int(args.patch_size_hint or 14),
            preprocessing={**preproc, "encoder_input": int(encoder_input_size)},
        )
    print(f"[cache:{args.backbone}] DONE — wrote/checked {total} FOVs, "
          f"skipped {skipped_no_parquet} no-parquet samples", flush=True)


def _train_eval_split(args) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[str]]:
    """Return (train_keys, val_keys, panel) for the chosen mode.

    Keys are ``(sample, fov_id)``; resolved on the data cache (the only cache
    guaranteed to be populated for every cohort run).
    """
    panel = json.load(open(Path(args.panel_dir) / "gene_panel.json"))["panel"]
    if args.data_cache_dir is None:
        sys.exit("--data_cache_dir required for phase=train|eval")

    if args.smoke_slide is not None:
        all_keys = list_fov_keys(args.data_cache_dir, [args.smoke_slide])
        if not all_keys:
            sys.exit(f"[train] no cache for smoke slide {args.smoke_slide}")
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(all_keys))
        rng.shuffle(idx)
        n_val = max(1, int(0.2 * len(all_keys)))
        val_keys = [all_keys[i] for i in idx[:n_val]]
        train_keys = [all_keys[i] for i in idx[n_val:]]
        return train_keys, val_keys, panel

    train_samples = manifest_sample_names(args.train_manifest)
    val_samples = manifest_sample_names(args.val_manifest)
    train_keys = list_fov_keys(args.data_cache_dir, train_samples)
    val_keys = list_fov_keys(args.data_cache_dir, val_samples)
    return train_keys, val_keys, panel


def _validate_caches_for_train(args, *, need_uni: bool, need_sigmma: bool,
                                need_sigmma_lora: bool = False) -> None:
    """Hard-error if any required cache is missing or has mismatched meta."""
    from sigmma.caches import validate_cache_meta

    required_data = {
        "kind": "shared_data",
        "task": "gex_2d",
        "preprocessing": {
            "fov_size": int(args.fov_size),
            "target_grid": int(args.cache_target_grid or args.target_grid),
            # hvg_method deliberately not validated — see note in phase_cache.
            # Old caches written with hvg_method in their meta still validate
            # (validate_cache_meta only checks keys present in `required`).
        },
    }
    validate_cache_meta(args.data_cache_dir, required=required_data)

    if need_uni:
        if args.uni_cache_dir is None:
            sys.exit("--uni_cache_dir required for input_repr=uni")
        validate_cache_meta(
            args.uni_cache_dir,
            required={"kind": "uni_features", "task": "gex_2d"},
        )
    if need_sigmma:
        if args.sigmma_cache_dir is None:
            sys.exit("--sigmma_cache_dir required for input_repr in {concat,micro,meso,macro}")
        validate_cache_meta(
            args.sigmma_cache_dir,
            required={"kind": "sigmma_features", "task": "gex_2d"},
        )
    if need_sigmma_lora:
        if args.sigmma_lora_cache_dir is None:
            sys.exit("--sigmma_lora_cache_dir required for input_repr=lora_uni")
        validate_cache_meta(
            args.sigmma_lora_cache_dir,
            required={"kind": "sigmma_lora_features", "task": "gex_2d"},
        )


def _set_head_seed(seed: int) -> None:
    """Deterministic seeding for head-side stochasticity (init, shuffle, dropout).

    Does not enable torch.use_deterministic_algorithms — cuDNN kernel non-
    determinism in conv ops is left in place because the per-bin head uses
    only 1x1 convs / Linear and AdamW, where the residual noise is small.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_train(args):
    _set_head_seed(int(args.head_seed))
    need_uni = args.input_repr == "uni"
    need_sigmma = args.input_repr in {"concat", "micro", "meso", "macro"}
    need_sigmma_lora = args.input_repr == "lora_uni"
    _validate_caches_for_train(args, need_uni=need_uni, need_sigmma=need_sigmma,
                               need_sigmma_lora=need_sigmma_lora)
    train_keys, val_keys, panel = _train_eval_split(args)
    print(f"[train] input_repr={args.input_repr}  seed={args.head_seed}  "
          f"|train|={len(train_keys)} |val|={len(val_keys)}", flush=True)
    if not train_keys:
        sys.exit("[train] no training keys — nothing to do")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = in_channels_for(
        args.input_repr,
        uni_cache_dir=args.uni_cache_dir,
        sigmma_cache_dir=args.sigmma_cache_dir,
        sigmma_lora_cache_dir=args.sigmma_lora_cache_dir,
        sample_key=train_keys[0],
    )
    sweep_lrs = _parse_sweep_lrs(args.sweep_lrs) if args.head_type == "linear_sweep" else None
    head = build_head(args.head_type, in_ch, len(panel), args.hidden_dim,
                      sweep_lrs=sweep_lrs, loss=args.loss)
    print(f"  [head] type={args.head_type} loss={args.loss} in_ch={in_ch} G={len(panel)} "
          f"params={sum(p.numel() for p in head.parameters()):,}", flush=True)
    common_kwargs = dict(
        data_cache_dir=args.data_cache_dir,
        uni_cache_dir=args.uni_cache_dir,
        sigmma_cache_dir=args.sigmma_cache_dir,
        sigmma_lora_cache_dir=args.sigmma_lora_cache_dir,
        target_grid=args.target_grid,
        device=device,
        input_repr=args.input_repr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        log_every=args.log_every,
        val_every=args.val_every,
        in_memory=args.in_memory,
    )
    if args.head_type == "linear_sweep":
        print(f"  [sweep] {len(sweep_lrs)} LRs: {[f'{x:.4g}' for x in sweep_lrs]}", flush=True)
        history = train_linear_sweep(
            head, train_keys=train_keys, val_keys=val_keys if val_keys else None,
            **common_kwargs,
        )
    else:
        history = train_probing_head(
            head, train_keys=train_keys, val_keys=val_keys if val_keys else None,
            lr=args.lr, loss=args.loss, **common_kwargs,
        )
    out_dir = Path(args.out_root) / _heads_subdir(args.head_type, args.loss) / args.input_repr
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"seed{int(args.head_seed)}"
    save_history(
        history,
        str(out_dir),
        state_filename=f"best_{seed_tag}.pt",
        history_filename=f"history_{seed_tag}.json",
    )
    print(f"[train] DONE — best val loss {history['best_val_loss']}", flush=True)
    if args.head_type == "linear_sweep":
        print(f"  [sweep] best lr={history['best_key']} "
              f"per-head val losses={history['best_val_per_head']}", flush=True)
    print(f"  [save] {out_dir}/best_{seed_tag}.pt + history_{seed_tag}.json", flush=True)


def _aggregate_per_seed_summaries(per_seed: list[dict]) -> dict:
    """Reduce a list of per-seed ``summarize`` dicts to mean/std over scalars.

    Scalar keys (top-level float / int) get a {mean, std, values} sub-dict.
    Non-scalar keys (lists, dicts, strings, None) are kept from the first seed
    verbatim so the overall JSON layout still mirrors the single-seed schema
    for any downstream consumer that does not need cross-seed aggregation.
    """
    if not per_seed:
        return {}
    keys = list(per_seed[0].keys())
    agg: dict = {}
    for k in keys:
        vals = [d.get(k) for d in per_seed]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            arr = np.asarray(vals, dtype=np.float64)
            agg[k] = {
                "mean": float(np.nanmean(arr)),
                "std":  float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "values": [float(x) for x in arr],
            }
        else:
            agg[k] = vals[0]
    return agg


def phase_eval(args):
    need_uni = args.input_repr == "uni"
    need_sigmma = args.input_repr in {"concat", "micro", "meso", "macro"}
    need_sigmma_lora = args.input_repr == "lora_uni"
    _validate_caches_for_train(args, need_uni=need_uni, need_sigmma=need_sigmma,
                               need_sigmma_lora=need_sigmma_lora)
    train_keys, val_keys, panel = _train_eval_split(args)
    if not val_keys:
        sys.exit("[eval] no held-out keys to evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = in_channels_for(
        args.input_repr,
        uni_cache_dir=args.uni_cache_dir,
        sigmma_cache_dir=args.sigmma_cache_dir,
        sigmma_lora_cache_dir=args.sigmma_lora_cache_dir,
        sample_key=val_keys[0],
    )
    # For linear_sweep, the train phase saves the *inner* LinearProbingHead2D
    # state of the best LR (not the wrapper). Build a single linear head for
    # loading.
    eval_head_type = "linear" if args.head_type == "linear_sweep" else args.head_type
    head_dir = Path(args.out_root) / _heads_subdir(args.head_type, args.loss) / args.input_repr
    seed_ckpts = sorted(head_dir.glob("best_seed*.pt"))
    if not seed_ckpts:
        # Backwards compat: fall back to single-seed legacy file if present.
        legacy = head_dir / "best.pt"
        if legacy.exists():
            seed_ckpts = [legacy]
        else:
            sys.exit(f"[eval] no trained heads at {head_dir}/best_seed*.pt; run phase=train first")

    eval_grids = _parse_eval_target_grids(args.eval_target_grids, args.target_grid)
    # per-grid list of per-seed summary dicts, keyed by target_grid
    per_seed_by_grid: dict[int, list[dict]] = {tg: [] for tg in eval_grids}

    for ck in seed_ckpts:
        head = build_head(eval_head_type, in_ch, len(panel), args.hidden_dim, loss=args.loss)
        head.load_state_dict(torch.load(ck, map_location="cpu"))
        out = predict_all(
            head, val_keys, target_grid=args.target_grid, device=device,
            input_repr=args.input_repr,
            data_cache_dir=args.data_cache_dir,
            uni_cache_dir=args.uni_cache_dir,
            sigmma_cache_dir=args.sigmma_cache_dir,
            sigmma_lora_cache_dir=args.sigmma_lora_cache_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            in_memory=args.in_memory,
        )
        for tg in eval_grids:
            if tg == args.target_grid:
                pred_g, target_g, mask_g = out["pred"], out["target"], out["mask"]
            else:
                # Pool in count space (expm1 → sum → log1p) so the coarse-grid
                # metric matches what a baseline trained at tg would see.
                pred_counts = np.clip(np.expm1(out["pred"].astype(np.float64)), 0, None)
                target_counts = np.expm1(out["target"].astype(np.float64))
                pred_g = np.log1p(pool_predictions(pred_counts, tg)).astype(np.float32)
                target_g = np.log1p(pool_predictions(target_counts, tg)).astype(np.float32)
                mask_g = (
                    pool_predictions(out["mask"][:, None].astype(np.float64), tg)
                    / ((args.target_grid // tg) ** 2)
                )[:, 0].astype(np.float32)
            s = summarize(
                pred_g, target_g, mask_g, gene_panel=panel,
                sample_ids=out.get("sample_ids"),
                x_starts=out.get("x_starts"),
                y_starts=out.get("y_starts"),
                fov_size=int(out.get("fov_size", 0)),
                target_grid=int(tg),
            )
            s["input_repr"] = args.input_repr
            s["head_type"] = args.head_type
            s["loss"] = args.loss
            s["num_val_fovs"] = int(pred_g.shape[0])
            s["smoke_slide"] = args.smoke_slide
            s["target_grid"] = int(tg)
            s["pred_native_grid"] = int(args.target_grid)
            s["seed_ckpt"] = ck.name
            per_seed_by_grid[tg].append(s)
        print(f"[eval] {ck.name}: median per-gene Pearson @ g{args.target_grid} = "
              f"{per_seed_by_grid[eval_grids[0]][-1]['median_per_gene_pearson']:.4f}", flush=True)

    out_dir = Path(args.out_root) / _results_subdir(args.head_type, args.loss)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tg in eval_grids:
        agg = _aggregate_per_seed_summaries(per_seed_by_grid[tg])
        agg["num_seeds"] = len(seed_ckpts)
        agg["seed_ckpts"] = [c.name for c in seed_ckpts]
        if args.eval_target_grids:
            res_path = out_dir / f"{args.input_repr}_g{tg}.json"
        else:
            res_path = out_dir / f"{args.input_repr}.json"
        with open(res_path, "w") as f:
            json.dump(agg, f, indent=2)
        m = agg.get("median_per_gene_pearson")
        if isinstance(m, dict):
            print(f"[eval] g{tg} median per-gene Pearson (N={agg['num_seeds']}) = "
                  f"{m['mean']:.4f} ± {m['std']:.4f}", flush=True)
        else:
            print(f"[eval] g{tg} median per-gene Pearson = {m:.4f}", flush=True)
        print(f"  [save] {res_path}", flush=True)
    print(f"[eval] DONE — {len(eval_grids)} grid(s) evaluated, N={len(seed_ckpts)} seeds", flush=True)


def _parse_eval_target_grids(spec: Optional[str], fallback: int) -> list[int]:
    """Parse --eval_target_grids comma-separated list; return [fallback] if empty."""
    if not spec:
        return [int(fallback)]
    grids = [int(x) for x in spec.split(",") if x.strip()]
    for g in grids:
        if g <= 0 or fallback % g != 0:
            raise ValueError(
                f"eval grid {g} must be a positive divisor of --target_grid={fallback}"
            )
    return grids


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["hvg", "cache", "train", "eval", "all"], required=True)
    ap.add_argument("--backbone",
                    choices=["data", "sigmma", "sigmma_lora", "uni", "bleep_uni", "clip_uni", "omiclip_uni", "plip_uni"],
                    default=None,
                    help="Required for phase=cache. Sub-stage selector: "
                         "'data' → target+mask only (ckpt + backbone independent); "
                         "'sigmma' → SIGMMA per-scale features (reads data cache); "
                         "'sigmma_lora' → LoRA-adapted UNI patches at full spatial "
                         "resolution, no scale pooling; "
                         "'uni'/'*_uni' → UNI patch tokens (reads data cache).")
    ap.add_argument("--ft_ckpt_path", default=None,
                    help="Path to fine-tuned UNI checkpoint (e.g. BLEEP/CLIP/PLIP). "
                         "Required for --backbone in {bleep,clip,omiclip,plip}_uni.")
    ap.add_argument("--data_cache_dir", default=None,
                    help="Shared target+mask cache dir (kind=shared_data). Required "
                         "for every phase. Use sigmma.caches.data_cache_dir_gex_2d().")
    ap.add_argument("--uni_cache_dir", default=None,
                    help="UNI features cache dir (kind=uni_features). Required for "
                         "--backbone=uni|*_uni at cache time, and for input_repr=uni "
                         "at train/eval time.")
    ap.add_argument("--sigmma_cache_dir", default=None,
                    help="SIGMMA features cache dir (kind=sigmma_features). Required "
                         "for --backbone=sigmma at cache time, and for input_repr in "
                         "{concat,micro,meso,macro} at train/eval.")
    ap.add_argument("--sigmma_lora_cache_dir", default=None,
                    help="SIGMMA LoRA-only patch tokens cache dir "
                         "(kind=sigmma_lora_features). Required for "
                         "--backbone=sigmma_lora at cache time, and for "
                         "input_repr=lora_uni at train/eval.")
    ap.add_argument("--patch_size_hint", type=int, default=None,
                    help="Patch size to record in cache_meta (UNI v1=16, UNI2-h=14). "
                         "Best-effort metadata — actual size is auto-detected per "
                         "backbone at forward time.")
    ap.add_argument("--input_repr", choices=["concat", "micro", "meso", "macro", "uni", "lora_uni"],
                    default=None, help="Required for phase=train/eval/all.")
    ap.add_argument("--sigmma_config", default="configs/cell_niche_transfer.yaml")
    ap.add_argument("--train_manifest",
                    default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv")
    ap.add_argument("--val_manifest",
                    default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_val.csv")
    ap.add_argument("--out_root", default="outputs/gex_2d")
    ap.add_argument("--panel_dir", default=None,
                    help="Directory that holds gene_panel.json. Defaults to "
                         "--out_root. Set explicitly to share a single panel across "
                         "multiple out_roots (e.g. different target_grids).")

    ap.add_argument("--top_k_genes", type=int, default=50)
    ap.add_argument("--qv_min", type=float, default=20.0)
    ap.add_argument("--hvg_method", choices=list(HVG_SELECTORS.keys()), default="mean",
                    help="Gene-panel selection: 'mean' (default, by total count = "
                         "abundance), or 'dispersion' (cross-slide CV gated by abundance floor).")
    ap.add_argument("--hvg_abundance_floor_top", type=int, default=300,
                    help="dispersion-only: keep top-N most abundant genes before "
                         "ranking by CV. Removes transcript-floor noise.")
    ap.add_argument("--um_per_px", type=float, default=0.5)
    ap.add_argument("--fov_size", type=int, default=256)
    ap.add_argument("--target_grid", type=int, default=56,
                    help="Bin grid for training target. Cache stores raw counts at the "
                         "grid used at caching time; reducing --target_grid for "
                         "train/eval triggers sum-pooling on the fly (must divide "
                         "cache-native grid).")
    ap.add_argument("--cache_target_grid", type=int, default=None,
                    help="Native binning grid used at phase=cache. Defaults to "
                         "--target_grid for backward compatibility. Set explicitly "
                         "(e.g. 56) to cache once at a fine grid and downsample at "
                         "train/eval via --target_grid.")
    ap.add_argument("--tissue_threshold", type=float, default=20000.0)

    ap.add_argument("--num_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--hidden_dim", type=int, default=256,
                    help="MLP head hidden width. Ignored for --head_type=linear.")
    ap.add_argument("--loss", choices=["mse", "lognormal", "zilognormal"], default="mse",
                    help="Training objective over log1p-count targets. 'mse' = legacy point "
                         "regression. 'lognormal' = heteroscedastic Gaussian NLL (per-bin "
                         "predicted sigma; y~N(mu,sigma^2)). 'zilognormal' = zero-inflated "
                         "lognormal (extra P(y>0) classifier for the empty-bin spike). "
                         "Non-MSE losses require --head_type=mlp and write to a "
                         "'<heads|results>_<loss>' subdir so MSE outputs are preserved.")
    ap.add_argument("--head_type", choices=["mlp", "linear", "linear_sweep"], default="mlp",
                    help="Probing head architecture. 'mlp' = 1x1 conv -> GN -> ReLU -> 1x1 conv "
                         "(decodability/containability mix). 'linear' = BN(no-affine) -> 1x1 conv "
                         "(strict per-pixel linear regressor; isolates linear decodability). "
                         "'linear_sweep' = DINOv2-style joint training of K linear heads with "
                         "different LRs (SGD + cosine to 0), best by val MSE saved as best.pt.")
    ap.add_argument("--sweep_lrs", type=str, default="logspace:1e-3:1e0:7",
                    help="LR sweep spec for head_type=linear_sweep. "
                         "'logspace:lo:hi:n' or comma-separated floats.")
    ap.add_argument("--eval_target_grids", type=str, default=None,
                    help="Comma-separated list of target_grid values to evaluate "
                         "post-hoc at phase=eval. Each grid must divide "
                         "--target_grid (the native fine prediction grid). "
                         "Pooling is performed in count space (expm1→sum→log1p). "
                         "Writes one JSON per grid: {repr}_g{tg}.json. "
                         "If unset, falls back to legacy single-grid eval at "
                         "--target_grid with filename {repr}.json.")
    ap.add_argument("--in_memory", action="store_true",
                    help="Eagerly preload all cached FOVs into RAM at dataset construction. "
                         "Avoids per-batch lustre reads (~3.5s/batch for SIGMMA cache); makes "
                         "training GPU-bound. Adds ~325 MB RAM per slide; safe up to the job "
                         "memory budget.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the train/val FOV split (smoke mode) "
                         "and other split-time randomness. Use --head_seed to "
                         "vary head-training stochasticity instead.")
    ap.add_argument("--head_seed", type=int, default=0,
                    help="Seed applied at the start of phase=train (head init, "
                         "DataLoader shuffle, dropout). Saved/loaded as "
                         "best_seed{N}.pt; phase=eval aggregates all "
                         "best_seed*.pt to mean ± std.")
    ap.add_argument("--smoke_slide", default=None,
                    help="If set, restrict cache+train+eval to this single slide "
                         "with an intra-slide 80/20 FOV split (for Phase 0b smoke).")
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    if args.panel_dir is None:
        args.panel_dir = args.out_root

    if args.phase == "hvg":
        phase_hvg(args)
    elif args.phase == "cache":
        if args.backbone is None:
            sys.exit("--backbone {data,sigmma,uni,*_uni} required for phase=cache")
        phase_cache(args)
    elif args.phase == "train":
        if args.input_repr is None:
            sys.exit("--input_repr required for phase=train")
        phase_train(args)
    elif args.phase == "eval":
        if args.input_repr is None:
            sys.exit("--input_repr required for phase=eval")
        phase_eval(args)
    elif args.phase == "all":
        sys.exit("phase=all removed; run hvg → cache(data) → cache(uni|sigmma) → train → eval explicitly")


if __name__ == "__main__":
    main()
