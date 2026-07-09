"""Shared SIGMMA-manifest-driven dataset for H&E→ST baseline training.

Each item is ``(image_uint8 [3, fov_size, fov_size], gene_target [G], meta)``
where ``gene_target`` is the **FOV-aggregated** log1p transcript count over
the chosen gene panel.

Aggregation: we sum the per-bin (G, K, K) tensor returned by ``bin_fov``
over the K×K grid to get the total transcript count per gene in the FOV,
then apply ``log1p``. This is mathematically equivalent to bin-aggregating
raw counts (no information lost by going through K×K first).

Caching: slide transcripts are loaded once per slide via
``load_slide_transcripts`` and held while we iterate its FOVs (caller is
responsible — see ``FovGexIndex``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from sigmma.evaluation.gex_cache_io import (
    SectionHandleCache,
    read_fov_ids,
    section_h5_exists,
)
from sigmma.evaluation.transcript_binning import (
    bin_fov,
    iter_fov_bboxes,
    load_slide_transcripts,
    manifest_path_to_transcripts_parquet,
)


def load_gene_panel(panel_path: str) -> list[str]:
    """Read the gene panel JSON written by ``probe_gex_2d.py`` phase=hvg."""
    panel = json.load(open(panel_path))
    return panel["panel"]


class FovGexIndex:
    """Resolve all (slide, fov_id, x_start, y_start) pairs from a manifest.

    Filters slides whose transcripts parquet cannot be resolved on NFS.
    The materialised list of FOVs is what the Dataset will sample over.
    """

    def __init__(self, manifest_path: str, fov_size: int, tissue_threshold: float):
        self.manifest_path = manifest_path
        self.fov_size = int(fov_size)
        self.tissue_threshold = float(tissue_threshold)
        self._build()

    def _build(self) -> None:
        df = pd.read_csv(self.manifest_path)
        self.slides: list[dict] = []
        self.fovs: list[tuple[int, int, int, int]] = []  # (slide_idx, fov_id, x, y)
        skipped = 0
        t0 = time.time()
        for _, row in df.iterrows():
            sample = row["sample_name"]
            he_zarr = row["hematoxylin_eosin_image"]
            idx_zarr = row["index"]
            parquet = manifest_path_to_transcripts_parquet(he_zarr)
            if parquet is None:
                skipped += 1
                continue
            slide_idx = len(self.slides)
            try:
                im = zarr.open(he_zarr, mode="r")
                H, W = im.shape[1], im.shape[2]
            except Exception:
                skipped += 1
                continue
            n_added = 0
            for fov_id, x_start, y_start, _ in iter_fov_bboxes(
                idx_zarr, tissue_filtering_threshold=self.tissue_threshold
            ):
                if (x_start + self.fov_size > W) or (y_start + self.fov_size > H):
                    continue
                self.fovs.append((slide_idx, int(fov_id), int(x_start), int(y_start)))
                n_added += 1
            self.slides.append({
                "sample": sample,
                "he_zarr": he_zarr,
                "parquet": parquet,
                "n_fovs": n_added,
            })
        print(f"[FovGexIndex] {len(self.slides)} slides, {len(self.fovs)} FOVs "
              f"(skipped {skipped} samples); built in {time.time() - t0:.1f}s",
              flush=True)

    def __len__(self) -> int:
        return len(self.fovs)


class FovGexDataset(Dataset):
    """Lazy-load per-FOV H&E image + bin-aggregated gene target.

    Loads each slide's transcripts on first access and caches them so
    subsequent FOVs from the same slide reuse the load. zarr H&E reads are
    lazy and per-FOV (no full-slide preload).

    Returns ``image_uint8 (3, fov, fov) int8``, ``target_log1p (G,) float32``,
    and ``meta`` dict (slide_idx, fov_id, x, y).
    """

    def __init__(
        self,
        index: FovGexIndex,
        gene_panel: list[str],
        fov_size: int = 256,
        target_grid: int = 14,
        qv_min: float = 20.0,
        um_per_px: float = 0.5,
        max_slides_in_cache: int = 4,
    ):
        self.index = index
        self.gene_panel = list(gene_panel)
        self.fov_size = int(fov_size)
        self.target_grid = int(target_grid)
        self.qv_min = float(qv_min)
        self.um_per_px = float(um_per_px)
        self.max_slides_in_cache = int(max_slides_in_cache)

        # Per-slide caches keyed by slide_idx. _zarr_cache holds zarr arrays;
        # _transcript_cache holds bin_fov-ready transcript dicts.
        self._zarr_cache: dict[int, object] = {}
        self._transcript_cache: dict[int, dict] = {}
        self._lru: list[int] = []

    def __len__(self) -> int:
        return len(self.index)

    def _get_slide(self, slide_idx: int):
        if slide_idx in self._zarr_cache and slide_idx in self._transcript_cache:
            return self._zarr_cache[slide_idx], self._transcript_cache[slide_idx]
        slide = self.index.slides[slide_idx]
        z = self._zarr_cache.get(slide_idx)
        if z is None:
            z = zarr.open(slide["he_zarr"], mode="r")
            self._zarr_cache[slide_idx] = z
        loaded = self._transcript_cache.get(slide_idx)
        if loaded is None:
            loaded = load_slide_transcripts(
                slide["parquet"], self.gene_panel,
                qv_min=self.qv_min, um_per_px=self.um_per_px,
            )
            self._transcript_cache[slide_idx] = loaded
        # LRU bump
        if slide_idx in self._lru:
            self._lru.remove(slide_idx)
        self._lru.append(slide_idx)
        while len(self._lru) > self.max_slides_in_cache:
            evict = self._lru.pop(0)
            self._zarr_cache.pop(evict, None)
            self._transcript_cache.pop(evict, None)
        return z, loaded

    def __getitem__(self, i: int):
        slide_idx, fov_id, x_start, y_start = self.index.fovs[i]
        z, loaded = self._get_slide(slide_idx)
        tile = np.asarray(z[:, y_start:y_start + self.fov_size,
                            x_start:x_start + self.fov_size])  # (3, fov, fov) uint8
        bin_counts = bin_fov(loaded, x_start, y_start, self.fov_size, self.target_grid)
        gene_total = bin_counts.sum(axis=(1, 2)).astype(np.float32)
        target = np.log1p(gene_total)
        meta = {
            "slide_idx": int(slide_idx),
            "sample": self.index.slides[slide_idx]["sample"],
            "fov_id": int(fov_id),
            "x_start": int(x_start),
            "y_start": int(y_start),
        }
        return torch.from_numpy(tile), torch.from_numpy(target), meta


class FovGexCacheIndex:
    """FOV index backed by the Stage-1 data cache (``probe_gex_2d`` phase=cache).

    Skips parquet I/O entirely — FOVs come from the per-section ``<sample>.h5``
    cache, bbox + binned target read on demand from one chunk per FOV. Image
    bytes still come from zarr per-FOV (cheap chunked read).
    """

    def __init__(self, manifest_path: str, data_cache_dir: str):
        self.manifest_path = manifest_path
        self.data_cache_dir = str(data_cache_dir)
        self._build()

    def _build(self) -> None:
        df = pd.read_csv(self.manifest_path)
        self.slides: list[dict] = []
        self.fovs: list[tuple[int, int]] = []  # (slide_idx, fov_id)
        skipped = 0
        t0 = time.time()
        for _, row in df.iterrows():
            sample = row["sample_name"]
            he_zarr = row["hematoxylin_eosin_image"]
            if not section_h5_exists(self.data_cache_dir, sample):
                skipped += 1
                continue
            fov_ids = sorted(read_fov_ids(self.data_cache_dir, sample))
            if not fov_ids:
                skipped += 1
                continue
            slide_idx = len(self.slides)
            self.slides.append({
                "sample": sample,
                "he_zarr": he_zarr,
            })
            for fov_id in fov_ids:
                self.fovs.append((slide_idx, int(fov_id)))
        print(f"[FovGexCacheIndex] {len(self.slides)} slides, {len(self.fovs)} FOVs "
              f"(skipped {skipped} samples); built in {time.time() - t0:.1f}s",
              flush=True)

    def __len__(self) -> int:
        return len(self.fovs)


class FovGexCacheDataset(Dataset):
    """Cache-backed counterpart of :class:`FovGexDataset`.

    Returns the same ``(image_uint8 (3, fov, fov), target_log1p (G,), meta)``
    tuple as :class:`FovGexDataset` — train loop is identical. Target is the
    FOV total transcript count summed over the cached K×K binned target,
    then log1p. No parquet load, no ``bin_fov`` call per iteration.
    """

    def __init__(
        self,
        index: FovGexCacheIndex,
        gene_panel: list[str],
        fov_size: int = 256,
    ):
        self.index = index
        self.gene_panel = list(gene_panel)
        self.fov_size = int(fov_size)
        self._zarr_cache: dict[int, object] = {}
        # Fork-safe: empty here, populated lazily per DataLoader worker.
        self._handles = SectionHandleCache()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        slide_idx, fov_id = self.index.fovs[i]
        slide = self.index.slides[slide_idx]
        z = self._zarr_cache.get(slide_idx)
        if z is None:
            z = zarr.open(slide["he_zarr"], mode="r")
            self._zarr_cache[slide_idx] = z
        r = self._handles.reader(self.index.data_cache_dir, slide["sample"])
        row = r.row(fov_id)
        target_bins = r.f["target"][row]  # (G, K, K) float32
        x_start = int(r.f["x_start"][row])
        y_start = int(r.f["y_start"][row])
        fov_size = r.fov_size
        gene_total = target_bins.sum(axis=(1, 2)).astype(np.float32)
        target_log1p = np.log1p(gene_total)
        tile = np.asarray(
            z[:, y_start:y_start + fov_size, x_start:x_start + fov_size]
        )
        meta = {
            "slide_idx": int(slide_idx),
            "sample": slide["sample"],
            "fov_id": int(fov_id),
            "x_start": x_start,
            "y_start": y_start,
        }
        return torch.from_numpy(tile), torch.from_numpy(target_log1p), meta


def collate_baseline(batch):
    """Stack (image, target) and gather meta dicts into a list."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return images, targets, metas


def compute_mean_target(dataset: FovGexDataset, max_samples: int = 256) -> torch.Tensor:
    """Estimate the per-gene mean of the log1p target on a random subset.

    Used to initialise ST-Net's regression-head bias (mirrors the original
    ``mean_expression`` initialisation).
    """
    n = min(len(dataset), max_samples)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset), size=n, replace=False)
    acc = None
    for k, i in enumerate(idx):
        _, target, _ = dataset[int(i)]
        target = target.numpy()
        if acc is None:
            acc = np.zeros_like(target)
        acc += target
    return torch.from_numpy(acc / max(1, len(idx))).float()
