"""Pack a legacy per-FOV ``.npz`` gex_2d cache into per-section ``.h5``.

The old layout wrote one ``<cache_dir>/<sample>/fov_<id>.npz`` per FOV, producing
10^5–10^6 tiny files on Lustre. This packs every ``fov_*.npz`` under each sample
into a single ``<cache_dir>/<sample>.h5`` (the format written by
``sigmma.evaluation.gex_cache_io.SectionCacheWriter``) **without recomputing** —
it just re-containers the bytes, so it is cheap (disk-bound) and exact.

Usage (one cache dir at a time — data / uni / sigmma / sigmma_lora):

    python -m scripts.convert_gex_cache_npz_to_h5 \
        --cache_dir /lustre/.../cache/data/gex_2d/svg0529_panel_..._g14_fov256

    # also remove the per-FOV npz dirs after a verified pack:
    python -m scripts.convert_gex_cache_npz_to_h5 --cache_dir <dir> --delete-npz

The npz key set is auto-detected per sample (``target``/``mask``/``fov_id``/
``x_start``/``y_start`` for the data cache; ``features``+``fov_id`` for uni /
sigmma_lora; ``micro``/``meso``/``macro``+``fov_id`` for sigmma). ``fov_size``
becomes an attr. A sample is skipped if its ``<sample>.h5`` already exists.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from sigmma.evaluation.gex_cache_io import SectionCacheWriter, section_h5_path

# Per-FOV scalar metadata stored as 1-D datasets (not chunked/compressed).
_SCALAR_KEYS = ("fov_id", "x_start", "y_start")
# fov_size is constant per section → stored as an attr, not a dataset.
_ATTR_KEYS = ("fov_size",)


def _sample_dirs(cache_dir: Path) -> list[Path]:
    return sorted(p for p in cache_dir.iterdir()
                  if p.is_dir() and any(p.glob("fov_*.npz")))


def convert_sample(cache_dir: Path, sample_dir: Path, *, compression: str | None) -> int:
    sample = sample_dir.name
    out = section_h5_path(cache_dir, sample)
    if out.exists():
        print(f"  [skip] {sample}: {out.name} already exists", flush=True)
        return 0

    files = sorted(sample_dir.glob("fov_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        return 0

    # Inspect the first FOV to learn the dataset layout (array keys + shapes).
    with np.load(files[0]) as f0:
        keys = list(f0.files)
        array_keys = [k for k in keys if k not in _SCALAR_KEYS and k not in _ATTR_KEYS]
        datasets: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        for k in array_keys:
            datasets[k] = (f0[k].shape, f0[k].dtype)
        for k in _SCALAR_KEYS:
            if k in keys:
                datasets[k] = ((), np.int32)
        attrs = {}
        for k in _ATTR_KEYS:
            if k in keys:
                attrs[k] = int(f0[k])

    with SectionCacheWriter(cache_dir, sample, len(files),
                            datasets=datasets, attrs=attrs,
                            compression=compression) as w:
        for i, p in enumerate(files):
            with np.load(p) as f:
                row = {k: f[k] for k in array_keys}
                for k in _SCALAR_KEYS:
                    if k in datasets:
                        row[k] = int(f[k])
                w.write_row(i, **row)
    print(f"  [ok] {sample}: packed {len(files)} FOVs → {out.name}", flush=True)
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="A single gex_2d cache dir (data / uni / sigmma / "
                         "sigmma_lora) containing <sample>/fov_*.npz subdirs.")
    ap.add_argument("--compression", default="lzf",
                    help="HDF5 compression for array datasets (lzf|gzip|none).")
    ap.add_argument("--delete-npz", action="store_true",
                    help="After packing a sample, delete its fov_*.npz dir. "
                         "Only the per-sample dir whose .h5 was just written is "
                         "removed; existing .h5 are never touched.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f"not a dir: {cache_dir}")
    compression = None if args.compression.lower() == "none" else args.compression

    sample_dirs = _sample_dirs(cache_dir)
    print(f"[convert] {cache_dir} — {len(sample_dirs)} samples with npz FOVs", flush=True)
    total = 0
    for sd in sample_dirs:
        n = convert_sample(cache_dir, sd, compression=compression)
        total += n
        if args.delete_npz and section_h5_path(cache_dir, sd.name).exists():
            shutil.rmtree(sd)
            print(f"  [rm] {sd.name}/ ({sd})", flush=True)
    print(f"[convert] DONE — packed {total} FOVs across {len(sample_dirs)} samples",
          flush=True)


if __name__ == "__main__":
    main()
