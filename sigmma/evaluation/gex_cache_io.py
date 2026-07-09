"""Per-section HDF5 cache I/O for the gex_2d probe.

Replaces the legacy *one-.npz-per-FOV* layout — which produced 10^5–10^6 tiny
files on Lustre (a 506k-file data cache was observed) and an MDS round-trip per
``__getitem__`` — with **one ``.h5`` per section**:

  * Each per-FOV array (``target`` / ``features`` / ``micro`` …) is one HDF5
    dataset shaped ``(n_fov, *per_fov_shape)`` and chunked ``(1, *per_fov_shape)``
    so a single FOV read fetches exactly one chunk (no read amplification).
  * 1-D per-FOV metadata (``fov_id`` / ``x_start`` / ``y_start``) is stored
    contiguous + uncompressed; ``fov_size`` (constant per section) is an attr.
  * Handles are opened **lazily inside each DataLoader worker** and cached for
    reuse — HDF5 file handles must not be inherited across ``fork`` (build the
    empty :class:`SectionHandleCache` in the parent; first access per worker
    opens that worker's own handles). ``ulimit -n`` on the farm is 131072 so
    keeping every section handle open per worker is safe.

Writes are atomic: a section is staged to ``<sample>.h5.tmp`` and ``os.replace``-d
on success, so a killed job never leaves a partial file that looks complete.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

CACHE_EXT = ".h5"


# =====================================================================
# Paths / enumeration
# =====================================================================
def section_h5_path(cache_dir: str | Path, sample: str) -> Path:
    return Path(cache_dir) / f"{sample}{CACHE_EXT}"


def section_h5_exists(cache_dir: str | Path, sample: str) -> bool:
    return section_h5_path(cache_dir, sample).exists()


def list_section_samples(cache_dir: str | Path) -> list[str]:
    """Sample names with a completed section cache under ``cache_dir``."""
    d = Path(cache_dir)
    if not d.is_dir():
        return []
    return sorted(p.name[: -len(CACHE_EXT)] for p in d.glob(f"*{CACHE_EXT}"))


def read_fov_ids(cache_dir: str | Path, sample: str) -> list[int]:
    with h5py.File(section_h5_path(cache_dir, sample), "r") as f:
        return [int(x) for x in f["fov_id"][:]]


def iter_section_bboxes(
    cache_dir: str | Path, sample: str
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(fov_id, x_start, y_start, fov_size)`` for every FOV in a section."""
    p = section_h5_path(cache_dir, sample)
    if not p.exists():
        return
    with h5py.File(p, "r") as f:
        fov_size = int(f.attrs["fov_size"])
        fids = f["fov_id"][:]
        xs = f["x_start"][:]
        ys = f["y_start"][:]
    for fid, x, y in zip(fids, xs, ys):
        yield int(fid), int(x), int(y), fov_size


def iter_section_fovs(
    cache_dir: str | Path, sample: str, fields: Iterable[str] = ()
) -> Iterator[dict]:
    """Yield one dict per FOV (sequential read over a single section file).

    Each dict always carries ``fov_id`` + ``fov_size``, plus ``x_start`` /
    ``y_start`` when present (only the data cache stores bboxes), plus every
    name in ``fields`` that exists as a dataset (e.g. ``target`` / ``mask`` /
    ``features`` / ``micro`` / ``meso`` / ``macro``). Mirrors the per-key access
    of the old per-FOV ``.npz`` dicts so diagnostic scripts port cleanly.
    """
    p = section_h5_path(cache_dir, sample)
    if not p.exists():
        return
    fields = list(fields)
    with h5py.File(p, "r") as f:
        fov_size = int(f.attrs["fov_size"])
        fids = f["fov_id"][:]
        xs = f["x_start"][:] if "x_start" in f else None
        ys = f["y_start"][:] if "y_start" in f else None
        for i in range(len(fids)):
            d = {"fov_id": int(fids[i]), "fov_size": fov_size}
            if xs is not None:
                d["x_start"] = int(xs[i])
            if ys is not None:
                d["y_start"] = int(ys[i])
            for name in fields:
                if name in f:
                    d[name] = f[name][i]
            yield d


# =====================================================================
# Writer
# =====================================================================
class SectionCacheWriter:
    """Streaming, atomic writer for one section's per-FOV cache.

    ``datasets`` maps name → ``(per_fov_shape, dtype)``. A scalar per-FOV field
    uses ``per_fov_shape == ()`` (stored contiguous, uncompressed); arrays use
    per-FOV chunking + ``compression``. Fill rows in any order with
    :meth:`write_row`, then :meth:`close` (or use as a context manager, which
    aborts + unlinks the temp file on exception).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        sample: str,
        n_fov: int,
        datasets: dict[str, tuple[tuple[int, ...], np.dtype | type]],
        *,
        attrs: dict | None = None,
        compression: str | None = "lzf",
    ):
        self.path = section_h5_path(cache_dir, sample)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self.n_fov = int(n_fov)
        self.f = h5py.File(self._tmp, "w")
        for name, (shp, dt) in datasets.items():
            shp = tuple(shp)
            self.f.create_dataset(
                name,
                shape=(self.n_fov, *shp),
                dtype=dt,
                chunks=((1, *shp) if shp else None),
                compression=(compression if shp else None),
            )
        for k, v in (attrs or {}).items():
            self.f.attrs[k] = v

    def write_row(self, i: int, **arrays) -> None:
        for k, v in arrays.items():
            self.f[k][i] = v

    def close(self) -> None:
        self.f.close()
        os.replace(self._tmp, self.path)

    def abort(self) -> None:
        try:
            self.f.close()
        finally:
            try:
                self._tmp.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "SectionCacheWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


# =====================================================================
# Reader + fork-safe handle cache
# =====================================================================
class _SectionReader:
    """One open read-only section file plus its ``fov_id → row`` map."""

    __slots__ = ("f", "_row")

    def __init__(self, path: str | Path):
        self.f = h5py.File(path, "r")
        self._row = {int(fid): i for i, fid in enumerate(self.f["fov_id"][:])}

    def row(self, fov_id: int) -> int:
        return self._row[int(fov_id)]

    def get(self, name: str, fov_id: int) -> np.ndarray:
        return self.f[name][self.row(fov_id)]

    def has(self, name: str) -> bool:
        return name in self.f

    @property
    def fov_size(self) -> int:
        return int(self.f.attrs["fov_size"])

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


class SectionHandleCache:
    """Lazy, fork-safe cache of open :class:`_SectionReader` handles keyed by
    ``(cache_dir, sample)``.

    Construct the (empty) cache in the parent process; the first access in each
    DataLoader worker opens that worker's own handles. Never open handles before
    a ``fork`` — see module docstring.
    """

    def __init__(self):
        self._readers: dict[tuple[str, str], _SectionReader] = {}

    def reader(self, cache_dir: str | Path, sample: str) -> _SectionReader:
        key = (str(cache_dir), str(sample))
        r = self._readers.get(key)
        if r is None:
            r = _SectionReader(section_h5_path(cache_dir, sample))
            self._readers[key] = r
        return r

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers.clear()

    def __del__(self):
        self.close()
