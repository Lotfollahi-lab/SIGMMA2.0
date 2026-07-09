"""Cache path helpers + per-dir meta sanity-check for downstream probes.

Three layers of cache, each rooted under ``CACHE_ROOT`` (lustre):

  * ``data/<task>/<tag>/``    — ckpt + backbone independent (targets, masks,
                                crops, labels). Computed once forever.
  * ``uni/<task>/<tag>/``     — backbone-variant + preprocessing dependent
                                UNI features (e.g. 1024-d UNI v1 vs 1536-d
                                UNI2-h, fov_size, patch_size). Computed once
                                per UNI variant + preprocessing combo.
  * ``sigmma/<task>/<ckpt_tag>/`` — per-SIGMMA-checkpoint features. The only
                                    layer that recomputes per α-sweep element.

Every directory carries a ``cache_meta.json`` that ``validate_cache_meta``
hard-checks at read time: any mismatch raises ``RuntimeError`` so silent
backbone/version/ckpt drift cannot pollute results.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CACHE_ROOT = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
)
REPO_ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")

_META_NAME = "cache_meta.json"


# =====================================================================
# Path helpers
# =====================================================================

def data_cache_dir_gex_2d(
    *,
    hvg_method: str,
    target_grid: int,
    fov_size: int,
    root: Path | str = CACHE_ROOT,
) -> Path:
    """Targets + tissue masks for gex_2d FOVs (ckpt + backbone independent)."""
    return Path(root) / "data" / "gex_2d" / f"{hvg_method}_g{target_grid}_fov{fov_size}"


def data_cache_dir_niche(
    *,
    scale_sizes: Iterable[int],
    root: Path | str = CACHE_ROOT,
) -> Path:
    """Cell crops + polygon masks + niche labels (ckpt + backbone independent)."""
    tag = "scales_" + "_".join(str(s) for s in scale_sizes)
    return Path(root) / "data" / "cell_niche" / tag


def uni_cache_dir_gex_2d(
    *,
    backbone_variant: str,
    fov_size: int,
    patch_size: int,
    root: Path | str = CACHE_ROOT,
) -> Path:
    return Path(root) / "uni" / "gex_2d" / f"{backbone_variant}_fov{fov_size}_p{patch_size}"


def uni_cache_dir_niche(
    *,
    backbone_variant: str,
    encoder_input: int,
    patch_size: int,
    root: Path | str = CACHE_ROOT,
) -> Path:
    return Path(root) / "uni" / "cell_niche" / f"{backbone_variant}_input{encoder_input}_p{patch_size}"


def sigmma_cache_dir_gex_2d(
    *,
    ckpt_tag: str,
    root: Path | str = CACHE_ROOT,
) -> Path:
    """``ckpt_tag`` must uniquely identify a SIGMMA checkpoint+step combination
    (e.g. ``25923_1_step500``). Caller's responsibility to keep unique."""
    return Path(root) / "sigmma" / "gex_2d" / ckpt_tag


def sigmma_cache_dir_niche(
    *,
    ckpt_tag: str,
    root: Path | str = CACHE_ROOT,
) -> Path:
    return Path(root) / "sigmma" / "cell_niche" / ckpt_tag


# =====================================================================
# Meta read / write / validate
# =====================================================================

def _git_sha(repo: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _ckpt_fingerprint(ckpt_path: str | Path) -> dict[str, Any]:
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"ckpt not found for fingerprint: {p}")
    st = p.stat()
    return {
        "ckpt_path": str(p),
        "ckpt_mtime_ns": st.st_mtime_ns,
        "ckpt_size_bytes": st.st_size,
    }


def write_cache_meta(
    cache_dir: Path | str,
    *,
    kind: str,
    task: str,
    backbone_variant: str | None = None,
    backbone_input_dim: int | None = None,
    patch_size: int | None = None,
    preprocessing: dict[str, Any] | None = None,
    ckpt_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write cache_meta.json. Call after the cache phase completes (so an
    interrupted cache run leaves the dir unmarked → forces re-cache).

    ``kind`` ∈ {"shared_data", "uni_features", "sigmma_features"}.
    ``task`` ∈ {"gex_2d", "cell_niche"}.
    """
    meta: dict[str, Any] = {
        "kind": kind,
        "task": task,
        "backbone_variant": backbone_variant,
        "backbone_input_dim": backbone_input_dim,
        "patch_size": patch_size,
        "preprocessing": preprocessing or {},
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if ckpt_path is not None:
        meta.update(_ckpt_fingerprint(ckpt_path))
    if extra:
        meta.update(extra)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / _META_NAME
    p.write_text(json.dumps(meta, indent=2))
    return p


def read_cache_meta(cache_dir: Path | str) -> dict[str, Any]:
    p = Path(cache_dir) / _META_NAME
    if not p.exists():
        raise RuntimeError(
            f"cache_meta.json not found at {cache_dir} — cache is missing, "
            "stale, or was written by old code. Re-run the seed/cache phase."
        )
    return json.loads(p.read_text())


def validate_cache_meta(
    cache_dir: Path | str,
    *,
    required: dict[str, Any],
) -> None:
    """Hard-error on any mismatch between on-disk meta and ``required``.

    ``required`` keys are matched at the top level. The ``preprocessing`` key
    (if present in ``required``) is matched field-by-field against the cache's
    ``preprocessing`` block. Missing keys in the cache → mismatch.

    Use this at read time before train/eval to refuse loading mismatched
    backbones, preprocessing, or ckpts. Cheap (one JSON parse).
    """
    meta = read_cache_meta(cache_dir)
    errors = []
    for k, want in required.items():
        if k == "preprocessing":
            got_pp = meta.get("preprocessing", {})
            for kk, vv in (want or {}).items():
                got = got_pp.get(kk)
                if got != vv:
                    errors.append(f"  preprocessing.{kk}: cache={got!r} required={vv!r}")
        else:
            got = meta.get(k)
            if got != want:
                errors.append(f"  {k}: cache={got!r} required={want!r}")
    if errors:
        raise RuntimeError(
            f"cache_meta mismatch at {cache_dir}:\n"
            + "\n".join(errors)
            + f"\n(full meta: {meta})"
        )


# =====================================================================
# Convenience: panel hashing (gex_2d only)
# =====================================================================

def panel_hash(gene_names: Iterable[str]) -> str:
    """Stable SHA-256 of a sorted gene list. Goes into preprocessing meta so
    a panel swap forces a cache rebuild."""
    import hashlib
    h = hashlib.sha256()
    for g in sorted(gene_names):
        h.update(g.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()[:16]
