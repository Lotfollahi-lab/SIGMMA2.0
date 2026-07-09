"""Compute section-level retrieval R@K mean±std from cached score_matrices.npz.

The benchmark_retrieval.py script saves
``scores_img2spatial`` / ``scores_spatial2img`` of shape ``(N_query, N_ref)``.
Rows of N follow the SequentialSampler order = manifest order of slides ×
per-slide ``SpatialPatchDataset.valid_indices`` order.

This script:
1. Resolves the row -> section_id mapping by instantiating the same
   MultiSectionDataset once (header-loads only — no image fetch).
2. For each section, computes R@K (image_recall@*, spatial_recall@*) treating
   the section's FOVs as queries against the full reference set, plus the
   matched-section subset for completeness.
3. Aggregates mean ± std across sections, plus pooled (current default) for
   reference.

Usage:
  python -m scripts.retrieval_section_level \
      --config configs/in-house_retrieval_bleep_uni_finetuned_step1000.yaml \
      --scores outputs/retrieval/baselines_uni2h_step1000/.../score_matrices.npz \
      --out_json outputs/retrieval/.../recall_metrics_section_level.json \
      [--section_cache outputs/retrieval/section_assignments_test.npz]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, SequentialSampler

from sigmma.utils import load_dataset
from sigmma.datasets import multimodal_collate


def build_section_assignments(cfg, model_name: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Iterate the retrieval dataloader and record (section_id) per surviving row.

    Mirrors ``benchmark_retrieval.py`` / ``load_embeddings``: rows with no image
    tensor are skipped, so the score matrix is shorter than ``len(dataset)``.
    """
    if model_name is None:
        model_name = cfg.model.get("name")
    ds = load_dataset(cfg.dataset.query_dataset, is_train=False, model_name=model_name)
    sampler = SequentialSampler(ds)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.test.get("batch_size", 128)),
        sampler=sampler,
        num_workers=0,  # head-node safe; image decode is the cost anyway
        collate_fn=multimodal_collate,
    )

    section_id_per_row: list[str] = []
    for batch in loader:
        img = batch.get("image")
        if not isinstance(img, torch.Tensor) or img.numel() == 0:
            continue
        # metadata.section_id may be a list (batched) of str
        meta = batch.get("metadata", {})
        sids = meta.get("section_id") if isinstance(meta, dict) else None
        if sids is None:
            raise RuntimeError("batch missing metadata.section_id; cannot derive sections")
        section_id_per_row.extend(list(sids))

    unique = sorted(set(section_id_per_row))
    name_to_idx = {s: i for i, s in enumerate(unique)}
    section_idx = np.array([name_to_idx[s] for s in section_id_per_row], dtype=np.int32)
    return section_idx, unique


def recall_at(scores: np.ndarray, target_idx: np.ndarray, k: int) -> float:
    """scores: (Q, R); target_idx: (Q,) with the GT reference row index per query.
    Returns the fraction of queries whose target_idx is in the top-k of scores[q]."""
    Q, R = scores.shape
    k = min(k, R)
    top_k = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = (top_k == target_idx[:, None]).any(axis=1)
    return float(hits.mean())


def recall_at_percent(scores: np.ndarray, target_idx: np.ndarray, p: float) -> float:
    Q, R = scores.shape
    k = max(1, int(round(R * p)))
    return recall_at(scores, target_idx, k)


def compute_metrics(scores_i2s: np.ndarray, scores_s2i: np.ndarray, target_idx: np.ndarray):
    out = {}
    for p in (0.01, 0.05, 0.10):
        out[f"image_recall@{int(p * 100)}percent"] = recall_at_percent(scores_i2s, target_idx, p)
        out[f"spatial_recall@{int(p * 100)}percent"] = recall_at_percent(scores_s2i, target_idx, p)
    for k in (1, 5, 10):
        out[f"image_recall@{k}"] = recall_at(scores_i2s, target_idx, k)
        out[f"spatial_recall@{k}"] = recall_at(scores_s2i, target_idx, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--section_cache", default="outputs/retrieval/section_assignments_test.npz",
                    help="Cached row->section mapping; built once from config if absent.")
    args = ap.parse_args()

    cfg = ConfigDict(yaml.safe_load(open(args.config)))

    cache = Path(args.section_cache)
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        section_idx = c["section_idx"]
        section_ids = list(c["section_ids"])
        print(f"[section] loaded cache: {len(section_ids)} sections, N={len(section_idx)}")
    else:
        section_idx, section_ids = build_section_assignments(cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, section_idx=section_idx, section_ids=np.array(section_ids, dtype=object))
        print(f"[section] built and cached: {len(section_ids)} sections, N={len(section_idx)} -> {cache}")

    z = np.load(args.scores)
    s_i2s = z["scores_img2spatial"]
    s_s2i = z["scores_spatial2img"]
    Q, R = s_i2s.shape
    assert Q == len(section_idx), f"score Q={Q} mismatches section_idx len={len(section_idx)}"
    # Reference order == query order (copy.deepcopy in benchmark_retrieval.py)
    target_idx = np.arange(Q)

    # Pooled (current default reporting)
    pooled = compute_metrics(s_i2s, s_s2i, target_idx)

    # Per-section: treat each section's rows as the query subset.
    per_section = {}
    for i, sid in enumerate(section_ids):
        mask = section_idx == i
        if mask.sum() < 2:
            continue
        sub_i2s = s_i2s[mask]
        sub_s2i = s_s2i[mask]
        sub_target = target_idx[mask]
        per_section[sid] = compute_metrics(sub_i2s, sub_s2i, sub_target)
        per_section[sid]["n_fov"] = int(mask.sum())

    # mean ± std across sections (each section equal weight)
    keys = [k for k in pooled.keys()]
    stats = {}
    for k in keys:
        vals = np.array([per_section[sid][k] for sid in per_section])
        stats[k] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_sections": int(len(vals)),
            "median": float(np.median(vals)),
        }

    out = {
        "pooled": pooled,
        "section_level": stats,
        "per_section": per_section,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print(f"\n=== Pooled R@K (current default) ===")
    for k, v in pooled.items():
        print(f"  {k:30s} {v:.4f}")
    print(f"\n=== Section-level mean±std (n_sections={stats[keys[0]]['n_sections']}) ===")
    for k in keys:
        print(f"  {k:30s} {stats[k]['mean']:.4f} ± {stats[k]['std']:.4f}  (median {stats[k]['median']:.4f})")


if __name__ == "__main__":
    main()
