"""
Decision experiment for "are the per-scale embeddings really orthogonal,
or is the spatial encoder undertrained?"

For one slide, run SIGMMA's spatial encoder and compute three cosine-similarity
distributions over the per-cell embeddings:

  (A) cross-cell, same-scale   — different cells at the same scale.
                                 If the model has learned cell-specific signal,
                                 this should peak well above 0 (cells share
                                 niche structure → directions correlate).
  (B) same-cell, cross-scale   — same cell across two different scales.
                                 This is what scale_entropy already saw via Gram
                                 spectrum; reported here directly per pair.
  (C) cross-cell, cross-scale  — different cells at different scales (null).
                                 Bounds chance correlation given finite D.

Decision:
  - (A) ≈ 0 and (B) ≈ 0  → graph layers are still random projections;
                            checkpoint is undertrained.
  - (A) ≫ 0 and (B) ≈ 0   → scales really are orthogonalised by the architecture.
  - (A) and (B) both ≫ 0  → scales share signal as expected.

Usage:
    python scripts/diagnose_scale_correlation.py \
      --config   configs/analyze_scale_divergence_sigmma.yaml \
      --slide    BK39-SKI-27-FO-1-S3 \
      --manifest /lustre/.../sanger_MANIFEST_test.csv \
      --out_dir  /lustre/.../scale_correlation_<slide>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader

from sigmma.utils import load_dataset, load_model
from sigmma.datasets.dataset import multimodal_collate
from sigmma.misc.extract_embeddings import (
    extract_graphs,
    extract_cell_embeddings_and_coordinates,
)


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-12)


def _summary(name: str, vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"name": name, "n": 0}
    qs = np.quantile(vals, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "name": name,
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p01": float(qs[0]), "p05": float(qs[1]), "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]), "p95": float(qs[5]), "p99": float(qs[6]),
        "min": float(vals.min()), "max": float(vals.max()),
    }


def _offdiag_pairs(M: np.ndarray, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """All-pairs cosine over rows of (already-normalised) M, returning a
    flat 1-D array of off-diagonal cosines, optionally subsampled."""
    n = M.shape[0]
    if n < 2:
        return np.array([], dtype=np.float32)
    # Full Gram is N x N. For N=5000 that's 25M floats — fine. Subsample at
    # output if needed.
    G = M @ M.T
    iu = np.triu_indices(n, k=1)
    vals = G[iu].astype(np.float32)
    if vals.size > max_pairs:
        idx = rng.choice(vals.size, size=max_pairs, replace=False)
        vals = vals[idx]
    return vals


def _cross_pairs(A: np.ndarray, B: np.ndarray, n_pairs: int,
                 rng: np.random.Generator, exclude_diag: bool = True) -> np.ndarray:
    """Sample n_pairs of (i, j) with i != j (if exclude_diag) and return
    cos(A[i], B[j]) where A and B are already L2-normalised."""
    n_a, n_b = A.shape[0], B.shape[0]
    if n_a == 0 or n_b == 0:
        return np.array([], dtype=np.float32)
    out = np.empty(n_pairs, dtype=np.float32)
    filled = 0
    while filled < n_pairs:
        need = n_pairs - filled
        i = rng.integers(0, n_a, size=need)
        j = rng.integers(0, n_b, size=need)
        if exclude_diag:
            keep = i != j
            i, j = i[keep], j[keep]
        if i.size == 0:
            continue
        c = (A[i] * B[j]).sum(axis=1)
        out[filled:filled + i.size] = c
        filled += i.size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cell_subsample", type=int, default=4000,
                    help="Random subsample of cells for cross-cell same-scale Gram.")
    ap.add_argument("--cross_pairs", type=int, default=200_000,
                    help="Number of random (i!=j) pairs sampled for cross-cell "
                         "cross-scale null distribution.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ConfigDict(yaml.safe_load(open(args.config)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # 1) Single-slide manifest.
    manifest = pd.read_csv(args.manifest)
    row = manifest[manifest["sample_name"] == args.slide]
    if row.empty:
        raise SystemExit(f"Slide '{args.slide}' not in {args.manifest}")
    single_csv = out_dir / f"_single_slide_{args.slide}.csv"
    row.to_csv(single_csv, index=False)
    print(f"[load] single-slide manifest -> {single_csv}")

    # 2) Dataset / dataloader (mirror analyze_scale_divergence.py).
    val_cfg = cfg.dataset.get("validation", cfg.dataset.get("test", None))
    if val_cfg is None:
        raise SystemExit("config.dataset must have a 'validation' or 'test' block.")
    ds_cfg = ConfigDict({
        "manifest_path": str(single_csv),
        "fov_size": val_cfg.get("fov_size", 256),
        "pretrained_cell_embedding": val_cfg.get("pretrained_cell_embedding", True),
        "type_pretrained_cell_embedding": val_cfg.get("type_pretrained_cell_embedding", "nemo"),
        "num_resolutions": cfg.model.get("num_resolutions", 3),
    })
    dataset = load_dataset(ds_cfg, is_train=False, model_name="sigmma")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=multimodal_collate,
    )
    print(f"[load] dataset = {len(dataset)} tiles")

    # 3) Load SIGMMA + ckpt.
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.get("pretrained_image_encoder", "UNI"),
        pretrained_ckpt_path=cfg.model.get("checkpoint_path", None),
        train=False,
    )
    model.to(device).eval()

    # 4) Per-scale per-cell embeddings.
    print("[forward] extract_graphs(multi_scale=True)")
    graphs_per_scale, _ = extract_graphs(
        loader, model, cfg, device, multi_scale=True,
    )

    n_scales = cfg.model.num_resolutions
    embs: dict[int, np.ndarray] = {}
    cell_ids_ref = None
    for k in range(n_scales):
        cell_ids, e, _ = extract_cell_embeddings_and_coordinates(
            graphs_per_scale[f"scale_{k}"]
        )
        if cell_ids_ref is None:
            cell_ids_ref = cell_ids
        else:
            assert (cell_ids == cell_ids_ref).all(), \
                f"cell ordering mismatch at scale_{k}"
        embs[k] = _l2_normalise(e.astype(np.float32))
        print(f"  scale_{k}: cells={e.shape[0]}, dim={e.shape[1]}, "
              f"||x|| pre-norm mean={np.linalg.norm(e, axis=1).mean():.4f}")
    n_cells = embs[0].shape[0]

    # 5) (A) cross-cell, same-scale  ─ subsample cells, full pairwise on subsample.
    summaries: list[dict] = []
    if n_cells > args.cell_subsample:
        sub_idx = rng.choice(n_cells, size=args.cell_subsample, replace=False)
    else:
        sub_idx = np.arange(n_cells)
    print(f"[A] cross-cell same-scale (subsample={sub_idx.size})")
    for k in range(n_scales):
        v = _offdiag_pairs(embs[k][sub_idx], max_pairs=500_000, rng=rng)
        s = _summary(f"crosscell_samescale_scale{k}", v)
        summaries.append(s)
        print(f"  scale_{k}: n={s['n']:>8}  mean={s['mean']:+.4f}  "
              f"std={s['std']:.4f}  p05={s['p05']:+.4f}  p95={s['p95']:+.4f}")
        np.save(out_dir / f"crosscell_samescale_scale{k}.npy", v)

    # 6) (B) same-cell, cross-scale  ─ exact, all cells.
    print("[B] same-cell cross-scale")
    for k1 in range(n_scales):
        for k2 in range(k1 + 1, n_scales):
            v = (embs[k1] * embs[k2]).sum(axis=1).astype(np.float32)
            s = _summary(f"samecell_crossscale_{k1}vs{k2}", v)
            summaries.append(s)
            print(f"  {k1} vs {k2}: n={s['n']:>8}  mean={s['mean']:+.4f}  "
                  f"std={s['std']:.4f}  p05={s['p05']:+.4f}  p95={s['p95']:+.4f}")
            np.save(out_dir / f"samecell_crossscale_{k1}vs{k2}.npy", v)

    # 7) (C) cross-cell, cross-scale  ─ random null.
    print(f"[C] cross-cell cross-scale null (n_pairs={args.cross_pairs})")
    for k1 in range(n_scales):
        for k2 in range(k1 + 1, n_scales):
            v = _cross_pairs(embs[k1], embs[k2],
                             n_pairs=args.cross_pairs, rng=rng, exclude_diag=True)
            s = _summary(f"crosscell_crossscale_{k1}vs{k2}", v)
            summaries.append(s)
            print(f"  {k1} vs {k2}: n={s['n']:>8}  mean={s['mean']:+.4f}  "
                  f"std={s['std']:.4f}  p05={s['p05']:+.4f}  p95={s['p95']:+.4f}")
            np.save(out_dir / f"crosscell_crossscale_{k1}vs{k2}.npy", v)

    # 8) Summary CSV + decision hint.
    df = pd.DataFrame(summaries)
    csv_path = out_dir / f"scale_correlation_summary_{args.slide}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[csv] wrote {csv_path}")

    # Quick decision hint.
    a_means = [s["mean"] for s in summaries if s["name"].startswith("crosscell_samescale_")]
    b_means = [s["mean"] for s in summaries if s["name"].startswith("samecell_crossscale_")]
    c_means = [s["mean"] for s in summaries if s["name"].startswith("crosscell_crossscale_")]
    print("\n=== Decision hint ===")
    print(f"(A) cross-cell same-scale  mean across scales : {np.mean(a_means):+.4f}")
    print(f"(B) same-cell cross-scale  mean across pairs  : {np.mean(b_means):+.4f}")
    print(f"(C) cross-cell cross-scale mean (null)        : {np.mean(c_means):+.4f}")
    expected_random = 1.0 / np.sqrt(embs[0].shape[1])
    print(f"     expected |cos| for random unit vecs in {embs[0].shape[1]}-D ≈ ±{expected_random:.4f}")
    if abs(np.mean(a_means)) < 3 * expected_random and abs(np.mean(b_means)) < 3 * expected_random:
        print("→ Both A and B are near random-projection level: spatial encoder is "
              "likely undertrained, not orthogonalised by design.")
    elif abs(np.mean(a_means)) >= 3 * expected_random and abs(np.mean(b_means)) < 3 * expected_random:
        print("→ A ≫ 0 but B ≈ 0: scales really are orthogonalised by the architecture.")
    else:
        print("→ A and B both meaningfully > 0: scales share signal as expected.")


if __name__ == "__main__":
    main()
