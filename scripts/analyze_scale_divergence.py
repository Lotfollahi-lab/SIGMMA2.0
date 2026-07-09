"""
Per-cell spectral scale-agreement entropy from SIGMMA's spatial encoder.

For one slide, forward the cell graph through SIGMMA's spatial encoder and
collect the per-cell embeddings produced after each ``GRAPHLayer`` (one per
resolution: micro / meso / macro). Stack as a (3, D) matrix per cell, take
the eigen-spectrum of its (3 × 3) Gram matrix, normalise the eigenvalues to
probabilities, and compute the Shannon entropy:

  H = 0          → all three scales agree (rank-1 Gram, top eigenvalue ≈ 3)
  H = log 3 ≈ 1.099 → scales are mutually orthogonal (Gram = I)

Cells are written out sorted by H **descending** — i.e. **most scale-disagreeing
cells first**.

Usage:
    python scripts/analyze_scale_divergence.py \
      --config   configs/in-house_train_sigmma.yaml \
      --slide    BK39-SKI-27-FO-1-S3 \
      --manifest /lustre/.../sanger_MANIFEST_test.csv \
      --out_dir  /lustre/.../scale_divergence_BK39
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
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


def _spectral_entropy(M: np.ndarray) -> tuple[float, np.ndarray]:
    """M: (3, D) — three per-cell embeddings.

    Row-normalises to unit vectors, computes Gram (3, 3), takes eigenvalues,
    normalises to probabilities, and returns (Shannon entropy, eigvals desc).
    """
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    G = M @ M.T
    eigs = np.linalg.eigvalsh(G)
    eigs = np.maximum(eigs, 0.0)
    s = float(eigs.sum())
    if s <= 0:
        return 0.0, eigs[::-1]
    p = eigs / s
    p = np.clip(p, 1e-12, None)
    H = float(-(p * np.log(p)).sum())
    return H, eigs[::-1]


def _load_xenium_cell_ids(preproc_h5: Path) -> list[str]:
    with h5py.File(preproc_h5, "r") as f:
        return [c.decode() if isinstance(c, bytes) else c
                for c in f["obs/cell_id"][:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="SIGMMA config (must carry the trained checkpoint_path "
                         "+ matching lora block).")
    ap.add_argument("--slide", required=True)
    ap.add_argument("--manifest", required=True,
                    help="Source manifest CSV containing the slide row.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    cfg = ConfigDict(yaml.safe_load(open(args.config)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Single-slide manifest.
    manifest = pd.read_csv(args.manifest)
    row = manifest[manifest["sample_name"] == args.slide]
    if row.empty:
        raise SystemExit(f"Slide '{args.slide}' not in {args.manifest}")
    single_csv = out_dir / f"_single_slide_{args.slide}.csv"
    row.to_csv(single_csv, index=False)
    print(f"[load] single-slide manifest -> {single_csv}")

    # 2) Build dataset/dataloader. Reuse the validation dataset shape from cfg.
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

    # 4) Spatial-encoder forward, capturing per-scale per-cell embeddings.
    print("[forward] extract_graphs(multi_scale=True)")
    graphs_per_scale, _ = extract_graphs(
        loader, model, cfg, device, multi_scale=True,
    )

    embs_per_scale: dict[str, np.ndarray] = {}
    coords_ref = None
    cell_ids_ref = None
    for k in range(cfg.model.num_resolutions):
        s_key = f"scale_{k}"
        cell_ids, embs, coords = extract_cell_embeddings_and_coordinates(
            graphs_per_scale[s_key]
        )
        embs_per_scale[s_key] = embs
        if cell_ids_ref is None:
            cell_ids_ref = cell_ids
            coords_ref = coords
        else:
            assert (cell_ids == cell_ids_ref).all(), \
                f"cell ordering mismatch at {s_key}"
        print(f"  {s_key}: cells={cell_ids.shape[0]}, dim={embs.shape[1]}")

    n = cell_ids_ref.shape[0]
    z = np.stack([embs_per_scale[f"scale_{k}"] for k in range(cfg.model.num_resolutions)],
                 axis=1)  # (N, num_scales, D)
    if z.shape[1] != 3:
        raise SystemExit(f"Expected 3 scales, got {z.shape[1]}")

    # 5) Per-cell spectral entropy.
    print("[entropy] computing per-cell H...")
    H = np.empty(n, dtype=np.float32)
    eigs = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        H[i], eigs[i] = _spectral_entropy(z[i])
    print(f"  H: mean={H.mean():.4f}, min={H.min():.4f}, max={H.max():.4f} "
          f"(theoretical max = log 3 = {np.log(3):.4f})")

    # 6) Map graph cell_id (positional index into spatial_data_processed.h5)
    # back to the Xenium string cell_id.
    preproc_path = Path(row.iloc[0]["spatial_preprocessed"])
    xenium_ids = None
    if preproc_path.exists():
        try:
            xenium_ids_all = _load_xenium_cell_ids(preproc_path)
            cell_idx_int = cell_ids_ref.astype(int)
            in_range = (cell_idx_int >= 0) & (cell_idx_int < len(xenium_ids_all))
            xenium_ids = np.array([
                xenium_ids_all[i] if in_range[k] else ""
                for k, i in enumerate(cell_idx_int)
            ])
            print(f"  resolved {in_range.sum()}/{n} cell_ids to Xenium strings")
        except Exception as e:
            print(f"  [warn] could not resolve Xenium cell_ids: {e}")

    df = pd.DataFrame({
        "cell_idx": cell_ids_ref.astype(int),
        "xenium_cell_id": xenium_ids if xenium_ids is not None else "",
        "x": coords_ref[:, 0],
        "y": coords_ref[:, 1],
        "scale_entropy": H,
        "eig_1": eigs[:, 0],
        "eig_2": eigs[:, 1],
        "eig_3": eigs[:, 2],
    })
    # Most scale-disagreeing cells first.
    df = df.sort_values("scale_entropy", ascending=False).reset_index(drop=True)

    csv_path = out_dir / f"scale_divergence_{args.slide}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[csv] wrote {csv_path}  (n={len(df)}, sorted by scale_entropy desc)")
    pd.set_option("display.float_format", "{:.4f}".format)
    print("Top 5 (highest disagreement):")
    print(df.head(5).to_string(index=False))
    print("\nBottom 5 (highest agreement):")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
