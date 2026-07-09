"""
Probe whether SIGMMA's per-scale graph layers collapse onto a 'scale prototype'.

Hypothesis (H_collapse):
    For each scale k, z_k(i) ≈ μ_k + ε_i where ε_i is small noise.
    Per-cell representation is dominated by a scale-wide direction; cell-specific
    information is suppressed.

Tests:
  T1. cos(z_k(i), μ_k) distribution.
      H_collapse → tight concentration near 1.
  T2. Effective rank (participation ratio) of centered Z_k:
        PR = (Σ λ_i)² / Σ λ_i²       (1 ≤ PR ≤ D)
      H_collapse → ≈ 1.
  T3. σ_within(k) = mean ||z_k(i) - μ_k||
      σ_between(k) = mean_{j≠k} ||μ_j - μ_k||
      Ratio σ_within / σ_between.
      H_collapse → ≪ 1.
  T4. Top-K nearest-neighbour overlap across scales for the SAME cell.
      Among shuffled (i ≠ i) baselines, expected overlap is K/N.
      H_collapse → observed overlap close to that chance level.

Reuses the same dataloader/checkpoint pipeline as analyze_scale_divergence.py.
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


def _participation_ratio(Z: np.ndarray) -> tuple[float, np.ndarray]:
    """PR of centered Z (N, D). Uses singular values (Z = USV).
    PR = (Σ s_i²)² / Σ s_i⁴   on s_i² (eigenvalues of cov)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # compute eigenvalues of (Zc.T @ Zc) — work in min(N,D) x min(N,D) for speed
    if Zc.shape[0] >= Zc.shape[1]:
        cov = (Zc.T @ Zc) / max(Zc.shape[0] - 1, 1)
        evals = np.linalg.eigvalsh(cov)
    else:
        gram = (Zc @ Zc.T) / max(Zc.shape[0] - 1, 1)
        evals = np.linalg.eigvalsh(gram)
    evals = np.maximum(evals, 0.0)
    s = float(evals.sum())
    if s <= 0:
        return 0.0, evals[::-1]
    pr = (s ** 2) / float((evals ** 2).sum() + 1e-30)
    return float(pr), evals[::-1]


def _topk_neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) indices of top-k cosine NN excluding self.
    X must already be L2-normalised."""
    N = X.shape[0]
    sim = X @ X.T            # (N, N)
    np.fill_diagonal(sim, -np.inf)
    # argpartition to get top-k
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    return idx


def _nn_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """For each row, |idx_a ∩ idx_b| as a fraction of k."""
    k = idx_a.shape[1]
    out = np.empty(idx_a.shape[0], dtype=np.float32)
    for i in range(idx_a.shape[0]):
        out[i] = len(set(idx_a[i]).intersection(idx_b[i])) / k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--knn_k", type=int, default=10,
                    help="K for top-K nearest-neighbour overlap test (T4).")
    ap.add_argument("--knn_subsample", type=int, default=4000,
                    help="Random subsample of cells for the K-NN overlap test "
                         "(full N×N is fine for N≤~10k).")
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

    # 2) Dataset / dataloader.
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
    raw: dict[int, np.ndarray] = {}      # unnormalised
    norm: dict[int, np.ndarray] = {}     # L2 normalised
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
        raw[k] = e.astype(np.float32)
        norm[k] = _l2_normalise(raw[k])
        print(f"  scale_{k}: cells={e.shape[0]}, dim={e.shape[1]}, "
              f"||x|| pre-norm mean={np.linalg.norm(e, axis=1).mean():.4f}")
    n_cells, D = raw[0].shape
    K_NN = args.knn_k

    # === T1. cos(z_k(i), μ_k) ===
    print("\n[T1] cos(z_k(i), μ̂_k) — concentration near 1 → prototype collapse")
    t1_rows = []
    for k in range(n_scales):
        # Use direction of the *normalised* mean to avoid magnitude bias.
        mu_dir = norm[k].mean(axis=0)
        mu_dir /= np.linalg.norm(mu_dir) + 1e-12
        cos_to_mean = norm[k] @ mu_dir
        np.save(out_dir / f"T1_cos_to_protomean_scale{k}.npy", cos_to_mean.astype(np.float32))
        t1_rows.append({
            "scale": k,
            "mean_cos_to_protomean": float(cos_to_mean.mean()),
            "std_cos_to_protomean": float(cos_to_mean.std()),
            "p05": float(np.quantile(cos_to_mean, 0.05)),
            "p50": float(np.quantile(cos_to_mean, 0.50)),
            "p95": float(np.quantile(cos_to_mean, 0.95)),
            "min": float(cos_to_mean.min()),
            "max": float(cos_to_mean.max()),
        })
        print(f"  scale_{k}: mean={cos_to_mean.mean():+.4f}  "
              f"std={cos_to_mean.std():.4f}  "
              f"p05={np.quantile(cos_to_mean, 0.05):+.4f}  "
              f"p95={np.quantile(cos_to_mean, 0.95):+.4f}")

    # === T2. Participation ratio of centered Z_k ===
    print("\n[T2] participation ratio of centered Z_k — D-dim isotropic ≈ D, collapse ≈ 1")
    t2_rows = []
    for k in range(n_scales):
        pr, evals = _participation_ratio(raw[k])
        # Normalised variant (drop magnitude information).
        pr_n, _ = _participation_ratio(norm[k])
        np.save(out_dir / f"T2_eigs_scale{k}.npy", evals.astype(np.float32))
        t2_rows.append({"scale": k, "PR_raw": pr, "PR_normalised": pr_n,
                        "D": D, "frac_top1_var": float(evals[0] / (evals.sum() + 1e-30))})
        print(f"  scale_{k}: PR_raw={pr:.2f} / D={D}   "
              f"PR_normalised={pr_n:.2f}   "
              f"top-1 eig fraction={evals[0] / (evals.sum() + 1e-30):.3f}")

    # === T3. σ_within vs σ_between ===
    print("\n[T3] σ_within(k) vs σ_between (centroid distances)")
    centroids = {k: raw[k].mean(axis=0) for k in range(n_scales)}
    sigma_within = {k: float(np.linalg.norm(raw[k] - centroids[k], axis=1).mean())
                    for k in range(n_scales)}
    inter_centroid = []
    for j in range(n_scales):
        for k in range(j + 1, n_scales):
            d = float(np.linalg.norm(centroids[j] - centroids[k]))
            inter_centroid.append({"i": j, "j": k, "centroid_dist": d})
            print(f"  ||μ_{j} - μ_{k}|| = {d:.4f}")
    sigma_between = float(np.mean([r["centroid_dist"] for r in inter_centroid]))
    t3_rows = []
    for k in range(n_scales):
        t3_rows.append({
            "scale": k,
            "sigma_within": sigma_within[k],
            "sigma_between_mean": sigma_between,
            "ratio_within_over_between": sigma_within[k] / (sigma_between + 1e-12),
        })
        print(f"  scale_{k}: σ_within={sigma_within[k]:.4f}   "
              f"σ_within / σ_between = {sigma_within[k] / (sigma_between + 1e-12):.3f}")

    # === T4. K-NN overlap across scales ===
    print(f"\n[T4] top-{K_NN} NN overlap across scales (subsample={args.knn_subsample})")
    if n_cells > args.knn_subsample:
        sub_idx = rng.choice(n_cells, size=args.knn_subsample, replace=False)
    else:
        sub_idx = np.arange(n_cells)
    sub_norm = {k: norm[k][sub_idx] for k in range(n_scales)}
    nn_idx = {k: _topk_neighbours(sub_norm[k], K_NN) for k in range(n_scales)}
    chance = K_NN / sub_idx.size
    t4_rows = []
    for j in range(n_scales):
        for k in range(j + 1, n_scales):
            ovl = _nn_overlap(nn_idx[j], nn_idx[k])
            np.save(out_dir / f"T4_nnoverlap_{j}vs{k}.npy", ovl.astype(np.float32))
            t4_rows.append({
                "i": j, "j": k,
                "mean_overlap": float(ovl.mean()),
                "median_overlap": float(np.median(ovl)),
                "fraction_zero_overlap": float((ovl == 0).mean()),
                "chance_overlap": chance,
                "ratio_observed_over_chance": float(ovl.mean() / chance),
            })
            print(f"  scale_{j} vs scale_{k}: mean overlap={ovl.mean():.4f}   "
                  f"median={np.median(ovl):.4f}   "
                  f"chance={chance:.6f}   "
                  f"observed/chance={ovl.mean() / chance:.2f}x")

    # === Decision summary ===
    print("\n=== Hypothesis verdict ===")
    pr_norm_mean = float(np.mean([r["PR_normalised"] for r in t2_rows]))
    cos_to_mean_mean = float(np.mean([r["mean_cos_to_protomean"] for r in t1_rows]))
    sigma_ratio = float(np.mean([r["ratio_within_over_between"] for r in t3_rows]))
    nn_overlap_obs = float(np.mean([r["mean_overlap"] for r in t4_rows]))
    nn_chance = float(np.mean([r["chance_overlap"] for r in t4_rows]))

    print(f"  T1 mean cos(z, μ̂) across scales : {cos_to_mean_mean:+.4f}    "
          f"(collapse → ≈ 1)")
    print(f"  T2 PR_normalised mean / D         : {pr_norm_mean:.2f} / {D}  "
          f"(collapse → ≈ 1)")
    print(f"  T3 σ_within / σ_between mean      : {sigma_ratio:.3f}        "
          f"(collapse → ≪ 1)")
    print(f"  T4 NN-overlap obs / chance        : "
          f"{nn_overlap_obs:.4f} vs {nn_chance:.6f}  "
          f"({nn_overlap_obs / nn_chance:.2f}× chance)  "
          f"(collapse → ≈ 1× chance)")

    # Heuristic verdict.
    score = 0
    if cos_to_mean_mean > 0.95: score += 1
    if pr_norm_mean < 0.05 * D: score += 1
    if sigma_ratio < 0.5: score += 1
    if nn_overlap_obs / max(nn_chance, 1e-12) < 3.0: score += 1
    if score >= 3:
        verdict = "STRONGLY supports prototype collapse"
    elif score == 2:
        verdict = "PARTIALLY supports prototype collapse"
    else:
        verdict = "DOES NOT support prototype collapse"
    print(f"\n  Verdict: {verdict}  (score {score}/4)")

    # Persist combined CSV.
    out_csv = out_dir / f"scale_collapse_summary_{args.slide}.csv"
    with open(out_csv, "w") as f:
        f.write("# T1 cos to scale prototype mean direction\n")
        pd.DataFrame(t1_rows).to_csv(f, index=False)
        f.write("# T2 participation ratio per scale\n")
        pd.DataFrame(t2_rows).to_csv(f, index=False)
        f.write("# T3 within-scale spread vs between-scale centroid distance\n")
        pd.DataFrame(t3_rows).to_csv(f, index=False)
        f.write("# T4 K-NN overlap across scales\n")
        pd.DataFrame(t4_rows).to_csv(f, index=False)
    print(f"\n[csv] wrote {out_csv}")


if __name__ == "__main__":
    main()
