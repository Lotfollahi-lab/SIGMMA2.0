"""Path A + Path B 2D GEX evaluation for trained H&E→ST baselines.

Reads val/test manifests, runs the chosen baseline at each FOV, and writes
JSON summaries with the same schema as SIGMMA's ``probe_gex_2d.py`` phase=eval
so downstream plotting tools work unchanged.

Path B (``--mode bin``):  (G, K, K) per FOV via K² bin-centered crops.
Path A (``--mode fov``):  (G,)     per FOV via a single forward.
``both`` runs both and writes two JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import zarr

from sigmma.baselines.inference import build_adapter
from sigmma.evaluation.gex_cache_io import iter_section_fovs, section_h5_exists
from sigmma.evaluation.gex_2d_probing import (
    per_bin_mse,
    per_bin_r2,
    per_fov_morans_i,
    per_gene_pearson,
    summarize,
)
from sigmma.evaluation.transcript_binning import (
    bin_fov,
    iter_fov_bboxes,
    load_slide_transcripts,
    manifest_path_to_transcripts_parquet,
)


def _summarize_fov_level(pred: np.ndarray, target: np.ndarray,
                          gene_panel: list[str]) -> dict:
    """pred/target: (N, G). Per-gene Pearson over FOVs + per-bin (here per-FOV) MSE."""
    G = pred.shape[1]
    pearson = np.full(G, np.nan, dtype=np.float64)
    for g in range(G):
        p = pred[:, g]; t = target[:, g]
        sp = p.std(); st = t.std()
        if sp < 1e-8 or st < 1e-8:
            continue
        pearson[g] = float(((p - p.mean()) * (t - t.mean())).mean() / (sp * st))
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else float("nan")
    mse_per_fov = ((pred - target) ** 2).mean(axis=1).astype(np.float64)
    return {
        "per_gene_pearson": {g: (None if np.isnan(p) else float(p))
                             for g, p in zip(gene_panel, pearson)},
        "median_per_gene_pearson": float(np.nanmedian(pearson)),
        "iqr_per_gene_pearson": [float(np.nanpercentile(pearson, 25)),
                                 float(np.nanpercentile(pearson, 75))],
        "per_bin_r2_pooled": r2,
        "mse_per_fov": mse_per_fov.tolist(),
        "mse_per_fov_summary": {
            "mean":   float(mse_per_fov.mean()),
            "median": float(np.median(mse_per_fov)),
            "p25":    float(np.percentile(mse_per_fov, 25)),
            "p75":    float(np.percentile(mse_per_fov, 75)),
            "p90":    float(np.percentile(mse_per_fov, 90)),
            "p95":    float(np.percentile(mse_per_fov, 95)),
            "n_fovs": int(mse_per_fov.size),
        },
    }


def _build_baseline_model(baseline: str, n_genes: int, ckpt_path: str,
                          patch_size: int, n_pos: int):
    if baseline == "stnet":
        from sigmma.baselines.stnet import build_model
        return build_model(n_genes=n_genes)
    if baseline == "histogene":
        from sigmma.baselines.histogene import build_model
        return build_model(n_genes=n_genes, patch_size=patch_size, n_pos=n_pos)
    if baseline == "deeppt":
        from sigmma.baselines.deeppt import build_model
        return build_model(n_genes=n_genes)
    raise ValueError(f"unknown baseline={baseline!r}")


def iter_eval_fovs(manifest_path: str, fov_size: int, tissue_threshold: float,
                   smoke_slide: Optional[str] = None):
    """Yield (sample, he_zarr_array, parquet_path, fov_id, x_start, y_start) per valid FOV."""
    df = pd.read_csv(manifest_path)
    for _, row in df.iterrows():
        sample = row["sample_name"]
        if smoke_slide is not None and sample != smoke_slide:
            continue
        he_zarr = row["hematoxylin_eosin_image"]
        idx_zarr = row["index"]
        parquet = manifest_path_to_transcripts_parquet(he_zarr)
        if parquet is None:
            continue
        try:
            arr = zarr.open(he_zarr, mode="r")
            H, W = arr.shape[1], arr.shape[2]
        except Exception:
            continue
        for fov_id, x_start, y_start, _ in iter_fov_bboxes(
            idx_zarr, tissue_filtering_threshold=tissue_threshold
        ):
            if (x_start + fov_size > W) or (y_start + fov_size > H):
                continue
            yield sample, arr, parquet, int(fov_id), int(x_start), int(y_start)


def iter_eval_fovs_from_cache(manifest_path: str, data_cache_dir: str,
                               smoke_slide: Optional[str] = None,
                               max_fovs_per_slide: int = 0,
                               fov_sample_seed: int = 0):
    """Cache-backed variant. Reads FOV bbox + binned target from Stage-1 cache.

    Yields (sample, he_zarr_array, target_bins (G, K_cache, K_cache),
    fov_id, x_start, y_start, fov_size_cached) per FOV.

    ``max_fovs_per_slide > 0`` keeps a seeded random subset of that many FOVs
    per slide (needed for the expensive Phoenix generative baseline — see the
    PhoenixAdapter docstring). The subset is drawn with a per-slide RNG seeded
    by ``(fov_sample_seed, sha1(sample))`` so it is reproducible and stable
    across reruns, and yielded in sorted order. Pooled per-gene Pearson and
    per-slide SSIM are unbiased estimators on a representative FOV subset.
    """
    df = pd.read_csv(manifest_path)
    for _, row in df.iterrows():
        sample = row["sample_name"]
        if smoke_slide is not None and sample != smoke_slide:
            continue
        if not section_h5_exists(data_cache_dir, sample):
            continue
        he_zarr = row["hematoxylin_eosin_image"]
        try:
            arr = zarr.open(he_zarr, mode="r")
        except Exception:
            continue
        fovs = list(iter_section_fovs(data_cache_dir, sample, fields=["target"]))
        if max_fovs_per_slide and len(fovs) > max_fovs_per_slide:
            slide_seed = int.from_bytes(hashlib.sha1(sample.encode()).digest()[:4], "little")
            rng = np.random.default_rng([int(fov_sample_seed), slide_seed])
            keep = np.sort(rng.choice(len(fovs), size=max_fovs_per_slide, replace=False))
            fovs = [fovs[i] for i in keep]
            print(f"  [subsample] {sample}: {max_fovs_per_slide} FOVs "
                  f"(seed={fov_sample_seed})", flush=True)
        for d in fovs:
            yield (sample, arr, d["target"], d["fov_id"],
                   d["x_start"], d["y_start"], d["fov_size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["stnet", "histogene", "deeppt", "phoenix"], required=True)
    ap.add_argument("--ckpt", default=None,
                    help="best.pt produced by training (regressor baselines). For "
                         "phoenix, optional override of assets/phoenix/flow_model.pth.")
    ap.add_argument("--panel_path", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--out_root", default="outputs/gex_2d/results/baselines")
    ap.add_argument("--mode", choices=["bin", "fov", "both"], default="both")
    ap.add_argument("--fov_size", type=int, default=256)
    ap.add_argument("--target_grid", type=int, default=14)
    ap.add_argument("--tissue_threshold", type=float, default=20000.0)
    ap.add_argument("--qv_min", type=float, default=20.0)
    ap.add_argument("--um_per_px", type=float, default=0.5)
    # HisToGene-only knobs
    ap.add_argument("--patch_size", type=int, default=112)
    ap.add_argument("--n_pos", type=int, default=64)
    # Phoenix-only knobs (zero-shot flow-matching sampling)
    ap.add_argument("--crop_px", type=int, default=112,
                    help="phoenix: per-bin crop size before resize to 224")
    ap.add_argument("--n_samples", type=int, default=8,
                    help="phoenix: ODE draws averaged per crop (conditional mean)")
    ap.add_argument("--atol", type=float, default=1e-1, help="phoenix: ODE solver atol")
    ap.add_argument("--rtol", type=float, default=1e-1, help="phoenix: ODE solver rtol")
    ap.add_argument("--phoenix_batch", type=int, default=64,
                    help="phoenix: crops per vision/flow forward chunk")
    ap.add_argument("--save_raw", action="store_true",
                    help="Save raw per-FOV bin arrays to {out_root}/_raw_bin/{shard}.npz "
                         "and skip per-shard summarize. Lets a section-parallel array be "
                         "merged into one globally-pooled JSON by "
                         "merge_phoenix_raw_gex_2d.py (so per_gene_pearson pools over all "
                         "sections, matching the stored baselines).")
    ap.add_argument("--smoke_slide", default=None)
    ap.add_argument("--max_fovs_per_slide", type=int, default=0,
                    help="If >0, evaluate only this many seeded-random FOVs per "
                         "slide (cache path only). Needed for the Phoenix "
                         "generative baseline, which is ~70 s/FOV (196 bins x "
                         "n_samples ODE solves); see PhoenixAdapter docstring.")
    ap.add_argument("--fov_sample_seed", type=int, default=0,
                    help="Seed for --max_fovs_per_slide subsampling.")
    ap.add_argument("--data_cache_dir", default=None,
                    help="If set, read binned targets from Stage-1 cache "
                         "(probe_gex_2d phase=cache) instead of re-binning "
                         "transcripts parquet. Hugely faster on dense panels.")
    ap.add_argument("--shared_panel_path", default=None,
                    help="STRICT zero-shot eval. Path to a gene_panel.json whose "
                         "'panel' is the target gene set for an EXTERNAL section "
                         "(e.g. svgB_panel_<hash>/gene_panel.json). The model still "
                         "predicts the full --panel_path genes, but pred+target are "
                         "sliced to (panel ∩ shared) in panel order before summarize, "
                         "matching scripts.eval_gex_2d_public_zeroshot. Use the "
                         "non-cache (re-bin) path so non-measured genes bin to 0.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    panel = json.load(open(args.panel_path))["panel"]
    # STRICT zero-shot: indices into the full panel that survive into the eval.
    eval_idx = None
    eval_panel = panel
    if args.shared_panel_path:
        shared_set = set(json.load(open(args.shared_panel_path))["panel"])
        eval_idx = [i for i, g in enumerate(panel) if g in shared_set]
        eval_panel = [panel[i] for i in eval_idx]
        if not eval_idx:
            raise SystemExit(f"no shared genes between --panel_path and "
                             f"--shared_panel_path ({args.shared_panel_path})")
        print(f"[eval] STRICT zero-shot: {len(eval_panel)} shared genes "
              f"of {len(panel)}: {eval_panel}", flush=True)
    print(f"[eval] baseline={args.baseline} mode={args.mode} |panel|={len(panel)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.baseline == "phoenix":
        # Zero-shot: pretrained weights loaded inside build_model (no ckpt state load).
        from sigmma.baselines.phoenix import build_model as build_phoenix
        phoenix = build_phoenix(weights_path=args.ckpt, device=device)
        n_cov = int((np.isin(panel, phoenix.gene_list)).sum())
        print(f"[eval] phoenix: {len(phoenix.gene_list)} genes, "
              f"{n_cov}/{len(panel)} overlap with panel; crop_px={args.crop_px} "
              f"n_samples={args.n_samples}", flush=True)
        adapter = build_adapter("phoenix", phoenix, device, panel=panel,
                                crop_px=args.crop_px, n_samples=args.n_samples,
                                atol=args.atol, rtol=args.rtol,
                                batch_size=args.phoenix_batch)
    else:
        if args.ckpt is None:
            raise SystemExit(f"--ckpt required for baseline={args.baseline}")
        model = _build_baseline_model(args.baseline, n_genes=len(panel),
                                      ckpt_path=args.ckpt,
                                      patch_size=args.patch_size, n_pos=args.n_pos)
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state)
        adapter = build_adapter(args.baseline, model, device, n_pos=args.n_pos)

    do_bin = args.mode in {"bin", "both"}
    do_fov = args.mode in {"fov", "both"}

    bin_preds, bin_targets, bin_masks = [], [], []
    bin_samples, bin_fov_ids, bin_x_starts, bin_y_starts = [], [], [], []
    fov_preds, fov_targets = [], []

    cur_sample = None
    cur_loaded = None
    n_fovs = 0
    t0 = time.time()

    if args.data_cache_dir:
        for sample, arr, target_bins_cache, fov_id, x_start, y_start, fov_size_cached in \
                iter_eval_fovs_from_cache(args.val_manifest, args.data_cache_dir,
                                          smoke_slide=args.smoke_slide,
                                          max_fovs_per_slide=args.max_fovs_per_slide,
                                          fov_sample_seed=args.fov_sample_seed):
            if sample != cur_sample:
                print(f"  [slide] {sample}", flush=True)
                cur_sample = sample
            # Sum-pool the cache grid (G, K_cache, K_cache) → (G, K, K)
            G, Kc, Kc2 = target_bins_cache.shape
            assert Kc == Kc2, f"cache target shape mismatch: {target_bins_cache.shape}"
            K = int(args.target_grid)
            if Kc != K:
                if Kc % K != 0:
                    raise ValueError(f"cache K={Kc} not divisible by target_grid={K}")
                f = Kc // K
                target_grid = target_bins_cache.reshape(G, K, f, K, f).sum(axis=(2, 4))
            else:
                target_grid = target_bins_cache
            target_log1p = np.log1p(target_grid).astype(np.float32)
            mask = np.ones((K, K), dtype=np.float32)
            tile = np.asarray(arr[:, y_start:y_start + fov_size_cached,
                                  x_start:x_start + fov_size_cached])
            tile_t = torch.from_numpy(tile)

            if do_bin:
                pred = adapter.predict_bin_grid(tile_t, K).numpy()
                bin_preds.append(pred.astype(np.float32))
                bin_targets.append(target_log1p)
                bin_masks.append(mask)
                bin_samples.append(str(sample))
                bin_fov_ids.append(int(fov_id))
                bin_x_starts.append(int(x_start))
                bin_y_starts.append(int(y_start))
            if do_fov:
                pred_fov = adapter.predict_fov(tile_t).numpy().astype(np.float32)
                target_fov = np.log1p(target_grid.sum(axis=(1, 2))).astype(np.float32)
                fov_preds.append(pred_fov)
                fov_targets.append(target_fov)

            n_fovs += 1
            if n_fovs % 50 == 0:
                print(f"  [progress] {n_fovs} FOVs in {time.time() - t0:.0f}s", flush=True)
    else:
        for sample, arr, parquet, fov_id, x_start, y_start in iter_eval_fovs(
            args.val_manifest, args.fov_size, args.tissue_threshold,
            smoke_slide=args.smoke_slide,
        ):
            if sample != cur_sample:
                print(f"  [slide] {sample}", flush=True)
                cur_loaded = load_slide_transcripts(
                    parquet, panel, qv_min=args.qv_min, um_per_px=args.um_per_px,
                )
                cur_sample = sample
            tile = np.asarray(arr[:, y_start:y_start + args.fov_size,
                                  x_start:x_start + args.fov_size])
            target_grid = bin_fov(cur_loaded, x_start, y_start, args.fov_size, args.target_grid)
            target_log1p = np.log1p(target_grid).astype(np.float32)
            mask = np.ones((args.target_grid, args.target_grid), dtype=np.float32)
            tile_t = torch.from_numpy(tile)

            if do_bin:
                pred = adapter.predict_bin_grid(tile_t, args.target_grid).numpy()
                bin_preds.append(pred.astype(np.float32))
                bin_targets.append(target_log1p)
                bin_masks.append(mask)
                bin_samples.append(str(sample))
                bin_fov_ids.append(int(fov_id))
                bin_x_starts.append(int(x_start))
                bin_y_starts.append(int(y_start))
            if do_fov:
                pred_fov = adapter.predict_fov(tile_t).numpy().astype(np.float32)
                target_fov = np.log1p(target_grid.sum(axis=(1, 2))).astype(np.float32)
                fov_preds.append(pred_fov)
                fov_targets.append(target_fov)

            n_fovs += 1
            if n_fovs % 50 == 0:
                print(f"  [progress] {n_fovs} FOVs in {time.time() - t0:.0f}s", flush=True)

    print(f"[eval] processed {n_fovs} FOVs in {time.time() - t0:.0f}s", flush=True)

    suffix = f"_smoke_{args.smoke_slide}" if args.smoke_slide else ""

    if args.save_raw:
        shard = args.smoke_slide if args.smoke_slide else "all"
        if do_bin and bin_preds:
            raw_dir = out_root / "_raw_bin"
            raw_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_dir / f"{shard}.npz",
                preds=np.stack(bin_preds, axis=0),
                targets=np.stack(bin_targets, axis=0),
                masks=np.stack(bin_masks, axis=0),
                samples=np.asarray(bin_samples, dtype=object),
                fov_ids=np.asarray(bin_fov_ids, dtype=np.int64),
                x_starts=np.asarray(bin_x_starts, dtype=np.int64),
                y_starts=np.asarray(bin_y_starts, dtype=np.int64),
            )
            print(f"[save_raw] wrote {raw_dir / f'{shard}.npz'} "
                  f"({len(bin_preds)} FOVs)", flush=True)
        if do_fov and fov_preds:
            raw_dir = out_root / "_raw_fov"
            raw_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_dir / f"{shard}.npz",
                preds=np.stack(fov_preds, axis=0),
                targets=np.stack(fov_targets, axis=0),
            )
        return

    if do_bin:
        P = np.stack(bin_preds, axis=0)
        T = np.stack(bin_targets, axis=0)
        M = np.stack(bin_masks, axis=0)
        if eval_idx is not None:
            P = P[:, eval_idx]
            T = T[:, eval_idx]
        s = summarize(
            P, T, M, gene_panel=eval_panel,
            sample_ids=np.asarray(bin_samples, dtype=object),
            x_starts=np.asarray(bin_x_starts, dtype=np.int64),
            y_starts=np.asarray(bin_y_starts, dtype=np.int64),
            fov_size=int(args.fov_size),
            target_grid=int(args.target_grid),
        )
        s["mode"] = "bin"
        s["baseline"] = args.baseline
        s["target_grid"] = args.target_grid
        s["num_val_fovs"] = int(P.shape[0])
        s["smoke_slide"] = args.smoke_slide
        with open(out_root / f"{args.baseline}_bin{suffix}.json", "w") as f:
            json.dump(s, f, indent=2)
        print(f"[bin] median per-gene Pearson = {s['median_per_gene_pearson']:.4f}", flush=True)

    if do_fov:
        P = np.stack(fov_preds, axis=0)
        T = np.stack(fov_targets, axis=0)
        if eval_idx is not None:
            P = P[:, eval_idx]
            T = T[:, eval_idx]
        s = _summarize_fov_level(P, T, gene_panel=eval_panel)
        s["mode"] = "fov"
        s["baseline"] = args.baseline
        s["num_val_fovs"] = int(P.shape[0])
        s["smoke_slide"] = args.smoke_slide
        with open(out_root / f"{args.baseline}_fov{suffix}.json", "w") as f:
            json.dump(s, f, indent=2)
        print(f"[fov] median per-gene Pearson = {s['median_per_gene_pearson']:.4f}", flush=True)


if __name__ == "__main__":
    main()
