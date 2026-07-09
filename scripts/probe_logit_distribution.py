"""
Snapshot the InfoNCE logit / softmax distribution under the trained SIGMMA
checkpoint, for a single batch of validation cells.

Why: we hypothesised cone-shaped representations were caused by InfoNCE τ=10
(too-flat logits → no uniformity pressure). Code audit shows the SIGMMA
training loop actually uses τ=0.07 (default of `symmetric_InfoNCE()` instantiated
inline in train_utils.py), so the config's `contrastive_temperature: 10.0` is
dead. This script confirms that empirically by showing for one batch:

  - logit std at τ=0.07 (actual training) — should be 'sharp', std ≫ 1.
  - logit std at τ=10  (dead config)      — should be flat, std < 0.1.
  - softmax mass on the diagonal (positive) at each τ — sharp τ → mass
    concentrated near 1; flat τ → mass uniform ≈ 1/N.

If τ=0.07 logits are sharp AND positives still aren't dominating the
softmax (i.e., the model can't tell the right pair), then the model is
genuinely undertrained, not victim of a broken temperature.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader

from sigmma.utils import load_dataset, load_model
from sigmma.datasets.dataset import multimodal_collate


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def _summary(name: str, t: torch.Tensor) -> dict:
    a = t.detach().cpu().float().flatten().numpy()
    return {
        "name": name,
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p01": float(np.quantile(a, 0.01)),
        "p50": float(np.quantile(a, 0.50)),
        "p99": float(np.quantile(a, 0.99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Match training batch_size for a fair logit-matrix snapshot.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=4,
                    help="Average across this many usable batches.")
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

    # 2) Dataset / dataloader.
    val_cfg = cfg.dataset.get("validation", cfg.dataset.get("test", None))
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
    print(f"[load] dataset = {len(dataset)} tiles, batch_size = {args.batch_size}")

    # 3) Load SIGMMA + ckpt.
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.get("pretrained_image_encoder", "UNI"),
        pretrained_ckpt_path=cfg.model.get("checkpoint_path", None),
        train=False,
    )
    model.to(device).eval()
    n_scales = cfg.model.num_resolutions
    taus = [0.07, 1.0, 10.0]

    # 4) Forward a few full-size batches and collect logits.
    print("[forward] grabbing batches...")
    rows: list[dict] = []
    softmax_mass_diag: dict[tuple, list[float]] = {}
    used = 0
    for batch in loader:
        if not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0:
            continue
        if batch["image"].shape[0] < 8:
            continue  # too small for a meaningful logit matrix
        used += 1
        image = batch["image"].to(device)
        spatial = batch["spatial_graph"].to(device)
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=True):
                out = model(image, spatial)
            embedding_imgs = out["image_embedding"]      # list[(B, D)] per scale
            embedding_spatial = out["spatial_embedding"] # list[(B, D)] per scale

        B = embedding_imgs[0].shape[0]
        for k in range(n_scales):
            zi = _normalise(embedding_imgs[k]).float()
            zj = _normalise(embedding_spatial[k]).float()
            cos = zi @ zj.t()      # (B, B)
            cos_diag = cos.diag()
            cos_off = cos.flatten()[~torch.eye(B, dtype=bool, device=cos.device).flatten()]
            rows.append({
                "scale": k, "tau": None, "metric": "cos_diag", "batch": used,
                **{f"_{kk}": vv for kk, vv in _summary(f"cos_diag_s{k}", cos_diag).items()},
            })
            rows.append({
                "scale": k, "tau": None, "metric": "cos_off", "batch": used,
                **{f"_{kk}": vv for kk, vv in _summary(f"cos_off_s{k}", cos_off).items()},
            })

            for tau in taus:
                logits = cos / tau
                logits_summary = _summary(f"logits_s{k}_tau{tau}", logits)
                rows.append({"scale": k, "tau": tau, "metric": "logits", "batch": used,
                             **{f"_{kk}": vv for kk, vv in logits_summary.items()}})

                # softmax row-wise (image→spatial direction).
                p = torch.softmax(logits, dim=-1)
                p_diag = p.diag()                       # (B,) prob given to true positive
                p_off  = p.flatten()[~torch.eye(B, dtype=bool, device=p.device).flatten()]
                rows.append({"scale": k, "tau": tau, "metric": "softmax_p_pos", "batch": used,
                             **{f"_{kk}": vv for kk, vv in _summary(f"p_pos_s{k}_tau{tau}", p_diag).items()}})
                rows.append({"scale": k, "tau": tau, "metric": "softmax_p_neg", "batch": used,
                             **{f"_{kk}": vv for kk, vv in _summary(f"p_neg_s{k}_tau{tau}", p_off).items()}})
                key = (k, tau)
                softmax_mass_diag.setdefault(key, []).append(float(p_diag.mean()))
        if used >= args.n_batches:
            break

    print(f"[forward] used {used} batches.\n")

    # 5) Pretty per-scale, per-τ summary across batches.
    df = pd.DataFrame(rows)
    out_csv = out_dir / f"logit_distribution_{args.slide}.csv"
    df.to_csv(out_csv, index=False)

    print("=== Per-batch averaged logit/softmax stats ===")
    print(f"{'scale':>6} {'tau':>8} {'logits_std':>12} {'logits_p99':>12} "
          f"{'p_pos_mean':>12} {'p_neg_mean':>12} {'p_pos_max':>12}")
    for k in range(n_scales):
        for tau in taus:
            lr = df[(df.scale == k) & (df.tau == tau) & (df.metric == "logits")]
            pp = df[(df.scale == k) & (df.tau == tau) & (df.metric == "softmax_p_pos")]
            pn = df[(df.scale == k) & (df.tau == tau) & (df.metric == "softmax_p_neg")]
            print(f"{k:>6} {tau:>8.3f} "
                  f"{lr['_std'].mean():>12.4f} {lr['_p99'].mean():>12.4f} "
                  f"{pp['_mean'].mean():>12.4f} {pn['_mean'].mean():>12.4e} "
                  f"{pp['_max'].mean():>12.4f}")

    # 6) Decision summary.
    B = args.batch_size
    print("\n=== Decision ===")
    print(f"Random-guess softmax probability per cell = 1/B = {1.0 / B:.4f}")
    print(f"(τ=10 'flat' regime) p_pos_mean ≈ 1/B  → softmax is uniform → no uniformity pressure")
    print(f"(τ=0.07 'sharp')     p_pos_mean ≈ 1     → model fully discriminates positives")
    print()
    for k in range(n_scales):
        for tau in taus:
            mass = float(np.mean(softmax_mass_diag.get((k, tau), [np.nan])))
            ratio = mass * B
            label = ""
            if tau == 0.07:
                if ratio > 0.5 * B: label = "  ← sharp, uniformity active"
                else: label = "  ← sharp τ but model can't separate (undertrained)"
            elif tau == 10.0:
                if ratio < 2: label = "  ← FLAT — softmax uniform → no negative pressure"
                else: label = "  ← τ=10 still sharp (would be surprising)"
            print(f"  scale_{k} τ={tau:>5}:  E[p(positive)] = {mass:.4f}   "
                  f"({ratio:.1f}× chance){label}")
    print(f"\n[csv] wrote {out_csv}")


if __name__ == "__main__":
    main()
