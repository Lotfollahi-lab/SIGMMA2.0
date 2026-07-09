"""Fine-tune HisToGene (ViT + spatial PE) on SIGMMA's Xenium manifest.

Training mode: single-spot (one FOV → one position in the model). The
multi-spot variant (K² bin patches per FOV) is exercised at eval time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sigmma.baselines.data import (
    FovGexCacheDataset,
    FovGexCacheIndex,
    FovGexDataset,
    FovGexIndex,
    load_gene_panel,
)
from sigmma.baselines.histogene import build_model
from sigmma.baselines.histogene.transforms import (
    make_center_default,
    to_model_input,
)
from sigmma.baselines.trainer import train_baseline


def make_image_transform(patch_size: int):
    def _fn(images, device=None):
        return to_model_input(images, patch_size=patch_size, device=device)
    return _fn


def make_extra_forward_args(n_pos_default: int):
    def _fn(batch_size, device=None):
        return {"centers": make_center_default(batch_size, n_pos_default=n_pos_default,
                                                device=device)}
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--panel_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pretrained_ckpt", default=None)
    ap.add_argument("--fov_size", type=int, default=256)
    ap.add_argument("--target_grid", type=int, default=14)
    ap.add_argument("--tissue_threshold", type=float, default=20000.0)
    ap.add_argument("--patch_size", type=int, default=112)
    ap.add_argument("--n_pos", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_slides_in_cache", type=int, default=4)
    ap.add_argument("--data_cache_dir", default=None,
                    help="If set, use Stage-1 binned-target cache "
                         "(probe_gex_2d phase=cache) instead of re-binning "
                         "parquet transcripts per iteration.")
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    panel = load_gene_panel(args.panel_path)
    print(f"[histogene] gene panel: {len(panel)} genes")

    if args.data_cache_dir:
        train_index = FovGexCacheIndex(args.train_manifest, args.data_cache_dir)
        val_index = FovGexCacheIndex(args.val_manifest, args.data_cache_dir)
        train_ds = FovGexCacheDataset(train_index, gene_panel=panel,
                                      fov_size=args.fov_size)
        val_ds = FovGexCacheDataset(val_index, gene_panel=panel,
                                    fov_size=args.fov_size)
    else:
        train_index = FovGexIndex(args.train_manifest, fov_size=args.fov_size,
                                  tissue_threshold=args.tissue_threshold)
        val_index = FovGexIndex(args.val_manifest, fov_size=args.fov_size,
                                tissue_threshold=args.tissue_threshold)
        train_ds = FovGexDataset(train_index, gene_panel=panel,
                                 fov_size=args.fov_size, target_grid=args.target_grid,
                                 max_slides_in_cache=args.max_slides_in_cache)
        val_ds = FovGexDataset(val_index, gene_panel=panel,
                               fov_size=args.fov_size, target_grid=args.target_grid,
                               max_slides_in_cache=args.max_slides_in_cache)

    model = build_model(n_genes=len(panel), patch_size=args.patch_size,
                        pretrained_ckpt=args.pretrained_ckpt,
                        n_layers=args.n_layers, dim=args.dim, n_pos=args.n_pos)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=Path(args.out_dir).name,
                               config=vars(args))

    train_baseline(
        model=model,
        image_transform=make_image_transform(patch_size=args.patch_size),
        extra_forward_args=make_extra_forward_args(n_pos_default=args.n_pos // 2),
        train_dataset=train_ds,
        val_dataset=val_ds,
        out_dir=args.out_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        log_every=args.log_every,
        val_every=args.val_every,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()
