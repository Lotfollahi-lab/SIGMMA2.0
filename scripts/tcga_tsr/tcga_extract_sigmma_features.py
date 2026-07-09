#!/usr/bin/env python
"""Stage 3 (SIGMMA env): encode TCGA nucleus crops -> z_sigmma_* per slide.

Thin driver around the unmodified ``process_slide_features`` so the TCGA
features live in the SAME space as the RCC probe's training features. Reads
a manifest (sample_name, hematoxylin_eosin_image, data_h5) and writes one
feature HDF5 per slide (z_sigmma_{micro,meso,macro} + _dense_noproj + _cls).

  python tcga_extract_sigmma_features.py \
      --config configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml \
      --manifest <tcga_manifest.csv> --out_dir <sigmma_feat_dir>

Run in the sigmma poetry venv (load_model + UNI2-h + ckpt 36852 step2000).
"""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict

from sigmma.utils import load_model
from sigmma.application.cell_niche_transfer.extract import process_slide_features


def _subsample_data_h5(src: Path, dst: Path, max_cells: int, seed: int = 42) -> Path:
    """Write a seeded random subsample of a data_h5 (same schema) if it exceeds
    max_cells. TSR is a ratio, so a few thousand cells estimate it tightly; this
    caps the per-slide ViT-H forward cost. Returns the path to use."""
    with h5py.File(src, "r") as f:
        n = f["cell_id"].shape[0]
        if not max_cells or n <= max_cells:
            return src
        idx = np.sort(np.random.default_rng(seed).choice(n, max_cells, replace=False))
        with h5py.File(dst, "w") as g:
            for k, v in f.attrs.items():
                g.attrs[k] = v
            for name in f:
                d = f[name]
                g.create_dataset(name, data=d[:][idx] if d.shape[0] == n else d[:])
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True,
                    help="CSV: sample_name, hematoxylin_eosin_image, data_h5")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--slide", default=None)
    ap.add_argument("--max_cells", type=int, default=0,
                    help="subsample to this many cells/slide (0 = all).")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="override cfg.extract.batch_size (0 = use config).")
    args = ap.parse_args()

    cfg = ConfigDict(yaml.safe_load(open(args.config)))
    man = pd.read_csv(args.manifest)
    if args.slide:
        man = man[man["sample_name"] == args.slide]
        if man.empty:
            raise SystemExit(f"slide {args.slide} not in manifest")

    encoder_input = int(cfg.extract.encoder_input)
    patch_size = int(cfg.extract.patch_size)
    batch_size = args.batch_size or int(cfg.extract.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=cfg.model.checkpoint_path,
        train=False,
    ).to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for _, row in man.iterrows():
        sample = row["sample_name"]
        data_h5 = Path(row["data_h5"])
        if not data_h5.exists():
            print(f"[skip] {sample}: data_h5 missing {data_h5}")
            continue
        out_path = out_dir / f"{sample}.h5"
        if out_path.exists():
            print(f"[skip] {out_path} exists")
            continue
        use_h5 = data_h5
        tmp = None
        if args.max_cells:
            tmp = Path(tempfile.gettempdir()) / f"{sample}_sub{args.max_cells}.h5"
            use_h5 = _subsample_data_h5(data_h5, tmp, args.max_cells)
            if use_h5 != data_h5:
                print(f"[subsample] {sample} -> {args.max_cells} cells")
        n = process_slide_features(
            slide_row=row,                       # needs hematoxylin_eosin_image
            data_h5_path=use_h5,
            backbone_model=model,
            kind="sigmma",
            sigmma_model=model,
            encoder_input=encoder_input,
            patch_size=patch_size,
            batch_size=batch_size,
            out_path=out_path,
        )
        total += n
        if tmp is not None and tmp.exists():
            tmp.unlink()
        print(f"[done] {sample}: {n} cells -> {out_path}", flush=True)
    print(f"\n[sigmma] {total:,} cells across {len(man)} slides -> {out_dir}")


if __name__ == "__main__":
    main()
