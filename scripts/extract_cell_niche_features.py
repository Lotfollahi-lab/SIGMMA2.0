"""Extract per-cell H&E embeddings for the niche label-transfer task.

Three independent stages, dispatched via ``--stage``:

  * ``data``    — ckpt + backbone independent. Writes per-slide HDF5s with
                  cell_id, niche labels, per-scale bbox + polygon mask.
  * ``uni``     — raw UNI backbone. Reads data HDF5, encodes pixel crops
                  from zarr on demand, writes z_uni_* features.
  * ``sigmma``  — SIGMMA wrapper around the LoRA-adapted UNI. Same I/O
                  pattern as ``uni``; writes z_sigmma_* (+_dense_noproj,
                  +_cls) features.

Both feature stages REQUIRE the data stage to have already produced one
HDF5 per evaluable slide. Output dirs are passed independently:
  --data_dir   (always required)
  --uni_dir    (--stage uni)
  --sigmma_dir (--stage sigmma)

A ``cache_meta.json`` is written next to each dir after the sweep
completes (see ``sigmma.caches.write_cache_meta``) so the probe can hard-
check that the three sources agree on backbone + preprocessing.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict

from sigmma.utils import load_model
from sigmma.model.models_image_backbones import UNI
from sigmma.application.cell_niche_transfer.extract import (
    label_config_from_cfg,
    process_slide_data,
    process_slide_features,
)
from sigmma.caches import write_cache_meta


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.upper()
    s = re.sub(r"-FO-0+(\d)", r"-FO-\1", s)
    s = re.sub(r"-S\d+(-[A-Z]\d+)?$", "", s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["data", "uni", "sigmma"], required=True,
                        help="Which extraction stage to run.")
    parser.add_argument("--config", required=True,
                        help="YAML with model + extract blocks. For --stage=data only "
                             "the extract block is read; for uni/sigmma the model block "
                             "drives encoder + (for sigmma) checkpoint loading.")
    parser.add_argument("--data_dir", required=True,
                        help="Shared per-slide data HDF5 dir (kind=shared_data). "
                             "Written by --stage=data; read by uni/sigmma.")
    parser.add_argument("--uni_dir", default=None,
                        help="Required for --stage=uni. Per-slide uni feature HDF5 dir.")
    parser.add_argument("--sigmma_dir", default=None,
                        help="Required for --stage=sigmma. Per-slide sigmma feature HDF5 dir.")
    parser.add_argument("--slide", default=None,
                        help="Optional sample_name to process a single slide.")
    args = parser.parse_args()

    cfg = ConfigDict(yaml.safe_load(open(args.config)))
    manifest = pd.read_csv(cfg.data.evaluable_slides_csv)
    if "norm" not in manifest.columns:
        manifest["norm"] = manifest["sample_name"].map(_normalize)
    if args.slide is not None:
        manifest = manifest[manifest["sample_name"] == args.slide]
        if manifest.empty:
            raise SystemExit(f"slide '{args.slide}' not in evaluable manifest")

    scale_sizes = tuple(cfg.extract.scale_sizes)
    encoder_input = int(cfg.extract.encoder_input)
    patch_size = int(cfg.extract.patch_size)
    batch_size = int(cfg.extract.batch_size)

    if args.stage == "data":
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        label_cfg = label_config_from_cfg(cfg.data)
        max_cells = (int(cfg.extract["max_cells_per_slide"])
                     if "max_cells_per_slide" in cfg.extract else None)
        sub_seed = (int(cfg.extract["subsample_seed"])
                    if "subsample_seed" in cfg.extract else 42)
        total = 0
        for _, row in manifest.iterrows():
            sample = row["sample_name"]
            out_path = data_dir / f"{sample}.h5"
            if out_path.exists():
                print(f"[skip] {out_path} exists")
                continue
            n = process_slide_data(
                slide_row=row,
                labels_h5ad=cfg.data.labels_h5ad,
                scale_sizes=scale_sizes,
                out_path=out_path,
                max_cells_per_slide=max_cells,
                subsample_seed=sub_seed,
                **label_cfg,
            )
            total += n
        if args.slide is None:
            write_cache_meta(
                data_dir, kind="shared_data", task="cell_niche",
                preprocessing={"scale_sizes": list(scale_sizes)},
            )
        print(f"\n[data] {total:,} cells written under {data_dir}")
        return

    # Feature stages share I/O pattern but differ in model + projection logic.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "uni":
        if args.uni_dir is None:
            raise SystemExit("--uni_dir required for --stage=uni")
        feat_dir = Path(args.uni_dir)
        feat_dir.mkdir(parents=True, exist_ok=True)
        raw_uni = UNI(checkpoint_path=cfg.model.pretrained_image_encoder_ckpt_path)
        raw_uni = raw_uni.to(device).eval()
        backbone, sigmma_model = raw_uni, None
        kind = "uni"
        backbone_input_dim = int(cfg.model.image_encoder_input_dim)
    else:  # sigmma
        if args.sigmma_dir is None:
            raise SystemExit("--sigmma_dir required for --stage=sigmma")
        feat_dir = Path(args.sigmma_dir)
        feat_dir.mkdir(parents=True, exist_ok=True)
        sigmma_model = load_model(
            cfg,
            pretrained_image_encoder=cfg.model.pretrained_image_encoder,
            pretrained_ckpt_path=cfg.model.checkpoint_path,
            train=False,
        ).to(device).eval()
        backbone = sigmma_model
        kind = "sigmma"
        backbone_input_dim = int(cfg.model.image_encoder_input_dim)

    data_dir = Path(args.data_dir)
    total = 0
    for _, row in manifest.iterrows():
        sample = row["sample_name"]
        data_h5 = data_dir / f"{sample}.h5"
        if not data_h5.exists():
            print(f"[skip] {sample}: data h5 missing at {data_h5} — run --stage=data first")
            continue
        out_path = feat_dir / f"{sample}.h5"
        if out_path.exists():
            print(f"[skip] {out_path} exists")
            continue
        n = process_slide_features(
            slide_row=row,
            data_h5_path=data_h5,
            backbone_model=backbone,
            kind=kind,
            sigmma_model=sigmma_model,
            encoder_input=encoder_input,
            patch_size=patch_size,
            batch_size=batch_size,
            out_path=out_path,
        )
        total += n

    if args.slide is None:
        preproc = {
            "scale_sizes": list(scale_sizes),
            "encoder_input": encoder_input,
        }
        if kind == "uni":
            write_cache_meta(
                feat_dir, kind="uni_features", task="cell_niche",
                backbone_variant="uni2h",
                backbone_input_dim=backbone_input_dim,
                patch_size=patch_size,
                preprocessing=preproc,
            )
        else:
            write_cache_meta(
                feat_dir, kind="sigmma_features", task="cell_niche",
                backbone_variant="uni2h",
                backbone_input_dim=backbone_input_dim,
                patch_size=patch_size,
                preprocessing=preproc,
                ckpt_path=cfg.model.checkpoint_path,
            )

    print(f"\n[{kind}] {total:,} cells written under {feat_dir}")


if __name__ == "__main__":
    main()
