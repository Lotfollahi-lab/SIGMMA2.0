"""Phase A1: cross-tissue patch-level embedding cache.

For one tissue (or all 8) of the in-house Sanger Xenium cohort, run the SIGMMA
model over the test split and write per-scale image / spatial embeddings to
HDF5 keyed by tissue. Mirrors the forward pass in
`scripts/benchmark_retrieval.py` + `sigmma.evaluation.evaluate_retrieval.load_embeddings`,
but keeps **all three scales** (micro/meso/macro) and persists results instead
of only computing recall scores.

Usage:
    python scripts/cross_tissue_extract_patches.py \
        --config configs/in-house_retrieval_sigmma.yaml \
        --tissue pancreas \
        --out_dir outputs/cross_tissue_retrieval/embeddings

If `--tissue all` is passed, the driver iterates the 8 tissues sequentially
using a single loaded model.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, SequentialSampler

from sigmma.datasets import multimodal_collate
from sigmma.utils import load_dataset, load_model

DATA_ROOT = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium")
PUBLIC_ROOT = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_xenium")

# tissue -> test manifest path. In-house (Sanger 5K/variant panels) + HEST-1k publics
# (each public dataset = its own small 10x panel) so the cohort spans many panels.
MANIFESTS: dict[str, Path] = {
    # in-house
    "skin": DATA_ROOT / "sanger_beacon_MANIFEST_test.csv",
    "glioblastoma": DATA_ROOT / "sanger_glioblastoma_MANIFEST_test.csv",
    "liver": DATA_ROOT / "sanger_liver_MANIFEST_test.csv",
    "lung": DATA_ROOT / "sanger_lung_MANIFEST_test.csv",
    "pancreas": DATA_ROOT / "sanger_pancreas_MANIFEST_test.csv",
    "pediatric_ovary": DATA_ROOT / "sanger_pediatric_ovary_MANIFEST_test.csv",
    "renal_cell_carcinoma": DATA_ROOT / "sanger_renal_cell_carcinoma_MANIFEST_test.csv",
    "whole_embryo": DATA_ROOT / "sanger_whole_embryo_MANIFEST_test.csv",
    # HEST-1k public (out-of-panel)
    "hest_PAAD": PUBLIC_ROOT / "MANIFEST_hest_1k_PAAD_test.csv",
    "hest_IDC": PUBLIC_ROOT / "MANIFEST_hest_1k_IDC_test.csv",
    "hest_LUAD": PUBLIC_ROOT / "MANIFEST_hest_1k_LUAD_test.csv",
    "hest_SKCM": PUBLIC_ROOT / "MANIFEST_hest_1k_SKCM_test.csv",
}
# tissue -> full (train+val+test) manifest. In-house tissues ship a combined
# `sanger_<tissue>_MANIFEST.csv`; HEST-1k publics are zero-shot OOD (test only),
# so they have no full manifest and stay test-only even when split="all".
FULL_MANIFESTS: dict[str, Path] = {
    "skin": DATA_ROOT / "sanger_beacon_MANIFEST.csv",
    "glioblastoma": DATA_ROOT / "sanger_glioblastoma_MANIFEST.csv",
    "liver": DATA_ROOT / "sanger_liver_MANIFEST.csv",
    "lung": DATA_ROOT / "sanger_lung_MANIFEST.csv",
    "pancreas": DATA_ROOT / "sanger_pancreas_MANIFEST.csv",
    "pediatric_ovary": DATA_ROOT / "sanger_pediatric_ovary_MANIFEST.csv",
    "renal_cell_carcinoma": DATA_ROOT / "sanger_renal_cell_carcinoma_MANIFEST.csv",
    "whole_embryo": DATA_ROOT / "sanger_whole_embryo_MANIFEST.csv",
}
TISSUE_BY_NAME = MANIFESTS  # name presence check
ALL_TISSUES = list(MANIFESTS.keys())


def manifest_path_for(tissue: str, split: str = "test") -> Path:
    """Resolve the manifest for a tissue.

    split="test" -> held-out test manifest (default, original behaviour).
    split="all"  -> full train+val+test manifest when available; falls back to
                    the test manifest for tissues without one (e.g. HEST-1k).
    """
    if split == "all" and tissue in FULL_MANIFESTS:
        return FULL_MANIFESTS[tissue]
    return MANIFESTS[tissue]


def make_per_slide_manifest(slide_row: pd.Series, tmp_dir: Path) -> Path:
    """Write a one-row CSV mirroring the original schema, return its path."""
    out = tmp_dir / f"slide_{slide_row['sample_name']}.csv"
    pd.DataFrame([slide_row]).to_csv(out, index=False)
    return out


def per_scale_normed_embeds(out: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pull image_embedding / spatial_embedding lists from a SIGMMA forward output
    and return them L2-normalized (matching `load_embeddings` behaviour)."""
    image_embeds = out["image_embedding"]
    spatial_embeds = out["spatial_embedding"]
    assert len(image_embeds) == 3 and len(spatial_embeds) == 3, (
        "Expected SIGMMA per-scale outputs of length 3 (micro/meso/macro)."
    )
    img_normed, st_normed = [], []
    for img, st in zip(image_embeds, spatial_embeds):
        img = img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        st = st / st.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        img_normed.append(img.detach().cpu())
        st_normed.append(st.detach().cpu())
    return img_normed, st_normed


@torch.no_grad()
def forward_one_slide(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict | None:
    """Run SIGMMA over a single-slide MultiSectionDataset; return arrays or None."""
    if len(dataset) == 0:
        return None
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multimodal_collate,
        persistent_workers=False,
    )

    img_per_scale = [[], [], []]
    st_per_scale = [[], [], []]
    section_ids: list[str] = []
    y_starts: list[int] = []
    x_starts: list[int] = []
    n_seen = 0

    for batch in loader:
        if not batch or "image" not in batch:
            continue
        image_obj = batch["image"]
        if not torch.is_tensor(image_obj) or image_obj.numel() == 0:
            # multimodal_collate returns an empty-list image when all samples
            # in the batch were filtered (e.g. zero-node spatial graphs).
            continue
        image = image_obj.to(device)
        spatial = batch["spatial_graph"].to(device)

        with torch.amp.autocast(device.type, enabled=True):
            out = model(image, spatial)

        img_normed, st_normed = per_scale_normed_embeds(out)
        for s in range(3):
            img_per_scale[s].append(img_normed[s])
            st_per_scale[s].append(st_normed[s])

        meta = batch.get("metadata", {})
        sec = meta.get("section_id", [])
        ys = meta.get("y_start", [])
        xs = meta.get("x_start", [])
        if isinstance(sec, list):
            section_ids.extend(sec)
        else:
            section_ids.extend([str(s) for s in sec])
        ys_iter = ys.tolist() if isinstance(ys, torch.Tensor) else list(ys)
        xs_iter = xs.tolist() if isinstance(xs, torch.Tensor) else list(xs)
        y_starts.extend(int(v) for v in ys_iter)
        x_starts.extend(int(v) for v in xs_iter)
        n_seen += image.shape[0]

    if n_seen == 0:
        return None

    return {
        "image_emb_micro": torch.cat(img_per_scale[0]).numpy().astype(np.float32),
        "image_emb_meso": torch.cat(img_per_scale[1]).numpy().astype(np.float32),
        "image_emb_macro": torch.cat(img_per_scale[2]).numpy().astype(np.float32),
        "st_emb_micro": torch.cat(st_per_scale[0]).numpy().astype(np.float32),
        "st_emb_meso": torch.cat(st_per_scale[1]).numpy().astype(np.float32),
        "st_emb_macro": torch.cat(st_per_scale[2]).numpy().astype(np.float32),
        "section_id": np.array(section_ids, dtype="S64"),
        "y_start": np.array(y_starts, dtype=np.int32),
        "x_start": np.array(x_starts, dtype=np.int32),
    }


def write_tissue_h5(
    out_path: Path,
    tissue: str,
    per_slide_results: list[tuple[str, dict]],
) -> None:
    """Concat per-slide arrays into a flat tissue-level HDF5."""
    if not per_slide_results:
        print(f"[{tissue}] no patches extracted; skipping write.")
        return
    keys = [
        "image_emb_micro", "image_emb_meso", "image_emb_macro",
        "st_emb_micro", "st_emb_meso", "st_emb_macro",
        "section_id", "y_start", "x_start",
    ]
    pooled = {k: [] for k in keys}
    slide_index_offsets = [0]
    slide_ids: list[str] = []
    for slide_id, arrays in per_slide_results:
        for k in keys:
            pooled[k].append(arrays[k])
        slide_ids.append(slide_id)
        slide_index_offsets.append(slide_index_offsets[-1] + arrays["image_emb_micro"].shape[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["tissue"] = tissue
        f.attrs["n_slides"] = len(slide_ids)
        f.attrs["n_patches"] = slide_index_offsets[-1]
        for k in keys:
            f.create_dataset(k, data=np.concatenate(pooled[k], axis=0))
        f.create_dataset("slide_id_list", data=np.array(slide_ids, dtype="S64"))
        f.create_dataset("slide_offsets", data=np.array(slide_index_offsets, dtype=np.int64))
    print(
        f"[{tissue}] wrote {out_path} "
        f"({slide_index_offsets[-1]} patches across {len(slide_ids)} slides)"
    )


def build_per_slide_dataset_config(template: ConfigDict, manifest_path: Path) -> ConfigDict:
    """Clone the existing dataset config and swap the manifest path."""
    new_cfg = ConfigDict(template.to_dict())
    new_cfg.manifest_path = str(manifest_path)
    new_cfg.hier_cache_dir = template.get("hier_cache_dir", None)
    return new_cfg


def run_tissue(
    *,
    tissue: str,
    model,
    dataset_template: ConfigDict,
    batch_size: int,
    num_workers: int,
    out_dir: Path,
    device: torch.device,
    max_slides: int | None,
    split: str = "test",
) -> None:
    manifest = manifest_path_for(tissue, split)
    if not manifest.exists():
        print(f"[{tissue}] no {split} manifest at {manifest}; skip.")
        return
    df = pd.read_csv(manifest)
    if "annotation" not in df.columns:
        df["annotation"] = ""  # HEST manifests lack this optional column
    if max_slides is not None:
        df = df.head(max_slides)
    print(f"[{tissue}] {len(df)} {split} slides from {manifest}")

    per_slide: list[tuple[str, dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for _, row in df.iterrows():
            sample = row["sample_name"]
            mini = make_per_slide_manifest(row, tmp_path)
            cfg = build_per_slide_dataset_config(dataset_template, mini)
            try:
                ds = load_dataset(cfg, is_train=False, model_name="sigmma")
            except ValueError as e:
                print(f"  [{sample}] dataset load failed: {e}")
                continue
            print(f"  [{sample}] valid patches: {len(ds)}")
            arr = forward_one_slide(
                model=model, dataset=ds, device=device,
                batch_size=batch_size, num_workers=num_workers,
            )
            if arr is None:
                print(f"  [{sample}] no patches survived forward pass")
                continue
            per_slide.append((sample, arr))
            print(
                f"  [{sample}] extracted {arr['image_emb_micro'].shape[0]} patches"
            )
    out_path = out_dir / f"{tissue}.h5"
    write_tissue_h5(out_path, tissue, per_slide)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Existing retrieval config (manifest path overridden per tissue).")
    p.add_argument("--tissue", default="all",
                   help="Tissue name from the audit, or 'all'.")
    p.add_argument("--out_dir", default="outputs/cross_tissue_retrieval/embeddings")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_slides", type=int, default=None,
                   help="Cap slides per tissue (smoke test convenience).")
    p.add_argument("--split", choices=["test", "all"], default="test",
                   help="'test' = held-out test slides (default); "
                        "'all' = full train+val+test for in-house tissues "
                        "(HEST-1k publics stay test-only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigDict(yaml.load(open(args.config), yaml.FullLoader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading SIGMMA from {cfg.model.checkpoint_path}")
    model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.get("pretrained_image_encoder", None),
        pretrained_ckpt_path=cfg.model.get("checkpoint_path", None),
        train=False,
    )
    model = model.to(device).eval()

    dataset_template = cfg.dataset.query_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tissues = [args.tissue] if args.tissue != "all" else list(ALL_TISSUES)
    for tissue in tissues:
        if tissue not in TISSUE_BY_NAME:
            print(f"[{tissue}] not in TISSUE_BY_NAME; skip.")
            continue
        run_tissue(
            tissue=tissue,
            model=model,
            dataset_template=dataset_template,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            out_dir=out_dir,
            device=device,
            max_slides=args.max_slides,
            split=args.split,
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
