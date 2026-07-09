"""Extract per-FOV (h_off, h_on, s_emb, gex) for the dense-attnpool ckpt.

Iterates the dataset for a single slide, runs sigmma.forward(image, spatial_graph)
both with LoRA active and with LoRA scaling zeroed, and concatenates the per-scale
512-dim normalized embeddings to a (1, 1536) vector per modality. GEX is read from
the matching probe cache npz (dispersion HVG panel, 50 genes, summed over tissue
bins per FOV) so the gene panel is fixed across all FOVs.

Output npz:
    h_off       (N, 1536) float32
    h_on        (N, 1536) float32
    s_emb       (N, 1536) float32
    gex         (N, G=50) float32
    fov_id      (N,)      int64
    x_start     (N,)      int64
    y_start     (N,)      int64
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from ml_collections.config_dict import ConfigDict
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.data import DataLoader

from sigmma.datasets import multimodal_collate
from sigmma.evaluation.gex_cache_io import iter_section_fovs
from sigmma.utils.train_utils import load_dataset, load_model


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, BaseTunerLayer)]


def save_lora_scaling(model):
    return {n: dict(m.scaling) for n, m in lora_modules(model)}


def zero_lora_scaling(model):
    for _, m in lora_modules(model):
        for k in list(m.scaling):
            m.scaling[k] = 0.0


def restore_lora_scaling(model, saved):
    for n, m in lora_modules(model):
        m.scaling.update(saved[n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", default="configs/in-house_train_sigmma_dense_attnpool.yaml")
    ap.add_argument("--ckpt", default="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool/23617/checkpoint-step1000.pth")
    ap.add_argument("--manifest_override", required=True,
                    help="1-row manifest with the slide to extract.")
    ap.add_argument("--cache_dir", required=True,
                    help="Data-cache root holding per-section `<sample>.h5` for the dispersion HVG panel "
                         "(probe_gex_2d phase=cache). Each section file carries `target` (N,G,K,K), "
                         "`mask` (N,K,K), `x_start`, `y_start`, `fov_id`.")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ConfigDict(yaml.safe_load(open(args.train_config)))
    cfg.model.checkpoint_path = args.ckpt
    cfg.dataset.train.manifest_path = args.manifest_override

    print(f"[load] model from {args.ckpt}", flush=True)
    model = load_model(cfg, pretrained_image_encoder=cfg.model.pretrained_image_encoder,
                       pretrained_ckpt_path=args.ckpt, train=False)
    model = model.to(device)
    for n, p in model.named_parameters():
        p.data = p.data.to(device)
    for n, b in model.named_buffers():
        b.data = b.data.to(device)
    model.eval()

    # Build cache index by (x_start, y_start) → dict so we can attach GEX to
    # each FOV emitted by the dataset.
    sample_name = str(pd.read_csv(args.manifest_override)["sample_name"].iloc[0])
    print(f"[cache] indexing {args.cache_dir} / {sample_name}.h5", flush=True)
    cache_index = {}
    for d in iter_section_fovs(args.cache_dir, sample_name, fields=["target", "mask"]):
        cache_index[(d["x_start"], d["y_start"])] = {
            "target": d["target"], "mask": d["mask"], "fov_id": d["fov_id"],
        }
    print(f"[cache] {len(cache_index)} FOVs indexed", flush=True)

    # Dataset
    ds = load_dataset(cfg.dataset.train, is_train=False, model_name="sigmma")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=multimodal_collate, drop_last=False)

    saved_scaling = save_lora_scaling(model)

    h_on_chunks, h_off_chunks, s_chunks, gex_chunks = [], [], [], []
    fov_ids, xs, ys = [], [], []
    n_seen = 0
    n_dropped = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            if not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0:
                continue
            img = batch["image"].to(device)
            graph = batch["spatial_graph"].to(device)
            meta = batch["metadata"]
            xs_b = [int(v) for v in meta["x_start"]]
            ys_b = [int(v) for v in meta["y_start"]]

            # GEX from cache for each row in the batch; drop rows w/o cache match
            keep = []
            gex_b = []
            fov_b = []
            for k in range(len(xs_b)):
                ck = (xs_b[k], ys_b[k])
                if ck not in cache_index:
                    n_dropped += 1
                    continue
                c = cache_index[ck]
                # FOV-level GEX = sum over tissue bins of log1p-counts.
                # mask is per-bin tissue fraction in [0,1]; weighting by mask down-
                # weights partial-tissue bins.
                t = c["target"]                       # (G, K, K)
                m = c["mask"]                          # (K, K)
                gex_b.append((t * m[None, :, :]).sum(axis=(1, 2)))
                fov_b.append(c["fov_id"])
                keep.append(k)

            if not keep:
                continue
            img_k = img[keep]
            # Sub-batch graphs — DGL.batch needs per-row graphs; use the convenience
            # of the original full batch then mask. Simpler: re-forward only on the
            # kept rows by reconstructing a new batched graph.
            # Easiest: just forward the full image+graph batch and slice outputs.
            out_on = model(img, graph)
            zero_lora_scaling(model)
            out_off = model(img, graph)
            restore_lora_scaling(model, saved_scaling)

            # Concatenate per-scale (B, D) along feature dim
            h_on  = torch.cat([e.float() for e in out_on["image_embedding"]],   dim=-1).cpu().numpy()
            h_off = torch.cat([e.float() for e in out_off["image_embedding"]],  dim=-1).cpu().numpy()
            s_e   = torch.cat([e.float() for e in out_on["spatial_embedding"]], dim=-1).cpu().numpy()

            h_on_chunks.append(h_on[keep])
            h_off_chunks.append(h_off[keep])
            s_chunks.append(s_e[keep])
            gex_chunks.append(np.stack(gex_b, axis=0))
            fov_ids.extend(fov_b)
            xs.extend(xs_b[k] for k in keep)
            ys.extend(ys_b[k] for k in keep)
            n_seen += len(keep)
            if batch_idx % 5 == 0:
                print(f"  [batch {batch_idx}] n_kept={n_seen} (dropped {n_dropped})", flush=True)

    h_on = np.concatenate(h_on_chunks, axis=0)
    h_off = np.concatenate(h_off_chunks, axis=0)
    s_emb = np.concatenate(s_chunks, axis=0)
    gex = np.concatenate(gex_chunks, axis=0)
    print(f"[done] N={len(fov_ids)} FOVs; h_on={h_on.shape} h_off={h_off.shape} s_emb={s_emb.shape} gex={gex.shape}",
          flush=True)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        h_off=h_off.astype(np.float32),
        h_on=h_on.astype(np.float32),
        s_emb=s_emb.astype(np.float32),
        gex=gex.astype(np.float32),
        fov_id=np.asarray(fov_ids, dtype=np.int64),
        x_start=np.asarray(xs, dtype=np.int64),
        y_start=np.asarray(ys, dtype=np.int64),
    )
    print(f"[save] {args.out_npz}", flush=True)


if __name__ == "__main__":
    main()
