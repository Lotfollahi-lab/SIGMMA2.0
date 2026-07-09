"""H3 Level-1 diagnostic — per-module gradient flow on SIGMMA contrastive loss.

Loads the SIGMMA checkpoint (job 18328 step 1000), runs a few real training
forwards on a 1-slide val manifest, backprops the symmetric_InfoNCE loss
(summed across 3 scales like the training loop), and reports per-module
gradient L2 norms.

H3 prediction: if the spatial encoder (STEMO + GRAPHLayer + READOUTLayer)
absorbs gradient, its per-parameter ‖∇‖ should dominate the UNI LoRA
adapters' per-parameter ‖∇‖.
"""

from __future__ import annotations
import argparse, os, yaml, math
import torch
import pandas as pd
from pathlib import Path
from ml_collections.config_dict import ConfigDict

from sigmma.utils.train_utils import load_model, load_dataset, load_checkpoint
from sigmma.model import symmetric_InfoNCE
from sigmma.datasets import multimodal_collate


def module_grad_norm(module):
    sq = 0.0
    n = 0
    for p in module.parameters():
        if p.grad is None or not p.requires_grad:
            continue
        sq += p.grad.detach().float().pow(2).sum().item()
        n  += p.numel()
    return math.sqrt(sq), n


def per_param_groups(model):
    g = {}
    # Image-side projection heads (per scale)
    for i, head in enumerate(model.image_projection_heads):
        g[f"image_projection_{i}"] = head
    # Spatial-side GRAPHLayer + READOUTLayer pairs (per scale)
    sp = model.spatial_encoder
    for s in range(len(sp) // 2):
        g[f"spatial_graph_{s}"]   = sp[s * 2]
        g[f"spatial_readout_{s}"] = sp[s * 2 + 1]
    # UNI image encoder (LoRA adapters are the only trainable params here)
    g["image_encoder_uni"] = model.image_encoder
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/in-house_train_sigmma.yaml")
    ap.add_argument("--ckpt",
                    default="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma/18328/checkpoint-step1000.pth")
    ap.add_argument("--manifest", default="/tmp/h3_diag_manifest.csv",
                    help="Small (e.g. 1-slide) manifest to avoid 30-min "
                         "MultiSectionDataset load.")
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="outputs/eva_critique/h3_grad_flow.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[h3] loading config from {args.config}")
    cfg = ConfigDict(yaml.safe_load(open(args.config)))
    # Override dataset to a tiny manifest, keep hier_cache available.
    cfg.dataset.validation.manifest_path = args.manifest
    cfg.train.batch_size = args.batch_size
    # train=True so we get an optimizer w/ requires_grad params, but we will
    # only use the model (not the optimizer).
    cfg.train.max_steps = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[h3] loading model + LoRA adapters")
    model, optimizer, scheduler = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        train=True,
    )
    print(f"[h3] loading checkpoint {args.ckpt}")
    load_checkpoint(model, None, None, Path(args.ckpt), device)
    model.train()

    # Trainable param summary
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[h3] trainable params: {n_trainable:,}")

    print(f"[h3] loading val dataset from {args.manifest}")
    ds = load_dataset(cfg.dataset.validation, is_train=False, model_name=cfg.model.name)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=multimodal_collate, num_workers=2, drop_last=True,
    )

    loss_fn = symmetric_InfoNCE(temperature=float(cfg.model.contrastive_temperature))

    agg: dict[str, list[float]] = {}  # tag -> [sum_norm, n_params, count]
    n_done = 0
    for batch in loader:
        if not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0:
            continue
        if n_done >= args.n_batches:
            break

        image   = batch["image"].to(device)
        spatial = batch["spatial_graph"].to(device)

        # Zero grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
            out = model(image, spatial)
            num_res = len(out["image_embedding"])
            loss = sum(
                loss_fn(out["image_embedding"][s], out["spatial_embedding"][s])
                for s in range(num_res)
            ) / num_res

        loss.backward()
        n_done += 1
        print(f"[h3] batch {n_done}: loss={loss.item():.4f}")

        groups = per_param_groups(model)
        for tag, mod in groups.items():
            norm, n = module_grad_norm(mod)
            if tag not in agg:
                agg[tag] = [0.0, n, 0]
            agg[tag][0] += norm
            agg[tag][2] += 1

    rows = []
    for tag, (s, n, c) in agg.items():
        avg_norm = s / max(c, 1)
        per_param = avg_norm / math.sqrt(max(n, 1))
        rows.append({
            "tag": tag,
            "n_trainable_params": n,
            "avg_grad_norm": round(avg_norm, 6),
            "per_param_grad_norm": round(per_param, 8),
        })

    df = pd.DataFrame(rows).sort_values("avg_grad_norm", ascending=False)
    print("\n=== Per-module gradient norms (avg over batches) ===")
    print(df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\n[h3] saved {args.out}")

    # Roll-up: spatial side vs UNI LoRA
    sp_keys = [k for k in agg if k.startswith("spatial_")]
    proj_keys = [k for k in agg if k.startswith("image_projection_")]
    uni_keys = ["image_encoder_uni"]

    def rollup(keys):
        nor = sum(agg[k][0] for k in keys) / max(n_done, 1)
        n   = sum(agg[k][1] for k in keys)
        return nor, n, nor / math.sqrt(max(n, 1))

    print("\n=== Rolled-up (side-vs-side) ===")
    for label, keys in [("spatial_encoder (GNN+READOUT, 3 scales)", sp_keys),
                         ("image_projection_heads (3 scales)",       proj_keys),
                         ("UNI image_encoder (LoRA adapters)",       uni_keys)]:
        nor, n, pp = rollup(keys)
        print(f"  {label:<50}  n_params={n:>10,}  ‖∇‖={nor:>10.4f}  per-param={pp:>10.2e}")


if __name__ == "__main__":
    main()
