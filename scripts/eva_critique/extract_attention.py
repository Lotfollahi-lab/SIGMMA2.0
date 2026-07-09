"""Extract per-tile CLS->patch attention maps for a given encoder.

Output is a single .npz at {out_dir}/attention.npz with:
  image_id : (N,)         tile id
  attn     : (N, H, 14, 14)  per-head attention (float16, last block)
  attn_mean: (N, 14, 14)  head-averaged
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.eva_critique.models import load_sigmma, load_uni
from scripts.eva_critique.train_decoder import _MoNuSACTransformed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmma_config", required=True)
    ap.add_argument("--model", choices=["uni", "sigmma"], required=True)
    ap.add_argument("--scale", type=int, default=0)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = (
        load_uni(args.sigmma_config, device)
        if args.model == "uni"
        else load_sigmma(args.sigmma_config, device, scale_idx=args.scale)
    )

    ds = _MoNuSACTransformed(os.path.join(args.data_root, "monusac"),
                             "test", train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2)

    image_ids, attns = [], []
    with torch.no_grad():
        for x, _y, fn in loader:
            x = x.to(device, non_blocking=True)
            a = encoder.extract_cls_attention(x)  # (B, 16, 14, 14)
            attns.append(a.cpu().numpy().astype(np.float16))
            image_ids.extend([f.rsplit("/", 1)[-1].replace(".tif", "") for f in fn])

    attn = np.concatenate(attns, axis=0)
    attn_mean = attn.astype(np.float32).mean(axis=1).astype(np.float16)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "attention.npz")
    np.savez_compressed(out,
                        image_id=np.array(image_ids),
                        attn=attn,
                        attn_mean=attn_mean)
    print(f"[done] {out}  attn shape={attn.shape}")


if __name__ == "__main__":
    main()
