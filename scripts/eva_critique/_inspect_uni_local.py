"""Inspect local UNI ckpt structure — verify it's vanilla UNI ViT-L.

Checks:
  - state_dict key set (no LoRA adapters, no EMA, no head-specific keys)
  - key parity vs fresh timm vit_large_patch16_224 + init_values
  - shape/dtype of select tensors
  - first-element values for a sanity check
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import timm


LOCAL_DIR = (
    "/nfs/team361/dj16/projects/img_spatial_alignment/assets/ckpts/"
    "vit_large_patch16_224.dinov2.uni_mass100k"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default=LOCAL_DIR)
    args = ap.parse_args()

    p = Path(args.local_dir, "pytorch_model.bin")
    print(f"[local] loading {p}  ({p.stat().st_size / 1e9:.3f} GB)")
    sd = torch.load(p, map_location="cpu")
    print(f"[local] keys={len(sd)}")

    # Fresh UNI-shaped timm reference.
    m_ref = timm.create_model(
        "vit_large_patch16_224",
        img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0,
        dynamic_img_size=True,
    )
    ref_sd = m_ref.state_dict()
    print(f"[ref ] keys={len(ref_sd)}")

    local_keys = set(sd.keys())
    ref_keys = set(ref_sd.keys())
    only_local = local_keys - ref_keys
    only_ref = ref_keys - local_keys
    common = local_keys & ref_keys
    print(f"\n[key diff] local-only={len(only_local)}  ref-only={len(only_ref)}  common={len(common)}")
    if only_local:
        print(f"  local-only (first 10): {sorted(only_local)[:10]}")
    if only_ref:
        print(f"  ref-only   (first 10): {sorted(only_ref)[:10]}")

    print(f"\n[structural check]")
    must_have = [
        ("cls_token",                (1, 1, 1024)),
        ("pos_embed",                (1, 197, 1024)),
        ("patch_embed.proj.weight",  (1024, 3, 16, 16)),
        ("blocks.0.attn.qkv.weight", (3072, 1024)),
        ("blocks.23.attn.qkv.weight",(3072, 1024)),
        ("norm.weight",              (1024,)),
    ]
    for k, expected in must_have:
        if k not in sd:
            print(f"  ✗ {k} MISSING")
            continue
        t = sd[k]
        ok = tuple(t.shape) == expected
        print(f"  {'✓' if ok else '✗'} {k:35s} shape={tuple(t.shape)}  "
              f"(expected {expected})  dtype={t.dtype}  "
              f"mean={t.mean().item():+.4e} std={t.std().item():.4e}")

    # Block count.
    block_indices = sorted({
        int(k.split(".")[1]) for k in sd if k.startswith("blocks.")
        and k.split(".")[1].isdigit()
    })
    print(f"\n[block count] {len(block_indices)} blocks, indices={block_indices[0]}..{block_indices[-1]}")

    # Look for any suspicious keys (LoRA, EMA, mask token, registers).
    suspicious = [k for k in sd if any(s in k.lower() for s in
                  ("lora", "ema", "mask_token", "register", "_head", "linear_head"))]
    print(f"\n[suspicious keys] {len(suspicious)}")
    for k in suspicious[:10]:
        print(f"  - {k}  shape={tuple(sd[k].shape)}")

    # Tensor hashes for reference (so future comparisons can match).
    import hashlib
    sample = ["cls_token", "patch_embed.proj.weight", "blocks.23.attn.qkv.weight",
              "norm.weight"]
    print("\n[reference hashes — for future comparison]")
    for k in sample:
        if k in sd:
            h = hashlib.sha256(sd[k].contiguous().cpu().to(torch.float32).numpy().tobytes()).hexdigest()
            print(f"  {k:40s} sha256={h[:32]}")


if __name__ == "__main__":
    main()
