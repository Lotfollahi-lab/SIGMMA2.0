"""Inspect new UNI2-h local ckpt structure."""
import argparse
import hashlib
from pathlib import Path
import torch
import timm

DEFAULT = "/nfs/team361/dj16/projects/SIGMMA/assets/ckpts/uni_h/pytorch_model.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT)
    args = ap.parse_args()

    p = Path(args.ckpt)
    print(f"[file] {p}  size={p.stat().st_size / 1e6:.2f} MB")

    print("[load] torch.load (cpu)...")
    sd = torch.load(p, map_location="cpu")
    if not isinstance(sd, dict):
        print(f"  unexpected type: {type(sd)}")
        return

    keys = list(sd.keys())
    print(f"[keys] {len(keys)} total")
    print(f"[keys] first 10: {keys[:10]}")
    print(f"[keys] last  5: {keys[-5:]}")

    # Block count
    block_idx = sorted({int(k.split('.')[1]) for k in keys if k.startswith('blocks.') and k.split('.')[1].isdigit()})
    print(f"[blocks] {len(block_idx)} blocks, indices {block_idx[:3]}..{block_idx[-3:] if block_idx else 'none'}")

    # Embed dim
    if 'cls_token' in sd:
        print(f"[cls_token] shape={tuple(sd['cls_token'].shape)}")
    if 'pos_embed' in sd:
        print(f"[pos_embed] shape={tuple(sd['pos_embed'].shape)}")
    if 'patch_embed.proj.weight' in sd:
        print(f"[patch_embed.proj.weight] shape={tuple(sd['patch_embed.proj.weight'].shape)}")
    if 'norm.weight' in sd:
        print(f"[norm.weight] shape={tuple(sd['norm.weight'].shape)}")

    # UNI2-h specific keys
    print("\n[UNI2-h specific markers]")
    has_reg = any('reg_token' in k for k in keys)
    has_mlp_w12 = any('mlp.fc1_g.weight' in k or 'mlp.w12' in k for k in keys)  # SwiGLU
    print(f"  reg_tokens key present?     {has_reg}")
    print(f"  SwiGLU mlp keys present?    {has_mlp_w12}")

    # Total param count
    total = sum(t.numel() for t in sd.values() if isinstance(t, torch.Tensor))
    print(f"\n[params] total = {total:,}  ({total * 4 / 1e9:.2f} GB if fp32)")

    # Sample tensor stats
    print("\n[sample tensors]")
    for k in ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'blocks.0.attn.qkv.weight', 'norm.weight']:
        if k in sd:
            t = sd[k]
            h = hashlib.sha256(t.contiguous().cpu().to(torch.float32).numpy().tobytes()).hexdigest()[:16]
            print(f"  {k:40s} shape={str(tuple(t.shape)):25s} sha256={h}")


if __name__ == "__main__":
    main()
