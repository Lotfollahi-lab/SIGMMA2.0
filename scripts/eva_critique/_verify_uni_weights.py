"""Compare our local UNI checkpoint to HuggingFace MahmoodLab/uni.

Two paths:
  - LOCAL: load `pytorch_model.bin` from
    /nfs/team361/dj16/projects/img_spatial_alignment/assets/ckpts/
    vit_large_patch16_224.dinov2.uni_mass100k/
  - HF: timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True) — downloads
    from HuggingFace Hub the first time.

Compare:
  - state_dict keys (set difference)
  - SHA256 of select tensors (cls_token, pos_embed, patch_embed.proj.weight,
    blocks.{0,11,23}.{norm1.weight, attn.qkv.weight, mlp.fc1.weight}, norm.weight)
  - Overall max abs diff across all common keys.

If all tensors match bit-perfectly → our local ckpt IS the same as EVA uses.
If they differ → we've been using a wrong/different UNI variant.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import torch
import timm


LOCAL_DIR = (
    "/nfs/team361/dj16/projects/img_spatial_alignment/assets/ckpts/"
    "vit_large_patch16_224.dinov2.uni_mass100k"
)


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA256 over bytes of a float32 tensor (after .contiguous().cpu())."""
    arr = t.detach().contiguous().cpu().to(torch.float32).numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def load_local(local_dir: str) -> dict:
    p = Path(local_dir, "pytorch_model.bin")
    print(f"[local] loading {p}")
    sd = torch.load(p, map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError(f"expected dict, got {type(sd)}")
    return sd


def load_hf() -> dict:
    print("[hf] timm.create_model('hf-hub:MahmoodLab/uni', pretrained=True)")
    m = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    return m.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default=LOCAL_DIR)
    args = ap.parse_args()

    sd_a = load_local(args.local_dir)
    sd_b = load_hf()

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b
    print(f"\n[keys] local={len(keys_a)}  hf={len(keys_b)}  common={len(common)}")
    print(f"[keys] local-only ({len(only_a)}): {sorted(only_a)[:5]}")
    print(f"[keys] hf-only ({len(only_b)}): {sorted(only_b)[:5]}")

    # Compare select tensor stats + hashes.
    sample_keys = [
        "cls_token",
        "pos_embed",
        "patch_embed.proj.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.11.attn.qkv.weight",
        "blocks.23.attn.qkv.weight",
        "blocks.23.mlp.fc1.weight",
        "norm.weight",
    ]
    print(f"\n{'key':40s} {'shape':25s} {'local_h':10s} {'hf_h':10s} {'eq?':4s}")
    print("-" * 95)
    eq_all = True
    for k in sample_keys:
        if k not in common:
            print(f"{k:40s} (MISSING IN ONE)")
            eq_all = False
            continue
        ta, tb = sd_a[k], sd_b[k]
        h_a = tensor_sha256(ta)
        h_b = tensor_sha256(tb)
        eq = h_a == h_b
        eq_all &= eq
        print(f"{k:40s} {str(tuple(ta.shape)):25s} {h_a:10s} {h_b:10s} {str(eq):4s}")

    # Whole-state diff over common keys (skip CLS/pos_embed that may shape-differ).
    print(f"\n[whole-state diff over {len(common)} common keys]")
    max_diff = 0.0
    mismatched_shape = 0
    mismatched_value = 0
    for k in common:
        ta, tb = sd_a[k], sd_b[k]
        if ta.shape != tb.shape:
            mismatched_shape += 1
            continue
        d = (ta.float() - tb.float()).abs().max().item()
        max_diff = max(max_diff, d)
        if d > 1e-6:
            mismatched_value += 1
    print(f"  mismatched_shape   = {mismatched_shape}")
    print(f"  mismatched_value   = {mismatched_value} (max_abs > 1e-6)")
    print(f"  max_abs_diff       = {max_diff:.6e}")

    print("\n[verdict]")
    if max_diff < 1e-6 and not only_a and not only_b and not mismatched_shape:
        print("  ✓ IDENTICAL — local ckpt == HuggingFace MahmoodLab/uni")
    else:
        print("  ✗ DIFFERENT — local ckpt is NOT bit-equal to MahmoodLab/uni")


if __name__ == "__main__":
    main()
