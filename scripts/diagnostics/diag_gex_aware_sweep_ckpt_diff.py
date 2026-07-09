"""Compare the 4 step500 checkpoints in the gex-aware α sweep.

For each pair (a00, a01), (a00, a02), (a00, a04), (a01, a04), report:
  * cosine similarity between concatenated weight vectors (per group: lora,
    scale_pools, scale_mlps, spatial_encoder, head_total)
  * l2 distance / l2 norm of the baseline (relative drift)

Output: stdout table.
"""
from pathlib import Path
import torch

# NOTE: a00 (23617) is UNI v1 (1024-d) — predates the UNI2-h migration of 2026-05-15.
# Cannot be directly compared against a01/a02/a04 (1536-d UNI2-h) at the weight level.
CKPTS = {
    "a01": "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_a01/25923_1/checkpoint-step500.pth",
    "a02": "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_a02/25923_2/checkpoint-step500.pth",
    "a04": "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_a04/25923_3/checkpoint-step500.pth",
}

GROUPS = {
    "lora": lambda k: "lora_" in k,
    "scale_pools": lambda k: "scale_pools" in k,
    "scale_mlps": lambda k: "scale_mlps" in k,
    "spatial_encoder": lambda k: "spatial_encoder" in k,
    "image_attn_pool": lambda k: "scale_pools" in k or "scale_mlps" in k,
}


def load_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
    if not isinstance(sd, dict):
        raise RuntimeError(f"could not find state_dict in {path}")
    # strip module./_orig_mod. prefixes
    out = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ("module.", "_orig_mod."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        out[k2] = v
    return out


def concat_group(sd, keep):
    pieces = [sd[k].flatten().float() for k in sorted(sd) if keep(k)]
    if not pieces:
        return None, 0
    return torch.cat(pieces), sum(p.numel() for p in pieces)


def main():
    states = {tag: load_state(p) for tag, p in CKPTS.items()}
    # Verify same keys (modulo non-relevant ones)
    keys = {tag: set(s.keys()) for tag, s in states.items()}
    base = keys["a01"]
    for tag in ("a02", "a04"):
        diff = (keys[tag] ^ base)
        if diff:
            print(f"[warn] {tag} ↔ a00 key-set diff size {len(diff)}; first 5: {sorted(diff)[:5]}")

    pairs = [("a01", "a02"), ("a01", "a04"), ("a02", "a04")]

    for group, keep in GROUPS.items():
        print(f"\n=== {group} ===")
        v_by_tag = {}
        for tag, sd in states.items():
            v, n = concat_group(sd, keep)
            v_by_tag[tag] = (v, n)
            if v is None:
                print(f"  {tag}: NO MATCHING KEYS")
                continue
            print(f"  {tag}: {n:,} params, ‖w‖={v.norm().item():.3f}")
        print(f"  {'pair':<12} {'cos_sim':>9} {'l2_dist':>12} {'l2_dist/‖base‖':>16}")
        for a, b in pairs:
            va, _ = v_by_tag[a]; vb, _ = v_by_tag[b]
            if va is None or vb is None:
                continue
            cos = torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            d = (va - vb).norm().item()
            rel = d / va.norm().item()
            print(f"  {a}-{b:<10} {cos:>9.5f} {d:>12.4f} {rel:>16.5f}")


if __name__ == "__main__":
    main()
