"""Compare two SIGMMA checkpoints tensor-by-tensor.

prev = previous benchmarking ckpt (gex_a01 / job 25923, old pipeline)
curr = current  benchmarking ckpt (gex_svg_a01 / job 36852, corrected pipeline)
"""
import sys, re, collections
import torch

PREV = "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_a01/25923_1/checkpoint-step1000.pth"
CURR = "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_svg_a01/36852/checkpoint-step1000.pth"


def load_sd(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                print(f"  [{path.split('/')[-3]}] top-level keys: {list(obj.keys())}")
                return obj[k]
        # maybe it's already a state_dict (tensor values)
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
        print(f"  [{path.split('/')[-3]}] top-level keys: {list(obj.keys())}")
    return obj


print("Loading prev ...")
prev = load_sd(PREV)
print("Loading curr ...")
curr = load_sd(CURR)

pk, ck = set(prev), set(curr)
print(f"\nprev tensors: {len(pk)}   curr tensors: {len(ck)}")
print(f"only in prev: {len(pk-ck)}   only in curr: {len(ck-pk)}   shared: {len(pk&ck)}")
for k in sorted(pk - ck)[:30]:
    print("  - prev-only:", k, tuple(prev[k].shape))
for k in sorted(ck - pk)[:30]:
    print("  + curr-only:", k, tuple(curr[k].shape))


def group_of(name):
    if "lora" in name.lower():
        return "image_encoder.LoRA (trainable)"
    if name.startswith("image_encoder"):
        return "image_encoder base (frozen)"
    if name.startswith("spatial_encoder"):
        if ".pool." in name:
            return "ST: scale attn-pool"
        if ".mlp." in name:
            return "ST: scale MLP"
        return "ST: graph conv (SAGE/STEMO)"
    if "proj" in name or "head" in name or "logit" in name or "temp" in name:
        return "projection/logit heads"
    return "other: " + name.split(".")[0]


rows = []
shape_mismatch = []
grp_num = collections.defaultdict(float)
grp_den = collections.defaultdict(float)
grp_cnt = collections.defaultdict(int)
grp_changed = collections.defaultdict(int)

for k in sorted(pk & ck):
    a, b = prev[k], curr[k]
    if a.shape != b.shape:
        shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
        continue
    if not torch.is_floating_point(a):
        continue
    a = a.float(); b = b.float()
    d = (b - a)
    dn = d.norm().item()
    an = a.norm().item()
    rel = dn / (an + 1e-12)
    g = group_of(k)
    grp_num[g] += dn ** 2
    grp_den[g] += an ** 2
    grp_cnt[g] += 1
    if dn > 0:
        grp_changed[g] += 1
    rows.append((rel, dn, an, k, tuple(a.shape)))

print("\n=== shape mismatches (shared name, diff shape) ===")
for k, sa, sb in shape_mismatch[:40]:
    print(f"  {k}: {sa} -> {sb}")
print(f"  total shape mismatches: {len(shape_mismatch)}")

print("\n=== per-group aggregate relative change (sqrt(sum dn^2)/sqrt(sum an^2)) ===")
print(f"{'group':<28}{'rel_change':>12}{'tensors':>9}{'changed':>9}")
for g in sorted(grp_den, key=lambda x: -(grp_num[x]/(grp_den[x]+1e-12))):
    rel = (grp_num[g] ** 0.5) / (grp_den[g] ** 0.5 + 1e-12)
    print(f"{g:<28}{rel:>12.4%}{grp_cnt[g]:>9}{grp_changed[g]:>9}")

print("\n=== top 30 tensors by RELATIVE change ===")
rows.sort(reverse=True)
print(f"{'rel':>9}  {'||d||':>10}  {'||w||':>10}  name")
for rel, dn, an, k, sh in rows[:30]:
    print(f"{rel:>9.3%}  {dn:>10.3g}  {an:>10.3g}  {k}  {sh}")

n_identical = sum(1 for rel, dn, *_ in rows if dn == 0)
print(f"\ntensors bit-identical (||d||==0): {n_identical} / {len(rows)} float tensors")
