"""
Diagnose how a LoRA-fine-tuned UNI inside a SIGMMA checkpoint differs from
the raw pretrained UNI.

Why this script exists
----------------------
PEFT LoRA wraps target ``nn.Linear`` modules with ``LoraLayer`` (renames the
original ``weight`` to ``base_layer.weight`` and adds ``lora_A``/``lora_B``).
A naïve state-dict diff fails (different keys) and even if patched would only
show base-vs-raw weights — which are identical, since the base is frozen
(see ``apply_lora_to_backbone`` setting ``requires_grad=False`` on non-LoRA
params). What actually moved is the LoRA delta

    delta_W = scaling * lora_B.weight @ lora_A.weight

added on top of the frozen base at forward time. This script reports those
deltas per-adapter (scale_0/1/2 when ``per_scale: true``), per-block, and
per-target-module, plus a sanity check that base layers are byte-identical
to raw UNI.

Usage
-----
    python scripts/diff_uni_weights.py \
        --sigmma_config /path/to/sigmma_config.yaml         \
        [--top_k 20]                                        \
        [--csv_out outputs/diff_uni_weights/diff_uni_weights.csv]               \
        [--cross_adapter]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from ml_collections.config_dict import ConfigDict

from peft.tuners.lora import LoraLayer

from sigmma.utils import load_model
from sigmma.model.models_image_backbones import UNI


def lora_delta(layer: LoraLayer, adapter: str) -> torch.Tensor:
    """Effective weight delta added by `adapter` on top of base.

    Delegates to PEFT's ``LoraLayer.get_delta_weight`` so that Linear
    (delta shape ``(out, in)``) and Conv2d (delta shape ``(out_c, in_c,
    kH, kW)`` — which timm ViT's ``patch_embed.proj`` is) are handled
    uniformly. The norm-based statistics downstream are dim-agnostic.
    """
    return layer.get_delta_weight(adapter).detach().float().cpu()


def collect_lora_layers(image_encoder_model):
    """Return list of (qualified_name, LoraLayer) for every patched module."""
    out = []
    for name, m in image_encoder_model.named_modules():
        if isinstance(m, LoraLayer):
            out.append((name, m))
    return out


_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


def block_idx(qualified_name: str):
    m = _BLOCK_RE.search(qualified_name)
    return int(m.group(1)) if m else None


def target_module_kind(qualified_name: str) -> str:
    # qkv vs proj for ViT attention; fc1/fc2 for MLP if present.
    for kind in ("qkv", "proj", "fc1", "fc2"):
        if qualified_name.endswith("." + kind):
            return kind
    return qualified_name.split(".")[-1]


def report_adapter(adapter, lora_layers, top_k: int, csv_rows):
    print(f"\n{'=' * 72}")
    print(f"Adapter: {adapter}")
    print(f"{'=' * 72}")

    rows = []
    block_acc = defaultdict(lambda: [0.0, 0.0])  # [sum delta^2, sum base^2]
    kind_acc = defaultdict(lambda: [0.0, 0.0])

    total_d2 = 0.0
    total_b2 = 0.0

    for name, layer in lora_layers:
        if adapter not in layer.lora_A:
            continue
        delta = lora_delta(layer, adapter)
        base = layer.base_layer.weight.detach().float().cpu()
        d_norm = delta.norm().item()
        b_norm = base.norm().item()
        rel = d_norm / (b_norm + 1e-12)
        max_abs = delta.abs().max().item()
        rank_r = layer.r[adapter]

        rows.append((name, rel, d_norm, max_abs, b_norm, rank_r,
                     tuple(base.shape)))
        bidx = block_idx(name)
        kind = target_module_kind(name)
        block_acc[bidx][0] += d_norm ** 2
        block_acc[bidx][1] += b_norm ** 2
        kind_acc[kind][0] += d_norm ** 2
        kind_acc[kind][1] += b_norm ** 2
        total_d2 += d_norm ** 2
        total_b2 += b_norm ** 2

        csv_rows.append({
            "adapter": adapter, "param": name, "rel": rel, "delta_l2": d_norm,
            "max_abs_delta": max_abs, "base_l2": b_norm, "rank": rank_r,
            "shape": "x".join(map(str, base.shape)),
        })

    if not rows:
        print(f"  (no patched layers carry adapter '{adapter}')")
        return

    global_rel = (total_d2 ** 0.5) / ((total_b2 ** 0.5) + 1e-12)
    print(f"Patched layers     : {len(rows)}")
    print(f"Global ||ΔW||      : {total_d2 ** 0.5:.4f}")
    print(f"Global ||W_base||  : {total_b2 ** 0.5:.4f}")
    print(f"Global ||ΔW||/||W||: {global_rel:.4%}")

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nTop-{min(top_k, len(rows))} most-changed layers (relative L2):")
    print(f"  {'param':50s}  {'rel':>8s}  {'|Δ|':>10s}  {'max|Δ|':>10s}  {'|W|':>10s}  {'r':>3s}")
    for name, rel, d_norm, max_abs, b_norm, r, _ in rows[:top_k]:
        print(f"  {name[:50]:50s}  {rel:>8.4%}  {d_norm:>10.4f}  {max_abs:>10.4f}  {b_norm:>10.4f}  {r:>3d}")

    print("\nPer-block aggregate (rel = ||Δ||_block / ||W||_block):")
    for b in sorted(k for k in block_acc.keys() if k is not None):
        d2, b2 = block_acc[b]
        print(f"  block {b:>2d}: rel={ (d2 ** 0.5) / ((b2 ** 0.5) + 1e-12):>8.4%}"
              f"   |Δ|={d2 ** 0.5:>8.4f}")
    if None in block_acc:
        d2, b2 = block_acc[None]
        print(f"  (non-block) rel={(d2 ** 0.5) / ((b2 ** 0.5) + 1e-12):>8.4%}")

    print("\nPer-target-module aggregate:")
    for kind in sorted(kind_acc.keys()):
        d2, b2 = kind_acc[kind]
        print(f"  {kind:>6s}: rel={(d2 ** 0.5) / ((b2 ** 0.5) + 1e-12):>8.4%}"
              f"   |Δ|={d2 ** 0.5:>8.4f}")


def cross_adapter_similarity(adapters, lora_layers):
    """Cosine similarity of ΔW vectors, averaged across patched layers."""
    if len(adapters) < 2:
        return
    print(f"\n{'=' * 72}")
    print("Cross-adapter ΔW cosine similarity (per layer, then mean)")
    print(f"{'=' * 72}")
    pair_sums = defaultdict(list)
    for name, layer in lora_layers:
        deltas = {}
        for a in adapters:
            if a in layer.lora_A:
                deltas[a] = lora_delta(layer, a).flatten()
        names = list(deltas.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                cos = torch.nn.functional.cosine_similarity(
                    deltas[a], deltas[b], dim=0).item()
                pair_sums[(a, b)].append(cos)
    print(f"  {'pair':30s}  {'mean cos':>10s}  {'min':>8s}  {'max':>8s}")
    for (a, b), vals in pair_sums.items():
        t = torch.tensor(vals)
        print(f"  {a + ' vs ' + b:30s}  {t.mean().item():>10.4f}  "
              f"{t.min().item():>8.4f}  {t.max().item():>8.4f}")


def verify_base_layers_unchanged(image_encoder_model, raw_state):
    """Sanity: every weight that was frozen (base_layer + non-LoRA modules)
    must be byte-identical to raw UNI."""
    print(f"\n{'=' * 72}")
    print("Sanity: base / non-LoRA params identical to raw UNI?")
    print(f"{'=' * 72}")

    sigmma_state = image_encoder_model.state_dict()
    # PEFT renames `<...>.weight` to `<...>.base_layer.weight` for patched
    # Linear modules; unpatched params keep their original key.
    aligned = {}
    for k, v in sigmma_state.items():
        if "lora_" in k:
            continue
        # strip the inserted ".base_layer" prefix wherever it appears.
        canonical = k.replace(".base_layer.", ".")
        aligned[canonical] = v

    missing_in_raw = [k for k in aligned if k not in raw_state]
    missing_in_sigmma = [k for k in raw_state if k not in aligned]
    n_checked = 0
    n_drifted = 0
    drifted = []
    for k, v in aligned.items():
        if k not in raw_state:
            continue
        n_checked += 1
        diff = (v.detach().float().cpu() - raw_state[k].detach().float().cpu()).abs().max().item()
        if diff > 0:
            n_drifted += 1
            drifted.append((k, diff))

    print(f"Frozen params checked        : {n_checked}")
    print(f"Drifted (max|Δ| > 0)         : {n_drifted}")
    if drifted:
        print("  Top drifts:")
        for k, d in sorted(drifted, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {k:60s}  max|Δ|={d:.3e}")
    if missing_in_raw:
        print(f"Keys in SIGMMA not in raw UNI ({len(missing_in_raw)}): "
              f"{missing_in_raw[:5]}{'...' if len(missing_in_raw) > 5 else ''}")
    if missing_in_sigmma:
        print(f"Keys in raw UNI not in SIGMMA ({len(missing_in_sigmma)}): "
              f"{missing_in_sigmma[:5]}{'...' if len(missing_in_sigmma) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmma_config", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--csv_out", type=str, default=None,
                        help="If set, dump per-layer per-adapter rows to CSV "
                             "for downstream plotting / notebook analysis.")
    parser.add_argument("--cross_adapter", action="store_true",
                        help="Report pairwise cosine similarity of ΔW across "
                             "adapters (only meaningful for per_scale models).")
    args = parser.parse_args()

    cfg = ConfigDict(yaml.load(open(args.sigmma_config), yaml.FullLoader))
    uni_ckpt_dir = cfg.model.pretrained_image_encoder_ckpt_path
    sigmma_ckpt_path = cfg.model.checkpoint_path

    print(f"SIGMMA config     : {args.sigmma_config}")
    print(f"SIGMMA checkpoint : {sigmma_ckpt_path}")
    print(f"Raw UNI directory : {uni_ckpt_dir}")
    lora_cfg = cfg.model.get("lora", None)
    per_scale = bool(lora_cfg.get("per_scale", False)) if lora_cfg else False
    print(f"LoRA per_scale    : {per_scale}")

    # --- Build SIGMMA (loads checkpoint, injects LoRA adapters) ---
    sigmma_model = load_model(
        cfg,
        pretrained_image_encoder=cfg.model.pretrained_image_encoder,
        pretrained_ckpt_path=sigmma_ckpt_path,
        train=False,
    )
    sigmma_model.eval()
    image_enc = sigmma_model.image_encoder.model

    # --- Sanity: base layers untouched ---
    raw_uni = UNI(checkpoint_path=uni_ckpt_dir)
    raw_state = raw_uni.model.state_dict()
    verify_base_layers_unchanged(image_enc, raw_state)

    # --- Collect LoRA layers and adapters ---
    lora_layers = collect_lora_layers(image_enc)
    adapters = sorted({a for _, m in lora_layers for a in m.lora_A.keys()})
    print(f"\nDetected {len(lora_layers)} LoRA-patched modules, "
          f"{len(adapters)} adapter(s): {adapters}")

    csv_rows = []
    for adapter in adapters:
        report_adapter(adapter, lora_layers, args.top_k, csv_rows)

    if args.cross_adapter:
        cross_adapter_similarity(adapters, lora_layers)

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote per-layer CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
