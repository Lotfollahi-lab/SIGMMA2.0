"""Diagnose LoRA effect on the dense-attnpool ckpt 23617.

Measurements (all on a single fixed batch from the train manifest):
  (1) Linear CKA between LoRA-on vs LoRA-off ViT pooled features.
  (2) Per-block activation cosine similarity (LoRA on vs off).
  (3) Image / spatial alignment metric change (LoRA on vs off): per-scale
      mean paired cosine + InfoNCE top-1 accuracy.
  (4) |delta W|_F per LoRA module (state_dict only).
  (5) Gradient norm of lora_A / lora_B per module after one forward+backward
      on the symmetric InfoNCE loss.
  (6) Singular value spectrum of (alpha/r) * B @ A per LoRA module.

Outputs:
  - outputs/diagnostics/dense_attnpool_23617/diagnose.json (all numbers)
  - prints a compact summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ml_collections.config_dict import ConfigDict
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.data import DataLoader

from sigmma.datasets import multimodal_collate
from sigmma.model import symmetric_InfoNCE
from sigmma.utils.train_utils import load_dataset, load_model


# ---------------------------------------------------------------- helpers ---
def lora_modules(model) -> list[tuple[str, BaseTunerLayer]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, BaseTunerLayer)]


def save_lora_scaling(model) -> dict:
    return {n: dict(m.scaling) for n, m in lora_modules(model)}


def zero_lora_scaling(model) -> None:
    for _, m in lora_modules(model):
        for adapter in list(m.scaling):
            m.scaling[adapter] = 0.0


def restore_lora_scaling(model, saved: dict) -> None:
    for n, m in lora_modules(model):
        m.scaling.update(saved[n])


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two centered matrices of shape (N, D)."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = (x.t() @ y).pow(2).sum()
    xx = (x.t() @ x).pow(2).sum()
    yy = (y.t() @ y).pow(2).sum()
    denom = (xx.sqrt() * yy.sqrt()).clamp_min(1e-12)
    return float(xy / denom)


# ----------------------------------------------------------------- runners --
@torch.no_grad()
def run_vit_with_hook(model, x_resized, device):
    """One ViT forward; capture per-block output tokens (B, P^2+1, D)."""
    acts: dict[int, torch.Tensor] = {}
    handles = []
    vit = model.image_encoder.model

    def _make_hook(idx: int):
        def _hook(mod, inp, out):
            # Side-effect only; must NOT return a value, otherwise PyTorch
            # replaces the block's output with the returned tensor.
            acts[idx] = out.detach().float().cpu()
        return _hook

    for i, blk in enumerate(vit.blocks):
        handles.append(blk.register_forward_hook(_make_hook(i)))
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x_resized.is_cuda):
            tokens = vit.forward_features(x_resized)  # (B, 1+P^2, D)
    finally:
        for h in handles:
            h.remove()
    patches = tokens[:, 1:, :].float().cpu()
    return patches, acts  # patches (B, P^2, D), acts: int -> (B, P^2+1, D)


def compute_alignment(image_embs, spatial_embs):
    """Per-scale mean paired cosine + InfoNCE top-1 accuracy (image->spatial)."""
    out = []
    for img, spat in zip(image_embs, spatial_embs):
        # img/spat: (B, D) already L2-normalized in the model
        cos_paired = (img * spat).sum(dim=-1)
        sim = img @ spat.t()  # (B, B)
        top1 = (sim.argmax(dim=-1) == torch.arange(sim.size(0), device=sim.device)).float().mean()
        out.append({
            "mean_paired_cosine": float(cos_paired.mean()),
            "infonce_top1_i2s": float(top1),
        })
    return out


def make_batch(cfg_dataset, batch_size: int, device, seed: int = 0):
    ds = load_dataset(cfg_dataset, is_train=False, model_name="sigmma")
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                    num_workers=0, collate_fn=multimodal_collate, drop_last=True)
    return next(iter(dl))


# ------------------------------------------------------------- measurements -
def measure_state_dict_only(state_dict, lora_r: int, lora_alpha: int):
    """(4) |delta W|_F per module and (6) singular value spectrum."""
    scale = lora_alpha / lora_r
    parents = defaultdict(dict)
    for k, v in state_dict.items():
        if ".lora_A.default.weight" in k:
            parents[k.replace(".lora_A.default.weight", "")]["A"] = v
        if ".lora_B.default.weight" in k:
            parents[k.replace(".lora_B.default.weight", "")]["B"] = v
        if ".base_layer.weight" in k:
            parents[k.replace(".base_layer.weight", "")]["W"] = v

    delta_norm = {}
    sv_spectrum = {}
    for par, ab in sorted(parents.items()):
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].float()
        B = ab["B"].float()
        if B.dim() == 2 and A.dim() == 2:
            delta = scale * (B @ A)
        elif B.dim() == 4 and A.dim() == 4:
            Bm = B.view(B.shape[0], B.shape[1])
            Am = A.view(A.shape[0], -1)
            delta = scale * (Bm @ Am)
        else:
            continue
        delta_norm[par] = {
            "delta_frob": float(delta.pow(2).sum().sqrt()),
            "base_frob": float(ab["W"].float().pow(2).sum().sqrt()) if "W" in ab else None,
        }
        # Singular values of delta (rank <= r=16)
        try:
            sv = torch.linalg.svdvals(delta).cpu().numpy()
        except Exception:
            sv = np.array([])
        sv_spectrum[par] = sv.tolist()
    return delta_norm, sv_spectrum


def measure_cka_and_layer_cosine(model, x1, x2, device):
    """(1) CKA on per-scale pooled image features; (2) per-block cosine."""
    # Resize x1 to the model's encoder input size for ViT forward
    enc = int(model.encoder_input_size)
    x_resized = F.interpolate(x1, size=(enc, enc), mode="bilinear", align_corners=False)
    if model.use_imagenet_norm:
        x_resized = (x_resized - model.mean_rgb) / model.std_rgb

    # LoRA on
    _, acts_on = run_vit_with_hook(model, x_resized, device)
    # Per-scale pooled features (post scale_pools, pre MLP)
    pooled_on = _pooled_per_scale(model, x_resized)

    # LoRA off
    saved = save_lora_scaling(model)
    zero_lora_scaling(model)
    try:
        _, acts_off = run_vit_with_hook(model, x_resized, device)
        pooled_off = _pooled_per_scale(model, x_resized)
    finally:
        restore_lora_scaling(model, saved)

    # CKA on per-scale pooled features (post scale_pools, pre MLP) — shape (B, D)
    cka_per_scale = []
    for s in range(len(pooled_on)):
        cka_per_scale.append(linear_cka(pooled_on[s], pooled_off[s]))

    # Layer-wise mean cosine over (B, tokens)
    layer_cos = []
    for i in sorted(acts_on.keys()):
        a = acts_on[i].flatten(0, 1)   # (B*N, D)
        b = acts_off[i].flatten(0, 1)
        c = F.cosine_similarity(a, b, dim=-1).mean()
        layer_cos.append({"block": int(i), "cosine": float(c)})
    return cka_per_scale, layer_cos


@torch.no_grad()
def _pooled_per_scale(model, x_resized):
    """Per-scale post-MHA pre-MLP token: (B, D). Same path as forward except
    we stop before scale_mlps. Returns list of cpu tensors."""
    vit = model.image_encoder.model
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x_resized.is_cuda):
        tokens_all = vit.forward_features(x_resized)
    patches = tokens_all[:, 1:, :].float()
    P = model.encoder_input_size // 16
    B = x_resized.shape[0]
    D = patches.shape[-1]
    patches_2d = patches.transpose(1, 2).contiguous().view(B, D, P, P)
    out = []
    for s in range(model.num_resolutions):
        g = 2 ** (model.num_resolutions - 1 - s)
        region = F.adaptive_avg_pool2d(patches_2d, g).flatten(2).transpose(1, 2)  # (B, g^2, D)
        q = model.scale_queries[s].expand(B, -1, -1)
        pooled, _ = model.scale_pools[s](q, region, region)
        out.append(pooled.squeeze(1).cpu().float())
    return out


def measure_alignment(model, x1, x2, device, temperature: float):
    """(3) Image/spatial alignment with LoRA on vs off."""
    model.eval()
    with torch.no_grad():
        out_on = model(x1, x2)
    saved = save_lora_scaling(model)
    zero_lora_scaling(model)
    try:
        with torch.no_grad():
            out_off = model(x1, x2)
    finally:
        restore_lora_scaling(model, saved)

    align_on = compute_alignment(out_on["image_embedding"], out_on["spatial_embedding"])
    align_off = compute_alignment(out_off["image_embedding"], out_off["spatial_embedding"])
    return align_on, align_off


def measure_grad_norms(model, x1, x2, temperature: float, device):
    """(5) Run one forward+backward with the symmetric InfoNCE; collect grad norms."""
    # Make sure only LoRA + sigmma heads need_grad; freeze ViT base layers.
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    model.train()
    model.zero_grad(set_to_none=True)
    contrastive = symmetric_InfoNCE(temperature=temperature).to(device)
    out = model(x1, x2)
    loss = sum(contrastive(out["image_embedding"][s], out["spatial_embedding"][s])
               for s in range(len(out["image_embedding"])))
    loss.backward()

    grad_norms = {}
    for n, p in model.named_parameters():
        if "lora_" in n and p.grad is not None:
            grad_norms[n] = float(p.grad.detach().float().pow(2).sum().sqrt())
    return float(loss.detach()), grad_norms


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", default="configs/in-house_train_sigmma_dense_attnpool.yaml")
    ap.add_argument("--ckpt", default="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool/23617/checkpoint-step1000.pth")
    ap.add_argument("--out_dir", default="outputs/diagnostics/dense_attnpool_23617")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest_override", default=None,
                    help="Path to a CSV overriding cfg.dataset.train.manifest_path. "
                         "Use a 1-row file to skip the ~25min full-manifest slide load.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ConfigDict(yaml.safe_load(open(args.train_config)))
    # Override checkpoint path to point at the ckpt of interest.
    cfg.model.checkpoint_path = args.ckpt
    if args.manifest_override:
        cfg.dataset.train.manifest_path = args.manifest_override

    print(f"[load] model from {args.ckpt}", flush=True)
    model = load_model(cfg, pretrained_image_encoder=cfg.model.pretrained_image_encoder,
                       pretrained_ckpt_path=args.ckpt, train=False)
    # Aggressive device placement. load_checkpoint(map_location=device) restores
    # parameters to device, but `_orig_mod` references inside peft tuner layers
    # (the saved base_layer.weight reference) can be left on CPU after
    # load_state_dict. We move every parameter + buffer + the `module._buffers`
    # of every submodule explicitly.
    model = model.to(device)
    moved = 0
    cpu_params_before = []
    for n, p in model.named_parameters():
        if p.device.type != "cuda":
            cpu_params_before.append(n)
        p.data = p.data.to(device, non_blocking=False)
        moved += 1
    for n, b in model.named_buffers():
        if b.device.type != "cuda":
            cpu_params_before.append(f"[buf] {n}")
        b.data = b.data.to(device, non_blocking=False)
    print(f"[device] forced {moved} params + buffers to {device}; "
          f"were-cpu pre-fix: {len(cpu_params_before)} (first 5: {cpu_params_before[:5]})", flush=True)
    # Final sanity sweep
    remaining_cpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    remaining_cpu_buf = [n for n, b in model.named_buffers() if b.device.type != "cuda"]
    if remaining_cpu or remaining_cpu_buf:
        print(f"[device][warn] still CPU: params={remaining_cpu[:5]} bufs={remaining_cpu_buf[:5]}",
              flush=True)
    model.eval()

    # ------- (4) + (6): state-dict-only measurements -------
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    lora_r = int(cfg.model.lora.r)
    lora_alpha = int(cfg.model.lora.alpha)
    delta_norm, sv_spectrum = measure_state_dict_only(sd, lora_r, lora_alpha)
    print(f"[4] LoRA modules with delta-norm: {len(delta_norm)}", flush=True)

    # ------- build one batch -------
    print(f"[batch] B={args.batch_size}", flush=True)
    batch = make_batch(cfg.dataset.train, args.batch_size, device, seed=args.seed)
    # Move image to device; spatial_graph (DGL) goes via its own .to.
    x1 = batch["image"].to(device)
    x2 = batch["spatial_graph"].to(device)

    # ------- (1) + (2): CKA + per-block cosine -------
    cka_per_scale, layer_cos = measure_cka_and_layer_cosine(model, x1, x2, device)
    print(f"[1] CKA(LoRA-on, LoRA-off) per scale: {[f'{c:.3f}' for c in cka_per_scale]}", flush=True)
    sample_blocks = layer_cos[:3] + layer_cos[-3:]
    sample_str = " ".join(f"b{r['block']}={r['cosine']:.3f}" for r in sample_blocks)
    print(f"[2] per-block cosine (first/last 3): {sample_str}", flush=True)

    # ------- (3) alignment with LoRA on/off -------
    T = float(cfg.model.get("contrastive_temperature", 1.0))
    align_on, align_off = measure_alignment(model, x1, x2, device, T)
    def fmt_align(arr):
        return " | ".join(
            f"s{i}: cos={a['mean_paired_cosine']:.3f} acc={a['infonce_top1_i2s']:.3f}"
            for i, a in enumerate(arr)
        )
    print(f"[3] alignment (LoRA on)  per scale: {fmt_align(align_on)}", flush=True)
    print(f"[3] alignment (LoRA off) per scale: {fmt_align(align_off)}", flush=True)

    # ------- (5) grad norms via one forward+backward -------
    # Rebuild fresh batch on device (the alignment forwards may have moved state).
    loss_val, grad_norms = measure_grad_norms(model, x1, x2, T, device)
    print(f"[5] InfoNCE loss = {loss_val:.4f}; grad norms collected for {len(grad_norms)} params", flush=True)

    # ------- write everything -------
    record = {
        "ckpt": args.ckpt,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "contrastive_temperature": T,
        "1_cka_per_scale": [{"scale": s, "linear_cka": c} for s, c in enumerate(cka_per_scale)],
        "2_per_block_cosine": layer_cos,
        "3_alignment_lora_on": align_on,
        "3_alignment_lora_off": align_off,
        "4_delta_norm": delta_norm,
        "5_loss": loss_val,
        "5_grad_norms": grad_norms,
        "6_sv_spectrum": sv_spectrum,
    }
    with open(out_dir / "diagnose.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"[done] {out_dir / 'diagnose.json'}", flush=True)


if __name__ == "__main__":
    main()
