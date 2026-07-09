"""Test 1 — alignment-localization ablation for retrieval.

Two subcommands run the standard retrieval pipeline (same as
``scripts/benchmark_retrieval.py``) on a model whose weights have been
selectively reverted between ``load_model()`` and ``model.eval()``.

Test 1a — ``sigmma-no-lora``: walk every PEFT LoraLayer and zero the
``scaling`` entry for the active adapter(s) so the LoRA delta is
effectively disabled while keeping the trained image_projection_heads and
spatial encoder intact.

Test 1b — ``bleep-uni-reset``: load the trained baseline-UNI checkpoint as
usual, then overwrite the inner timm vit_large_patch16_224 state dict with
the original pretrained UNI weights, preserving image_projection and
spot_projection.

Outputs ``recall_metrics.json`` under ``out_dir``. The decision rules in
``plans/floofy-enchanting-avalanche.md`` apply.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from ml_collections.config_dict import ConfigDict
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.data import DataLoader, SequentialSampler

import numpy as np

from sigmma.datasets import multimodal_collate
from sigmma.utils import (
    eval_retrieval,
    load_dataset,
    load_model,
)


def disable_lora_scaling(model: torch.nn.Module) -> int:
    """Zero the per-adapter scaling on every BaseTunerLayer inside ``model``.

    Returns the number of LoRA layers touched. The original scaling is not
    restored — call only on a transient model instance.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, BaseTunerLayer) and hasattr(m, "scaling"):
            for k in list(m.scaling.keys()):
                m.scaling[k] = 0.0
            n += 1
    return n


def reset_uni_backbone(model: torch.nn.Module, uni_ckpt_dir: str) -> dict:
    """Find the ``_UNIImageBackbone`` inside ``model`` and overwrite its
    inner timm ViT-L state dict with the pretrained UNI weights at
    ``<uni_ckpt_dir>/pytorch_model.bin``.

    Returns a dict with ``missing_keys`` and ``unexpected_keys`` counts.
    """
    raw_sd = torch.load(
        Path(uni_ckpt_dir, "pytorch_model.bin"), map_location="cpu"
    )

    target = None
    for _, sub in model.named_modules():
        if (
            sub.__class__.__name__ == "_UNIImageBackbone"
            and hasattr(sub, "uni")
            and hasattr(sub.uni, "model")
        ):
            target = sub
            break
    if target is None:
        raise RuntimeError("Could not find _UNIImageBackbone in model")

    msg = target.uni.model.load_state_dict(raw_sd, strict=False)
    return {
        "missing_keys": len(msg.missing_keys),
        "unexpected_keys": len(msg.unexpected_keys),
        "missing_sample": msg.missing_keys[:5],
        "unexpected_sample": msg.unexpected_keys[:5],
    }


def _load_embeddings_fp32(model, dataloader, device, config) -> tuple:
    """Mirror of ``sigmma.evaluation.evaluate_retrieval.load_embeddings`` but
    with autocast disabled. DGL's CPU segment_reduce kernel does not support
    BF16 in the current install, and the SIGMMA spatial branch's
    ``softmax_nodes`` hits that kernel under autocast — running fp32 sidesteps
    the issue at the cost of ~2x runtime.
    """
    model_name = config.model.get("name")
    img_li, spat_li = [], []
    for batch in dataloader:
        img_tensor = batch.get("image")
        if not isinstance(img_tensor, torch.Tensor) or img_tensor.numel() == 0:
            continue
        image = img_tensor.to(device)
        if config.model.name == "sigmma":
            spatial = batch["spatial_graph"].to(device)
            inputs = [image, spatial]
        elif model_name in ["openAI_CLIP", "PLIP", "loki_OmiCLIP"]:
            inputs = [image, batch["spatial_text"]]
        elif model_name == "BLEEP":
            inputs = [image, batch["spatial"].to(device)]
        else:
            raise ValueError(f"Unknown model {model_name}")

        with torch.no_grad():
            out = model(*inputs)
        img_emb = out["image_embedding"]
        spat_emb = out["spatial_embedding"]

        if config.model.name == "sigmma":
            scale_idx = (
                int(config.test.get("scale_idx", -1)) if "test" in config else -1
            )
            img_emb = img_emb[scale_idx]
            spat_emb = spat_emb[scale_idx]
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        spat_emb = spat_emb / spat_emb.norm(dim=-1, keepdim=True)
        img_li.append(img_emb.detach().cpu())
        spat_li.append(spat_emb.detach().cpu())
    return torch.cat(img_li, dim=0), torch.cat(spat_li, dim=0)


def _retrieval_score_fp32(model, q_loader, r_loader, device, config):
    img_q, spat_q = _load_embeddings_fp32(model, q_loader, device, config)
    img_r, spat_r = _load_embeddings_fp32(model, r_loader, device, config)
    sim_i2s = (img_q @ spat_r.T).numpy()
    sim_s2i = (spat_q @ img_r.T).numpy()
    return sim_i2s, sim_s2i


def run_retrieval(config, model, device, out_dir) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model_name = config.model.get("name")

    # Section-split: query and reference are the same dataset, evaluated
    # against itself with FOV-level retrieval (same as benchmark_retrieval.py
    # for ``split_type == "section_split"``).
    query_dataset = load_dataset(
        config.dataset.query_dataset, is_train=False, model_name=model_name
    )
    q_sampler = SequentialSampler(query_dataset)
    q_loader = DataLoader(
        query_dataset,
        batch_size=config.test.batch_size,
        sampler=q_sampler,
        num_workers=config.test.num_workers,
        collate_fn=multimodal_collate,
    )
    ref_dataset = copy.deepcopy(query_dataset)
    r_sampler = SequentialSampler(ref_dataset)
    r_loader = DataLoader(
        ref_dataset,
        batch_size=config.test.batch_size,
        sampler=r_sampler,
        num_workers=config.test.num_workers,
        collate_fn=multimodal_collate,
    )

    model.eval()
    scores_i2s, scores_s2i = _retrieval_score_fp32(
        model, q_loader, r_loader, device, config
    )
    result = eval_retrieval(scores_i2s, scores_s2i)

    out_path = os.path.join(out_dir, "recall_metrics.json")
    json.dump({k: float(v) for k, v in result.items()}, open(out_path, "w"), indent=2)
    print(f"\n[ablation] wrote {out_path}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def cmd_sigmma_no_lora(args) -> None:
    cfg = yaml.load(open(args.config_file), yaml.FullLoader)
    config = ConfigDict(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available — SIGMMA retrieval on CPU is infeasible. "
            "Check LSF GPU allocation."
        )
    device = torch.device("cuda")

    model = load_model(
        config,
        pretrained_image_encoder=config.model.get("pretrained_image_encoder", "UNI"),
        pretrained_ckpt_path=config.model.get("checkpoint_path", None),
        train=False,
    )

    if args.disable_lora:
        n = disable_lora_scaling(model)
        print(f"[ablation] disabled LoRA scaling on {n} layers")
    else:
        print("[ablation] sanity mode: LoRA enabled (no mutation)")

    model.to(device)
    run_retrieval(config, model, device, args.out_dir)


def cmd_bleep_uni_reset(args) -> None:
    cfg = yaml.load(open(args.config_file), yaml.FullLoader)
    config = ConfigDict(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available — BLEEP-UNI retrieval on CPU is infeasible. "
            "Check LSF GPU allocation."
        )
    device = torch.device("cuda")

    model = load_model(
        config,
        pretrained_image_encoder=config.model.get("pretrained_image_encoder", None),
        pretrained_ckpt_path=config.model.get("checkpoint_path", None),
        train=False,
    )

    if args.reset_uni:
        info = reset_uni_backbone(model, args.uni_ckpt_dir)
        print(f"[ablation] reset UNI backbone: {info}")
    else:
        print("[ablation] sanity mode: trained UNI retained (no mutation)")

    model.to(device)
    run_retrieval(config, model, device, args.out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="subcommand", required=True)

    p1 = sub.add_parser("sigmma-no-lora")
    p1.add_argument("--config_file", required=True)
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--disable_lora", action="store_true", default=True)
    p1.add_argument("--no_disable_lora", dest="disable_lora", action="store_false")
    p1.set_defaults(func=cmd_sigmma_no_lora)

    p2 = sub.add_parser("bleep-uni-reset")
    p2.add_argument("--config_file", required=True)
    p2.add_argument("--out_dir", required=True)
    p2.add_argument(
        "--uni_ckpt_dir",
        default="/nfs/team361/dj16/projects/SIGMMA/assets/ckpts/uni_h/",
    )
    p2.add_argument("--reset_uni", action="store_true", default=True)
    p2.add_argument("--no_reset_uni", dest="reset_uni", action="store_false")
    p2.set_defaults(func=cmd_bleep_uni_reset)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
