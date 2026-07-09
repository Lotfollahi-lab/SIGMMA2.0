import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from ml_collections.config_dict import ConfigDict
import torch.distributed as dist

from sigmma.datasets.dataset import MultiSectionDataset
from sigmma.model import sigmma
from sigmma.model import UNI, ResNet50
from sigmma.model.models_image_backbones import H_optimus_zero, apply_lora_to_backbone
from sigmma.model import symmetric_InfoNCE, bleep_cross_entropy
from sigmma.model.losses import GEXAwareSoftContrastiveLoss
from sigmma.baselines.openai_clip import openAI_CLIP
from sigmma.baselines.plip import PLIP
from sigmma.baselines.omiclip import loki_OmiCLIP
from sigmma.baselines.bleep import BLEEP
from ..evaluation.evaluate_retrieval import retrieval_score, eval_retrieval

from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

import wandb

_IMAGE_BACKBONE_REGISTRY = {
    "UNI": lambda cfg: UNI(checkpoint_path=cfg.model.pretrained_image_encoder_ckpt_path),
    "H_optimus_zero": lambda cfg: H_optimus_zero(
        checkpoint_path=cfg.model.get("pretrained_image_encoder_ckpt_path", None) or None
    ),
    "ResNet50": lambda cfg: ResNet50(),
}

def load_dataset(config_dataset, is_train=False, model_name=None):
    """
    Load the training and evaluation datasets based on the configuration.
    
    Args:
        config_dataset: Configuration for the dataset
        is_train: Whether this is a training dataset (for data augmentation)
        model_name: Name of the model being used (e.g., 'BLEEP')
    """
    dataset = MultiSectionDataset(manifest_path=Path(config_dataset.manifest_path),
                                  num_cell_types=config_dataset.get("num_cell_types", None),
                                  fov_size=config_dataset.get("fov_size", 256),
                                  number_of_genes=config_dataset.get("number_of_genes", 50),
                                  num_resolutions=config_dataset.get("num_resolutions", 3),
                                  hierarchical_graph=config_dataset.get("hierarchical_graph", True),
                                  housekeeping_genes=config_dataset.get("housekeeping_genes", None),
                                  tissue_filtering_threshold=config_dataset.get("tissue_filtering_threshold", 20000),
                                  pretrained_cell_embedding=config_dataset.pretrained_cell_embedding,
                                  type_pretrained_cell_embedding=config_dataset.type_pretrained_cell_embedding,
                                  is_train=is_train,
                                  model_name=model_name,
                                  hier_cache_dir=config_dataset.get("hier_cache_dir", None),
                                  raw_gex_panel_npy=config_dataset.get("raw_gex_panel_npy", None),
                                  svg_panel_dir=config_dataset.get("svg_panel_dir", None),
                                  panel_index_json=config_dataset.get("panel_index_json", None),
                                  k_max=int(config_dataset.get("k_max", 100)),
                                  use_full_panel=bool(config_dataset.get("use_full_panel", False)),
                                  gex_lognorm=bool(config_dataset.get("gex_lognorm", False)),
                                  gex_lognorm_target_sum=float(config_dataset.get("gex_lognorm_target_sum", 1e4)))

    return dataset


def load_model(config:ConfigDict, 
               pretrained_image_encoder:str="UNI",
               pretrained_ckpt_path:Optional[str]=None,
               train: Optional[bool]=True,
               rank:Optional[int]=None):
    """
    Load the model, optimizer, and learning rate scheduler based on the configuration.
    """
    if config.model.name == "sigmma":
        if pretrained_image_encoder not in _IMAGE_BACKBONE_REGISTRY:
            raise NotImplementedError(
                f"Pretrained image encoder '{pretrained_image_encoder}' is not implemented. "
                f"Available: {sorted(_IMAGE_BACKBONE_REGISTRY)}"
            )
        if pretrained_image_encoder == "UNI":
            pretrained_ckpt_dir = config.model.pretrained_image_encoder_ckpt_path
            assert Path(pretrained_ckpt_dir).exists(), f"Pretrained image encoder checkpoint directory {pretrained_ckpt_dir} does not exist."
        pretrained_image_encoder = _IMAGE_BACKBONE_REGISTRY[pretrained_image_encoder](config)

        lora_cfg = config.model.get("lora", None)
        if lora_cfg is not None and bool(lora_cfg.get("enabled", False)):
            num_adapters = int(config.model.num_resolutions) if bool(lora_cfg.get("per_scale", False)) else 1
            pretrained_image_encoder = apply_lora_to_backbone(pretrained_image_encoder, lora_cfg, num_adapters=num_adapters)

        model = sigmma(config.model, pretrained_image_encoder)

    elif config.model.name == "openAI_CLIP":
        model = openAI_CLIP(
            hf_model_id=config.model.get("hf_model_id", "openai/clip-vit-base-patch32"),
            freeze_encoders=not bool(train),
            loss_temperature=(float(config.model.contrastive_temperature)
                              if train and config.model.get("contrastive_temperature") is not None
                              else None),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    elif config.model.name == "PLIP":
        model = PLIP(
            hf_model_id=config.model.get("hf_model_id", "vinid/plip"),
            freeze_encoders=not bool(train),
            loss_temperature=(float(config.model.contrastive_temperature)
                              if train and config.model.get("contrastive_temperature") is not None
                              else None),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    elif config.model.name == "loki_OmiCLIP":
        ckpt = config.model.pretrained_ckpt_path
        assert Path(ckpt).exists(), f"OmiCLIP checkpoint not found at {ckpt}"
        model = loki_OmiCLIP(
            pretrained_ckpt_path=str(ckpt),
            freeze_encoders=not bool(train),
            loss_temperature=(float(config.model.contrastive_temperature)
                              if train and config.model.get("contrastive_temperature") is not None
                              else None),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    elif config.model.name == "BLEEP":
        model = BLEEP(
            spot_input_dim=int(config.model.spot_input_dim),
            image_embedding_dim=int(config.model.get("image_embedding_dim", 2048)),
            projection_dim=int(config.model.get("projection_dim", 256)),
            spot_hidden_dim=int(config.model.get("spot_hidden_dim", 512)),
            temperature=float(config.model.get("contrastive_temperature", 1.0)),
            dropout=float(config.model.get("dropout", 0.1)),
            image_size=int(config.model.get("image_size", 224)),
            pretrained_image_encoder=bool(config.model.get("pretrained_image_encoder_imagenet", True)),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")

    if rank is not None:
        model = model.to(torch.cuda.current_device())
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    if int(os.environ.get("RANK", "0")) == 0:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"{config.model.name} parameters: trainable={n_trainable:,} / total={n_total:,} ({100.0 * n_trainable / max(n_total, 1):.2f}%)")

    if train:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        if config.train.lr_scheduler == "constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=config.train.max_steps)

        elif config.train.lr_scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.train.lr_step_size,
                gamma=config.train.lr_gamma,
        )

        elif config.train.lr_scheduler == "warmup_cosine":
            warmup_steps = int(config.train.get("lr_warmup_steps", 0))
            min_lr = float(config.train.get("lr_min", 0.0))
            base_lr = float(config.train.learning_rate)
            min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=max(1, warmup_steps),
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(config.train.max_steps) - warmup_steps),
                eta_min=min_lr,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[max(1, warmup_steps)],
            )

        return model, optimizer, lr_scheduler

    elif pretrained_ckpt_path is not None:
        assert Path(pretrained_ckpt_path).exists(), f"Pretrained checkpoint path {pretrained_ckpt_path} does not exist."
        load_checkpoint(model, None, None, Path(pretrained_ckpt_path), device)
        return model

    else:
        return model


def train(
    config,
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    max_steps,
    device,
    wandb_logger,
):
    """
    Step-based training loop. Runs until global_step reaches max_steps,
    cycling through the dataloader as needed.
    """
    model.train()

    # Identify rank in DDP setting
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        wandb.watch(model, log=None, log_freq=10)

    accum_freq = config.get("train.gradient_accumulation_steps", 1)
    if accum_freq > 1:
        accum_images, accum_spatial, accum_features = [], [], {}
        accum_raw_gex = []
        accum_raw_gex_valid_mask = []
        accum_panel_ids = []

    global_step = 0
    i = -1
    pass_index = 0
    steps_per_pass = len(train_dataloader)
    train_sampler = getattr(train_dataloader, "sampler", None)
    # Sync per-batch skips across DDP ranks — a unilateral `continue` desyncs
    # the next DDP allreduce/eval all_gather and hangs NCCL.
    train_is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    while global_step < max_steps:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(pass_index)
        for batch in train_dataloader:
            if global_step >= max_steps:
                break
            local_skip = int(
                not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0
            )
            if train_is_distributed:
                skip_t = torch.tensor([local_skip], device=device, dtype=torch.long)
                dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                if int(skip_t.item()) == 1:
                    continue
            elif local_skip:
                continue
            i += 1
            image = batch["image"].to(device)
            if config.model.name == "sigmma":
                spatial = batch['spatial_graph'].to(device)
                inputs = [image, spatial]
            elif config.model.name == "BLEEP":
                spatial = batch['spatial'].to(device)
                inputs = [image, spatial]
            elif config.model.name in ("openAI_CLIP", "PLIP", "loki_OmiCLIP"):
                spatial = batch["spatial_text"]
                inputs = [image, spatial]
            else:
                raise NotImplementedError(f"Model {config.model.name} is not implemented for training.")

            # Gradient accumulation code adapted from OpenCLIP (https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/train.py)
            if accum_freq == 1:
                # Forward pass
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
                    model_out = model(*inputs)
                    num_resolutions = len(model_out['image_embedding'])
                if config.model.name == "sigmma":
                    loss = 0
                    gex_cfg = config.model.get("gex_aware_soft_loss", None)
                    use_gex_aware = bool(gex_cfg is not None and gex_cfg.get("enabled", False))
                    if use_gex_aware:
                        contrastive_loss = GEXAwareSoftContrastiveLoss(
                            tau=float(config.model.contrastive_temperature),
                            tau_gex=float(gex_cfg.get("tau_gex", 0.1)),
                            alpha=float(gex_cfg.get("alpha", 0.2)),
                            mode=str(gex_cfg.get("mode", "topk")),
                            topk=int(gex_cfg.get("topk", 5)),
                            gex_preprocess=str(gex_cfg.get("gex_preprocess", "log1p_zscore")),
                            clamp_negative_gex_sim=bool(gex_cfg.get("clamp_negative_gex_sim", True)),
                            symmetric=bool(gex_cfg.get("symmetric", True)),
                            detach_gex_targets=bool(gex_cfg.get("detach_gex_targets", True)),
                        )
                        gex_batch = batch["raw_gex"].to(device)
                        # Panel-aware extras (None when legacy raw_gex_panel_npy used)
                        gex_mask_batch = batch.get("raw_gex_valid_mask")
                        if gex_mask_batch is not None and gex_mask_batch.numel() > 0:
                            gex_mask_batch = gex_mask_batch.to(device)
                        else:
                            gex_mask_batch = None
                        panel_ids_batch = batch.get("panel_id")
                        if panel_ids_batch is not None and panel_ids_batch.numel() > 0:
                            panel_ids_batch = panel_ids_batch.to(device)
                            # int default (-1) is the "no panel" sentinel — panel-aware
                            # path only activates if all entries are >=0.
                            if (panel_ids_batch < 0).any():
                                panel_ids_batch = None
                        else:
                            panel_ids_batch = None
                        for s in range(num_resolutions):
                            loss += contrastive_loss(
                                model_out['image_embedding'][s],
                                model_out['spatial_embedding'][s],
                                gex=gex_batch,
                                raw_gex_valid_mask=gex_mask_batch,
                                panel_ids=panel_ids_batch,
                            )
                    else:
                        contrastive_loss = symmetric_InfoNCE(
                            temperature=float(config.model.contrastive_temperature)
                        )
                        for s in range(num_resolutions):
                            loss += contrastive_loss(model_out['image_embedding'][s], model_out['spatial_embedding'][s])
                    loss = loss / num_resolutions
                    image_embedding = model_out["image_embedding"]
                    spatial_embedding = model_out["spatial_embedding"]
                else:
                    if rank == 0:
                        print('Loss', model_out['loss'])
                    loss = model_out["loss"]
                    image_embedding = model_out["image_embedding"] # Tensor (B, D)
                    spatial_embedding = model_out["spatial_embedding"] # Tensor (B, D)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            else:
                with torch.no_grad():
                    # Forward pass
                    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
                        model_out = model(*inputs)

                    image_embedding = model_out["image_embedding"] # List of Tensors [(B, D), (B, D), ...] or Tensor (B, D)
                    spatial_embedding = model_out["spatial_embedding"] # List of Tensors [(B, D), (B, D), ...] or Tensor (B, D)

                    accum_images.append(image)
                    accum_spatial.append(spatial)
                    if config.model.name == "sigmma":
                        accum_raw_gex.append(batch["raw_gex"].to(device))
                        # Panel-aware extras (graceful if missing)
                        rg_mask = batch.get("raw_gex_valid_mask")
                        if rg_mask is not None and rg_mask.numel() > 0:
                            accum_raw_gex_valid_mask.append(rg_mask.to(device))
                        pid_b = batch.get("panel_id")
                        if pid_b is not None and pid_b.numel() > 0:
                            accum_panel_ids.append(pid_b.to(device))
                        feat_keys = ("image_embedding", "spatial_embedding")
                    else:
                        # Baselines: cache loss-space embeddings (BLEEP's loss uses
                        # UNnormalized projections, so we cannot reuse the normalized
                        # "image_embedding"). The grad step pools these into the full
                        # effective-batch contrastive matrix.
                        feat_keys = ("image_embedding_for_loss", "spatial_embedding_for_loss")
                    for key in feat_keys:
                        if key not in accum_features:
                            accum_features[key] = [model_out[key]]
                        else:
                            accum_features[key].append(model_out[key])

                if (i+1) % accum_freq > 0:
                    if rank == 0:
                        print(f"Skipping optimizer step at iter {i}, accumulating more gradients...")
                    continue

                # Take gradients for the last accum_freq batches
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                #
                # DDP `no_sync` optimisation: when there are more than one accum
                # micro-batches AND the model is DDP-wrapped, wrap all but the
                # FINAL micro-batch's backward in `model.no_sync()`. This skips
                # the per-backward gradient all-reduce; the single sync at the
                # final backward all-reduces the accumulated bucket contents,
                # giving the same mathematical result
                #   avg_ranks(sum_j local_grad_j) == sum_j avg_ranks(local_grad_j)
                # but at ~1/accum_freq the gradient-comm volume.
                # `nullcontext` is used in the non-DDP path (single-GPU) and on
                # the last iteration (which must trigger the sync).
                optimizer.zero_grad()
                total_loss = None
                n_accum = len(accum_images)
                ddp_has_no_sync = hasattr(model, "no_sync")
                for j in range(n_accum):
                    image = to_device_if_tensor(accum_images[j], device)
                    spatial = to_device_if_tensor(accum_spatial[j], device)
                    is_last_accum = (j == n_accum - 1)
                    sync_ctx = (
                        model.no_sync()
                        if ddp_has_no_sync and not is_last_accum
                        else nullcontext()
                    )
                    with sync_ctx:
                        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
                            model_out = model(image, spatial)
                            inputs_for_loss = {}
                            if config.model.name == "sigmma":
                                loss = 0
                                num_resolutions = len(model_out['image_embedding'])
                                gex_cfg = config.model.get("gex_aware_soft_loss", None)
                                use_gex_aware = bool(
                                    gex_cfg is not None and gex_cfg.get("enabled", False)
                                )
                                if use_gex_aware:
                                    contrastive_loss = GEXAwareSoftContrastiveLoss(
                                        tau=float(config.model.contrastive_temperature),
                                        tau_gex=float(gex_cfg.get("tau_gex", 0.1)),
                                        alpha=float(gex_cfg.get("alpha", 0.2)),
                                        mode=str(gex_cfg.get("mode", "topk")),
                                        topk=int(gex_cfg.get("topk", 5)),
                                        gex_preprocess=str(gex_cfg.get("gex_preprocess", "log1p_zscore")),
                                        clamp_negative_gex_sim=bool(gex_cfg.get("clamp_negative_gex_sim", True)),
                                        symmetric=bool(gex_cfg.get("symmetric", True)),
                                        detach_gex_targets=bool(gex_cfg.get("detach_gex_targets", True)),
                                    )
                                    # raw_gex has no gradient dependence; concat across
                                    # accum batches in the original ordering so diagonal
                                    # positions match image/spatial concat.
                                    gex_cat = torch.cat(accum_raw_gex, dim=0)
                                    # Panel-aware extras: only valid if every accum
                                    # batch provided them (else fall back to None).
                                    if len(accum_raw_gex_valid_mask) == len(accum_raw_gex):
                                        gex_mask_cat = torch.cat(accum_raw_gex_valid_mask, dim=0)
                                    else:
                                        gex_mask_cat = None
                                    if len(accum_panel_ids) == len(accum_raw_gex):
                                        panel_ids_cat = torch.cat(accum_panel_ids, dim=0)
                                        if (panel_ids_cat < 0).any():
                                            panel_ids_cat = None
                                    else:
                                        panel_ids_cat = None
                                else:
                                    contrastive_loss = symmetric_InfoNCE(
                                        temperature=float(config.model.contrastive_temperature)
                                    )
                                    gex_cat = None
                                    gex_mask_cat = None
                                    panel_ids_cat = None
                                for s in range(num_resolutions):
                                    for key in accum_features:
                                        accumulated = accum_features[key]
                                        parts = [accumulated[k][s] for k in range(j)] + [model_out[key][s]] + [accumulated[k][s] for k in range(j+1, len(accumulated))]
                                        inputs_for_loss[key] = torch.cat(parts)
                                    if use_gex_aware:
                                        loss += contrastive_loss(
                                            inputs_for_loss['image_embedding'],
                                            inputs_for_loss['spatial_embedding'],
                                            gex=gex_cat,
                                            raw_gex_valid_mask=gex_mask_cat,
                                            panel_ids=panel_ids_cat,
                                        )
                                    else:
                                        loss += contrastive_loss(
                                            inputs_for_loss['image_embedding'],
                                            inputs_for_loss['spatial_embedding'],
                                        )
                                loss = loss / num_resolutions
                                losses = {'contrastive_loss': loss}
                                total_loss = sum(losses.values()) #+ model_out["loss_csc_img"] #+ model_out["loss_kl"]
                                losses['loss'] = total_loss
                            else:
                                # Baselines: GradCache-style negative pooling. The
                                # current micro-batch j carries gradients; the other
                                # accumulated micro-batches are cached/detached
                                # negatives. Compute the model's own contrastive loss
                                # over the concatenated full effective batch.
                                base_model = model.module if hasattr(model, "module") else model
                                pooled = {}
                                for key in ("image_embedding_for_loss", "spatial_embedding_for_loss"):
                                    cached = accum_features[key]
                                    parts = (
                                        [cached[k] for k in range(j)]
                                        + [model_out[key]]
                                        + [cached[k] for k in range(j + 1, len(cached))]
                                    )
                                    pooled[key] = torch.cat(parts)
                                loss = base_model.contrastive_loss_from_embeddings(
                                    pooled["image_embedding_for_loss"],
                                    pooled["spatial_embedding_for_loss"],
                                )
                                losses = {'contrastive_loss': loss}
                                total_loss = loss
                                losses['loss'] = total_loss
                            del inputs_for_loss
                        total_loss.backward()
                loss = total_loss.item()

                optimizer.step()
                lr_scheduler.step()

                # Reset accumulators
                if accum_freq > 1:
                    accum_images, accum_spatial, accum_features = [], [], {}
                    accum_raw_gex = []
                    accum_raw_gex_valid_mask = []
                    accum_panel_ids = []

            if (global_step + 1) % config.train.save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    config.outdir,
                    global_step + 1,
                )
            do_eval = eval_dataloader is not None and (global_step + 1) % config.train.eval_every == 0
            if do_eval:
                if rank == 0:
                    print(f"Evaluating at step {global_step + 1}...")
                model.eval()
                with torch.no_grad():
                    eval_loss, _, _, _, _, _ = evaluate(config, model, eval_dataloader, device)
                if rank == 0:
                    if wandb_logger is not None:
                        # Log eval loss first, separately
                        wandb_logger.update({"(eval)loss": eval_loss})

                    if config.model.name == "sigmma":
                        image_embedding_plot = image_embedding[-1]
                        spatial_embedding_plot = spatial_embedding[-1]
                    else:
                        image_embedding_plot = image_embedding
                        spatial_embedding_plot = spatial_embedding

                    # Plot learnt embeddings only on rank 0
                    fig_train_img2spatial_heatmap = heatmap(
                        image_embedding_plot.detach().cpu(),
                        spatial_embedding_plot.detach().cpu(),
                        xlabel="Image Embeddings",
                        ylabel="Spatial Embeddings",
                    )
                    fig_train_img2img_heatmap = heatmap(
                        image_embedding_plot.detach().cpu(),
                        image_embedding_plot.detach().cpu(),
                        xlabel="Image Embeddings",
                        ylabel="Image Embeddings",
                    )
                    fig_train_spatial2spatial_heatmap = heatmap(
                        spatial_embedding_plot.detach().cpu(),
                        spatial_embedding_plot.detach().cpu(),
                        xlabel="Spatial Embeddings",
                        ylabel="Spatial Embeddings",
                    )
                    plt.close('all')
                    if wandb_logger is not None:
                        wandb_logger.update({
                            "(train)img2spatial heatmap": wandb.Image(fig_train_img2spatial_heatmap),
                            "(train)img2img heatmap": wandb.Image(fig_train_img2img_heatmap),
                            "(train)spatial2spatial heatmap": wandb.Image(fig_train_spatial2spatial_heatmap),
                        })

                model.train()

            if wandb_logger is not None and rank == 0:
                wandb_logger.update({
                    "(train)loss": loss,
                    "(train)lr": lr_scheduler.get_last_lr()[0],
                    "epoch": (global_step + 1) / steps_per_pass
                })
                wandb_logger.set_step()

            global_step += 1
        pass_index += 1

    save_checkpoint(model,
                    optimizer,
                    lr_scheduler,
                    config.outdir,
                    global_step,
                )
    return global_step


def save_checkpoint(
    model, optimizer, lr_scheduler, output_dir, global_step, logger=None
):
    save_path = Path(output_dir) / f"checkpoint-step{global_step}.pth"

    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "global_step": global_step,
    }
    torch.save(checkpoint, save_path)

    if int(os.environ.get("RANK", "0")) == 0:
        if logger:
            logger.info(f"Checkpoint saved at {save_path}")
        else:
            print(f"Checkpoint saved at {save_path}")
    return save_path


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    is_ddp_checkpoint = False

    if len(state_dict) > 0:
        first_key = next(iter(state_dict))
        is_ddp_checkpoint = first_key.startswith('module.')

    model_is_ddp = hasattr(model, 'module')

    print(f"Checkpoint DDP: {is_ddp_checkpoint}, Model DDP: {model_is_ddp}")

    if is_ddp_checkpoint and not model_is_ddp:
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print("Removed 'module.' prefix from DDP checkpoint")
    elif not is_ddp_checkpoint and model_is_ddp:
        new_state_dict = {"module." + k: v for k, v in state_dict.items()}
        model.module.load_state_dict(new_state_dict, strict=True)
        print("Added 'module.' prefix for DDP model")
    else:
        if model_is_ddp:
            model.module.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
        print("Direct load (matching DDP status)")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    global_step = checkpoint.get("global_step", 0)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Loaded checkpoint from {checkpoint_path} at step {global_step}")
    return global_step


def evaluate(config, model, eval_dataloader, device):
    """
    Evaluate the model on the validation dataset.

    DDP-aware: when launched under torchrun, the eval dataloader is sharded by
    DistributedSampler, so each rank only forwards its shard. Per-rank
    embeddings are then all_gather'd before computing loss / score matrices.
    """
    model.eval()
    forward_module = getattr(model, "module", model)
    all_image_embeddings = []
    all_spatial_embeddings = []

    # Empty-batch skips must be synchronised across DDP ranks: each rank sees
    # the same number of batches (DistributedSampler pads), but per-batch
    # all-None samples differ — a unilateral `continue` desyncs the downstream
    # all_gather and hangs NCCL.
    skip_is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    with torch.no_grad():
        for batch in eval_dataloader:
            local_skip = int(
                not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0
            )
            if skip_is_distributed:
                skip_t = torch.tensor([local_skip], device=device, dtype=torch.long)
                dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                if int(skip_t.item()) == 1:
                    continue
            elif local_skip:
                continue
            image = batch["image"].to(device)
            if config.model.name == "sigmma":
                spatial = batch['spatial_graph'].to(device)
                inputs = [image, spatial]
            elif config.model.name == "BLEEP":
                spatial = batch['spatial'].to(device)
                inputs = [image, spatial]
            elif config.model.name in ("openAI_CLIP", "PLIP", "loki_OmiCLIP"):
                spatial = batch["spatial_text"]
                inputs = [image, spatial]
            else:
                raise NotImplementedError(f"Model {config.model.name} is not implemented for evaluation.")

            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
                model_out = forward_module(*inputs)
            image_embedding = model_out["image_embedding"]
            spatial_embedding = model_out["spatial_embedding"]

            all_image_embeddings.append(image_embedding)
            all_spatial_embeddings.append(spatial_embedding)

    is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _all_gather(t: torch.Tensor) -> torch.Tensor:
        # Per-rank dim-0 length can differ when multimodal_collate drops
        # None/empty samples asymmetrically across ranks, so we cannot assume
        # shape symmetry: gather sizes first, pad to max, then trim.
        world = dist.get_world_size()
        local_n = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_n) for _ in range(world)]
        dist.all_gather(sizes, local_n)
        sizes = [int(s.item()) for s in sizes]
        max_n = max(sizes)
        if max_n == 0:
            return t[:0].contiguous()
        pad_n = max_n - t.shape[0]
        if pad_n > 0:
            pad = torch.zeros((pad_n, *t.shape[1:]), device=t.device, dtype=t.dtype)
            t_padded = torch.cat([t, pad], dim=0)
        else:
            t_padded = t
        tensors = [torch.zeros_like(t_padded) for _ in range(world)]
        dist.all_gather(tensors, t_padded.contiguous())
        return torch.cat([tensors[i][: sizes[i]] for i in range(world)], dim=0)

    if config.model.name == "sigmma":
        num_resolutions = len(all_image_embeddings[0])
        local_image = [torch.concat([all_image_embeddings[i][s] for i in range(len(all_image_embeddings))], dim=0).to(device) for s in range(num_resolutions)]
        local_spatial = [torch.concat([all_spatial_embeddings[i][s] for i in range(len(all_spatial_embeddings))], dim=0).to(device) for s in range(num_resolutions)]
        if is_distributed:
            all_image_embeddings = [_all_gather(t) for t in local_image]
            all_spatial_embeddings = [_all_gather(t) for t in local_spatial]
        else:
            all_image_embeddings = local_image
            all_spatial_embeddings = local_spatial
        loss = 0
        contrastive_loss = symmetric_InfoNCE(
            temperature=float(config.model.contrastive_temperature)
        )
        for s in range(num_resolutions):
            loss += contrastive_loss(all_image_embeddings[s], all_spatial_embeddings[s])
        loss = loss / num_resolutions
    else:
        local_image = torch.concat(all_image_embeddings, dim=0).to(device)
        local_spatial = torch.concat(all_spatial_embeddings, dim=0).to(device)
        if is_distributed:
            all_image_embeddings = _all_gather(local_image)
            all_spatial_embeddings = _all_gather(local_spatial)
        else:
            all_image_embeddings = local_image
            all_spatial_embeddings = local_spatial
        loss = symmetric_InfoNCE(
            temperature=float(config.model.contrastive_temperature)
        )(all_image_embeddings, all_spatial_embeddings)

    # retrieval_score does its own forward pass over the dataloader, which
    # under DDP would yield rank-local query/reference matrices. Callers in
    # train() and train_model.py only consume eval_loss, so skip the redundant
    # second pass when distributed.
    if is_distributed:
        score_matrix_img2spatial = score_matrix_spatial2image = None
        eval_results = None
    else:
        score_matrix_img2spatial, score_matrix_spatial2image = retrieval_score(model, eval_dataloader, eval_dataloader, device, config)
        eval_results = eval_retrieval(score_matrix_img2spatial, score_matrix_spatial2image)
    return loss, all_image_embeddings, all_spatial_embeddings, score_matrix_img2spatial, score_matrix_spatial2image, eval_results

def PCA_embeddings(image_embeddings, spatial_embeddings, num_samples=1000):
    """
    Plot the image and spatial embeddings using PCA.
    """
    pca = PCA(n_components=2)
    image_embeddings_pca = pca.fit_transform(image_embeddings[:num_samples])
    spatial_embeddings_pca = pca.fit_transform(spatial_embeddings[:num_samples])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(image_embeddings_pca[:, 0], image_embeddings_pca[:, 1], label="Image", alpha=0.5, color='blue')
    ax.scatter(spatial_embeddings_pca[:, 0], spatial_embeddings_pca[:, 1], label="Spatial", alpha=0.5, color='red')
    ax.set_title("PCA of Image vs Spatial Embeddings")
    ax.legend()

    return fig

def UMAP_embeddings(image_embeddings, spatial_embeddings, num_samples=1000):
    """
    Plot the image and spatial embeddings using UMAP.
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings = np.concatenate((image_embeddings[:num_samples], spatial_embeddings[:num_samples]), axis=0)
    embeddings_umap = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 8))
    n = min(num_samples, len(image_embeddings), len(spatial_embeddings))

    ax.scatter(
        embeddings_umap[:n, 0], embeddings_umap[:n, 1],
        color="blue", label="Image Embeddings", alpha=0.6
    )
    ax.scatter(
        embeddings_umap[n:2*n, 0], embeddings_umap[n:2*n, 1],
        color="red", label="Spatial Embeddings", alpha=0.6
    )

    ax.set_title("UMAP of Image vs Spatial Embeddings")
    ax.legend()
    return fig

def heatmap(image_embedding, spatial_embedding, xlabel="Spatial Embeddings", ylabel="Image Embeddings"):
    """
    Plot a heatmap of the score matrix.
    """
    score_matrix = torch.matmul(image_embedding, spatial_embedding.T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(score_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_title("Score Matrix Heatmap")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks([])
    plt.yticks([])
    return fig

def fix_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def to_device_if_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device_if_tensor(xx, device) for xx in x)
    if isinstance(x, dict):
        return {k: to_device_if_tensor(v, device) for k, v in x.items()}
    return x