import os
import torch
from pathlib import Path
from typing import Optional
from ml_collections.config_dict import ConfigDict

import timm
from model.datasets.dataset import SpatialPatchDataset
from model.model import base_contrastive, openAI_CLIP, PLIP, OmiCLIP
from transformers import CLIPProcessor, CLIPModel
import loki

import wandb

def load_dataset(config_dataset):
    """
    Load the training and evaluation datasets based on the configuration.
    """

    dataset = SpatialPatchDataset(
        image_file=Path(config_dataset.image_file),
        spatial_file=Path(config_dataset.spatial_file),
        indices_file=Path(config_dataset.indices_file),
        fov_size=config_dataset.fov_size,
    )

    return dataset


def load_model(config:ConfigDict, 
               device:Optional[str]="cuda"):
    """
    Load the model, optimizer, and learning rate scheduler based on the configuration.
    """
    if config.model.name == "basic_contrastive":
        pretrained_ckpt_dir = config.model.pretrained_image_encoder_ckpt_path
        assert Path(pretrained_ckpt_dir).exists(), f"Pretrained image encoder checkpoint directory {pretrained_ckpt_dir} does not exist."
        pretrained_image_encoder = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        pretrained_image_encoder.load_state_dict(
            torch.load(
                Path(pretrained_ckpt_dir, "pytorch_model.bin"), map_location="cpu"
            ),
            strict=True,
        )
        pretrained_image_encoder.eval()

        model = base_contrastive(config, pretrained_image_encoder)

    elif config.model.name == "openAI_CLIP":
        model_raw = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model = openAI_CLIP(config, model_raw, processor)

    elif config.model.name == "PLIP":
        model_raw = CLIPModel.from_pretrained("vinid/plip")
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        model = PLIP(config, model_raw, processor)

    elif config.model.name == "loki_omiclip":
        assert config.model.pretrained_omiclip_ckpt_path is not None, "Pretrained Loki checkpoint path must be provided."
        model_raw_path = Path(config.model.pretrained_loki_ckpt_path, "checkpoint.pt")
        model_raw, preprocess, tokenizer = loki.utils.load_model(model_raw_path, device)
        model = OmiCLIP(config, model_raw, preprocess, tokenizer)
    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.lr_step_size,
        gamma=config.train.lr_gamma,
    )

    model = model.to(device)
    optimizer = optimizer.to(device)
    lr_scheduler = lr_scheduler.to(device)

    return model, optimizer, lr_scheduler


def train_one_epoch(
    config,
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    global_step,
    device,
    wandb_logger,
):
    """
    One epoch of training for the model.
    """
    model = model.to(device)
    model.train()

    wandb.watch(model, log="all", log_freq=10)

    for i, batch in enumerate(train_dataloader):
        images = batch["image"].to(device)
        spatial = batch["spatial"].to(device)

        # Forward pass
        with torch.amp.autocast(device.type, enabled=False):
            loss, image_embedding, spatial_embedding = model(images, spatial)

        # Backward pass
        loss.backward()

        if (i+1) % config.get("train.gradient_accumulation_steps",1) == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        if (global_step + 1) % config.train.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                config.outdir,
                global_step + 1,
            )
            torch.save(
                {
                    "step": global_step,
                    "image_embedding": image_embedding.detach().cpu(),
                    "spatial_embedding": spatial_embedding.detach().cpu(),
                },
                f"{config.outdir}/embedding_step_{global_step}.pt",
            )

        if (
            eval_dataloader is not None
            and (global_step + 1) % config.train.eval_every == 0
        ):
            eval_loss, _, _ = evaluate(model, eval_dataloader, device)
            print(f"Eval Loss: {eval_loss}")
            wandb_logger.update({"eval_loss": eval_loss})

        wandb_logger.update({"train_loss": loss,
                             "lr": lr_scheduler.get_last_lr()[0]})
        wandb_logger.set_step()

        global_step += 1

    return global_step


def save_checkpoint(
    model, optimizer, lr_scheduler, output_dir, global_step, logger=None
):
    save_path = Path(output_dir) / f"checkpoint-{global_step}.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "global_step": global_step,
    }
    torch.save(checkpoint, save_path)
    if logger:
        logger.info(f"Checkpoint saved at {save_path}")
    else:
        print(f"Checkpoint saved at {save_path}")
    return save_path


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    global_step = checkpoint.get("global_step", 0)

    print(f"Loaded checkpoint from {checkpoint_path} at step {global_step}")
    return global_step


def evaluate(model, eval_dataloader, device):
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    all_image_embeddings = []
    all_spatial_embeddings = []

    with torch.no_grad():
        for batch in eval_dataloader:
            images = batch["image"].to(device)
            spatial = batch["spatial"].to(device)

            loss, image_embedding, spatial_embedding = model(images, spatial)
            all_image_embeddings.append(image_embedding.cpu())
            all_spatial_embeddings.append(spatial_embedding.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_spatial_embeddings = torch.cat(all_spatial_embeddings, dim=0)

    return avg_loss, all_image_embeddings, all_spatial_embeddings
