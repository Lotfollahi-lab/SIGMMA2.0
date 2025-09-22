import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from ml_collections.config_dict import ConfigDict

import pickle
import timm
from model.datasets.dataset import MultiSectionDataset
from model.model import hierarchical_img_spatial_graph, single_img_spatial, single_img_spatial_graph, openAI_CLIP, PLIP, OmiCLIP, symmetric_InfoNCE
from .evaluate_retrieval import retrieval_score, eval_retrieval
from transformers import CLIPProcessor, CLIPModel
import loki.utils

from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

import wandb

def load_dataset(config_dataset, test=False):
    """
    Load the training and evaluation datasets based on the configuration.
    """
    dataset = MultiSectionDataset(manifest_path=Path(config_dataset.manifest_path),
                                  num_cell_types=config_dataset.num_cell_types,
                                  pretrained_cell_embedding=config_dataset.pretrained_cell_embedding)

    return dataset


def load_model(config:ConfigDict, 
               device:Optional[str]="cuda",
               pretrained_ckpt_path:Optional[str]=None,
               train: Optional[bool]=True):
    """
    Load the model, optimizer, and learning rate scheduler based on the configuration.
    """
    if config.model.name == "hierarchical_img_spatial_graph":
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
        model = hierarchical_img_spatial_graph(config.model, pretrained_image_encoder)

    elif config.model.name == "single_img_spatial":
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
        model = single_img_spatial(config.model, pretrained_image_encoder)

    elif config.model.name == "single_img_spatial_graph":
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
        model = single_img_spatial_graph(config.model, pretrained_image_encoder)
    elif config.model.name == "openAI_CLIP":
        model_raw = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
        model = openAI_CLIP(config, model_raw, processor)

    elif config.model.name == "PLIP":
        model_raw = CLIPModel.from_pretrained("vinid/plip", use_safetensors=True)
        processor = CLIPProcessor.from_pretrained("vinid/plip", use_safetensors=True)
        model = PLIP(config, model_raw, processor)

    elif config.model.name == "loki_OmiCLIP":
        assert config.model.pretrained_omiclip_ckpt_path is not None, "Pretrained Loki checkpoint path must be provided."
        assert Path(config.model.pretrained_omiclip_ckpt_path).exists(), f"Pretrained OmiCLIP checkpoint path {config.model.pretrained_omiclip_ckpt_path} does not exist."
        model_raw_path = config.model.pretrained_omiclip_ckpt_path + "/checkpoint.pt"
        model_raw, preprocess, tokenizer = loki.utils.load_model(model_raw_path, device)
        model = OmiCLIP(config, model_raw, preprocess, tokenizer)
    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")

    model = model.to(device)

    print(f"Total parameters in the {config.model.name} model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if train:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        if config.train.lr_scheduler == "constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=config.train.epochs * config.train.batch_size)

        elif config.train.lr_scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.train.lr_step_size,
                gamma=config.train.lr_gamma,
        )

        # # + Warmup ---
        # gradient_accumulation_steps = config.get("train.gradient_accumulation_steps", 1)
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=config.train.learning_rate*100,
        #     total_steps=config.train.epochs*config.train.batch_size,
        #     epochs=config.train.epochs,
        #     pct_start=0.2,
        # )

        ## + Restart ---
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)

        return model, optimizer, lr_scheduler
    elif pretrained_ckpt_path is not None:
        assert Path(pretrained_ckpt_path).exists(), f"Pretrained checkpoint path {pretrained_ckpt_path} does not exist."
        load_checkpoint(model, None, None, Path(pretrained_ckpt_path), device)
        return model
    else:
        return model


def train_one_epoch(
    config,
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    global_step,
    epoch_index,
    device,
    wandb_logger,
):
    """
    One epoch of training for the model.
    """
    model = model.to(device)
    model.train()

    wandb.watch(model, log=None, log_freq=10)

    accum_freq = config.get("train.gradient_accumulation_steps", 1)
    if accum_freq > 1:
        accum_images, accum_spatial, accum_features = [], [], {}

    for i, batch in enumerate(train_dataloader):
        image = batch["image"].to(device)
        if config.model.name in ["hierarchical_img_spatial_graph", "single_img_spatial_graph"]:
            spatial = batch['spatial_graph'].to(device)
            inputs = [image, spatial]
        elif config.model.name in ["single_img_spatial"]:
            spatial = batch["spatial"].to(device)
            inputs = [image, spatial]
        elif config.model.name in ["openAI_CLIP", "PLIP", "loki_OmiCLIP"]:
            spatial = batch['spatial_text']
            inputs = [image, spatial]
        else:
            raise NotImplementedError(f"Model {model.name} is not implemented for training.")

        optimizer.zero_grad()
        # Gradient accumulation code adapted from OpenCLIP (https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/train.py)
        if accum_freq == 1:
            # Forward pass
            with torch.amp.autocast(device.type, enabled=False):
                model_out = model(*inputs)
            if config.model.name == "hierarchical_img_spatial_graph":
                loss = 0
                for s in range(num_resolutions):
                    loss += symmetric_InfoNCE()(inputs_for_loss['image_embedding'][s], inputs_for_loss['spatial_embedding'][s])
                loss = loss / num_resolutions
            else:
                loss = model_out["loss"]
                image_embedding = model_out["image_embedding"] # Tensor (B, D)
                spatial_embedding = model_out["spatial_embedding"] # Tensor (B, D)
            # Backward pass
            loss.backward()
        else:
            with torch.no_grad():
                # Forward pass
                with torch.amp.autocast(device.type, enabled=False):
                    model_out = model(*inputs)

                image_embedding = model_out["image_embedding"] # List of Tensors [(B, D), (B, D), ...] or Tensor (B, D)
                spatial_embedding = model_out["spatial_embedding"] # List of Tensors [(B, D), (B, D), ...] or Tensor (B, D)

                accum_images.append(image)
                accum_spatial.append(spatial)
                if "image_embedding" not in accum_features:
                    accum_features["image_embedding"] = [image_embedding]
                    accum_features["spatial_embedding"] = [spatial_embedding]
                else:
                    accum_features["image_embedding"].append(image_embedding)
                    accum_features["spatial_embedding"].append(spatial_embedding)

            if (i+1) % accum_freq > 0:
                print(f"Skipping optimizer step at iter {i}, accumulating more gradients...")
                continue
            
            # Take gradients for the last accum_freq batches
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            optimizer.zero_grad()
            total_loss = None
            for j in range(len(accum_images)):
                image = accum_images[j].to(device)
                spatial = accum_spatial[j].to(device)
                with torch.amp.autocast(device.type, enabled=False):
                    model_out = model(image, spatial)
                    inputs_for_loss = {}
                    if config.model.name == "hierarchical_img_spatial_graph":
                        loss = 0
                        num_resolutions = len(model_out['image_embedding'])
                        for s in range(num_resolutions):
                            for key in accum_features:
                                accumulated = accum_features[key]
                                parts = [accumulated[k][s] for k in range(j)] + [model_out[key][s]] + [accumulated[k][s] for k in range(j+1, len(accumulated))]
                                inputs_for_loss[key] = torch.cat(parts)
                            loss += symmetric_InfoNCE()(inputs_for_loss['image_embedding'], inputs_for_loss['spatial_embedding'])
                        loss = loss / num_resolutions
                    else:
                        for key in accum_features:
                            accumulated = accum_features[key]
                            inputs_for_loss[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j+1:])
                        loss = symmetric_InfoNCE()(inputs_for_loss['image_embedding'], inputs_for_loss['spatial_embedding'])
                    losses = {'contrastive_loss': loss}
                    total_loss = sum(losses.values())
                    losses['loss'] = total_loss
                    del inputs_for_loss
                total_loss.backward()
            loss = total_loss.item()

            optimizer.step()
            lr_scheduler.step()

            # Reset accumulators
            if accum_freq > 1:
                accum_images, accum_spatial, accum_features = [], [], {}

        if (global_step + 1) % config.train.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                config.outdir,
                global_step + 1,
                epoch_index + 1,
            )
        if (
            eval_dataloader is not None
            and (global_step + 1) % config.train.eval_every == 0
            ):
            eval_loss, eval_image_embedding, eval_spatial_embedding, score_matrix_img2spatial, score_matrix_spatial2image, eval_results = evaluate(config, model, eval_dataloader, device)
            if config.model.name == "hierarchical_img_spatial_graph":
                eval_image_embedding = eval_image_embedding[-1]
                eval_spatial_embedding = eval_spatial_embedding[-1]
            fig_eval_umap = UMAP_embeddings(eval_image_embedding.detach().cpu(), eval_spatial_embedding.detach().cpu())
            fig_eval_img2spatial_heatmap = heatmap(eval_image_embedding.detach().cpu(), eval_spatial_embedding.detach().cpu(), xlabel="Image Embeddings", ylabel="Spatial Embeddings")
            fig_eval_img2img_heatmap = heatmap(eval_image_embedding.detach().cpu(), eval_image_embedding.detach().cpu(), xlabel="Image Embeddings", ylabel="Image Embeddings")
            fig_eval_spatial2spatial_heatmap = heatmap(eval_spatial_embedding.detach().cpu(), eval_spatial_embedding.detach().cpu(), xlabel="Spatial Embeddings", ylabel="Spatial Embeddings")

            # Plot learnt embeddings
            fig_train_umap = UMAP_embeddings(image_embedding.detach().cpu(), spatial_embedding.detach().cpu())
            fig_train_img2spatial_heatmap = heatmap(image_embedding.detach().cpu(), spatial_embedding.detach().cpu(), xlabel="Image Embeddings", ylabel="Spatial Embeddings")
            fig_train_img2img_heatmap = heatmap(image_embedding.detach().cpu(), image_embedding.detach().cpu(), xlabel="Image Embeddings", ylabel="Image Embeddings")
            fig_train_spatial2spatial_heatmap = heatmap(spatial_embedding.detach().cpu(), spatial_embedding.detach().cpu(), xlabel="Spatial Embeddings", ylabel="Spatial Embeddings")

            model.train() # Set model back to training mode
            wandb_logger.update({"(train)UMAP_embeddings": wandb.Image(fig_train_umap),
                                 "(train)img2spatial heatmap": wandb.Image(fig_train_img2spatial_heatmap),
                                 "(train)img2img heatmap": wandb.Image(fig_train_img2img_heatmap),
                                 "(train)spatial2spatial heatmap": wandb.Image(fig_train_spatial2spatial_heatmap),
                                 "(eval)loss": eval_loss,
                                 "(eval)UMAP_embeddings": wandb.Image(fig_eval_umap),
                                 "(eval)img2spatial heatmap": wandb.Image(fig_eval_img2spatial_heatmap),
                                 "(eval)img2img heatmap": wandb.Image(fig_eval_img2img_heatmap),
                                 "(eval)spatial2spatial heatmap": wandb.Image(fig_eval_spatial2spatial_heatmap),
                                 "(eval)image recall@5percent": eval_results["image_recall@5percent"],
                                 "(eval)image recall@10percent": eval_results["image_recall@10percent"],
                                 "(eval)image recall@15percent": eval_results["image_recall@15percent"],
                                 "(eval)spatial recall@5percent": eval_results["spatial_recall@5percent"],
                                 "(eval)spatial recall@10percent": eval_results["spatial_recall@10percent"],
                                 "(eval)spatial recall@15percent": eval_results["spatial_recall@15percent"],
                                 "epoch": epoch_index + 1})
            torch.save({"eval_image_embeddings": eval_image_embedding,
                        "eval_spatial_embeddings": eval_spatial_embedding},
                       f"{config.outdir}/embeddings-{global_step + 1}.pt")

        wandb_logger.update({"(train)loss": loss,
                             "(train)lr": lr_scheduler.get_last_lr()[0],
                             "epoch": epoch_index + 1})
        wandb_logger.set_step()

        global_step += 1
    save_checkpoint(model,
                    optimizer,
                    lr_scheduler,
                    config.outdir,
                    global_step + 1,
                    epoch_index + 1,
                )
    return global_step


def save_checkpoint(
    model, optimizer, lr_scheduler, output_dir, global_step, epoch_index, logger=None
):
    save_path = Path(output_dir) / f"checkpoint-epoch{epoch_index}.pth"
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
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    global_step = checkpoint.get("global_step", 0)

    print(f"Loaded checkpoint from {checkpoint_path} at step {global_step}")
    return global_step


def evaluate(config, model, eval_dataloader, device):
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    all_image_embeddings = []
    all_spatial_embeddings = []

    with torch.no_grad():
        for batch in eval_dataloader:
            image = batch["image"].to(device)
            if config.model.name in ["hierarchical_img_spatial_graph", "single_img_spatial_graph"]:
                spatial = batch['spatial_graph'].to(device)
                inputs = [image, spatial]
            elif config.model.name in ["single_img_spatial"]:
                spatial = batch["spatial"].to(device)
                inputs = [image, spatial]
            elif config.model.name in ["openAI_CLIP", "PLIP", "loki_OmiCLIP"]:
                spatial = batch['spatial_text']
                inputs = [image, spatial]
            else:
                raise NotImplementedError(f"Model {model.name} is not implemented for training.")

            model_out = model(*inputs)
            image_embedding = model_out["image_embedding"]
            spatial_embedding = model_out["spatial_embedding"]

            all_image_embeddings.append(image_embedding)
            all_spatial_embeddings.append(spatial_embedding)

    if config.model.name == "hierarchical_img_spatial_graph":
        num_resolutions = len(all_image_embeddings[0])
        all_image_embeddings = [torch.concat([all_image_embeddings[i][s] for i in range(len(all_image_embeddings))], dim=0).to(device) for s in range(num_resolutions)]
        all_spatial_embeddings = [torch.concat([all_spatial_embeddings[i][s] for i in range(len(all_spatial_embeddings))], dim=0).to(device) for s in range(num_resolutions)]
        loss = 0
        for s in range(num_resolutions):
            loss += symmetric_InfoNCE()(all_image_embeddings[s], all_spatial_embeddings[s])
        loss = loss / num_resolutions
    else:
        all_image_embeddings = torch.concat(all_image_embeddings, dim=0).to(device)
        all_spatial_embeddings = torch.concat(all_spatial_embeddings, dim=0).to(device)
        loss = symmetric_InfoNCE()(all_image_embeddings, all_spatial_embeddings)

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
