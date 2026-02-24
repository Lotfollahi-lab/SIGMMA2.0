import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from ml_collections.config_dict import ConfigDict
import torch.distributed as dist

from sigmma.datasets.dataset import MultiSectionDataset
from sigmma.model import sigmma
from sigmma.model import UNI, ResNet50
from sigmma.model import symmetric_InfoNCE, bleep_cross_entropy
from .evaluate_retrieval import retrieval_score, eval_retrieval
import loki.utils

from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

import wandb

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
                                  model_name=model_name)

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
        if pretrained_image_encoder == "UNI":
            pretrained_ckpt_dir = config.model.pretrained_image_encoder_ckpt_path
            assert Path(pretrained_ckpt_dir).exists(), f"Pretrained image encoder checkpoint directory {pretrained_ckpt_dir} does not exist."
            pretrained_image_encoder = UNI(checkpoint_path=pretrained_ckpt_dir)
        else:
            raise NotImplementedError(f"Pretrained image encoder {pretrained_image_encoder} is not implemented.")

        model = sigmma(config.model, pretrained_image_encoder)

    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")

    if rank is not None: 
        model = model.to(torch.cuda.current_device())
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    if int(os.environ.get("RANK", "0")) == 0:
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
        
    for i, batch in enumerate(train_dataloader):
        image = batch["image"].to(device)
        if config.model.name == "sigmma":
            spatial = batch['spatial_graph'].to(device)
            inputs = [image, spatial]
        else:
            raise NotImplementedError(f"Model {config.model.name} is not implemented for training.")
        
        optimizer.zero_grad()
        # Gradient accumulation code adapted from OpenCLIP (https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/train.py)
        if accum_freq == 1:
            # Forward pass
            with torch.amp.autocast(device.type, enabled=True):
                model_out = model(*inputs)
                num_resolutions = len(model_out['image_embedding'])
            if config.model.name == "sigmma":
                loss = 0
                for s in range(num_resolutions):
                    loss += symmetric_InfoNCE()(model_out['image_embedding'][s], model_out['spatial_embedding'][s])
                loss = loss / num_resolutions
            else:
                if rank == 0:
                    print('Loss', model_out['loss'])
                loss = model_out["loss"]
                image_embedding = model_out["image_embedding"] # Tensor (B, D)
                spatial_embedding = model_out["spatial_embedding"] # Tensor (B, D)
            # Backward pass
            loss.backward()
            
        else:
            with torch.no_grad():
                # Forward pass
                with torch.amp.autocast(device.type, enabled=True):
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
                if rank == 0:
                    print(f"Skipping optimizer step at iter {i}, accumulating more gradients...")
                continue
            
            # Take gradients for the last accum_freq batches
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            optimizer.zero_grad()
            total_loss = None
            for j in range(len(accum_images)):
                image = to_device_if_tensor(accum_images[j], device)
                spatial = to_device_if_tensor(accum_spatial[j], device)
                with torch.amp.autocast(device.type, enabled=True):
                    model_out = model(image, spatial)
                    inputs_for_loss = {}
                    if config.model.name == "sigmma":
                        loss = 0
                        num_resolutions = len(model_out['image_embedding'])
                        for s in range(num_resolutions):
                            for key in accum_features:
                                accumulated = accum_features[key]
                                parts = [accumulated[k][s] for k in range(j)] + [model_out[key][s]] + [accumulated[k][s] for k in range(j+1, len(accumulated))]
                                inputs_for_loss[key] = torch.cat(parts)
                            loss += symmetric_InfoNCE()(inputs_for_loss['image_embedding'], inputs_for_loss['spatial_embedding'])
                        loss = loss / num_resolutions
                        losses = {'contrastive_loss': loss}
                        total_loss = sum(losses.values()) #+ model_out["loss_csc_img"] #+ model_out["loss_kl"]
                        losses['loss'] = total_loss
                    else:
                        raise NotImplementedError(f"Model {config.model.name} is not implemented for training.")
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
            # In DDP mode, synchronize all ranks
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            # Run eval
            if rank == 0:
                eval_loss, _, _, _, _, _ = evaluate(config, model, eval_dataloader, device)

            if config.model.name == "sigmma":
                image_embedding = image_embedding[-1]
                spatial_embedding = spatial_embedding[-1]

            # Plot learnt embeddings
            fig_train_img2spatial_heatmap = heatmap(image_embedding.detach().cpu(), spatial_embedding.detach().cpu(), xlabel="Image Embeddings", ylabel="Spatial Embeddings")
            fig_train_img2img_heatmap = heatmap(image_embedding.detach().cpu(), image_embedding.detach().cpu(), xlabel="Image Embeddings", ylabel="Image Embeddings")
            fig_train_spatial2spatial_heatmap = heatmap(spatial_embedding.detach().cpu(), spatial_embedding.detach().cpu(), xlabel="Spatial Embeddings", ylabel="Spatial Embeddings")

            if wandb_logger is not None:
                wandb_logger.update({"(train)img2spatial heatmap": wandb.Image(fig_train_img2spatial_heatmap),
                                    "(train)img2img heatmap": wandb.Image(fig_train_img2img_heatmap),
                                    "(train)spatial2spatial heatmap": wandb.Image(fig_train_spatial2spatial_heatmap),
                                    "(eval)loss": eval_loss,
                                    "epoch": epoch_index + 1})

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            model.train() # Set model back to training mode

        if wandb_logger is not None:
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
    if int(os.environ.get("RANK", "0")) == 0:
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

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Loaded checkpoint from {checkpoint_path} at step {global_step}")
    return global_step


def evaluate(config, model, eval_dataloader, device):
    """
    Evaluate the model on the validation dataset.
    """
    # DDP unwrap
    model = model.module if hasattr(model, "module") else model
    model.eval()
    all_image_embeddings = []
    all_spatial_embeddings = []

    with torch.no_grad():
        for batch in eval_dataloader:
            image = batch["image"].to(device)
            if config.model.name == "sigmma": 
                spatial = batch['spatial_graph'].to(device)
                inputs = [image, spatial]
            else:
                raise NotImplementedError(f"Model {model.name} is not implemented for training.")

            with torch.amp.autocast(device.type, enabled=True):
                model_out = model(*inputs)
            image_embedding = model_out["image_embedding"]
            spatial_embedding = model_out["spatial_embedding"]

            all_image_embeddings.append(image_embedding)
            all_spatial_embeddings.append(spatial_embedding)

    if config.model.name == "sigmma":
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

def to_device_if_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device_if_tensor(xx, device) for xx in x)
    if isinstance(x, dict):
        return {k: to_device_if_tensor(v, device) for k, v in x.items()}
    return x