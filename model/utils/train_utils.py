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
from model.model import hierarchical_img_cellgraph, base_contrastive, openAI_CLIP, PLIP, OmiCLIP
from .evaluate_retrieval import retrieval_score, eval_retrieval
from transformers import CLIPProcessor, CLIPModel
import loki.utils

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import wandb

def load_dataset(config_dataset):
    """
    Load the training and evaluation datasets based on the configuration.
    """
    dataset = MultiSectionDataset(manifest_path=Path(config_dataset.manifest_path),
                                  num_cell_types=config_dataset.num_cell_types)

    # valid_indices = []
    # for i in range(len(dataset)):
    #     spatial_graph = dataset[i]['spatial_graph']
    #     if not spatial_graph.num_nodes() == 0:
    #         valid_indices.append(i)
    # with open("valid_indices_skin.txt", "wb") as f:
    #     pickle.dump(valid_indices, f)
    with open("valid_indices_skin.txt", "rb") as f:
        valid_indices = pickle.load(f)
    dataset = torch.utils.data.Subset(dataset, valid_indices)
    return dataset


def load_model(config:ConfigDict, 
               device:Optional[str]="cuda",
               pretrained_ckpt_path:Optional[str]=None,
               train: Optional[bool]=True):
    """
    Load the model, optimizer, and learning rate scheduler based on the configuration.
    """
    if config.model.name == "hierarchical_img_cellgraph":
        model = hierarchical_img_cellgraph(config.model)

    elif config.model.name == "basic_contrastive":
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

        model = base_contrastive(config.model, pretrained_image_encoder)

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

        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=config.train.epochs * config.train.batch_size)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=config.train.lr_step_size,
        #     gamma=config.train.lr_gamma,
        # )

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

    wandb.watch(model, log="all", log_freq=10)

    for i, batch in enumerate(train_dataloader):
        image = batch["image"].to(device)
        cell_type_composition = batch["cell_type_composition"].to(device)

        if config.model.name == "hierarchical_img_cellgraph":
            spatial = batch['spatial_graph'].to(device)
            inputs = [image, spatial, cell_type_composition]
        elif config.model.name == "basic_contrastive":
            spatial = batch["spatial"].to(device)
            inputs = [image, spatial, cell_type_composition]
        elif config.model.name in ["openAI_CLIP", "PLIP", "loki_OmiCLIP"]:
            spatial = batch['spatial_text']
            inputs = [image, spatial]
        else:
            raise NotImplementedError(f"Model {model.name} is not implemented for training.")

        # Forward pass
        with torch.amp.autocast(device.type, enabled=False):
            if config.model.name == "hierarchical_img_cellgraph":
                loss_kl, loss_contrastive, loss_ce, loss, image_embedding, spatial_embedding = model(*inputs) # TODO: multi-resolution embeddings
            else:
                loss, image_embedding, spatial_embedding = model(*inputs)
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
                epoch_index + 1,
            )
        if (
            eval_dataloader is not None
            and (global_step + 1) % config.train.eval_every == 0
        ):
            eval_loss, eval_image_embedding, eval_spatial_embedding, score_matrix_img2spatial, score_matrix_spatial2image, eval_results = evaluate(config, model, eval_dataloader, device)
            fig_eval = PCA_embeddings(eval_image_embedding.detach().cpu(), eval_spatial_embedding.detach().cpu())

            fig_train = PCA_embeddings(image_embedding.detach().cpu(), spatial_embedding.detach().cpu())
            wandb_logger.update({"(train)PCA_embeddings": wandb.Image(fig_train),
                                 "(eval)loss": eval_loss,
                                 "(eval)PCA_embeddings": wandb.Image(fig_eval),
                                 "(eval)image recall@5percent": eval_results["image_recall_5@percent"],
                                 "(eval)image recall@10percent": eval_results["image_recall_10@percent"],
                                 "(eval)image recall@15percent": eval_results["image_recall_15@percent"],
                                 "(eval)spatial recall@5percent": eval_results["spatial_recall_5@percent"],
                                 "(eval)spatial recall@10percent": eval_results["spatial_recall_10@percent"],
                                 "(eval)spatial recall@15percent": eval_results["spatial_recall_15@percent"]})
            torch.save({"eval_image_embeddings": eval_image_embedding,
                        "eval_spatial_embeddings": eval_spatial_embedding},
                       f"{config.outdir}/embeddings-{global_step + 1}.pt")

        if config.model.name == "hierarchical_img_cellgraph":
            wandb_logger.update({"(train)loss": loss,
                                 "(train)loss_kl": loss_kl,
                                 "(train)loss_contrastive": loss_contrastive,
                                 "(train)loss_ce": loss_ce,
                                 "(train)lr": lr_scheduler.get_last_lr()[0],
                                 "(train)epoch": epoch_index})
        else:
            wandb_logger.update({"(train)loss": loss,
                                 "(train)lr": lr_scheduler.get_last_lr()[0],
                                 "(train)epoch": epoch_index})
        wandb_logger.set_step()

        global_step += 1

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
    total_loss = 0.0
    all_image_embeddings = []
    all_spatial_embeddings = []

    with torch.no_grad():
        for batch in eval_dataloader:
            image = batch["image"].to(device)
            cell_type_composition = batch["cell_type_composition"].to(device)
            if config.model.name == "hierarchical_img_cellgraph":
                spatial = batch['spatial_graph'].to(device)
                inputs = [image, spatial, cell_type_composition]
            elif config.model.name == "basic_contrastive":
                spatial = batch["spatial"].to(device)
                inputs = [image, spatial, cell_type_composition]
            elif config.model.name in ["openAI_CLIP", "PLIP", "loki_OmiCLIP"]:
                spatial = batch['spatial_text']
                inputs = [image, spatial]
            else:
                raise NotImplementedError(f"Model {model.name} is not implemented for training.")

            if config.model.name == "hierarchical_img_cellgraph":
                _, _, _, loss, image_embedding, spatial_embedding = model(*inputs)
            else:
                loss, image_embedding, spatial_embedding = model(*inputs)
            all_image_embeddings.append(image_embedding.cpu())
            all_spatial_embeddings.append(spatial_embedding.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_spatial_embeddings = torch.cat(all_spatial_embeddings, dim=0)

    score_matrix_img2spatial, score_matrix_spatial2image = retrieval_score(model, eval_dataloader, eval_dataloader, device, config)
    eval_results = eval_retrieval(score_matrix_img2spatial, score_matrix_spatial2image)

    return avg_loss, all_image_embeddings, all_spatial_embeddings, score_matrix_img2spatial, score_matrix_spatial2image, eval_results

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

def heatmap(score_matrix, xlabel="Spatial Embeddings", ylabel="Image Embeddings"):
    """
    Plot a heatmap of the score matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(score_matrix, cmap='binary')
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
