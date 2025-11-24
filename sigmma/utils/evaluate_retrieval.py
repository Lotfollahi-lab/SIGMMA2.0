import torch
import zarr
import scanpy
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

@torch.no_grad()
def retrieval_score(model, query_dataloader, reference_dataloader, device, config):
    model.eval()

    # Load embeddings for query and reference datasets
    img_embeds_query, spatial_embeds_query = load_embeddings(model, query_dataloader, device, config)
    img_embeds_reference, spatial_embeds_reference = load_embeddings(model, reference_dataloader, device, config)

    # Cosine similarity scores
    score_matrix_img2spatial = img_embeds_query @ spatial_embeds_reference.T
    score_matrix_spatial2img = spatial_embeds_query @ img_embeds_reference.T

    return score_matrix_img2spatial.cpu().numpy(), score_matrix_spatial2img.cpu().numpy()

def load_embeddings(model, dataloader, device, config):
    model_name = config.model.get("name")

    image_embed_li = []
    spatial_embed_li = []
    for batch in dataloader:
        image = batch["image"].to(device)

        if config.model.name in ["hierarchical_img_spatial_graph", "single_img_spatial_graph", "hierarchical_img_spatial_graph_wo_sparsification"]:
            spatial = batch["spatial_graph"].to(device)
            inputs = [image, spatial]
        elif model_name == "single_img_spatial":
            spatial = batch["spatial"].to(device)
            inputs = [image, spatial]
        elif model_name in ["openAI_CLIP", "PLIP", "loki_OmiCLIP"]:
            spatial = batch["spatial_text"]
            inputs = [image, spatial]
        elif model_name == "BLEEP":
            spatial = batch["spatial"].to(device)
            inputs = [image, spatial]
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Forward pass through the model
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=True):
                model_output = model(*inputs)
            image_embed = model_output["image_embedding"]
            spatial_embed = model_output["spatial_embedding"]

        # Normalize embeddings
        if config.model.name in ["hierarchical_img_spatial_graph", "hierarchical_img_spatial_graph_wo_sparsification"]:
            image_embed = image_embed[-1]
            spatial_embed = spatial_embed[-1]
        else:
            image_embed = image_embed
            spatial_embed = spatial_embed
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        spatial_embed = spatial_embed / spatial_embed.norm(dim=-1, keepdim=True)

        image_embed_li.append(image_embed.detach().cpu())
        spatial_embed_li.append(spatial_embed.detach().cpu())

    img_embeds = torch.cat(image_embed_li, dim=0)
    spatial_embeds = torch.cat(spatial_embed_li, dim=0)
    return img_embeds, spatial_embeds

def eval_retrieval(scores_img2spatial, scores_spatial2img):
    num_reference = scores_img2spatial.shape[1]
    gt_indices = np.arange(scores_img2spatial.shape[0])
    ranks = np.zeros(scores_img2spatial.shape[0])
    for index, score in enumerate(scores_img2spatial):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == gt_indices[index])[0][0]
    # == Top@percent Recall ==
    img_r_5percent = len(np.where(ranks < int(num_reference*0.05))[0]) / len(ranks)
    img_r_10percent = len(np.where(ranks < int(num_reference*0.1))[0]) / len(ranks)
    img_r_15percent = len(np.where(ranks < int(num_reference*0.15))[0]) / len(ranks)

    # == Top@k Recall ==
    img_r1 = np.mean(ranks == 0)
    img_r5 = np.mean(ranks < 5)
    img_r10 = np.mean(ranks < 10)

    for index, score in enumerate(scores_spatial2img):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == gt_indices[index])[0][0]
    
    # === Top@percent Recall ===
    spatial_r_5percent = len(np.where(ranks < int(num_reference*0.05))[0]) / len(ranks)
    spatial_r_10percent = len(np.where(ranks < int(num_reference*0.1))[0]) / len(ranks)
    spatial_r_15percent = len(np.where(ranks < int(num_reference*0.15))[0]) / len(ranks)

    # == Top@k Recall ==
    spatial_r1 = np.mean(ranks == 0)
    spatial_r5 = np.mean(ranks < 5)
    spatial_r10 = np.mean(ranks < 10)

    eval_results = {"image_recall@5percent": img_r_5percent,
                     "image_recall@10percent": img_r_10percent,
                     "image_recall@15percent": img_r_15percent,
                     "spatial_recall@5percent": spatial_r_5percent,
                     "spatial_recall@10percent": spatial_r_10percent,
                     "spatial_recall@15percent": spatial_r_15percent,
                     "image_recall@1": img_r1,
                     "image_recall@5": img_r5,
                     "image_recall@10": img_r10,
                     "spatial_recall@1": spatial_r1,
                     "spatial_recall@5": spatial_r5,
                     "spatial_recall@10": spatial_r10}
    return eval_results