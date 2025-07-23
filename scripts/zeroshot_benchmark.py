import os
import argparse
from pathlib import Path
import pandas as pd
import pickle
import copy
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ml_collections.config_dict import ConfigDict
import yaml

from model.utils import load_dataset, load_model, cross_modal_retrieval, compute_pcc, recall_at_K, WandbLogger
from model.datasets import multimodal_collate

def main(args):
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.outdir = config.get("outdir") if args.out_dir is None else args.out_dir
    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)
    # == Query dataset preparation ==
    query_dataset = load_dataset(config.query_dataset)
    # valid_indices = []
    # for i in range(len(query_dataset)):
    #     spatial = query_dataset[i]['spatial']
    #     if not torch.any(torch.isnan(spatial)):
    #         valid_indices.append(i)
    # with open("valid_indices_test.txt", "wb") as f:
    #     pickle.dump(valid_indices, f)
    with open("valid_indices_test.txt", "rb") as f:
        valid_indices = pickle.load(f)
    test_dataset = torch.utils.data.Subset(query_dataset, valid_indices)
    print(f"Total samples after filtering: {len(query_dataset)}")

    batch_size = config.get("test.batch_size", 1)
    num_samples = config.get("test.num_samples", len(test_dataset))
    random_sampler = RandomSampler(test_dataset, num_samples=num_samples, replacement=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=random_sampler, num_workers=0, collate_fn=multimodal_collate)

    # Add wandb logging
    wandb_config = {
        "entity": None,
        "project": "img-spatial-alignment",
        "group": None,
        "name": "benchmark",
        "tags": None,
        "mode": "online",
    }

    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    model = load_model(config, device, train=False)
    model_name = config.model.get("name")
    model.eval()
    reference_path = config.model.get("reference_path", None)
    assert reference_path is not None, f"Reference path for {model_name} is not set."
    
    # Retrieval
    # -- Reference dataset preparation --
    reference_images = torch.load(Path(reference_path, f"{model_name}_image_embeddings.pt"))
    reference_spatial = torch.load(Path(reference_path, f"{model_name}_spatial_embeddings.pt"))
    reference_metadata = pd.read_csv(Path(reference_path, f"{model_name}_reference_metadata.csv"))

    image_query_list, spatial_query_list = [], []
    for batch in test_dataloader:
        image = batch["image"].to(device)
        cell_type_composition = batch["cell_type_composition"].to(device)
        if  model_name == "hierarchical_img_cellgraph":
            spatial = batch["spatial_graph"].to(device)
            inputs = [image, spatial, cell_type_composition]
        elif model_name == "base_contrastive":
            spatial = batch["spatial"].to(device)
            inputs = [image, spatial, cell_type_composition]
        elif model_name in ["openAI_CLIP", "PLIP", "OmiCLIP"]:
            spatial = batch["spatial_text"]
            inputs = [image, spatial]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        with torch.no_grad():
            if config.model.name == "hierarchical_img_cellgraph":
                loss_kl, loss_contrastive, loss_ce, loss, image_embeds, spatial_embeds = model(*inputs) # TODO: multi-resolution embeddings
            else:
                loss, image_embeds, spatial_embeds = model(*inputs)

        image_embeds = image_embeds.detach().cpu()
        spatial_embeds = spatial_embeds.detach().cpu()

        image_query_list.append(image_embeds.detach().cpu())
        spatial_query_list.append(spatial_embeds.detach().cpu())

    image_query_tensor = torch.cat(image_query_list, dim=0)  # num_batches X N X D
    spatial_query_tensor = torch.cat(spatial_query_list, dim=0) # num_batches X N X D

    # Compute PCC
    pcc = compute_pcc(spatial_query_tensor, reference_spatial, reference_metadata, k=10) # compute mean pcc between query and k-retrieved spatial embedding
    print(f"{model_name} PCC: {pcc.mean():.4f}")
    wandb_logger.update({f"{(model_name)} pcc": pcc.mean()})
    # Compute Recall@K
    for k in [10, 50, 100]:
        recall = recall_at_K(spatial_query_tensor, reference_spatial, reference_metadata, k=k)
        print(f"{model_name} Recall@{k}: {recall:.4f}")
        wandb_logger.update({f"{model_name} recall@{k}": recall})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()
    main(args)