import os
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

import torch
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler, RandomSampler

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
    # == Reference dataset preparation ==
    dataset1 = load_dataset(config.dataset1)
    dataset2 = load_dataset(config.dataset2)

    reference_dataset = ConcatDataset([dataset1, dataset2])

    # valid_indices = []
    # for i in range(len(dataset)):
    #     spatial = dataset[i]['spatial']
    #     if not torch.any(torch.isnan(spatial)):
    #         valid_indices.append(i)
    # with open("valid_indices.txt", "wb") as f:
    #     pickle.dump(valid_indices, f)
    with open("valid_indices_lymph_node.txt", "rb") as f:
        valid_indices = pickle.load(f)
    reference_dataset = torch.utils.data.Subset(reference_dataset, valid_indices)
    print(f"Total samples after filtering: {len(reference_dataset)}")    

    batch_size = config.get("test.batch_size")
    sequential_sampler = SequentialSampler(reference_dataset)
    reference_dataloader = DataLoader(reference_dataset, batch_size=batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, collate_fn=multimodal_collate)

    # Add wandb logging
    wandb_config = {
        "entity": None,
        "project": "img-spatial-alignment",
        "group": None,
        "name": "benchmark_build_reference",
        "tags": None,
        "mode": "online",
    }

    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    model = load_model(config, device, pretrained_ckpt_path=config.model.pretrained_checkpoint_path, train=False)
    model_name = config.model.get("name")
    model.eval()

    for batch_idx, batch in enumerate(reference_dataloader):
        image = batch["image"].to(device)
        cell_type_composition = batch["cell_type_composition"].to(device)
        if model_name == "hierarchical_img_cellgraph":
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
    
        # Normalize embeddings
        image_embeds = image_embeds.detach().cpu()
        spatial_embeds = spatial_embeds.detach().cpu()
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        spatial_embeds = spatial_embeds / spatial_embeds.norm(dim=-1, keepdim=True)
    
        torch.save(image_embeds, Path(config.outdir, f"{model_name}_image_embeddings_batch{batch_idx}.pt"))
        torch.save(spatial_embeds, Path(config.outdir, f"{model_name}_spatial_embeddings_batch{batch_idx}.pt"))

        # Store reference embeddings
        if 'reference_image_embeddings' not in locals():
            assert 'reference_spatial_embeddings' not in locals(), "Both reference embeddings should be initialized together."
            reference_image_embeddings = image_embeds.cpu()
            reference_spatial_embeddings = spatial_embeds.cpu()
            df_metadata = pd.DataFrame(zip( batch["image_file"], batch["spatial_file"], batch["indices_file"], np.array(batch["index"])), 
                                        columns=['image_file', 'spatial_file', 'index_array_file', 'data_index'])
            df_metadata.to_csv(Path(config.outdir, f"{model_name}_metadata_batch{batch_idx}.csv"), index=False)
        else:
            reference_image_embeddings = torch.cat((reference_image_embeddings, image_embeds.cpu()), dim=0)
            reference_spatial_embeddings = torch.cat((reference_spatial_embeddings, spatial_embeds.cpu()), dim=0)
            df_metadata_tmp = pd.DataFrame(zip( batch["image_file"], batch["spatial_file"], batch["indices_file"], np.array(batch["index"])), 
                                          columns=['image_file', 'spatial_file', 'index_array_file', 'data_index'])
            df_metadata = pd.concat([df_metadata, df_metadata_tmp])

    torch.save(reference_image_embeddings, Path(config.outdir, f"{model_name}_image_embeddings.pt"))
    torch.save(reference_spatial_embeddings, Path(config.outdir, f"{model_name}_spatial_embeddings.pt"))
    df_metadata.to_csv(Path(config.outdir, f"{model_name}_reference_metadata.csv", index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", type=str, required=True, help="Path to the config file")
    parser.add_argument('-out_dir', type=str, default=None, help='Output directory path')
    args = parser.parse_args()
    main(args)