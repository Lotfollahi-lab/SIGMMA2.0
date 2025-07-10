import argparse
import pickle
import copy
import torch
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict
import yaml

from model.utils import load_dataset, load_model, compute_pcc, recall_at_K, WandbLogger

def main(args):
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(config.dataset)

    valid_indices = []
    for i in range(len(dataset)):
        spatial = dataset[i]['spatial']
        if not torch.any(torch.isnan(spatial)):
            valid_indices.append(i)
    with open("valid_indices_test.txt", "wb") as f:
        pickle.dump(valid_indices, f)
    with open("valid_indices_test.txt", "rb") as f:
        valid_indices = pickle.load(f)
    test_dataset = torch.utils.data.Subset(dataset, valid_indices)
    print(f"Total samples after filtering: {len(dataset)}")

    batch_size = config.get("train.batch_size")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

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

    models = []
    for i in [3]:#range(5):
        config.model = copy.deepcopy(config.get(f"model_{i}"))
        model = load_model(config, device, train=False)
        model_name = config.model.get("name")
        model.eval()
        models.append((model_name, model))
    
    # Retrieval
    for model_name, model in models:
        image_query_list, image_retrieved_list = [], []
        spatial_query_list, spatial_retrieved_list = [], []
        for batch in test_dataloader:
            image = batch["image"].to(device)
            if  model_name == "hierarchical_img_cellgraph":
                spatial = batch["spatial_graph"].to(device)
                inputs = [image, spatial]
            elif model_name == "base_contrastive":
                spatial = batch["spatial"].to(device)
                inputs = [image, spatial]
            elif model_name == ["openAI_CLIP", "PLIP", "OmiCLIP"]:
                spatial = batch["spatial_text"]
                inputs = [image, spatial]
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            with torch.no_grad():
                _, image_embeds, spatial_embeds = model(*inputs)

            # Query embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            spatial_embeds = spatial_embeds / spatial_embeds.norm(dim=-1, keepdim=True)
            image_query_list.append(image_embeds.cpu())
            spatial_query_list.append(spatial_embeds.cpu())

            # Retrieve embeddings given the query
            image_retrieved = None
            spatial_retrieved = None
            image_retrieved_list.append(image_retrieved)
            spatial_retrieved_list.append(spatial_retrieved)

        image_query_tensor = torch.cat(image_query_list, dim=0)
        spatial_query_tensor = torch.cat(spatial_query_list, dim=0)

        image_retrieved_tensor = torch.cat(image_retrieved_list, dim=0)
        spatial_retrieved_tensor = torch.cat(spatial_retrieved_list, dim=0)

        # Compute PCC
        pcc = compute_pcc(image_query_tensor, image_retrieved_tensor)
        print(f"{model_name} PCC: {pcc:.4f}")
        wandb_logger.log({"pcc": pcc}, step=0)
        # Compute Recall@K
        for k in [10, 50, 100]:
            recall = recall_at_K(image_query_tensor, image_retrieved_tensor, k=k)
            print(f"{model_name} Recall@{k}: {recall:.4f}")
            wandb_logger.log({f"recall@{k}": recall}, step=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    args = parser.parse_args()
    main(args)