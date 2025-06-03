import os
import torch
import pandas as pd

from ml_collections.config_dict import ConfigDict
import yaml

import timm

from model.utils.train_utils import train_one_epoch, save_checkpoint
from novae import Novae
from huggingface_hub import login, hf_hub_download

import wandb


def main(args):
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)

    model_name = config.get("model.name")
    out_dir = config.get("outdir")

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    train_dataset = None
    val_dataset = None

    model, optimizer, lr_scheduler = None, None, None

    # Load pretrained UNI model (UNI checkpoints "pytorch_model.bin" should be downloaded in assets)
    pretrained_ckpt_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(pretrained_ckpt_dir, exist_ok=True)
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
            os.path.join(pretrained_ckpt_dir, "pytorch_model.bin"), map_location="cpu"
        ),
        strict=True,
    )
    pretrained_image_encoder.eval()

    # Load pretrained spatial encoder
    pretrained_spatial_encoder = Novae.from_pretrained("MICS-Lab/novae-human-0")

    global_step, first_epoch, num_train_epochs = 0, 0, config.get("num_train_epochs", 10)
    
    for current_epoch in range(first_epoch, num_train_epochs):
        global_step = train_one_epoch(config, model, optimizer, lr_scheduler)

    save_checkpoint()


if __name__ == "__main__":
    main()
