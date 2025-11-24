import torch
import torch.nn as nn
from torchvision import models
import timm
from pathlib import Path


class UNI(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(UNI, self).__init__()
        uni_model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        if checkpoint_path:
            uni_model.load_state_dict(
                torch.load(
                    Path(checkpoint_path, "pytorch_model.bin"), map_location="cpu"
                ),
                strict=True,
            )
        self.model = uni_model

    @torch.no_grad()
    def get_image_features(self, x):
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(x)

class H_optimus_zero(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(H_optimus_zero, self).__init__()
        H_optimus_zero_model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)

        if checkpoint_path:
            H_optimus_zero_model.load_state_dict(
                torch.load(
                    Path(checkpoint_path, "pytorch_model.bin"), map_location="cpu"
                ),
                strict=True,
            )
        self.model = H_optimus_zero_model

    @torch.no_grad()
    def get_image_features(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.model(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        self.model = model

    @torch.no_grad()
    def get_image_features(self, x):
        with torch.inference_mode():
            image_embedding = self.model(x)
            image_embedding = image_embedding.flatten(1)
        return image_embedding

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        return x