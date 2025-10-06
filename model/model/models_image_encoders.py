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

    def get_image_features(self, x):
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding
    
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

    def get_image_features(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = nn.Sequential(*list(resnet50.children())[:-1])
        self.model = model

    def get_image_features(self, x):
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding