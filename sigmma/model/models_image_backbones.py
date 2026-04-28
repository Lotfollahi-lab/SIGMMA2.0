import torch
import torch.nn as nn
from torchvision import models
import timm
from pathlib import Path
from peft.tuners.lora import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft import LoraConfig, inject_adapter_in_model

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
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(x)

def apply_lora_to_backbone(backbone: nn.Module, lora_cfg, num_adapters: int = 1) -> nn.Module:
    """
    Inject LoRA adapters (in-place) into an image backbone's inner model, using
    HuggingFace peft. Works uniformly for any wrapper class in this module (UNI,
    H_optimus_zero, ResNet50, ...) that exposes the inner torch module as `.model`.

    The inner module's original type is preserved (no PeftModel wrapping), so
    native methods like `get_intermediate_layers` or `set_grad_checkpointing`
    remain callable.

    Base weights are frozen; only LoRA A/B matrices are left trainable. The
    appropriate `target_modules` depends on the backbone:
      - timm ViT family (UNI, H-optimus-0, ...):   ["qkv", "proj"] (attention)
                                                   + optional ["fc1", "fc2"] (mlp)
      - torchvision ResNet:                        ["conv1", "conv2", "conv3"]

    When `num_adapters > 1`, multiple named adapters ("scale_0", "scale_1", ...)
    are stacked on the same frozen backbone. Forward-time routing is controlled
    by `set_lora_adapter` below. The full list of adapter names is stored on
    `backbone.lora_adapter_names`; single-adapter case leaves it as None.

    Args:
        backbone: an nn.Module wrapper exposing `.model` (the target to patch).
        lora_cfg: object (e.g. ml_collections ConfigDict) with attributes
                  r, alpha, dropout, target_modules, bias.
        num_adapters: number of independent LoRA adapters to inject. >1 enables
                      per-scale routing.

    Returns:
        The same `backbone` instance, with LoRA layers injected into `backbone.model`.
    """
    if not hasattr(backbone, "model") or not isinstance(backbone.model, nn.Module):
        raise AttributeError(
            f"apply_lora_to_backbone expected `.model` (nn.Module) on "
            f"{type(backbone).__name__}; cannot inject LoRA."
        )

    target_modules = list(getattr(lora_cfg, "target_modules", ["qkv", "proj"]))

    def _make_config():
        # Build a fresh LoraConfig per adapter. peft mutates configs internally
        # (e.g. resolving target_modules against the model), so reusing one
        # instance across adapters is unsafe.
        return LoraConfig(
            r=int(getattr(lora_cfg, "r", 16)),
            lora_alpha=int(getattr(lora_cfg, "alpha", 32)),
            lora_dropout=float(getattr(lora_cfg, "dropout", 0.0)),
            target_modules=target_modules,
            bias=str(getattr(lora_cfg, "bias", "none")),
        )

    if num_adapters <= 1:
        # Single adapter: inject_adapter_in_model returns a LoraModel whose
        # `.model` is the (in-place patched) original module. We discard the
        # LoraModel wrapper and keep the patched base module to preserve type.
        lora_wrapper = inject_adapter_in_model(_make_config(), backbone.model)
        backbone.model = lora_wrapper.model
        backbone.lora_adapter_names = None
    else:
        adapter_names = [f"scale_{i}" for i in range(num_adapters)]

        # First adapter: inject_adapter_in_model creates the LoraModel and
        # patches base modules in place with adapter_names[0]. Note that
        # inject_adapter_in_model uses the default adapter name "default", so
        # we go through LoraModel directly to control the name.
        lora_wrapper = LoraModel(
            backbone.model, _make_config(), adapter_name=adapter_names[0]
        )

        # Additional adapters: register the config under the new name, then
        # call inject_adapter (BaseTuner method) to patch the same base module
        # in place with that adapter's A/B matrices. This is the supported
        # multi-adapter pattern when avoiding PeftModel wrapping.
        for name in adapter_names[1:]:
            lora_wrapper.peft_config[name] = _make_config()
            lora_wrapper.inject_adapter(lora_wrapper.model, name)

        # `lora_wrapper.model` is the same base module, now carrying all
        # adapters. Replace backbone.model with it (identity in practice, but
        # explicit) and drop the wrapper.
        backbone.model = lora_wrapper.model
        backbone.lora_adapter_names = adapter_names

    for name, p in backbone.model.named_parameters():
        p.requires_grad = "lora_" in name

    return backbone

def set_lora_adapter(backbone: nn.Module, adapter_name: str) -> None:
    """
    Switch the active LoRA adapter on every peft tuner layer inside `backbone`.
    Used to route each scale's forward pass through its own adapter when
    `apply_lora_to_backbone(..., num_adapters>1)` was used.

    `set_adapter` in peft also toggles requires_grad of A/B matrices to "only
    the active one is trainable". That is unhelpful here — we train all
    per-scale adapters jointly — and it can also confuse DDP, which snapshots
    trainable params at wrap time. So we restore requires_grad=True on every
    lora_* parameter after switching.
    """
    for m in backbone.modules():
        if isinstance(m, BaseTunerLayer):
            m.set_adapter(adapter_name)
    for n, p in backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

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
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            image_embedding = self.model(x)
        return image_embedding

    def forward(self, x):
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