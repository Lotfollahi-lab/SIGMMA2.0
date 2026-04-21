"""
EVA Benchmarking for SIGMMA using the kaiko-ai/eva library
============================================================
Supports:
  - Patch-level classification (BreakHis, CRC, Gleason, MHIST, PatchCamelyon)
  - WSI slide-level classification (Camelyon16, PANDA)
  - Semantic segmentation (CoNSeP, MoNuSAC)

Reproduces the EVA leaderboard protocol:
  - Patch classification: 5 runs, 12.5k steps, 20% patience, fit on train, model
    selection on val (MulticlassRecall = balanced accuracy), final report on test.
  - WSI classification: 20 runs, 100 epochs, 20% patience, ABMIL head.
  - Segmentation: 5 runs, 2k steps, 20% patience, Dice monitoring.

Reference:
  https://kaiko-ai.github.io/eva/main/leaderboards/
  https://kaiko-ai.github.io/eva/main/user-guide/advanced/replicate_evaluations/

Usage:
    # Patch-level classification
    python benchmark_image_pathology_task.py \
        --sigmma_config /path/to/config.yaml \
        --datasets breakhis crc gleason mhist patchcam

    # WSI slide-level
    python benchmark_image_pathology_task.py \
        --sigmma_config /path/to/config.yaml \
        --datasets camelyon16 panda

    # Segmentation
    python benchmark_image_pathology_task.py \
        --sigmma_config /path/to/config.yaml \
        --datasets consep monusac

    # All at once (uses task-specific default n_runs: 5 / 20 / 5)
    python benchmark_image_pathology_task.py \
        --sigmma_config /path/to/config.yaml \
        --datasets breakhis crc gleason mhist patchcam camelyon16 panda consep monusac \
        --download

Prerequisites:
    pip install 'kaiko-eva[vision]'
"""

import os
import math
import argparse
import inspect
import traceback
import yaml
from functools import partial

import torch
import torch.nn as nn
import lightning as L
from ml_collections.config_dict import ConfigDict
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import eva
from eva import core
from eva.vision import datasets as eva_datasets
from eva.vision.models.modules import SemanticSegmentationModule
from eva.vision.models.networks.decoders.segmentation import ConvDecoderWithImage
from eva.vision.losses import DiceLoss
from eva.vision.data.transforms.common import ResizeAndCrop
from eva.vision.metrics.defaults import MulticlassSegmentationMetrics
from eva.core.models.transforms import AsDiscrete
from eva.core.models.modules.utils.batch_postprocess import BatchPostProcess
from eva.core.data.transforms import Pad2DTensor
from eva.core.data.transforms.dtype import ArrayToFloatTensor
from eva.core.metrics.structs import MetricsSchema

from sigmma.utils import load_model


class _DebugTransform:
    """Print shape + dtype + uniqueness of a tensor once, pass-through."""
    _printed = False

    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, x):
        if not _DebugTransform._printed:
            import torch as _t
            print(f"[DBG-{self.tag}] shape={tuple(x.shape)}, dtype={x.dtype}, "
                  f"min={x.min().item()}, max={x.max().item()}, "
                  f"n_unique={len(_t.unique(x))}")
            _DebugTransform._printed = True
        return x


class _SqueezeArgmax:
    """argmax over channel dim, drop the channel dim explicitly."""
    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, x):
        import torch as _t
        return _t.argmax(x, dim=self.dim)  # [B, C, H, W] -> [B, H, W]


# ============================================================
# 1. SIGMMA backbone wrappers
# ============================================================
class SIGMMABackbone(nn.Module):
    """
    For classification tasks.
    Returns CLS token embedding: (B, D).
    """

    def __init__(self, config_path: str):
        super().__init__()
        config_file = yaml.load(open(config_path, "r"), yaml.FullLoader)
        config = ConfigDict(config_file)

        model = load_model(
            config,
            pretrained_image_encoder=config.model.pretrained_image_encoder,
            pretrained_ckpt_path=config.model.checkpoint_path,
            train=False,
        )
        self.encoder = model.image_encoder
        self.encoder.eval()

        # Defensive freeze (offline pipeline uses no_grad, but be explicit).
        for p in self.encoder.parameters():
            p.requires_grad = False

        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = self.encoder(dummy)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 3:
                out = out[:, 0, :]
            self._out_dim = out.shape[-1]

        print(f"[SIGMMABackbone] Loaded. Output dim = {self._out_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 3:
            out = out[:, 0, :]
        return out

    @property
    def out_features(self) -> int:
        return self._out_dim


class SIGMMABackboneSegmentation(nn.Module):
    """
    For segmentation tasks.
    Returns a list of multi-scale feature maps, each (B, D, H, W), extracted
    from the last `n_intermediate` ViT blocks via timm's get_intermediate_layers.
    EVA's SemanticSegmentationModule expects a list of feature tensors.
    """

    def __init__(self, config_path: str, n_intermediate: int = 1):
        super().__init__()
        config_file = yaml.load(open(config_path, "r"), yaml.FullLoader)
        config = ConfigDict(config_file)

        model = load_model(
            config,
            pretrained_image_encoder=config.model.pretrained_image_encoder,
            pretrained_ckpt_path=config.model.checkpoint_path,
            train=False,
        )
        self.model = model.image_encoder.model  # inner timm ViT
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # timm renamed the prefix-token kwarg between versions; pick whichever exists.
        sig = inspect.signature(self.model.get_intermediate_layers)
        self._intermediate_kwargs = dict(n=n_intermediate, reshape=True, norm=True)
        if "return_prefix_tokens" in sig.parameters:
            self._intermediate_kwargs["return_prefix_tokens"] = False
        elif "return_class_token" in sig.parameters:
            self._intermediate_kwargs["return_class_token"] = False

        device = next(self.model.parameters()).device
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            feats = self.model.get_intermediate_layers(dummy, **self._intermediate_kwargs)
            self._out_dim = feats[0].shape[1]
            self._n_layers = len(feats)

        print(f"[SIGMMABackboneSegmentation] Loaded. "
              f"Output: list of {self._n_layers} feature maps, dim={self._out_dim}")

    def forward(self, x: torch.Tensor):
        feats = self.model.get_intermediate_layers(x, **self._intermediate_kwargs)
        return list(feats)

    @property
    def out_features(self) -> int:
        # EVA's decoder concatenates the list of feature maps along channels,
        # so the decoder's first conv must accept D * n_layers input channels.
        return self._out_dim * self._n_layers


# ============================================================
# 2. Dataset configs
# ============================================================
EVA_CLASSIFICATION_TASKS = {
    "breakhis": {"dataset_cls": "BreaKHis", "num_classes": 4},
    "crc":      {"dataset_cls": "CRC",      "num_classes": 9},
    "gleason":  {"dataset_cls": "GleasonArvaniti", "num_classes": 4},
    "mhist":    {"dataset_cls": "MHIST",    "num_classes": 2},
    "patchcam": {"dataset_cls": "PatchCamelyon", "num_classes": 2},
}

EVA_WSI_TASKS = {
    # Camelyon16: leaderboard uses 1000 foreground patches per slide.
    "camelyon16": {"dataset_cls": "Camelyon16", "num_classes": 1, "n_patches": 1000},
    "panda":      {"dataset_cls": "PANDASmall", "num_classes": 6, "n_patches": 200},
}

EVA_SEGMENTATION_TASKS = {
    # CoNSeP: 0=bg + 4 nuclei = 5 classes, no ignore.
    "consep": {
        "dataset_cls": "CoNSeP",
        "num_classes": 5,
        "metric_num_classes": 5,
        "ignore_index": None,
    },
    # MoNuSAC: decoder predicts 5 classes (bg + 4 nuclei). The test set adds
    # a 6th "Ambiguous" class (=5) for fuzzy regions; EVA ignores it in both
    # loss and metrics (metric_num_classes=6 + ignore_index=5).
    "monusac": {
        "dataset_cls": "MoNuSAC",
        "num_classes": 5,
        "metric_num_classes": 6,
        "ignore_index": 5,
    },
}

# EVA leaderboard: 5 runs for patch/segmentation, 20 runs for WSI.
DEFAULT_N_RUNS = {
    **{k: 5 for k in EVA_CLASSIFICATION_TASKS},
    **{k: 20 for k in EVA_WSI_TASKS},
    **{k: 5 for k in EVA_SEGMENTATION_TASKS},
}


def _safe_load_split(dataset_cls, root, split, download, **kwargs):
    try:
        ds = dataset_cls(root=root, split=split, download=download, **kwargs)
        _ = len(ds)  # forces _image_files → _data_directory evaluation
        return ds
    except Exception as e:
        print(f"  [INFO] split='{split}' not available: {e}")
        return None

# ============================================================
# 3. Patch-level classification
# ============================================================
def run_classification(
    dataset_name: str,
    sigmma_config_path: str,
    data_root: str,
    output_root: str,
    download: bool = False,
    n_runs: int = 5,
    batch_size: int = 256,
    max_steps: int = 12500,
    lr: float = 0.0003,
):
    task = EVA_CLASSIFICATION_TASKS[dataset_name]
    num_classes = task["num_classes"]
    DatasetClass = getattr(eva_datasets, task["dataset_cls"])

    backbone = SIGMMABackbone(sigmma_config_path)
    in_features = backbone.out_features

    transform = ResizeAndCrop(
        size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    task_output_dir = os.path.join(output_root, dataset_name)
    embeddings_dir = os.path.join(task_output_dir, "embeddings")

    # EVA leaderboard: EarlyStopping patience=500 validation epochs (effectively loose).
    patience = 500

    # ------------------------------------------------------------
    # Step 1: Extract embeddings once (before any run).
    # ------------------------------------------------------------
    manifest_path = os.path.join(embeddings_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"\n[{dataset_name}] Extracting embeddings -> {embeddings_dir}")

        split_specs = [("train", download), ("val", False), ("test", False)]
        predict_datasets = []
        dataloader_idx_map = {}
        for i, (split, dl) in enumerate(split_specs):
            ds = _safe_load_split(
                DatasetClass,
                root=os.path.join(data_root, dataset_name),
                split=split,
                download=dl,
                transforms=transform,
            )
            if ds is not None:
                predict_datasets.append(ds)
                dataloader_idx_map[len(predict_datasets) - 1] = split

        trainer_predict = core.Trainer(
            default_root_dir=task_output_dir,
            max_steps=0,
            accelerator="auto",
            devices=1,
            callbacks=[
                eva.callbacks.ClassificationEmbeddingsWriter(
                    output_dir=embeddings_dir,
                    dataloader_idx_map=dataloader_idx_map,
                    backbone=backbone,
                    overwrite=True,
                ),
            ],
        )
        predict_datamodule = core.DataModule(
            datasets=core.DatasetsSchema(predict=predict_datasets),
            dataloaders=core.DataloadersSchema(
                predict=core.DataLoader(batch_size=64, num_workers=4),
            ),
        )
        trainer_predict.predict(
            model=core.HeadModule(
                head=nn.Linear(in_features, num_classes),
                criterion=nn.CrossEntropyLoss(),
                optimizer=partial(
                    torch.optim.AdamW, lr=lr,
                    weight_decay=0.0, betas=(0.9, 0.999),
                ),
            ),
            datamodule=predict_datamodule,
        )

    # Determine which splits actually have embeddings.
    available_splits = {"train", "val"}
    try:
        import csv
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            available_splits = {row["split"] for row in reader if row.get("split")}
    except Exception:
        pass
    has_test_embeddings = "test" in available_splits

    # ------------------------------------------------------------
    # Step 2: Fit linear probe (n_runs with different seeds).
    # ------------------------------------------------------------
    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Run {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        run_output_dir = os.path.join(task_output_dir, f"run_{run_idx}")

        trainer_fit = core.Trainer(
            default_root_dir=run_output_dir,
            max_steps=max_steps,
            accelerator="auto",
            devices=1,
            callbacks=[
                eva.callbacks.ConfigurationLogger(),
                ModelCheckpoint(
                    filename="best", save_top_k=1,
                    monitor="val/MulticlassAccuracy", mode="max",
                ),
                EarlyStopping(
                    min_delta=0, patience=patience,
                    monitor="val/MulticlassAccuracy", mode="max",
                ),
            ],
        )

        model = core.HeadModule(
            head=nn.Linear(in_features, num_classes),
            criterion=nn.CrossEntropyLoss(),
            optimizer=partial(
                torch.optim.AdamW, lr=lr,
                weight_decay=0.0, betas=(0.9, 0.999),
            ),
            metrics=MetricsSchema(
                common=[
                    eva.metrics.AverageLoss(),
                    eva.metrics.MulticlassClassificationMetrics(num_classes=num_classes),
                ]
            ),
        )

        fit_datasets = dict(
            train=eva.datasets.EmbeddingsClassificationDataset(
                root=embeddings_dir, manifest_file="manifest.csv", split="train",
            ),
            val=eva.datasets.EmbeddingsClassificationDataset(
                root=embeddings_dir, manifest_file="manifest.csv", split="val",
            ),
        )
        dataloaders_dict = dict(
            train=core.DataLoader(batch_size=batch_size, num_workers=4, shuffle=True),
            val=core.DataLoader(batch_size=batch_size, num_workers=4),
        )
        if has_test_embeddings:
            fit_datasets["test"] = eva.datasets.EmbeddingsClassificationDataset(
                root=embeddings_dir, manifest_file="manifest.csv", split="test",
            )
            dataloaders_dict["test"] = core.DataLoader(batch_size=batch_size, num_workers=4)

        fit_datamodule = core.DataModule(
            datasets=core.DatasetsSchema(**fit_datasets),
            dataloaders=core.DataloadersSchema(**dataloaders_dict),
        )

        trainer_fit.fit(model=model, datamodule=fit_datamodule)
        if has_test_embeddings:
            trainer_fit.test(model=model, datamodule=fit_datamodule, ckpt_path="best")
        else:
            # Fall back to val reporting, per EVA protocol when test is unavailable.
            trainer_fit.validate(model=model, datamodule=fit_datamodule, ckpt_path="best")

    print(f"\n[{dataset_name}] Done. Results in: {task_output_dir}")


# ============================================================
# 4. WSI slide-level classification
# ============================================================
def run_wsi(
    dataset_name: str,
    sigmma_config_path: str,
    data_root: str,
    output_root: str,
    download: bool = False,
    n_runs: int = 20,
    batch_size: int = 32,
    max_epochs: int = 100,
    lr: float = 0.001,
):
    task = EVA_WSI_TASKS[dataset_name]
    num_classes = task["num_classes"]
    n_patches = task["n_patches"]
    DatasetClass = getattr(eva_datasets, task["dataset_cls"])

    backbone = SIGMMABackbone(sigmma_config_path)
    in_features = backbone.out_features

    transform = ResizeAndCrop(
        size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    task_output_dir = os.path.join(output_root, dataset_name)
    embeddings_dir = os.path.join(task_output_dir, "embeddings")
    manifest_path = os.path.join(embeddings_dir, "manifest.csv")

    # ------------------------------------------------------------
    # Step 1: Extract embeddings once.
    # ------------------------------------------------------------
    if not os.path.exists(manifest_path):
        print(f"\n[{dataset_name}] Extracting WSI embeddings...")

        split_specs = [("train", download), ("val", False), ("test", False)]
        predict_datasets = []
        dataloader_idx_map = {}
        for i, (split, dl) in enumerate(split_specs):
            ds = _safe_load_split(
                DatasetClass,
                root=os.path.join(data_root, dataset_name),
                split=split,
                download=dl,
                sampler=eva.vision.data.wsi.patching.samplers.ForegroundGridSampler(
                    max_samples=n_patches,
                ),
                width=224,
                height=224,
                target_mpp=0.25,
                image_transforms=transform,
            )
            if ds is not None:
                predict_datasets.append(ds)
                dataloader_idx_map[len(predict_datasets) - 1] = split

        trainer_predict = core.Trainer(
            default_root_dir=task_output_dir,
            max_steps=0,
            accelerator="auto",
            devices=1,
            callbacks=[
                eva.callbacks.ClassificationEmbeddingsWriter(
                    output_dir=embeddings_dir,
                    save_every_n=10000,
                    dataloader_idx_map=dataloader_idx_map,
                    metadata_keys=["wsi_id"],
                    backbone=backbone,
                    overwrite=True,
                ),
            ],
        )
        predict_datamodule = core.DataModule(
            datasets=core.DatasetsSchema(predict=predict_datasets),
            dataloaders=core.DataloadersSchema(
                predict=core.DataLoader(batch_size=64, num_workers=4),
            ),
        )
        trainer_predict.predict(
            model=core.HeadModule(
                head=nn.Linear(in_features, num_classes),
                criterion=(nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()),
                optimizer=partial(
                    torch.optim.AdamW, lr=lr,
                    weight_decay=0.0, betas=(0.9, 0.999),
                ),
            ),
            datamodule=predict_datamodule,
        )

    # Detect available splits from the manifest.
    available_splits = {"train", "val"}
    try:
        import csv
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            available_splits = {row["split"] for row in reader if row.get("split")}
    except Exception:
        pass
    has_test_embeddings = "test" in available_splits

    # ------------------------------------------------------------
    # Step 2: Fit ABMIL head (n_runs with different seeds).
    # ------------------------------------------------------------
    is_binary = (num_classes == 1)
    monitor_metric = "val/BinaryBalancedAccuracy" if is_binary else "val/MulticlassRecall"

    # 20% of max_epochs, per leaderboard protocol.
    patience_epochs = max(1, int(0.2 * max_epochs))

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Run {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        run_output_dir = os.path.join(task_output_dir, f"run_{run_idx}")

        trainer_fit = core.Trainer(
            default_root_dir=run_output_dir,
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[
                eva.callbacks.ConfigurationLogger(),
                ModelCheckpoint(
                    filename="best", save_top_k=1,
                    monitor=monitor_metric, mode="max",
                ),
                EarlyStopping(
                    min_delta=0, patience=patience_epochs,
                    monitor=monitor_metric, mode="max",
                ),
            ],
        )

        head = eva.vision.models.networks.ABMIL(
            input_size=in_features,
            output_size=num_classes,
            projected_input_size=128,
        )

        if is_binary:
            metrics_schema = MetricsSchema(
                common=[
                    eva.metrics.AverageLoss(),
                    eva.metrics.BinaryClassificationMetrics(),
                ]
            )
            criterion = nn.BCEWithLogitsLoss()
            target_transforms = ArrayToFloatTensor()
        else:
            metrics_schema = MetricsSchema(
                common=[
                    eva.metrics.AverageLoss(),
                    eva.metrics.MulticlassClassificationMetrics(num_classes=num_classes),
                ]
            )
            criterion = nn.CrossEntropyLoss()
            target_transforms = None

        model = core.HeadModule(
            head=head,
            criterion=criterion,
            optimizer=partial(
                torch.optim.AdamW, lr=lr,
                weight_decay=0.0, betas=(0.9, 0.999),
            ),
            metrics=metrics_schema,
        )

        ds_kwargs = dict(
            root=embeddings_dir,
            manifest_file="manifest.csv",
            embeddings_transforms=Pad2DTensor(pad_size=n_patches),
        )
        if target_transforms is not None:
            ds_kwargs["target_transforms"] = target_transforms

        fit_datasets = {
            "train": eva.datasets.MultiEmbeddingsClassificationDataset(split="train", **ds_kwargs),
            "val":   eva.datasets.MultiEmbeddingsClassificationDataset(split="val",   **ds_kwargs),
        }
        dl_kwargs = dict(batch_size=batch_size, num_workers=4)
        dataloaders_dict = {
            "train": core.DataLoader(shuffle=True, **dl_kwargs),
            "val":   core.DataLoader(**dl_kwargs),
        }
        if has_test_embeddings:
            fit_datasets["test"] = eva.datasets.MultiEmbeddingsClassificationDataset(
                split="test", **ds_kwargs,
            )
            dataloaders_dict["test"] = core.DataLoader(**dl_kwargs)

        fit_datamodule = core.DataModule(
            datasets=core.DatasetsSchema(**fit_datasets),
            dataloaders=core.DataloadersSchema(**dataloaders_dict),
        )

        trainer_fit.fit(model=model, datamodule=fit_datamodule)
        if has_test_embeddings:
            trainer_fit.test(model=model, datamodule=fit_datamodule, ckpt_path="best")
        else:
            trainer_fit.validate(model=model, datamodule=fit_datamodule, ckpt_path="best")

    print(f"\n[{dataset_name}] Done. Results in: {task_output_dir}")


# ============================================================
# 5. Segmentation
# ============================================================
def run_segmentation(
    dataset_name: str,
    sigmma_config_path: str,
    data_root: str,
    output_root: str,
    download: bool = False,
    n_runs: int = 5,
    batch_size: int = 64,
    max_steps: int = 2000,
    lr: float = 0.002,
):
    task = EVA_SEGMENTATION_TASKS[dataset_name]
    num_classes = task["num_classes"]
    metric_num_classes = task.get("metric_num_classes", num_classes)
    ignore_index = task.get("ignore_index")
    DatasetClass = getattr(eva_datasets, task["dataset_cls"])

    seg_backbone = SIGMMABackboneSegmentation(sigmma_config_path)
    in_features = seg_backbone.out_features

    # Train vs val transforms (matches EVA online/monusac.yaml):
    #   train: RandomResizedCrop(224) + ToDtype(float32, scale=True) + Normalize
    #   val:   ResizeAndCrop(224, mean, std)
    from torchvision.transforms import v2 as tv2
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]
    train_transform = tv2.Compose([
        tv2.RandomResizedCrop(size=224),
        tv2.ToDtype(torch.float32, scale=True),
        tv2.Normalize(mean=_MEAN, std=_STD),
    ])
    eval_transform = ResizeAndCrop(size=224, mean=_MEAN, std=_STD)

    task_output_dir = os.path.join(output_root, dataset_name)

    # 20% of max_steps, per leaderboard protocol.
    patience_steps = int(0.2 * max_steps)

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"[{dataset_name}] Run {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        run_output_dir = os.path.join(task_output_dir, f"run_{run_idx}")

        trainer = core.Trainer(
            default_root_dir=run_output_dir,
            max_steps=max_steps,
            log_every_n_steps=6,
            accelerator="auto",
            devices=1,
            callbacks=[
                eva.callbacks.ConfigurationLogger(),
                ModelCheckpoint(
                    filename="best", save_top_k=1,
                    monitor="val/MonaiDiceScore", mode="max",
                ),
                EarlyStopping(
                    min_delta=0, patience=patience_steps,
                    monitor="val/MonaiDiceScore", mode="max",
                ),
            ],
        )

        decoder = ConvDecoderWithImage(
            in_features=in_features,
            num_classes=num_classes,
        )

        model = SemanticSegmentationModule(
            encoder=seg_backbone,
            decoder=decoder,
            criterion=DiceLoss(softmax=True, batch=True, ignore_index=ignore_index),
            lr_multiplier_encoder=0.0,  # freeze encoder
            optimizer=partial(
                torch.optim.AdamW, lr=lr,
                weight_decay=0.0, betas=(0.9, 0.999),
            ),
            postprocess=BatchPostProcess(
                predictions_transforms=[_SqueezeArgmax(dim=1)],
            ),
            metrics=MetricsSchema(
                common=[eva.metrics.AverageLoss()],
                evaluation=[MulticlassSegmentationMetrics(
                    num_classes=metric_num_classes,
                    ignore_index=ignore_index,
                )],
            ),
        )

        train_ds = DatasetClass(
            root=os.path.join(data_root, dataset_name),
            split="train",
            download=download,
            transforms=train_transform,
        )

        # Resolve val / test splits per dataset.
        #   - CoNSeP: train / val / test (all three available)
        #   - MoNuSAC: train / test only — follow EVA convention of using
        #     "test" for both early-stopping monitor and final reporting.
        val_split = "val"
        test_split = "test"
        val_ds = _safe_load_split(
            DatasetClass,
            root=os.path.join(data_root, dataset_name),
            split=val_split,
            download=False,
            transforms=eval_transform,
        )
        if val_ds is None:
            # Dataset has no val split (e.g., MoNuSAC). Use test for monitoring.
            val_split = "test"
            val_ds = DatasetClass(
                root=os.path.join(data_root, dataset_name),
                split=val_split,
                download=False,
                transforms=eval_transform,
            )
            test_ds = None  # val == test
        else:
            test_ds = _safe_load_split(
                DatasetClass,
                root=os.path.join(data_root, dataset_name),
                split=test_split,
                download=False,
                transforms=eval_transform,
            )

        # MoNuSAC train has ~4 batches/epoch; multi-worker loading on shared
        # Lustre hosts has been causing intermittent worker crashes
        # (exit code 1, details lost by multiprocessing). Fall back to
        # num_workers=0 — no perf loss given the tiny iter count.
        # prefetch_factor/persistent_workers must be disabled for num_workers=0.
        seg_dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
        )
        datasets_schema_kwargs = dict(train=train_ds, val=val_ds)
        dataloaders_schema_kwargs = dict(
            train=core.DataLoader(**seg_dl_kwargs, shuffle=True),
            val=core.DataLoader(**seg_dl_kwargs),
        )
        if test_ds is not None:
            datasets_schema_kwargs["test"] = test_ds
            dataloaders_schema_kwargs["test"] = core.DataLoader(**seg_dl_kwargs)

        datamodule = core.DataModule(
            datasets=core.DatasetsSchema(**datasets_schema_kwargs),
            dataloaders=core.DataloadersSchema(**dataloaders_schema_kwargs),
        )

        # ---- DIAGNOSTIC: inspect mask value range before training ----
        if os.environ.get("SIGMMA_MASK_DEBUG", "0") == "1":
            import numpy as _np
            import torch as _torch
            import sys as _sys

            print("\n[DIAG] Sweeping ALL train batches for target value range...")
            probe_loader = _torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, num_workers=2, shuffle=False,
            )
            g_min, g_max = float("inf"), float("-inf")
            g_unique = set()
            n_batches = 0
            sample_batch = None
            for batch in probe_loader:
                tgt = batch[1] if isinstance(batch, (tuple, list)) else batch["target"]
                g_min = min(g_min, tgt.min().item())
                g_max = max(g_max, tgt.max().item())
                g_unique.update(_torch.unique(tgt).tolist())
                n_batches += 1
                if sample_batch is None:
                    sample_batch = (batch[0] if isinstance(batch, (tuple, list)) else batch["data"], tgt)
            print(f"[DIAG] train: n_batches={n_batches}, global min={g_min}, max={g_max}, "
                  f"unique={sorted(g_unique)}")
            print(f"[DIAG] configured num_classes={num_classes}")

            print("\n[DIAG] Also sweeping val loader...")
            val_probe = _torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, num_workers=2, shuffle=False,
            )
            v_min, v_max = float("inf"), float("-inf")
            v_unique = set()
            for batch in val_probe:
                tgt = batch[1] if isinstance(batch, (tuple, list)) else batch["target"]
                v_min = min(v_min, tgt.min().item())
                v_max = max(v_max, tgt.max().item())
                v_unique.update(_torch.unique(tgt).tolist())
            print(f"[DIAG] val: min={v_min}, max={v_max}, unique={sorted(v_unique)}")

            print("\n[DIAG] Simulating DiceLoss forward with dummy prediction (CPU)...")
            data_b, tgt_b = sample_batch
            print(f"[DIAG] sample batch: data={tuple(data_b.shape)} {data_b.dtype}, "
                  f"target={tuple(tgt_b.shape)} {tgt_b.dtype}")
            dummy_pred = _torch.randn(tgt_b.shape[0], num_classes, *tgt_b.shape[-2:])
            try:
                loss_val = DiceLoss(softmax=True, batch=True)(dummy_pred, tgt_b)
                print(f"[DIAG] DiceLoss OK, loss={loss_val.item():.4f}")
            except Exception as _e:
                import traceback as _tb
                print(f"[DIAG] DiceLoss FAILED: {type(_e).__name__}: {_e}")
                _tb.print_exc()

            print("\n[DIAG] Simulating MulticlassSegmentationMetrics (CPU)...")
            try:
                _metrics = MulticlassSegmentationMetrics(num_classes=num_classes)
                # argmax'd predictions as metrics expect
                pred_cls = dummy_pred.argmax(dim=1, keepdim=True)
                _metrics.update(pred_cls, tgt_b)
                out = _metrics.compute()
                print(f"[DIAG] Metrics OK: keys={list(out.keys())}")
            except Exception as _e:
                import traceback as _tb
                print(f"[DIAG] Metrics FAILED: {type(_e).__name__}: {_e}")
                _tb.print_exc()

            print("[DIAG] done — exiting.\n")
            _sys.exit(0)
        # ---- END DIAGNOSTIC ----

        trainer.fit(model=model, datamodule=datamodule)
        if test_ds is not None:
            trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
        else:
            # MoNuSAC case: val == test; report the monitored (best) state.
            trainer.validate(model=model, datamodule=datamodule, ckpt_path="best")

    print(f"\n[{dataset_name}] Done. Results in: {task_output_dir}")


# ============================================================
# 6. Main
# ============================================================
def main():
    ALL_TASKS = (
        list(EVA_CLASSIFICATION_TASKS.keys())
        + list(EVA_WSI_TASKS.keys())
        + list(EVA_SEGMENTATION_TASKS.keys())
    )

    parser = argparse.ArgumentParser(description="EVA Benchmark for SIGMMA")
    parser.add_argument("--sigmma_config", type=str, required=True,
                        help="Path to SIGMMA config yaml")
    parser.add_argument("--datasets", nargs="+",
                        default=["breakhis", "crc", "gleason", "mhist", "patchcam"],
                        choices=ALL_TASKS)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_root", type=str, default="./logs/sigmma")
    parser.add_argument("--download", action="store_true",
                        help="Download datasets if not present")
    parser.add_argument("--n_runs", type=int, default=None,
                        help="Override the default number of runs. "
                             "Defaults follow the EVA leaderboard: "
                             "5 (patch/seg), 20 (WSI).")
    args = parser.parse_args()

    # EVA convention: seed ONCE at process startup (matches their
    # _configure_random_seed in src/eva/core/cli/setup.py which calls
    # pl_seed.seed_everything(42, workers=True) exactly once in setup()).
    # The per-task n_runs loops do NOT re-seed — successive runs advance
    # the RNG state naturally, so each run differs but the whole session
    # is fully reproducible from this single call.
    L.seed_everything(42, workers=True)

    for ds in args.datasets:
        n_runs = args.n_runs if args.n_runs is not None else DEFAULT_N_RUNS[ds]
        try:
            if ds in EVA_CLASSIFICATION_TASKS:
                run_classification(
                    dataset_name=ds,
                    sigmma_config_path=args.sigmma_config,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    download=args.download,
                    n_runs=n_runs,
                )
            elif ds in EVA_WSI_TASKS:
                run_wsi(
                    dataset_name=ds,
                    sigmma_config_path=args.sigmma_config,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    download=args.download,
                    n_runs=n_runs,
                )
            elif ds in EVA_SEGMENTATION_TASKS:
                run_segmentation(
                    dataset_name=ds,
                    sigmma_config_path=args.sigmma_config,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    download=args.download,
                    n_runs=n_runs,
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n[ERROR] {ds}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()