"""
Per-scale segmentation benchmark for SIGMMA + per-scale LoRA.

Loads a SIGMMA checkpoint trained with `lora.per_scale: true` (three independent
adapters: scale_0 / scale_1 / scale_2 = micro / meso / macro), and for each
scale activates the corresponding LoRA adapter on UNI before running the EVA
segmentation benchmark (CoNSeP / MoNuSAC). Produces three independent runs per
dataset, one per scale, with results placed under
    {output_root}/scale_{0,1,2}/{dataset}/run_*

Usage:
    python scripts/benchmark_segmentation_per_scale.py \
        --sigmma_config configs/image_pathology_segmentation_per_scale_sigmma.yaml \
        --datasets consep monusac \
        --data_root ./data \
        --output_root ./outputs/eva_bench/sigmma_per_scale \
        --n_runs 5
"""

import argparse
import os
import traceback

import lightning as L

from benchmark_image_pathology_task import (
    EVA_SEGMENTATION_TASKS,
    SIGMMABackboneSegmentation,
    run_segmentation,
)
from sigmma.model.models_image_backbones import set_lora_adapter


SCALE_NAMES = {0: "micro", 1: "meso", 2: "macro"}


class SIGMMABackboneSegmentationPerScale(SIGMMABackboneSegmentation):
    """SIGMMA segmentation backbone with a fixed active LoRA adapter.

    The base class already injects all three per-scale adapters (because
    ``lora.per_scale: true`` is read by ``load_model``). Here we additionally
    activate one adapter at construction time and never switch — so every
    forward pass routes UNI through that scale's LoRA delta only.
    """

    def __init__(self, config_path: str, scale_idx: int, n_intermediate: int = 1):
        super().__init__(config_path, n_intermediate=n_intermediate)
        adapter_name = f"scale_{scale_idx}"
        # set_lora_adapter walks `modules()` for PEFT BaseTunerLayer instances,
        # so passing the inner timm ViT (`self.model`) is sufficient — the
        # tuner layers were monkey-patched into its qkv / proj modules.
        set_lora_adapter(self.model, adapter_name)
        self.scale_idx = scale_idx
        self.scale_name = SCALE_NAMES.get(scale_idx, str(scale_idx))
        print(f"[PerScale] scale_idx={scale_idx} ({self.scale_name}); "
              f"active LoRA adapter: {adapter_name}")


def main():
    parser = argparse.ArgumentParser(description="Per-scale SIGMMA segmentation benchmark")
    parser.add_argument("--sigmma_config", required=True, type=str,
                        help="Path to SIGMMA pathology config (must have "
                             "lora.per_scale: true and a checkpoint_path).")
    parser.add_argument("--datasets", nargs="+",
                        default=list(EVA_SEGMENTATION_TASKS.keys()),
                        choices=list(EVA_SEGMENTATION_TASKS.keys()))
    parser.add_argument("--scales", nargs="+", type=int, default=[0, 1, 2],
                        choices=[0, 1, 2],
                        help="Which scale indices to benchmark "
                             "(0=micro, 1=meso, 2=macro).")
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--output_root", default="./outputs/eva_bench/sigmma_per_scale", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of seeds per (scale, dataset). EVA "
                             "leaderboard segmentation protocol is 5.")
    args = parser.parse_args()

    L.seed_everything(42, workers=True)

    for scale_idx in args.scales:
        scale_output_root = os.path.join(args.output_root, f"scale_{scale_idx}")
        # Build the backbone once per scale (loads LoRA-injected SIGMMA + sets
        # the active adapter); reuse it across datasets at this scale.
        seg_backbone = SIGMMABackboneSegmentationPerScale(
            args.sigmma_config, scale_idx=scale_idx,
        )

        for ds in args.datasets:
            try:
                run_segmentation(
                    dataset_name=ds,
                    sigmma_config_path=args.sigmma_config,
                    data_root=args.data_root,
                    output_root=scale_output_root,
                    download=args.download,
                    n_runs=args.n_runs,
                    backbone_type="sigmma",
                    seg_backbone=seg_backbone,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[ERROR] scale={scale_idx} dataset={ds}: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
