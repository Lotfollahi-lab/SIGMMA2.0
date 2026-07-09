# EVA-critique: does Dice on MoNuSAC miss biologically meaningful SIGMMA gains?

Five components compare pretrained UNI vs SIGMMA-fine-tuned UNI on the EVA
MoNuSAC test split, holding seed / aug / optimizer fixed. The encoder is
frozen everywhere.

## Layout
```
scripts/eva_critique/
  models.py           # frozen UNI vs SIGMMA wrappers; CLS-attention via forward hook
  data.py             # EVA MoNuSAC iterator + per-pixel R1/R2/R3/R4 region builder
  decoders.py         # D0 linear, D1 MLP, D2 light UNet, D3 UNet w/ skips from {6,12,18,24}
  metrics_seg.py      # Dice/IoU/Recall, AJI, PQ, HD95, ASD, BoundaryF1, AUPRC, ECE, entropy
  inst.py             # prob -> τ -> distance transform + watershed
  train_decoder.py    # train one (model, decoder) pair; cache test prob/pred/entropy
  extract_attention.py# cache per-tile (16, 14, 14) CLS->patch attention
  linear_probe.py     # CLS+mean ridge probe, image-grouped 5-fold + held-out test
  pipeline_eval.py    # consume artifacts, emit Components 1/2/3/5 CSVs + figures
  summarize.py        # aggregate CSVs into deltas + Wilcoxon (Holm) + 1-page summary
```

## Run end-to-end (LSF)

```
bsub < jobscripts/jobscript_eva_critique.sh
```

## Outputs (under `logs/eva_critique/`)

CSVs:
- `region_stratified_metrics.csv` — Component 1
- `alternative_metrics.csv`       — Component 2
- `instance_metrics.csv`          — Component 3 (τ sweep, then τ\* picked in summary.md)
- `decoder_sensitivity.csv`       — Component 4 (a)
- `linear_probe_results.csv`      — Component 4 (b)
- `attention_consistency.csv`     — Component 5

Figures:
- `figures/region_metric_forest_plot.pdf`
- `figures/decoder_ladder.pdf`
- `figures/attention_grid_<tile_id>.png` × ~20 tiles spanning R1/R2/R3-dominant cases

`summary.md` maps each result to the three hypotheses:
- **H1 decoder bottleneck**, **H2 average dilution**, **H3 Dice insensitivity**.

## Knobs

- `--scale`: which SIGMMA LoRA adapter to activate (0=micro, 1=meso, 2=macro).
  Default 0 — cell segmentation is a per-cell task, so micro is the natural
  fixed scale to compare per-tile attention/feature maps with UNI.
- `--max_steps` / `--lr` / `--seed`: held identical across all (model × decoder)
  pairs by the jobscript so component-4 deltas are paired by image.

## Model identity

`models.load_sigmma()` instantiates the SIGMMA backbone via the same
`load_model` path used by `scripts/benchmark_image_pathology_task.py`,
reading `model.checkpoint_path` from the YAML. The released anonymous
checkpoint at `AnonymousAuthor42/SIGMMA` (HF) is the same artefact; swap
the local path for an HF download URI if you want to verify externally.
