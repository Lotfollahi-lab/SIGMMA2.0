"""H&E → ST gene-expression prediction baselines vendored for SIGMMA benchmarking.

Each subpackage exposes a minimal ``build_model(n_genes, ...)`` factory that
returns a PyTorch ``nn.Module`` mapping a 3×224×224 (or 3×crop_px×crop_px)
H&E patch to an ``(G,)`` gene-expression vector. Loss is method-native MSE
on log1p (or method-specific z-scored) targets.

See ``plans/hazy-cooking-boole.md`` for the integration design.
"""
