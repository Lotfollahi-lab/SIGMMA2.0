"""Zero-shot cross-panel GEX-2D eval on external HEST-1k public sections.

Reuses the step2000 svg0529 MAIN-panel probe heads (panel_35a7bd78801c, 94
genes) WITHOUT any retraining, and applies them to the public HEST-1k test
sections (TENX*) that live in the merged panels 11bbd69265ed (PAAD/LUAD) and
80b4f72e3da8 (SKCM). The head predicts the 94 main-panel genes; we evaluate
only on the genes shared with each HEST section's SVG panel (main ∩ HEST-SVG):
  - 11bbd: 5 shared genes   - 80b4: 9 shared genes

This is genuine zero-shot transfer: frozen H&E encoder + frozen readout head
trained on the in-house cohort, evaluated on external public sections.

For each (panel × condition) we slice pred (main-gene order) and target
(HEST-gene order) to the shared genes, then call the standard ``summarize`` so
the output JSON matches every other GEX-2D result (per-gene Pearson / Moran /
SSIM / MSE_slide_std).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.probe_gex_2d import in_channels_for, build_head, _heads_subdir
from sigmma.evaluation.gex_cache_io import read_fov_ids, section_h5_exists
from sigmma.evaluation.gex_2d_probing import predict_all, summarize

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
CACHE = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache")
MAIN_HEADS = ROOT / "outputs/gex_2d/svg0529_step2000"          # source heads (94 genes)
# STRICT zero-shot: target genes = intersection of the main head's output with each
# panel's MEASURED var.index (read from transcripts.parquet), NOT the panel's
# SVG-100 (which was selected using expression and, for the public panels, the
# held-out test sections themselves -> leak). Targets are read from the svgB data
# caches (built on the shared-gene axis). Written to a NEW dir so the old (leaked)
# public_zeroshot_step2000 results are preserved for the leaked-vs-strict delta.
OUT_ROOT = ROOT / "outputs/gex_2d/public_zeroshot_strict_step2000"
TARGET_GRID = 14

MAIN_PANEL = json.load(open(ROOT / "outputs/gex_2d/svg0529_panel_35a7bd78801c/gene_panel.json"))["panel"]

UNI2H = str(CACHE / "uni/gex_2d/uni2h_svg0529_fov256_p14")
SIGMMA_CACHE = str(CACHE / "sigmma/gex_2d/36852_step2000")
SIGMMA_CFG = str(ROOT / "configs/cell_niche_transfer_dense_attnpool_a01_step2000.yaml")


def _svgb_eval_sections(panel_hash: str) -> list[str]:
    import csv
    m = ROOT / f"outputs/gex_2d/svgB_panel_{panel_hash}/MANIFEST_panel_{panel_hash}_eval.csv"
    return [r["sample_name"] for r in csv.DictReader(open(m))]


# Unseen panels for strict zero-shot. Sections come from the svgB eval manifests
# (ovary/lung/brain held-out test; TENX-only for the public-mixed panels).
PANELS = {h: _svgb_eval_sections(h) for h in
          ["1ca7fe57390a", "a87c22a4bcb9", "2df22e67cd31", "11bbd69265ed", "80b4f72e3da8",
           "2bbc576266c5"]}  # 2bbc = HEST-1k IDC (4 sections, 12 shared genes)

# Optional comma-separated panel filter (e.g. ZEROSHOT_PANELS=80b4f72e3da8) so a
# re-run can fill in a single panel without recomputing the others.
_panel_filter = os.environ.get("ZEROSHOT_PANELS")
if _panel_filter:
    keep = {p.strip() for p in _panel_filter.split(",") if p.strip()}
    PANELS = {p: s for p, s in PANELS.items() if p in keep}

# (cond_label, repr, head_type, head_subroot, uni_cache, sigmma_cache)
CONDITIONS = [
    ("uni2h_no_ft",        "uni",   "linear_sweep", "uni2h_no_ft",          UNI2H, None),
    ("bleep_uni_step2000", "uni",   "linear_sweep", "bleep_uni_step2000",   str(CACHE/"uni/gex_2d/bleep_uni_svg0529_step2000"), None),
    ("clip_uni_step2000",  "uni",   "linear_sweep", "clip_uni_step2000",    str(CACHE/"uni/gex_2d/clip_uni_svg0529_step2000"), None),
    ("plip_uni_step2000",  "uni",   "linear_sweep", "plip_uni_step2000",    str(CACHE/"uni/gex_2d/plip_uni_svg0529_step2000"), None),
    ("omiclip_uni_step2000","uni",  "linear_sweep", "omiclip_uni_step2000", str(CACHE/"uni/gex_2d/omiclip_uni_svg0529_step2000"), None),
    ("sigmma_a01_step2000","uni",   "mlp",          "sigmma_a01_step2000",  UNI2H, SIGMMA_CACHE),
    ("sigmma_a01_step2000","concat","mlp",          "sigmma_a01_step2000",  UNI2H, SIGMMA_CACHE),
    ("sigmma_a01_step2000","micro", "mlp",          "sigmma_a01_step2000",  UNI2H, SIGMMA_CACHE),
    ("sigmma_a01_step2000","meso",  "mlp",          "sigmma_a01_step2000",  UNI2H, SIGMMA_CACHE),
    ("sigmma_a01_step2000","macro", "mlp",          "sigmma_a01_step2000",  UNI2H, SIGMMA_CACHE),
]


def fov_keys_for(data_cache_dir: Path, sections: list[str]) -> list[tuple[str, int]]:
    """All (sample, fov_id) keys for the given sections, read from the data cache."""
    keys: list[tuple[str, int]] = []
    for s in sections:
        if not section_h5_exists(data_cache_dir, s):
            print(f"    [warn] no data-cache file for section {s}")
            continue
        for fid in read_fov_ids(data_cache_dir, s):
            keys.append((s, int(fid)))
    return keys


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 256  # default ProbingHead2D hidden dim (matches train)

    for panel, sections in PANELS.items():
        # Target gene axis = the svgB panel = shared (main-94 ∩ measured var.index),
        # already in main-panel order; the svgB data cache bins exactly these genes.
        svgb_panel = json.load(open(ROOT / f"outputs/gex_2d/svgB_panel_{panel}/gene_panel.json"))["panel"]
        shared = [g for g in MAIN_PANEL if g in set(svgb_panel)]
        pred_idx = [MAIN_PANEL.index(g) for g in shared]
        tgt_idx = [svgb_panel.index(g) for g in shared]
        data_cache_dir = CACHE / f"data/gex_2d/svgB_panel_{panel}_g14_fov256"
        val_keys = fov_keys_for(data_cache_dir, sections)
        print(f"\n=== panel {panel}: {len(sections)} public sections, "
              f"{len(val_keys)} FOVs, {len(shared)} shared genes: {shared} ===")
        if not val_keys or not shared:
            print("    [skip] no FOVs or no shared genes"); continue

        for cond, repr_, head_type, subroot, uni_cache, sig_cache in CONDITIONS:
            head_dir = MAIN_HEADS / subroot / _heads_subdir(head_type) / repr_
            ckpts = sorted(head_dir.glob("best_seed*.pt")) or ([head_dir / "best.pt"] if (head_dir/"best.pt").exists() else [])
            if not ckpts:
                print(f"    [skip] {cond}/{repr_}: no head at {head_dir}"); continue
            in_ch = in_channels_for(repr_, uni_cache_dir=uni_cache,
                                    sigmma_cache_dir=sig_cache, sample_key=val_keys[0])
            eval_head_type = "linear" if head_type == "linear_sweep" else head_type
            head = build_head(eval_head_type, in_ch, len(MAIN_PANEL), hidden_dim)
            head.load_state_dict(torch.load(ckpts[0], map_location="cpu"))

            out = predict_all(
                head, val_keys, target_grid=TARGET_GRID, device=device,
                input_repr=repr_, data_cache_dir=str(data_cache_dir),
                uni_cache_dir=uni_cache, sigmma_cache_dir=sig_cache,
                batch_size=32, num_workers=0,
            )
            pred = out["pred"][:, pred_idx]      # (N, |shared|, K, K) main-gene order
            target = out["target"][:, tgt_idx]   # (N, |shared|, K, K) svgB (=shared) order
            s = summarize(
                pred, target, out["mask"], gene_panel=shared,
                sample_ids=out.get("sample_ids"), x_starts=out.get("x_starts"),
                y_starts=out.get("y_starts"), fov_size=int(out.get("fov_size", 0)),
                target_grid=TARGET_GRID,
            )
            s.update(dict(input_repr=repr_, head_type=head_type, cond=cond,
                          panel=panel, n_shared_genes=len(shared),
                          shared_genes=shared, sections=sections,
                          num_val_fovs=int(pred.shape[0]),
                          gene_axis="measured_var_index_strict",
                          zero_shot_source="svg0529_main_35a7bd78801c_step2000"))
            tag = f"{cond}__{repr_}" if cond.startswith("sigmma") else cond
            res_dir = OUT_ROOT / panel
            res_dir.mkdir(parents=True, exist_ok=True)
            with open(res_dir / f"{tag}.json", "w") as f:
                json.dump(s, f, indent=2)
            mp = s["median_per_gene_pearson"]
            print(f"    {tag:30} median per-gene Pearson(shared) = {mp:.4f}  "
                  f"MSE_sl_std = {s.get('median_mse_per_slide_standardized')}")
    print("\n[done] wrote per-(panel,cond) JSONs under", OUT_ROOT)


if __name__ == "__main__":
    main()
