"""Merge per-section raw shards from a section-parallel Phoenix eval into one
globally-pooled summary JSON, identical in schema to a single-job
predict_baseline_gex_2d run.

Each shard ({out_root}/_raw_bin/{section}.npz) is one section's raw per-FOV bin
arrays (saved with --save_raw). We concatenate all sections and call the shared
``summarize()`` once, so per-gene Pearson pools over all sections exactly like
the stored baselines.

Usage:
  python -m scripts.merge_phoenix_raw_gex_2d \
      --out_root outputs/gex_2d/svg_panel_2d22ba31_step1000_test/phoenix \
      --panel_path outputs/gex_2d/svg_panel_2d22ba31/gene_panel.json \
      --target_grid 14 --fov_size 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sigmma.evaluation.gex_2d_probing import summarize
from scripts.predict_baseline_gex_2d import _summarize_fov_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--panel_path", required=True)
    ap.add_argument("--target_grid", type=int, default=14)
    ap.add_argument("--fov_size", type=int, default=256)
    ap.add_argument("--baseline", default="phoenix")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    panel = json.load(open(args.panel_path))["panel"]

    # --- Path B (bin) -------------------------------------------------------
    bin_dir = out_root / "_raw_bin"
    shards = sorted(bin_dir.glob("*.npz")) if bin_dir.is_dir() else []
    if shards:
        preds, targets, masks, samples, fov_ids, xs, ys = [], [], [], [], [], [], []
        for p in shards:
            with np.load(p, allow_pickle=True) as f:
                preds.append(f["preds"]); targets.append(f["targets"])
                masks.append(f["masks"]); samples.append(f["samples"])
                fov_ids.append(f["fov_ids"]); xs.append(f["x_starts"]); ys.append(f["y_starts"])
            print(f"  [shard] {p.name}: {preds[-1].shape[0]} FOVs", flush=True)
        P = np.concatenate(preds, 0); T = np.concatenate(targets, 0)
        M = np.concatenate(masks, 0)
        sample_ids = np.concatenate(samples, 0)
        x_starts = np.concatenate(xs, 0); y_starts = np.concatenate(ys, 0)
        print(f"[merge] {len(shards)} sections → {P.shape[0]} FOVs total", flush=True)
        s = summarize(
            P, T, M, gene_panel=panel,
            sample_ids=np.asarray(sample_ids, dtype=object),
            x_starts=np.asarray(x_starts, dtype=np.int64),
            y_starts=np.asarray(y_starts, dtype=np.int64),
            fov_size=int(args.fov_size), target_grid=int(args.target_grid),
        )
        s["mode"] = "bin"; s["baseline"] = args.baseline
        s["target_grid"] = args.target_grid
        s["num_val_fovs"] = int(P.shape[0]); s["smoke_slide"] = None
        s["n_sections_merged"] = len(shards)
        with open(out_root / f"{args.baseline}_bin.json", "w") as f:
            json.dump(s, f, indent=2)
        med = s["median_per_gene_pearson"]
        med = med.get("mean") if isinstance(med, dict) else med
        print(f"[bin] median per-gene Pearson = {med:.4f} "
              f"(pooled, all genes inc. NaN-dropped)", flush=True)
    else:
        print(f"[merge] no bin shards in {bin_dir}", flush=True)

    # --- Path A (fov), optional --------------------------------------------
    fov_dir = out_root / "_raw_fov"
    fshards = sorted(fov_dir.glob("*.npz")) if fov_dir.is_dir() else []
    if fshards:
        fp, ft = [], []
        for p in fshards:
            with np.load(p, allow_pickle=True) as f:
                fp.append(f["preds"]); ft.append(f["targets"])
        Pf = np.concatenate(fp, 0); Tf = np.concatenate(ft, 0)
        s = _summarize_fov_level(Pf, Tf, gene_panel=panel)
        s["mode"] = "fov"; s["baseline"] = args.baseline
        s["num_val_fovs"] = int(Pf.shape[0]); s["smoke_slide"] = None
        with open(out_root / f"{args.baseline}_fov.json", "w") as f:
            json.dump(s, f, indent=2)
        print(f"[fov] median per-gene Pearson = {s['median_per_gene_pearson']:.4f}", flush=True)


if __name__ == "__main__":
    main()
