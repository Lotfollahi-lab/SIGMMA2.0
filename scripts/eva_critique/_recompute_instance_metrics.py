"""Post-hoc re-compute instance_metrics_global.csv for all
conditions in the EVA-critique sweep, using the *current* defaults of
`prob_to_instances` in scripts/eva_critique/inst.py.

This is invoked after tweaking watershed defaults (seed_min_distance, min_size)
so we can re-evaluate without re-training the decoders. Test_artifacts.npz
(prob + gt + image_id) are reused; only the per-FOV instance metrics CSVs are
overwritten.

Conditions covered:
  - UNI2-h baseline             (eva_critique_uni2h/seed*/uni/...)
  - SIGMMA α-sweep              (eva_critique_gex_sweep_step1000/a0{0,1,2,4}/seed*/sigmma_shared_lora/...)
  - Baseline-FT (4 models)      (eva_critique_baselines_uni2h_step1000/{bleep,clip,plip,omiclip}_uni_ft/seed*/<model>/...)
  - Per-scale grid (micro+meso) (eva_critique_gex_sweep_step1000/a01_scale_grid/seed*/sigmma_scale_{micro,meso}/...)
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import glob
import numpy as np
import pandas as pd

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
sys.path.insert(0, str(ROOT))

from scripts.eva_critique.metrics_seg import (  # noqa: E402
    aji, detection_f1_hungarian, label_components, panoptic_quality,
)
from scripts.eva_critique.inst import prob_to_instances  # noqa: E402


def per_fov(prob, gt):
    pred = prob_to_instances(prob, threshold=0.5)
    gti = label_components(gt.astype(bool))
    PQ, SQ, DQ, *_ = panoptic_quality(pred, gti, iou_thr=0.5)
    return dict(
        AJI=aji(pred, gti),
        PQ=PQ, SQ=SQ, DQ=DQ,
        DetF1_07=detection_f1_hungarian(
            pred, gti, iou_thresholds=(0.5, 0.7)
        )["f1@0.7"],
    )


def _run_one(args):
    """Worker entrypoint (picklable: module-level)."""
    label, npz, out, prefix = args
    t0 = time.time()
    try:
        recompute_one(npz, out, prefix=prefix)
        return (label, True, time.time() - t0, None)
    except Exception as e:  # noqa: BLE001
        return (label, False, time.time() - t0, str(e))


def recompute_one(npz_path: Path, out_dir: Path, prefix: str = "") -> None:
    """Load npz, compute per-FOV global metrics, save CSV next
    to out_dir (NOT next to npz). `prefix` is added before
    'instance_metrics_{label}.csv' (used by per-scale grid which has multiple
    artifacts in the same seed dir)."""
    t0 = time.time()
    d = np.load(npz_path, allow_pickle=True)
    ids, probs, gts = d["image_id"], d["prob"].astype(np.float32), d["gt"]

    rows_g = []
    for i in range(len(ids)):
        p, g = probs[i], gts[i]
        rows_g.append({**per_fov(p, g), "image_id": str(ids[i])})

    out_dir.mkdir(parents=True, exist_ok=True)
    for label, rows in [("global", rows_g)]:
        df = pd.DataFrame(rows)
        name = f"{prefix}instance_metrics_{label}.csv" if prefix else f"instance_metrics_{label}.csv"
        out = out_dir / name
        df.to_csv(out, index=False)
    # dt printed by parent


def main():
    seeds = [42, 2, 6, 14, 21]
    tasks: list[tuple[str, Path, Path, str]] = []

    # UNI2-h baseline
    for s in seeds:
        npz = ROOT / f"outputs/eva_critique_uni2h/seed{s}/uni/convdec_image/test_artifacts.npz"
        out = ROOT / f"outputs/eva_critique_uni2h/seed{s}"
        tasks.append((f"UNI2-h seed{s}", npz, out, ""))

    # SIGMMA α-sweep
    for cond in ["a00", "a01", "a02", "a04"]:
        for s in seeds:
            npz = ROOT / f"outputs/eva_critique_gex_sweep_step1000/{cond}/seed{s}/sigmma_shared_lora/convdec_image/test_artifacts.npz"
            out = ROOT / f"outputs/eva_critique_gex_sweep_step1000/{cond}/seed{s}"
            tasks.append((f"SIGMMA {cond} seed{s}", npz, out, ""))

    # Baseline-FT
    for model in ["bleep_uni_ft", "clip_uni_ft", "plip_uni_ft", "omiclip_uni_ft"]:
        for s in seeds:
            npz = ROOT / f"outputs/eva_critique_baselines_uni2h_step1000/{model}/seed{s}/{model}/convdec_image/test_artifacts.npz"
            out = ROOT / f"outputs/eva_critique_baselines_uni2h_step1000/{model}/seed{s}"
            tasks.append((f"{model} seed{s}", npz, out, ""))

    # Per-scale grid (micro + meso)
    for scale in ["micro", "meso"]:
        for s in seeds:
            npz = ROOT / f"outputs/eva_critique_gex_sweep_step1000/a01_scale_grid/seed{s}/sigmma_scale_{scale}/convdec_image/test_artifacts.npz"
            out = ROOT / f"outputs/eva_critique_gex_sweep_step1000/a01_scale_grid/seed{s}"
            tasks.append((f"scale_{scale} seed{s}", npz, out, f"sigmma_scale_{scale}_"))

    print(f"Total tasks: {len(tasks)}", flush=True)
    print(f"prob_to_instances defaults: min_size=32, seed_min_distance=7 (current inst.py)\n", flush=True)

    missing = []
    for label, npz, _, _ in tasks:
        if not npz.exists():
            missing.append((label, npz))
    if missing:
        print(f"WARNING: {len(missing)} missing artifacts:", flush=True)
        for label, npz in missing[:10]:
            print(f"  {label}: {npz}", flush=True)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", flush=True)

    # Filter to existing tasks
    tasks = [t for t in tasks if t[1].exists()]

    # Parallelize across tasks (each task = one npz file's worth of FOVs).
    # Tasks are independent → process pool works well.
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = int(os.environ.get("RECOMPUTE_WORKERS", "8"))
    print(f"Using {n_workers} workers across {len(tasks)} tasks\n", flush=True)

    t_start = time.time()
    n_done = 0
    n_fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            label, ok, dt, err = fut.result()
            if ok:
                n_done += 1
                print(f"[{n_done+n_fail}/{len(tasks)}] {label}  ({dt:.1f}s)", flush=True)
            else:
                n_fail += 1
                print(f"[{n_done+n_fail}/{len(tasks)}] {label}  FAIL: {err}", flush=True)

    print(f"\nAll done. {n_done} ok / {n_fail} fail in {(time.time()-t_start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
