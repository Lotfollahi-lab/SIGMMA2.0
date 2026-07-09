"""Sweep watershed `seed_min_distance` ∈ {7, 9, 11, 13} for the conditions
relevant to the Step 1 analysis (SIGMMA α=0.1, PLIP-ft, BLEEP-ft, UNI2-h
baseline), reusing saved test_artifacts.npz (no retraining).

Output:
  outputs/eva_critique_seed_dist_sweep.csv
  columns: cond, seed, sd, fov, AJI, PQ, SQ, DQ
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
sys.path.insert(0, str(ROOT))

from scripts.eva_critique.metrics_seg import (  # noqa: E402
    aji, label_components, panoptic_quality,
)
from scripts.eva_critique.inst import prob_to_instances  # noqa: E402


SEEDS = [42, 2, 6, 14, 21]
SD_VALUES = [7, 9, 11, 13]
MIN_SIZE = 32

NPZ_PATHS = {
    "sigmma_a01": "outputs/eva_critique_gex_sweep_step1000/a01/seed{seed}/sigmma_shared_lora/convdec_image/test_artifacts.npz",
    "plip_uni_ft": "outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft/seed{seed}/plip_uni_ft/convdec_image/test_artifacts.npz",
    "bleep_uni_ft": "outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft/seed{seed}/bleep_uni_ft/convdec_image/test_artifacts.npz",
    "uni2h": "outputs/eva_critique_uni2h/seed{seed}/uni/convdec_image/test_artifacts.npz",
}


def per_fov(prob, gt, sd):
    pred = prob_to_instances(prob, threshold=0.5,
                             min_size=MIN_SIZE, seed_min_distance=sd)
    gti = label_components(gt.astype(bool))
    PQ, SQ, DQ, *_ = panoptic_quality(pred, gti, iou_thr=0.5)
    return dict(AJI=aji(pred, gti), PQ=PQ, SQ=SQ, DQ=DQ)


def process_one(args):
    """One (cond, seed, sd) → list of FOV-level rows."""
    cond, seed, sd, npz_path = args
    t0 = time.time()
    rows = []
    try:
        d = np.load(ROOT / npz_path, allow_pickle=True)
        ids, probs, gts = d["image_id"], d["prob"].astype(np.float32), d["gt"]
        for i, fov in enumerate(ids):
            m = per_fov(probs[i], gts[i], sd)
            rows.append({"cond": cond, "seed": seed, "sd": sd,
                         "fov": str(fov), **m})
    except Exception as e:
        return cond, seed, sd, None, str(e), time.time() - t0
    return cond, seed, sd, rows, None, time.time() - t0


def main():
    tasks = []
    for cond, tmpl in NPZ_PATHS.items():
        for seed in SEEDS:
            for sd in SD_VALUES:
                tasks.append((cond, seed, sd, tmpl.format(seed=seed)))
    print(f"Total tasks: {len(tasks)} ({len(NPZ_PATHS)} conds × {len(SEEDS)} seeds × {len(SD_VALUES)} sd values)",
          flush=True)
    print(f"min_size fixed at {MIN_SIZE}\n", flush=True)

    n_done = 0
    n_fail = 0
    all_rows = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            cond, seed, sd, rows, err, dt = fut.result()
            if err:
                n_fail += 1
                print(f"[{n_done+n_fail}/{len(tasks)}] FAIL {cond} seed{seed} sd{sd}: {err}",
                      flush=True)
            else:
                all_rows.extend(rows)
                n_done += 1
                print(f"[{n_done+n_fail}/{len(tasks)}] {cond:14s} seed{seed:2d} sd{sd:2d}  "
                      f"({dt:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    out = ROOT / "outputs/eva_critique_seed_dist_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nDone. {n_done} ok / {n_fail} fail in {(time.time()-t_start)/60:.1f} min.")
    print(f"Saved {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
