"""Joint (tau, sd, ms) watershed grid sweep to reduce SIGMMA oversegmentation.

Evaluates SIGMMA full-rerun step500 + UNI2-h (seed42, 85 FOVs) over a grid of
threshold / seed_min_distance / min_size. Reports mean global PQ + AJI, and
stratifies FOVs by GT mean cell size (large-sparse = failure-prone vs
small-dense) to expose the trade-off.
"""
from __future__ import annotations

import glob, sys, time
from pathlib import Path
from itertools import product
from multiprocessing import Pool
import numpy as np
import statistics as st

sys.path.insert(0, str(Path.cwd()))
from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.metrics_seg import aji, panoptic_quality, label_components

UNI = "outputs/eva_step2000/uni2h/seed42/uni/convdec_image/test_artifacts.npz"
SIG = "outputs/eva_step2000/a01_36852_step500/seed42/sigmma_shared_lora/convdec_image/test_artifacts.npz"

TAUS = [0.5, 0.6]
SDS = [7, 9, 11]
MSS = [32, 48, 64]

def load(path):
    d = np.load(path, allow_pickle=True)
    return ([str(x) for x in d["image_id"]], d["prob"].astype(np.float32), d["gt"])

def gt_size_class(gts):
    """Per-FOV mean instance area (fg px / n inst); split by median."""
    areas = []
    for g in gts:
        gi = label_components(g.astype(bool)); n = int(gi.max())
        areas.append((g > 0).sum() / n if n else 0.0)
    med = np.median([a for a in areas if a > 0])
    return areas, med

def eval_config(args):
    tau, sd, ms, probs_u, probs_s, gts = args
    def model_pq_aji(probs):
        pqs = []; ajis = []
        for i in range(len(probs)):
            pred = prob_to_instances(probs[i], threshold=tau, min_size=ms, seed_min_distance=sd)
            gti = label_components(gts[i].astype(bool))
            PQ, *_ = panoptic_quality(pred, gti, iou_thr=0.5)
            pqs.append(PQ); ajis.append(aji(pred, gti))
        return pqs, ajis
    pq_u, aji_u = model_pq_aji(probs_u)
    pq_s, aji_s = model_pq_aji(probs_s)
    return (tau, sd, ms, pq_u, aji_u, pq_s, aji_s)

def main():
    idu, pu, gu = load(UNI)
    ids, ps, gs = load(SIG)
    assert idu == ids
    areas, med = gt_size_class(gu)
    large = np.array([a > med for a in areas])  # large-sparse cells (failure-prone)

    grid = [(tau, sd, ms, pu, ps, gu) for tau, sd, ms in product(TAUS, SDS, MSS)]
    t0 = time.time()
    rows = []
    with Pool(7) as pool:
        for k, r in enumerate(pool.imap_unordered(eval_config, grid), 1):
            rows.append(r)
            print(f"[{time.time()-t0:.0f}s] {k}/{len(grid)} done", flush=True)

    def mean(v, mask=None):
        v = np.asarray(v)
        return float(v[mask].mean()) if mask is not None else float(v.mean())

    rows.sort(key=lambda r: -mean(r[5]))  # by SIGMMA PQ desc
    print(f"\nGT size median area/inst = {med:.0f} px ; large-cell FOVs n={int(large.sum())}, small-cell n={int((~large).sum())}\n")
    print(f"{'tau':>4}{'sd':>4}{'ms':>4} | {'SIG_PQ':>7}{'UNI_PQ':>7}{'dPQ':>7} | {'SIG_AJI':>8}{'UNI_AJI':>8} | {'SIG_PQ_large':>13}{'SIG_PQ_small':>13}")
    for tau, sd, ms, pq_u, aji_u, pq_s, aji_s in rows:
        flag = "  <-- current" if (tau==0.5 and sd==7 and ms==32) else ""
        print(f"{tau:>4}{sd:>4}{ms:>4} | {mean(pq_s):>7.4f}{mean(pq_u):>7.4f}{mean(pq_s)-mean(pq_u):>+7.4f} | "
              f"{mean(aji_s):>8.4f}{mean(aji_u):>8.4f} | {mean(pq_s,large):>13.4f}{mean(pq_s,~large):>13.4f}{flag}")
    print("\nDONE")

if __name__ == "__main__":
    main()
