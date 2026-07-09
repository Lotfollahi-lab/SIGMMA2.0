"""Recompute EVA MoNuSAC instance metrics with polygon-instance GT (one XML
polygon = one instance), parallel over npz files. CC metrics are read back from
the existing instance_metrics_global.csv (no recompute). Prediction side is
identical (prob_to_instances tau=0.5); only the GT instance map differs.
"""
from __future__ import annotations
import glob, sys, warnings, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from scripts.eva_critique.metrics_seg import aji, panoptic_quality
from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.polygon_gt import load_polygon_instance_label

DATA_ROOT = ROOT / "data" / "monusac"

# condition -> (era, glob-root, CC-csv-root)
CONDS = {
 # ERA1 (112sec, step1000)
 ("ERA1","UNI2-h(frozen)"): "outputs/eva_critique_uni2h",
 ("ERA1","BLEEP-FT"):       "outputs/eva_critique_baselines_uni2h_step1000/bleep_uni_ft",
 ("ERA1","CLIP-FT"):        "outputs/eva_critique_baselines_uni2h_step1000/clip_uni_ft",
 ("ERA1","PLIP-FT"):        "outputs/eva_critique_baselines_uni2h_step1000/plip_uni_ft",
 ("ERA1","OmiCLIP-FT"):     "outputs/eva_critique_baselines_uni2h_step1000/omiclip_uni_ft",
 ("ERA1","SIGMMA s1000"):   "outputs/eva_critique_gex_sweep_step1000/a01",
 # ERA2 (353sec, step2000)
 ("ERA2","UNI2-h(frozen)"): "outputs/eva_step2000/uni2h",
 ("ERA2","BLEEP-FT"):       "outputs/eva_step2000/bleep",
 ("ERA2","CLIP-FT"):        "outputs/eva_step2000/clip",
 ("ERA2","PLIP-FT"):        "outputs/eva_step2000/plip",
 ("ERA2","OmiCLIP-FT"):     "outputs/eva_step2000/omiclip",
 ("ERA2","SIGMMA s2000"):   "outputs/eva_step2000/a01",
 ("ERA2","SIGMMA s500*"):   "outputs/eva_step2000/a01_36852_step500",
}

POLY_GT = {}  # filled in parent, inherited by forked workers

def npz_for(root):
    out = {}
    for f in sorted(glob.glob(str(ROOT / root / "seed*" / "**" / "test_artifacts.npz"),
                              recursive=True)):
        seed = [p for p in Path(f).parts if p.startswith("seed")][0]
        out.setdefault(seed, f)
    return out

def work(task):
    key, seed, npz = task
    d = np.load(npz, allow_pickle=True)
    ids, probs = [str(x) for x in d["image_id"]], d["prob"]
    rows = []
    for i, iid in enumerate(ids):
        if iid not in POLY_GT:
            continue
        pred = prob_to_instances(probs[i].astype(np.float32), threshold=0.5)
        gp = POLY_GT[iid]
        rows.append((key[0], key[1], seed, iid,
                     aji(pred, gp), panoptic_quality(pred, gp, iou_thr=0.5)[0]))
    return rows

def main():
    print("building polygon GT cache ...", flush=True)
    any_npz = list(npz_for("outputs/eva_step2000/a01").values())[0]
    ids = [str(x) for x in np.load(any_npz, allow_pickle=True)["image_id"]]
    for iid in ids:
        try:
            POLY_GT[iid] = load_polygon_instance_label(iid, DATA_ROOT)
        except Exception as e:
            print(f"  poly GT FAIL {iid}: {str(e)[:50]}")
    print(f"  poly GT for {len(POLY_GT)}/{len(ids)} FOVs", flush=True)

    tasks = []
    for key, root in CONDS.items():
        for seed, npz in npz_for(root).items():
            tasks.append((key, seed, npz))
    print(f"running {len(tasks)} (condition,seed) tasks on 8 cores ...", flush=True)

    with mp.Pool(8) as pool:
        allrows = []
        for j, rows in enumerate(pool.imap_unordered(work, tasks), 1):
            allrows.extend(rows)
            print(f"  done {j}/{len(tasks)}", flush=True)

    df = pd.DataFrame(allrows, columns=["era","method","seed","image_id","AJI_poly","PQ_poly"])
    df.to_csv(ROOT / "outputs" / "eva_polygon_per_fov.csv", index=False)

    # aggregate: per-seed mean then mean across seeds
    g = (df.groupby(["era","method","seed"])[["AJI_poly","PQ_poly"]].mean()
           .groupby(["era","method"]).agg(["mean","std"]))
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()

    # CC reference from existing global CSVs (same prob_to_instances)
    def cc(root):
        a=[];q=[]
        for f in sorted(glob.glob(str(ROOT/root/"seed*"/"instance_metrics_global.csv"))):
            d=pd.read_csv(f); c={k.lower():k for k in d.columns}
            a.append(d[c['aji']].mean()); q.append(d[c['pq']].mean())
        return (np.mean(a),np.mean(q)) if a else (np.nan,np.nan)
    g["AJI_cc"]=np.nan; g["PQ_cc"]=np.nan
    for i,r in g.iterrows():
        ar,qr=cc(CONDS[(r["era"],r["method"])])
        g.at[i,"AJI_cc"]=ar; g.at[i,"PQ_cc"]=qr

    g.to_csv(ROOT / "outputs" / "eva_polygon_vs_cc_metrics.csv", index=False)
    print("\n=== AJI/PQ: CC GT vs POLYGON GT (N=5 seed mean) ===")
    for era in ["ERA1","ERA2"]:
        print(f"\n--- {era} ---")
        print(f"{'method':18s} {'AJI_cc':>8s} {'AJI_poly':>9s} | {'PQ_cc':>8s} {'PQ_poly':>9s}")
        for _,r in g[g.era==era].iterrows():
            print(f"{r['method']:18s} {r['AJI_cc']:8.4f} {r['AJI_poly_mean']:9.4f} | "
                  f"{r['PQ_cc']:8.4f} {r['PQ_poly_mean']:9.4f}")
    print(f"\nsaved -> outputs/eva_polygon_vs_cc_metrics.csv")

if __name__ == "__main__":
    main()
