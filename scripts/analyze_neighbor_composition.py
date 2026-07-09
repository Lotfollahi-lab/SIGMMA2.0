"""Stream the (102363 x 102363) img->spatial score matrix row-by-row (no full
load; 42GB array, only ~21GB RAM free) and characterise, for a sample of IDC
(public OOD) and renal_cell_carcinoma (in-house) image queries:
  - tissue composition of each query's top-5% retrieved ST neighbours
  - cosine-score distribution within that top-5% neighbourhood
  - cosine score + rank of the TRUE paired ST (diagonal)
"""
from __future__ import annotations
import sys, zipfile, collections
import numpy as np, pandas as pd
import numpy.lib.format as fmt

METHOD = sys.argv[1] if len(sys.argv) > 1 else "a01"
_BASE = "outputs/retrieval/full_rerun_step2000"
NPZ = {
    "a01":     f"{_BASE}/in-house_retrieval_sigmma_dense_attnpool_a01_step2000/score_matrices.npz",
    "bleep":   f"{_BASE}/in-house_retrieval_bleep_uni_finetuned_step2000/score_matrices.npz",
    "clip":    f"{_BASE}/in-house_retrieval_clip_uni_finetuned_step2000/score_matrices.npz",
    "plip":    f"{_BASE}/in-house_retrieval_plip_uni_finetuned_step2000/score_matrices.npz",
    "omiclip": f"{_BASE}/in-house_retrieval_omiclip_uni_finetuned_step2000/score_matrices.npz",
}[METHOD]
SA = "outputs/retrieval/full_rerun_step2000/section_assignments_test.npz"
META = "/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/sample_metadata.csv"
N_SAMPLE = 200
PCT = 0.05
BLOCK = 1500

# ---- row -> tissue ----
sa = np.load(SA, allow_pickle=True)
sec_idx = sa["section_idx"]            # (N,) section index per row
sec_ids = sa["section_ids"]            # (55,) section id per section index
tt = dict(zip(*[pd.read_csv(META)[c] for c in ["sample_id", "tissue_type"]]))
row_tissue = np.array([tt.get(sec_ids[i], "?") for i in sec_idx])
N = len(row_tissue)
K = int(N * PCT)
print(f"method={METHOD} N={N} top-{PCT:.0%}={K}", flush=True)

rng = np.random.default_rng(0)
targets = {}
for grp in ["IDC", "renal_cell_carcinoma"]:
    idx = np.where(row_tissue == grp)[0]
    take = rng.choice(idx, size=min(N_SAMPLE, len(idx)), replace=False)
    targets[grp] = set(int(x) for x in take)
    print(f"  {grp}: {len(idx)} rows, sampled {len(take)}", flush=True)
want = {r: grp for grp, rs in targets.items() for r in rs}
maxrow = max(want) + 1

# ---- accumulators ----
comp = {g: collections.Counter() for g in targets}     # neighbour tissue counts
nbr_scores = {g: [] for g in targets}                  # pooled neighbour cosine
self_score = {g: [] for g in targets}                  # true-match cosine
self_rank = {g: [] for g in targets}                   # true-match rank (0-based)

# ---- stream the compressed .npy row blocks ----
z = zipfile.ZipFile(NPZ)
fp = z.open("scores_img2spatial.npy")
ver = fmt.read_magic(fp)
shp, fo, dt = fmt._read_array_header(fp, ver)
assert shp == (N, N), shp
rowbytes = N * dt.itemsize

def readexact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)

r0 = 0
while r0 < maxrow:
    b = min(BLOCK, maxrow - r0)
    raw = readexact(fp, b * rowbytes)
    block = np.frombuffer(raw, dtype=dt).reshape(b, N)
    for rr in range(r0, r0 + b):
        if rr not in want:
            continue
        g = want[rr]
        srow = block[rr - r0]
        # top-K neighbours
        top = np.argpartition(-srow, K)[:K]
        comp[g].update(row_tissue[top].tolist())
        nbr_scores[g].append(srow[top])
        # true match (diagonal) score + rank
        s_true = srow[rr]
        self_score[g].append(float(s_true))
        self_rank[g].append(int((srow > s_true).sum()))
    r0 += b
    print(f"  ...streamed to row {r0}/{maxrow}", flush=True)

# ---- report ----
print("\n==================  RESULTS  ==================", flush=True)
for g in targets:
    ns = np.concatenate(nbr_scores[g])
    sr = np.array(self_score[g]); rk = np.array(self_rank[g])
    tot = sum(comp[g].values())
    print(f"\n### {g}  (n_query={len(self_score[g])}, top-{K} neighbours each)", flush=True)
    print("  neighbour TISSUE composition (% of retrieved):", flush=True)
    for tis, c in comp[g].most_common(8):
        print(f"      {tis:24s} {100*c/tot:6.2f}%", flush=True)
    print("  neighbour COSINE score  min/p25/med/p75/max: "
          f"{ns.min():.3f}/{np.percentile(ns,25):.3f}/{np.median(ns):.3f}/"
          f"{np.percentile(ns,75):.3f}/{ns.max():.3f}", flush=True)
    print(f"  TRUE-match cosine  med={np.median(sr):.3f}  "
          f"(p5={np.percentile(sr,5):.3f}, p95={np.percentile(sr,95):.3f})", flush=True)
    print(f"  TRUE-match rank    med={int(np.median(rk))}  "
          f"(best={rk.min()}, p25={int(np.percentile(rk,25))}, "
          f"in-top-{K}? {100*np.mean(rk<K):.1f}%)", flush=True)
