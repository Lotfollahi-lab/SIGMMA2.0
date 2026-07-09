#!/bin/bash
#BSUB -J verify_gbm_indist
#BSUB -o logs/job_logs/verify_gbm_indist.%J.out
#BSUB -e logs/job_logs/verify_gbm_indist.%J.err
#BSUB -q oversubscribed
#BSUB -G s10396
#BSUB -n 1
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000]"

set -euo pipefail
cd /nfs/team361/dj16/projects/SIGMMA
source .venv/bin/activate

python - <<'PY'
import h5py, numpy as np, pandas as pd, zarr, glob
from pathlib import Path

EMB_DIR = Path("/nfs/team361/dj16/projects/SIGMMA/outputs/cross_tissue_retrieval/"
               "sigmma_dense_attnpool_gex_svg_a01/36852/embeddings_step2000_allsplit/")
TISSUES = sorted(p.stem for p in EMB_DIR.glob("*.h5"))
SECTION = "AT15-BRA-4-FO-F1-S42-ii"
FOV = 256

def _l2(a): return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)

# Rebuild banks/meta exactly like the notebook (macro scale).
img, st, tissue, section, xs, ys = [], [], [], [], [], []
for t in TISSUES:
    with h5py.File(EMB_DIR / f"{t}.h5", "r") as h:
        im, spv = h["image_emb_macro"][:], h["st_emb_macro"][:]
        img.append(im); st.append(spv)
        tissue.extend([t]*im.shape[0])
        section.extend(s.decode() if isinstance(s, bytes) else s for s in h["section_id"][:])
        xs.append(h["x_start"][:]); ys.append(h["y_start"][:])
img_bank = _l2(np.concatenate(img).astype(np.float32))
st_bank  = _l2(np.concatenate(st).astype(np.float32))
meta = pd.DataFrame({"tissue": tissue, "section_id": section,
                     "x_start": np.concatenate(xs), "y_start": np.concatenate(ys)})
print(f"pooled tiles={len(meta):,}  img_bank={img_bank.shape}  st_bank={st_bank.shape}")

# (1) grid alignment assumption used by the selection binning.
xa, ya = meta["x_start"].to_numpy(), meta["y_start"].to_numpy()
print("ASSERT x_start,y_start multiples of 256:",
      bool((xa % FOV == 0).all() and (ya % FOV == 0).all()))

# (2) retrieve()'s ref_row == bank row index.
def retrieve(qv, topk):
    scores = qv @ st_bank.T
    top = np.argsort(-scores)[:topk]
    out = meta.iloc[top].copy(); out.insert(0,"rank",np.arange(len(top)))
    out["cos_sim"]=scores[top]
    return out.reset_index().rename(columns={"index":"ref_row"})

sec_rows = meta.index[meta["section_id"] == SECTION].to_numpy()
print(f"section {SECTION}: {len(sec_rows)} tiles in bank")
probe = int(sec_rows[len(sec_rows)//2])
# Query with the bank's stored IMAGE emb for `probe`; its GT ST pair is row `probe`.
r = retrieve(img_bank[probe], topk=len(meta))
gt_rank = int(np.where(r["ref_row"].to_numpy()==probe)[0][0])
print(f"ASSERT ref_row maps to bank rows: top1 ref_row={int(r['ref_row'].iloc[0])}")
print(f"GT-pair (row {probe}) rank when querying its own image emb = {gt_rank} / {len(meta)}")
print(f"  same-tissue@50 = {(r.head(50)['tissue']=='glioblastoma').mean():.2f}")

# (3) full-res H&E zarr crop shape at a real tile coord.
p = glob.glob(f"/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
              f"sanger_glioblastoma/*/{SECTION}/hematoxylin_eosin.zarr")[0]
he = zarr.open(p, mode="r")
x0, y0 = int(meta.at[probe,"x_start"]), int(meta.at[probe,"y_start"])
ctile = np.transpose(he[:, y0:y0+FOV, x0:x0+FOV], (1,2,0))
print(f"crop shape at (x0={x0},y0={y0}) = {ctile.shape} dtype={ctile.dtype}  (expect ({FOV},{FOV},3) uint8)")
print("ALL CHECKS DONE")
PY
