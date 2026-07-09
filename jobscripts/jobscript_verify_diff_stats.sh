#!/bin/bash
#BSUB -J verify-diff-stats
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/verify-diff-stats.%J.out
#BSUB -e logs/job_logs/verify-diff-stats.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6GB
#BSUB -R "select[mem>6GB] rusage[mem=6GB] span[hosts=1]"

set -e
module load python-3.11.6
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH"
cd "${run_dir}"

python3 - <<'PY'
import os, h5py, numpy as np
CR = "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
new = f"{CR}/sigmma/gex_2d/_verify_idc_36852"
ref = f"{CR}/sigmma/gex_2d/36852_step2000"
for s in ["NCBI783","NCBI785","TENX95","TENX99"]:
    a_p, b_p = f"{new}/{s}.h5", f"{ref}/{s}.h5"
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        print(f"{s}: MISSING"); continue
    with h5py.File(a_p) as a, h5py.File(b_p) as b:
        n = a["micro"].shape[0]
        fid_ok = np.array_equal(a["fov_id"][:], b["fov_id"][:])
        print(f"\n=== {s}  n_fov={n}  fov_id_match={fid_ok} ===")
        for k in ("micro","meso","macro"):
            tot=0; absd=0.0; mx=0.0; near=0; dot=0.0; na=0.0; nb=0.0
            for i in range(n):
                va = a[k][i].astype(np.float32).ravel()
                vb = b[k][i].astype(np.float32).ravel()
                d  = np.abs(va-vb)
                tot += d.size; absd += float(d.sum()); mx = max(mx, float(d.max()))
                near += int((d < 1e-3).sum())
                dot += float((va*vb).sum()); na += float((va*va).sum()); nb += float((vb*vb).sum())
            mean = absd/tot
            cos  = dot/((na**0.5)*(nb**0.5)+1e-12)
            print(f"  {k:5s} mean|d|={mean:.2e} max|d|={mx:.2e} "
                  f"frac|d|<1e-3={near/tot:.4f} global_cos={cos:.6f}")
PY
echo "=== diff-stats done ==="
