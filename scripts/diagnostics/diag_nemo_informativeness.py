"""Diagnostic: is the NEMO cell embedding informative on ACC (untrained tissue)
vs in-house skin (trained)?  Decides branch A (encoder collapses recoverable
signal) vs branch B (NEMO output is itself uninformative on ACC).

For each section measures, on a subsample of cells:
  1. NEMO INPUT effective rank (entropy of singular values) — is the 384-d input
     already degenerate (~3-dim) before the encoder ever runs?
  2. Linear-probe R^2: predict top-20 PCs of log1p gene expression from nemo_latent
     (5-fold, Ridge) — does NEMO retain the cell's expression structure?
  3. Linear-probe R^2: predict basic obs (transcript_counts, cell_area) — sanity.
  4. KMeans(k=10) silhouette in NEMO space — is there cluster structure at all?
"""
import numpy as np, scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.sparse as sp

RNG = np.random.RandomState(0)
NSUB = 8000

SECTIONS = [
    ("ACC",  "PX69156a-s3", "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_adrenocortical_carcinoma/", None),
    ("ACC",  "PD55831a",    None, None),
    ("skin", "CE4-SKI-27-FO-4-S22-A2", "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_beacon/20250404__122725__SGP237_Run1r/CE4-SKI-27-FO-4-S22-A2/spatial_data_processed.h5", None),
    ("skin", "BK39-SKI-27-FO-3-S3",    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_beacon/20241010__132615__SGP194_SKI_RUN1/BK39-SKI-27-FO-3-S3/spatial_data_processed.h5", None),
]
# resolve ACC paths from manifest
import pandas as pd
accm = pd.read_csv("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_adrenocortical_carcinoma_MANIFEST_test.csv")
acc_paths = dict(zip(accm["sample_name"], accm["spatial_preprocessed"]))

def eff_rank(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-9]; p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def probe_r2(Xemb, Y, name):
    Xs = StandardScaler().fit_transform(Xemb)
    sc_ = cross_val_score(Ridge(alpha=1.0), Xs, Y, cv=5, scoring="r2")
    return float(sc_.mean())

print(f"{'tissue':5s} {'section':24s} {'ncells':>7s} {'NEMO_effR':>9s} {'R2_gexPC':>9s} {'R2_tcount':>9s} {'R2_carea':>9s} {'silhou':>7s}")
for tis, sec, path, _ in SECTIONS:
    if tis == "ACC":
        path = acc_paths[sec]
    a = sc.read_h5ad(path)
    n = a.n_obs
    idx = RNG.choice(n, size=min(NSUB, n), replace=False)
    nemo = np.asarray(a.obsm["nemo_latent"], dtype=np.float32)[idx]
    # gene expression -> top20 PCs (log1p)
    X = a.X[idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = np.log1p(X.astype(np.float32))
    gex_pc = PCA(n_components=20, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    r2_gex = probe_r2(nemo, gex_pc, "gexPC")
    # basic obs
    def get(col):
        return a.obs[col].to_numpy()[idx].astype(np.float32) if col in a.obs else None
    tc = get("transcript_counts"); ca = get("cell_area")
    r2_tc = probe_r2(nemo, np.log1p(tc), "tc") if tc is not None else float("nan")
    r2_ca = probe_r2(nemo, np.log1p(ca), "ca") if ca is not None else float("nan")
    # cluster structure
    sub = RNG.choice(len(nemo), size=min(4000, len(nemo)), replace=False)
    km = KMeans(n_clusters=10, n_init=5, random_state=0).fit_predict(StandardScaler().fit_transform(nemo[sub]))
    sil = silhouette_score(StandardScaler().fit_transform(nemo[sub]), km)
    er = eff_rank(nemo)
    print(f"{tis:5s} {sec:24s} {len(idx):7d} {er:9.1f} {r2_gex:9.3f} {r2_tc:9.3f} {r2_ca:9.3f} {sil:7.3f}")
