"""Does harmonizing the nemo ST input into the in-house manifold recover
HEST-1k OOD retrieval? (within-section, public panels)

Diagnosis (project_within_panel_all_panel_retrieval): public nemo cell
embeddings sit OUTSIDE the in-house nemo manifold (100% beyond in-house p99,
median NN 20 vs 7), mostly a large global offset (~16) + residual shape diff.
The SIGMMA ST encoder, trained on in-house nemo, then extrapolates -> public
retrieval at chance EVEN within-section (MRR 0.0058).

Test: transform each public section's nemo into the in-house training manifold
and re-run within-section retrieval. If MRR rises well above chance, the OOD
failure is a (correctable) nemo batch shift; if it stays at chance, the gap is
deeper (shape / learned correspondence).

Variants (per-section source stats -> in-house global target):
  H0  none            : raw nemo (reproduces the chance result)
  H1  mean-shift      : x - mu_sec + mu_IH                  (kills global offset)
  H2  per-dim z-match : (x - mu_sec)/sd_sec * sd_IH + mu_IH (offset + per-dim scale)

Image embeddings are nemo-invariant but we re-forward per variant for simplicity
(one-off). scale_idx = -1 (macro) to match the cached benchmark.
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, SequentialSampler

from sigmma.utils import load_dataset, load_model
from sigmma.datasets import multimodal_collate

ROOT = "/nfs/team361/dj16/projects/SIGMMA"
META = ("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
        "sample_metadata.csv")
BASE = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium"
PUBLIC = {"IDC", "PAAD", "LUAD", "SKCM"}
OUT = f"{ROOT}/outputs/retrieval/nemo_harmonize_within_section.json"
N_IH_REF_SECTIONS = 8       # in-house train sections to estimate target manifold
IH_CONTROLS = 2             # in-house test sections to confirm harmonize doesn't break them


def inhouse_ref_stats(n_sections=8, cells_per=4000):
    """Global in-house nemo mean / per-dim std from a sample of TRAIN sections."""
    import h5py
    tr = pd.read_csv(f"{BASE}/sanger_MANIFEST_train.csv")
    tt = dict(zip(*[pd.read_csv(META)[c] for c in ["sample_id", "tissue_type"]]))
    tr["tt"] = tr["sample_name"].map(tt)
    tr = tr[~tr["tt"].isin(PUBLIC)]
    rng = np.random.default_rng(0)
    # spread across panels: take a stratified-ish sample
    rows = tr.sample(min(n_sections, len(tr)), random_state=0)
    chunks = []
    for _, r in rows.iterrows():
        try:
            with h5py.File(r["spatial_preprocessed"], "r") as f:
                e = f["obsm/nemo_latent"]
                idx = np.sort(rng.choice(e.shape[0], min(cells_per, e.shape[0]), replace=False))
                chunks.append(e[idx].astype(np.float64))
        except Exception as ex:
            print(f"  [ref skip {r['sample_name']}: {ex}]")
    X = np.concatenate(chunks, 0)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    print(f"[ref] in-house nemo: {X.shape[0]} cells, |mu|={np.linalg.norm(mu):.2f}, "
          f"mean sd={sd.mean():.3f}")
    return mu.astype(np.float32), sd.astype(np.float32)


def harmonize(nemo, mu_ih, sd_ih, variant):
    if variant == "H0":
        return nemo
    mu_s = nemo.mean(0)
    if variant == "H1":
        return (nemo - mu_s + mu_ih).astype(np.float32)
    if variant == "H2":
        sd_s = nemo.std(0) + 1e-6
        return ((nemo - mu_s) / sd_s * sd_ih + mu_ih).astype(np.float32)
    raise ValueError(variant)


SYM_REF_SECTIONS = 16       # in-house train sections to build the harmonized reference
SYM_CELLS_PER = 3000
SYM_N_COMPS = 100


def build_symphony_reference(n_sections=SYM_REF_SECTIONS, cells_per=SYM_CELLS_PER,
                             n_comps=SYM_N_COMPS, batch_key="batch"):
    """Symphony (Kang 2021): build a Harmony-integrated in-house NEMO reference ONCE,
    then return a mapper that projects a query section's NEMO into that fixed harmonized
    space and inverse-transforms back to 384-d for the frozen SIGMMA encoder.

    Reference is z-scored (sc.pp.scale stores var['mean']/['std'], which map_embedding
    needs) -> PCA-n_comps -> Harmony over in-house sections. The in-house reference is
    never moved; only the query is mapped onto it.
    """
    import warnings; warnings.filterwarnings("ignore")
    import h5py, scanpy as sc, anndata as ad, symphonypy as sp
    tr = pd.read_csv(f"{BASE}/sanger_MANIFEST_train.csv")
    tt = dict(zip(*[pd.read_csv(META)[c] for c in ["sample_id", "tissue_type"]]))
    tr["tt"] = tr["sample_name"].map(tt)
    tr = tr[~tr["tt"].isin(PUBLIC)]
    rng = np.random.default_rng(0)
    rows = tr.sample(min(n_sections, len(tr)), random_state=0)
    Xs, bat = [], []
    for _, r in rows.iterrows():
        try:
            with h5py.File(r["spatial_preprocessed"], "r") as f:
                e = f["obsm/nemo_latent"]
                idx = np.sort(rng.choice(e.shape[0], min(cells_per, e.shape[0]), replace=False))
                Xs.append(e[idx].astype(np.float32))
                bat += [r["sample_name"]] * len(idx)
        except Exception as ex:
            print(f"  [sym ref skip {r['sample_name']}: {ex}]")
    Xref = np.concatenate(Xs, 0)
    var = pd.DataFrame(index=[f"nemo_{i}" for i in range(Xref.shape[1])])
    ref = ad.AnnData(X=Xref.copy(), obs=pd.DataFrame({batch_key: bat}), var=var)
    sc.pp.scale(ref)
    sc.pp.pca(ref, n_comps=n_comps, zero_center=True)
    sp.pp.harmony_integrate(ref, key=batch_key, verbose=False)
    PCs = ref.varm["PCs"]
    mean = ref.var["mean"].values.astype(np.float32)
    std = ref.var["std"].values.astype(np.float32)
    print(f"[sym] reference: {Xref.shape[0]} cells / {len(rows)} sections, "
          f"PCA-{n_comps}, harmonized over '{batch_key}'", flush=True)

    def sym_map(nemo):
        q = ad.AnnData(X=nemo.astype(np.float32).copy(),
                       obs=pd.DataFrame({batch_key: ["q"] * len(nemo)}), var=var)
        sp.tl.map_embedding(q, ref, key=batch_key, use_genes_column=None)
        Z = q.obsm["X_pca_harmony"]
        return ((Z @ PCs.T) * std + mean).astype(np.float32)

    return sym_map


@torch.no_grad()
def section_embeds(model, ds, device, batch_size=256):
    """Return (img, spat) L2-normed embeddings at scale -1 for one section dataset."""
    dl = DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds),
                    num_workers=8, collate_fn=multimodal_collate)
    imgs, spats = [], []
    for batch in dl:
        if not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0:
            continue
        image = batch["image"].to(device)
        spatial = batch["spatial_graph"].to(device)
        with torch.amp.autocast(device.type, enabled=True):
            out = model(image, spatial)
        ie = out["image_embedding"][-1]
        se = out["spatial_embedding"][-1]
        ie = ie / ie.norm(dim=-1, keepdim=True)
        se = se / se.norm(dim=-1, keepdim=True)
        imgs.append(ie.float().cpu())
        spats.append(se.float().cpu())
    return torch.cat(imgs).numpy(), torch.cat(spats).numpy()


def within_section_metrics(img, spat):
    S = img @ spat.T                       # (n, n), GT = diagonal
    n = S.shape[0]
    gt = np.diag(S).copy()
    rank = (S > gt[:, None]).sum(1) + 1    # 1-indexed
    H = np.log(n) + 0.5772
    return {
        "n": int(n),
        "recall_at_1": float(np.mean(rank == 1)),
        "recall_at_1pct": float(np.mean(rank <= max(1, int(n * 0.01)))),
        "mrr": float(np.mean(1.0 / rank)),
        "chance_mrr": float(H / n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-config_file",
                    default=f"{ROOT}/configs/in-house_retrieval_sigmma_dense_attnpool_a01_step2000.yaml")
    ap.add_argument("--variants", default="H0,H1,H2")
    args = ap.parse_args()
    variants = args.variants.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ConfigDict(yaml.load(open(args.config_file), yaml.FullLoader))

    # --- reduced manifest: all public test sections + IH_CONTROLS in-house ---
    test = pd.read_csv(cfg.dataset.query_dataset.manifest_path)
    tt = dict(zip(*[pd.read_csv(META)[c] for c in ["sample_id", "tissue_type"]]))
    test["tt"] = test["sample_name"].map(tt)
    pub = test[test["tt"].isin(PUBLIC)]
    ctrl = test[~test["tt"].isin(PUBLIC)].head(IH_CONTROLS)
    red = pd.concat([pub, ctrl]).drop(columns="tt")
    red_path = f"{ROOT}/outputs/retrieval/_harmonize_manifest.csv"
    red.to_csv(red_path, index=False)
    print(f"reduced manifest: {len(pub)} public + {len(ctrl)} in-house controls -> {red_path}")
    cfg.dataset.query_dataset.manifest_path = red_path

    mu_ih, sd_ih = inhouse_ref_stats(N_IH_REF_SECTIONS)
    sym_map = build_symphony_reference() if "SYM" in variants else None

    print("loading model ...", flush=True)
    model = load_model(cfg, pretrained_image_encoder=cfg.model.get("pretrained_image_encoder"),
                       pretrained_ckpt_path=cfg.model.get("checkpoint_path"), train=False)
    model.eval().to(device)

    print("loading dataset ...", flush=True)
    ds = load_dataset(cfg.dataset.query_dataset, is_train=False, model_name="sigmma")
    sections = list(ds.modality_datasets.datasets)
    # snapshot original nemo per section
    orig = {i: sec._nemo.copy() for i, sec in enumerate(sections)}

    results = {}
    for v in variants:
        print(f"\n===== variant {v} =====", flush=True)
        for i, sec in enumerate(sections):
            sid = sec.section_id
            ttype = sec.tissue_type
            is_pub = ttype in PUBLIC
            sec._nemo = sym_map(orig[i]) if v == "SYM" else harmonize(orig[i], mu_ih, sd_ih, v)
            img, spat = section_embeds(model, sec, device)
            m = within_section_metrics(img, spat)
            m["tissue_type"] = ttype
            m["is_public"] = bool(is_pub)
            results.setdefault(v, {})[sid] = m
            print(f"  [{v}] {sid:14} {ttype[:5]:6} pub={is_pub} n={m['n']:6} "
                  f"MRR={m['mrr']:.4f} (chance {m['chance_mrr']:.4f}) "
                  f"R@1%={m['recall_at_1pct']:.4f} R@1={m['recall_at_1']:.4f}", flush=True)
        # restore for next variant
        for i, sec in enumerate(sections):
            sec._nemo = orig[i]

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")

    # summary: public tile-weighted MRR per variant
    print("\n=== public tile-weighted within-section (per variant) ===")
    print(f"{'variant':8}{'MRR':>10}{'R@1%':>10}{'R@1':>10}{'  (in-house ctrl MRR)':>22}")
    for v in variants:
        ps = results[v]
        pub_s = [s for s, m in ps.items() if m["is_public"]]
        ih_s = [s for s, m in ps.items() if not m["is_public"]]
        def tw(sl, key):
            w = np.array([ps[s]["n"] for s in sl], float)
            x = np.array([ps[s][key] for s in sl], float)
            return (x * w).sum() / w.sum() if len(sl) else float("nan")
        print(f"{v:8}{tw(pub_s,'mrr'):>10.4f}{tw(pub_s,'recall_at_1pct'):>10.4f}"
              f"{tw(pub_s,'recall_at_1'):>10.4f}{tw(ih_s,'mrr'):>22.4f}")


if __name__ == "__main__":
    main()
