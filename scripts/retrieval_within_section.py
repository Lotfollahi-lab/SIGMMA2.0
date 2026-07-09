"""Within-section H&E<->ST retrieval from cached global score matrices.

Decisive OOD test: the standard benchmark ranks each query against the FULL
pooled gallery (102,363 tiles). Public HEST-1k panels score ~chance there, but
that conflates two failure modes:
  (a) the model never aligns public H&E<->ST at all, or
  (b) it aligns them, but the correct match is drowned in a 100k global pool.

This script restricts the gallery to the query's OWN section (block-diagonal of
the cached score matrix), so chance = 1/n_section instead of 1%. If SIGMMA scores
high within-section on public panels, the alignment DOES transfer and the global
metric just reflects pool scale; if it's still ~chance, the alignment is genuinely
absent on unseen panels.

Reuses outputs/retrieval/full_rerun_step2000/<cond>/score_matrices.npz (no model
re-run). Row order matches section_assignments_test.npz (built from the a01
step2000 config; identical query manifest across all conditions).
"""
from __future__ import annotations
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
RUN = ROOT / "outputs/retrieval/full_rerun_step2000"
SECT = RUN / "section_assignments_test.npz"
META_CSV = Path("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
                "sample_metadata.csv")
OUT = ROOT / "outputs/retrieval/within_section_step2000.json"

CONDS = {
    "bleep":   "in-house_retrieval_bleep_uni_finetuned_step2000",
    "clip":    "in-house_retrieval_clip_uni_finetuned_step2000",
    "plip":    "in-house_retrieval_plip_uni_finetuned_step2000",
    "omiclip": "in-house_retrieval_omiclip_uni_finetuned_step2000",
    "a01":     "in-house_retrieval_sigmma_dense_attnpool_a01_step2000",
}
PUBLIC = {"IDC", "PAAD", "LUAD", "SKCM"}


def section_metrics(block):
    """block: (n,n) score sub-matrix, query rows vs same-section ref cols.
    Ground truth for local row j is local col j (diagonal). Returns dict."""
    n = block.shape[0]
    gt = np.diag(block).copy()
    # rank = #references scoring strictly higher than the true match (0-based).
    rank = (block > gt[:, None]).sum(axis=1)
    out = {
        "n_ref": int(n),
        "recall_at_1": float(np.mean(rank == 0)),
        "recall_at_5": float(np.mean(rank < 5)),
        "recall_at_10": float(np.mean(rank < 10)),
        "recall_at_1pct": float(np.mean(rank < max(1, int(n * 0.01)))),
        "recall_at_5pct": float(np.mean(rank < max(1, int(n * 0.05)))),
        "recall_at_10pct": float(np.mean(rank < max(1, int(n * 0.1)))),
        "mrr": float(np.mean(1.0 / (rank + 1.0))),
        "median_normalized_rank": float(np.median(rank / max(1, n - 1))),
        "chance_recall_at_1": float(1.0 / n),
    }
    return out


def main():
    z = np.load(SECT, allow_pickle=True)
    sect_idx = z["section_idx"]            # (102363,) -> index into section_ids
    sect_ids = z["section_ids"]            # (55,)
    meta = pd.read_csv(META_CSV)
    tt = dict(zip(meta["sample_id"], meta["tissue_type"]))

    # local row indices per section
    rows_of = {i: np.where(sect_idx == i)[0] for i in range(len(sect_ids))}

    results = {}
    for cond, sub in CONDS.items():
        npz_path = RUN / sub / "score_matrices.npz"
        print(f"\n=== {cond}: loading {npz_path}", flush=True)
        npz = np.load(npz_path)
        results[cond] = {"image": {}, "spatial": {}}
        for direction, key in [("image", "scores_img2spatial"),
                               ("spatial", "scores_spatial2img")]:
            print(f"  [{cond}] loading {key} ...", flush=True)
            S = npz[key]                    # (102363, 102363) float32, ~42 GB
            print(f"  [{cond}] {key} shape={S.shape} dtype={S.dtype}", flush=True)
            for i, sid in enumerate(sect_ids):
                r = rows_of[i]
                if len(r) < 2:
                    continue
                block = S[np.ix_(r, r)]
                m = section_metrics(block)
                m["tissue_type"] = str(tt.get(sid, "<no-meta>"))
                m["is_public"] = m["tissue_type"] in PUBLIC
                results[cond][direction][str(sid)] = m
                del block
            del S
            gc.collect()
        npz.close()

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}", flush=True)

    # quick summary: tile-weighted within-section metrics, public vs in-house
    print("\n=== within-section summary (image dir, tile-weighted) ===", flush=True)
    hdr = f"{'cond':8} {'group':9} {'R@1':>8} {'R@1%':>8} {'MRR':>8}"
    print(hdr, flush=True)
    for cond in CONDS:
        ps = results[cond]["image"]
        for grp, want_pub in [("in-house", False), ("public", True)]:
            secs = [s for s, m in ps.items() if m["is_public"] == want_pub]
            if not secs:
                continue
            w = np.array([ps[s]["n_ref"] for s in secs], float)
            vals = {}
            for met in ["recall_at_1", "recall_at_1pct", "mrr"]:
                v = np.array([ps[s][met] for s in secs], float)
                vals[met] = (v * w).sum() / w.sum()
            print(f"  {cond:8} {grp:9} {vals['recall_at_1']:8.4f} "
                  f"{vals['recall_at_1pct']:8.4f} {vals['mrr']:8.4f}", flush=True)


if __name__ == "__main__":
    main()
