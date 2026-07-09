"""Cross-tissue modality-gap + top-K cross-modal retrieval inspection.

Reuses the SAME cosine-similarity retrieval as
`sigmma.evaluation.evaluate_retrieval.retrieval_score`
(score = img_query @ st_reference.T on L2-normalized embeddings), but instead of
reporting recall metrics it reports:

  1) Modality-gap statistics per scale: paired vs random cross-modal cosine,
     within-modality cosine, and the L2 distance between image / ST centroids
     (Liang et al. 2022 "modality gap" definition).
  2) Top-K retrieved tiles' TISSUE COMPOSITION pooled across all 8 tissues
     (a query-tissue x retrieved-tissue mixing matrix) -> shows whether the
     shared space retrieves within-tissue or crosses tissues.
  3) Concrete example top-K tile tables (tissue / section / x,y / similarity /
     whether it is the ground-truth pair).

NOTE: runs on the 2026-05-12 embeddings = UNI v1 backbone (STALE — predates the
2026-05-15 UNI2-h migration). Geometry is indicative only; re-run after UNI2-h
re-extraction for any paper number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

EMB_DIR = Path("outputs/cross_tissue_retrieval/embeddings")
OUT_DIR = Path("outputs/cross_tissue_retrieval/modality_gap_retrieval")
TISSUES = [
    "skin", "glioblastoma", "liver", "lung", "pancreas",
    "pediatric_ovary", "renal_cell_carcinoma", "whole_embryo",
]
SCALES = ["micro", "meso", "macro"]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def load_all(scale: str):
    """Pool all tissues. Row i of img and st is the SAME physical tile (pair)."""
    img, st, tissue, section, xs, ys = [], [], [], [], [], []
    for t in TISSUES:
        p = EMB_DIR / f"{t}.h5"
        with h5py.File(p, "r") as h:
            im = h[f"image_emb_{scale}"][:]
            sp = h[f"st_emb_{scale}"][:]
            n = im.shape[0]
            img.append(im)
            st.append(sp)
            tissue.extend([t] * n)
            section.extend([s.decode() if isinstance(s, bytes) else s
                            for s in h["section_id"][:]])
            xs.append(h["x_start"][:])
            ys.append(h["y_start"][:])
    img = _l2(np.concatenate(img).astype(np.float32))
    st = _l2(np.concatenate(st).astype(np.float32))
    meta = pd.DataFrame({
        "tissue": tissue,
        "section_id": section,
        "x_start": np.concatenate(xs),
        "y_start": np.concatenate(ys),
    })
    return img, st, meta


def modality_gap(rng: np.random.Generator, n_sample: int = 20000) -> pd.DataFrame:
    rows = []
    for scale in SCALES:
        img, st, _ = load_all(scale)
        n = img.shape[0]
        paired = float((img * st).sum(1).mean())
        perm = rng.permutation(n)
        rand_cross = float((img * st[perm]).sum(1).mean())
        k = min(n_sample, n)
        a = rng.integers(0, n, k)
        b = rng.integers(0, n, k)
        within_img = float((img[a] * img[b]).sum(1).mean())
        within_st = float((st[a] * st[b]).sum(1).mean())
        gap = float(np.linalg.norm(img.mean(0) - st.mean(0)))
        rows.append({
            "scale": scale,
            "n_tiles": n,
            "paired_cos": paired,
            "rand_cross_cos": rand_cross,
            "within_img_cos": within_img,
            "within_st_cos": within_st,
            "centroid_gap_L2": gap,
            "paired_minus_rand": paired - rand_cross,
        })
    return pd.DataFrame(rows)


def topk_retrieval(img, st, meta, direction: str, topk: int,
                   query_idx: np.ndarray, batch: int = 1024):
    """Cosine retrieval mirroring evaluate_retrieval.retrieval_score.

    direction 'img2st': query = image emb, reference = ST emb (and vice versa).
    Returns per-query: GT rank, top-k reference row indices, top-k sims.
    """
    if direction == "img2st":
        q_bank, r_bank = img, st
    else:
        q_bank, r_bank = st, img
    n_ref = r_bank.shape[0]
    topk = min(topk, n_ref)
    gt_rank = np.empty(len(query_idx), dtype=np.int64)
    top_idx = np.empty((len(query_idx), topk), dtype=np.int64)
    top_sim = np.empty((len(query_idx), topk), dtype=np.float32)
    for s in range(0, len(query_idx), batch):
        qi = query_idx[s:s + batch]
        scores = q_bank[qi] @ r_bank.T  # (b, n_ref)
        gt_scores = scores[np.arange(len(qi)), qi]
        gt_rank[s:s + batch] = (scores > gt_scores[:, None]).sum(1)
        part = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        part_sim = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        top_idx[s:s + batch] = np.take_along_axis(part, order, axis=1)
        top_sim[s:s + batch] = np.take_along_axis(part_sim, order, axis=1)
    return gt_rank, top_idx, top_sim


def tissue_mixing(meta, query_idx, top_idx, exclude_gt=True):
    """query-tissue x retrieved-tissue mean fraction of top-K."""
    tis = meta["tissue"].to_numpy()
    q_tis = tis[query_idx]
    counts = {qt: {rt: 0.0 for rt in TISSUES} for qt in TISSUES}
    n_per_q = {qt: 0 for qt in TISSUES}
    k = top_idx.shape[1]
    for row, qi in enumerate(query_idx):
        qt = q_tis[row]
        retrieved = top_idx[row]
        if exclude_gt:
            retrieved = retrieved[retrieved != qi]
        rt = tis[retrieved]
        for t in TISSUES:
            counts[qt][t] += int((rt == t).sum())
        n_per_q[qt] += len(retrieved)
    mat = pd.DataFrame(
        [[counts[qt][rt] / max(n_per_q[qt], 1) for rt in TISSUES] for qt in TISSUES],
        index=TISSUES, columns=TISSUES,
    )
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="macro", choices=SCALES,
                    help="scale for the retrieval demo (gap is computed for all)")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--queries_per_tissue", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("MODALITY GAP (all scales)  [STALE UNI v1 embeddings — indicative]")
    print("=" * 70)
    gap_df = modality_gap(rng)
    print(gap_df.to_string(index=False))
    gap_df.to_csv(OUT_DIR / "modality_gap.csv", index=False)

    print()
    print("=" * 70)
    print(f"TOP-{args.topk} CROSS-MODAL RETRIEVAL  (scale={args.scale}, pooled 8 tissues)")
    print("=" * 70)
    img, st, meta = load_all(args.scale)
    n = img.shape[0]

    # sample queries proportionally per tissue
    q_list = []
    for t in TISSUES:
        idx_t = np.where(meta["tissue"].to_numpy() == t)[0]
        take = min(args.queries_per_tissue, len(idx_t))
        q_list.append(rng.choice(idx_t, size=take, replace=False))
    query_idx = np.sort(np.concatenate(q_list))

    for direction in ["img2st", "st2img"]:
        gt_rank, top_idx, top_sim = topk_retrieval(
            img, st, meta, direction, args.topk, query_idx)
        tis = meta["tissue"].to_numpy()
        same_tis = np.array([
            np.mean(tis[top_idx[r]] == tis[query_idx[r]])
            for r in range(len(query_idx))
        ])
        gt_in_topk = float(np.mean(gt_rank < args.topk))
        print(f"\n--- {direction} (n_query={len(query_idx)}, n_ref={n}) ---")
        print(f"  GT recall@{args.topk}      : {gt_in_topk:.3f}")
        print(f"  median GT rank        : {int(np.median(gt_rank))}")
        print(f"  mean same-tissue frac in top-{args.topk}: {same_tis.mean():.3f}")
        mat = tissue_mixing(meta, query_idx, top_idx, exclude_gt=True)
        print(f"  query-tissue x retrieved-tissue mean fraction of top-{args.topk}"
              f" (GT excluded):")
        print(mat.round(3).to_string())
        mat.to_csv(OUT_DIR / f"tissue_mixing_{direction}_{args.scale}.csv")

        # concrete example: one random query per tissue, full top-k tile table
        ex_rows = []
        for t in TISSUES:
            cand = np.where(tis[query_idx] == t)[0]
            if len(cand) == 0:
                continue
            r = int(rng.choice(cand))
            qi = query_idx[r]
            for rank, (ri, sim) in enumerate(zip(top_idx[r], top_sim[r])):
                ex_rows.append({
                    "direction": direction,
                    "query_tissue": t,
                    "query_section": meta.iloc[qi]["section_id"],
                    "query_x": int(meta.iloc[qi]["x_start"]),
                    "query_y": int(meta.iloc[qi]["y_start"]),
                    "rank": rank,
                    "ret_tissue": meta.iloc[ri]["tissue"],
                    "ret_section": meta.iloc[ri]["section_id"],
                    "ret_x": int(meta.iloc[ri]["x_start"]),
                    "ret_y": int(meta.iloc[ri]["y_start"]),
                    "cos_sim": float(sim),
                    "is_gt_pair": bool(ri == qi),
                })
        ex_df = pd.DataFrame(ex_rows)
        ex_df.to_csv(OUT_DIR / f"examples_top{args.topk}_{direction}_{args.scale}.csv",
                     index=False)

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
