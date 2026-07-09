"""Per-tissue gallery retrieval metrics. For each tissue T, queries from T are
ranked **only against the gallery cells from T**. Streaming-decompresses the
existing score_matrices.npz files (no full materialization).

Sibling of compute_retrieval_normalized_metrics.py (which uses the full
N=61,613 gallery for all tissues). This script answers: "within tissue T, can
the model identify the matching cell among other cells of the same tissue?"
"""
from __future__ import annotations
import argparse
import json
import re
import zipfile
from pathlib import Path
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
import numpy as np

from sigmma.utils.tissue_map import tissue_of as _tissue_of, GROUP_OF, GROUP_LABEL


def group_of(sid: str) -> str | None:
    """Authoritative display group for a section id (beacon+CTCL -> 'skin').

    Returns the GROUP_OF group of the section's tissue_type, so the two skin
    cohorts collapse into one per-tissue bucket. None if the section is not in
    sample_metadata.csv (e.g. an external public id)."""
    return GROUP_OF.get(_tissue_of(sid))


def stream_npy_rows(zip_path: str, npy_name: str, chunk_rows: int = 512):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(npy_name) as f:
            major, minor = read_magic(f)
            if (major, minor) == (1, 0):
                shape, _, dtype = read_array_header_1_0(f)
            elif (major, minor) == (2, 0):
                shape, _, dtype = read_array_header_2_0(f)
            else:
                raise ValueError(f"unsupported npy version {major}.{minor}")
            N0, N1 = shape
            row_bytes = int(N1 * dtype.itemsize)
            for start in range(0, N0, chunk_rows):
                n = min(chunk_rows, N0 - start)
                buf = f.read(n * row_bytes)
                if len(buf) != n * row_bytes:
                    raise IOError(f"short read at row {start}")
                chunk = np.frombuffer(buf, dtype=dtype).reshape(n, N1)
                yield start, chunk, (N0, N1)


def _metrics_from_ranks(ranks_arr: np.ndarray, n_references: int) -> dict:
    if ranks_arr.size == 0:
        return {k: float("nan") for k in
                ("mrr", "median_rank", "mean_rank", "median_normalized_rank",
                 "recall_at_1", "recall_at_5", "recall_at_10",
                 "recall_at_1pct", "recall_at_5pct", "recall_at_10pct")}
    return {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_normalized_rank": float(np.median(ranks_arr / n_references)),
        "recall_at_1":  float(np.mean(ranks_arr == 1)),
        "recall_at_5":  float(np.mean(ranks_arr <= 5)),
        "recall_at_10": float(np.mean(ranks_arr <= 10)),
        "recall_at_1pct":  float(np.mean(ranks_arr <= max(1, int(n_references * 0.01)))),
        "recall_at_5pct":  float(np.mean(ranks_arr <= max(1, int(n_references * 0.05)))),
        "recall_at_10pct": float(np.mean(ranks_arr <= max(1, int(n_references * 0.10)))),
    }


def compute_per_tissue_gallery_stats(
    zip_path: str, matrix_name: str,
    tissue_of_cell: np.ndarray,
    tissues: list[str],
    chunk_rows: int = 256,
) -> dict:
    """For each tissue T:
       - queries: cells with tissue T
       - gallery: cells with tissue T
       - rank of query cell i (1-indexed) = #cells in T's gallery whose score
         to query i exceeds the (i,i) diagonal score, + 1.
    """
    # Pre-compute column indices per tissue
    tissue_cols: dict[str, np.ndarray] = {}
    cell_local_idx = np.full(tissue_of_cell.shape, -1, dtype=np.int64)
    for t in tissues:
        cols = np.where(tissue_of_cell == t)[0]
        tissue_cols[t] = cols
        cell_local_idx[cols] = np.arange(len(cols))

    # Accumulate ranks per tissue
    ranks_per_tissue: dict[str, list[np.ndarray]] = {t: [] for t in tissues}

    for start, chunk, (N0, N1) in stream_npy_rows(zip_path, matrix_name, chunk_rows):
        n = chunk.shape[0]
        row_global_idx = np.arange(start, start + n)
        row_tissue = tissue_of_cell[row_global_idx]

        for t in tissues:
            mask = (row_tissue == t)
            if not mask.any():
                continue
            chunk_rows_t = chunk[mask]           # (k, N1)
            cols_t = tissue_cols[t]               # (N_T,)
            sub = chunk_rows_t[:, cols_t]         # (k, N_T)
            local_positions = cell_local_idx[row_global_idx[mask]]
            k = sub.shape[0]
            diag = sub[np.arange(k), local_positions]
            gt = sub > diag[:, None]
            rank = gt.sum(axis=1).astype(np.int64) + 1
            ranks_per_tissue[t].append(rank)

    out: dict[str, dict] = {}
    for t in tissues:
        if not ranks_per_tissue[t]:
            continue
        ranks_arr = np.concatenate(ranks_per_tissue[t])
        n_ref = int(tissue_cols[t].size)
        m = _metrics_from_ranks(ranks_arr, n_ref)
        m["n_queries"] = int(ranks_arr.size)
        m["n_references"] = n_ref
        out[t] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_rows", type=int, default=256)
    ap.add_argument(
        "--section_assignments",
        default="outputs/retrieval/section_assignments_test.npz",
    )
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of {a00,a01,a02,a04,bleep,clip,plip,omiclip}")
    args = ap.parse_args()

    sa = np.load(args.section_assignments, allow_pickle=True)
    section_idx = sa["section_idx"]            # (N,)
    section_ids = sa["section_ids"]            # (S,)
    # authoritative tissue group per section (beacon+CTCL -> 'skin'); external
    # public sections (no Sanger panel) collapse to None and are flagged below.
    tissue_per_section = np.array(
        [group_of(str(s)) for s in section_ids], dtype=object
    )
    if any(t is None for t in tissue_per_section):
        bad = [(i, s) for i, (s, t) in enumerate(zip(section_ids, tissue_per_section)) if t is None]
        raise ValueError(f"unmapped sections: {bad}")
    tissue_of_cell = tissue_per_section[section_idx]   # (N,)
    # tissue groups present, ordered by descending cell count (stable output order)
    groups, counts = np.unique(tissue_of_cell.astype(str), return_counts=True)
    tissues = list(groups[np.argsort(-counts)])
    print(f"[tissues] N={len(tissue_of_cell)}; per-tissue counts:", flush=True)
    for t in tissues:
        print(f"  {GROUP_LABEL.get(t, t)} ({t}): "
              f"{int((tissue_of_cell == t).sum())}", flush=True)

    paths = {
        "a00":     "outputs/retrieval/gex_sweep_step1000/a00_26881/score_matrices.npz",
        "a01":     "outputs/retrieval/gex_sweep_step1000/a01_698685/score_matrices.npz",
        "a02":     "outputs/retrieval/gex_sweep_step1000/a02_698685/score_matrices.npz",
        "a04":     "outputs/retrieval/gex_sweep_step1000/a04_698685/score_matrices.npz",
        "bleep":   "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_bleep_uni_finetuned_step1000/score_matrices.npz",
        "clip":    "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_clip_uni_finetuned_step1000/score_matrices.npz",
        "plip":    "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_plip_uni_finetuned_step1000/score_matrices.npz",
        "omiclip": "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_omiclip_uni_finetuned_step1000/score_matrices.npz",
    }
    conds = args.conditions or list(paths.keys())

    out = {}
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path) as f:
            out = json.load(f)
        print(f"[resume] loaded {len(out)} prior conditions from {out_path}", flush=True)

    for cond in conds:
        p = paths[cond]
        if not Path(p).exists():
            print(f"[skip] {cond}: missing  ({p})", flush=True)
            continue
        out.setdefault(cond, {})
        for direction, mat in [("image", "scores_img2spatial.npy"),
                               ("spatial", "scores_spatial2img.npy")]:
            if direction in out[cond] and out[cond][direction]:
                print(f"[skip] {cond} {direction}: already in out", flush=True)
                continue
            print(f"[run] {cond:8s} {direction:7s} ...", flush=True)
            stats = compute_per_tissue_gallery_stats(
                p, mat, tissue_of_cell, tissues, chunk_rows=args.chunk_rows,
            )
            out[cond][direction] = stats
            for t in tissues:
                if t in stats:
                    s = stats[t]
                    print(f"  {t}: N_ref={s['n_references']:5d}  N_q={s['n_queries']:5d}  "
                          f"R@1={s['recall_at_1']:.4f}  R@10={s['recall_at_10']:.4f}  "
                          f"MRR={s['mrr']:.4f}", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
