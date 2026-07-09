"""Compute MRR + median rank + median normalized rank from existing
score_matrices.npz (stream-decompress, chunked over rows so we don't
materialize the full 15 GB matrix at once).

Also computes per-section aggregation when ``--section_assignments`` is
provided (npz with ``section_idx`` (N,) and ``section_ids`` (S,) — produced
by the retrieval pipeline). Per-section gives mean±std error bars on every
metric across the 32 in-house test sections."""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
import numpy as np


def stream_npy_rows(zip_path: str, npy_name: str, chunk_rows: int = 512):
    """Stream-decompress a single .npy inside a .npz, yielding row chunks.

    Yields ``(row_offset, ndarray (n, N1))`` blocks.
    """
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
                ("mrr","mrr_at_10","median_rank","mean_rank","median_normalized_rank",
                 "recall_at_1","recall_at_5","recall_at_10",
                 "recall_at_1pct","recall_at_5pct","recall_at_10pct")}
    rr = 1.0 / ranks_arr
    # MRR@10: standard IR metric (MS-MARCO, BEIR). Reciprocal rank set to 0
    # for queries whose true match is beyond top-10.
    rr_at_10 = np.where(ranks_arr <= 10, rr, 0.0)
    return {
        "mrr": float(np.mean(rr)),
        "mrr_at_10": float(np.mean(rr_at_10)),
        "median_rank": float(np.median(ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_normalized_rank": float(np.median(ranks_arr / n_references)),
        "recall_at_1":  float(np.mean(ranks_arr == 1)),
        "recall_at_5":  float(np.mean(ranks_arr <= 5)),
        "recall_at_10": float(np.mean(ranks_arr <= 10)),
        "recall_at_1pct":  float(np.mean(ranks_arr <= int(n_references * 0.01))),
        "recall_at_5pct":  float(np.mean(ranks_arr <= int(n_references * 0.05))),
        "recall_at_10pct": float(np.mean(ranks_arr <= int(n_references * 0.10))),
    }


def _aggregate_across_sections(per_section: dict[str, dict]) -> dict:
    """Mean ± std across sections per metric. ``per_section`` is
    ``{section_id: metrics_dict}``."""
    metrics = list(next(iter(per_section.values())).keys())
    out: dict[str, dict] = {}
    for m in metrics:
        vals = np.asarray(
            [per_section[s][m] for s in per_section
             if per_section[s][m] == per_section[s][m]],  # drop NaN
            dtype=np.float64,
        )
        if vals.size == 0:
            out[m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            out[m] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "n":    int(vals.size),
                "min":  float(vals.min()),
                "max":  float(vals.max()),
            }
    return out


def compute_rank_stats(
    zip_path: str, matrix_name: str, chunk_rows: int = 512,
    section_idx: np.ndarray | None = None,
    section_ids: np.ndarray | None = None,
) -> dict:
    """Per-row, rank of the diagonal element (true match) in descending order.

    rank (1-indexed) = #scores strictly greater than the diagonal score, +1.
    Reports MRR, median/mean rank, median normalized rank, plus R@K (sanity).

    When ``section_idx`` is provided, also reports per-section breakdown
    (one metric dict per section) and ``per_section_mean_std`` aggregation
    (mean±std across sections).
    """
    ranks = []
    n_queries = 0
    n_references = 0
    for start, chunk, (N0, N1) in stream_npy_rows(zip_path, matrix_name, chunk_rows):
        n_queries = N0
        n_references = N1
        n = chunk.shape[0]
        diag_cols = np.arange(start, start + n)
        if diag_cols.max() >= N1:
            valid = diag_cols < N1
            chunk = chunk[valid]
            diag_cols = diag_cols[valid]
            n = chunk.shape[0]
        diag_scores = chunk[np.arange(n), diag_cols]
        gt = chunk > diag_scores[:, None]
        rank_per_row = gt.sum(axis=1).astype(np.int64) + 1
        ranks.append(rank_per_row)
    ranks_arr = np.concatenate(ranks)

    pooled = _metrics_from_ranks(ranks_arr, n_references)
    pooled["n_queries"] = int(n_queries)
    pooled["n_references"] = int(n_references)
    pooled["iqr_normalized_rank"] = [
        float(np.percentile(ranks_arr / n_references, 25)),
        float(np.percentile(ranks_arr / n_references, 75)),
    ]

    if section_idx is not None and section_ids is not None:
        assert len(section_idx) == n_queries, (
            f"section_idx length {len(section_idx)} != n_queries {n_queries}"
        )
        per_section: dict[str, dict] = {}
        for s_i, sid in enumerate(section_ids):
            sel = np.where(section_idx == s_i)[0]
            if sel.size == 0:
                continue
            sub_ranks = ranks_arr[sel]
            m = _metrics_from_ranks(sub_ranks, n_references)
            m["n_queries"] = int(sel.size)
            per_section[str(sid)] = m
        pooled["per_section"] = per_section
        pooled["per_section_mean_std"] = _aggregate_across_sections(per_section)

    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write JSON results")
    ap.add_argument("--chunk_rows", type=int, default=512)
    ap.add_argument(
        "--section_assignments",
        default="outputs/retrieval/section_assignments_test.npz",
        help="npz with section_idx (N,) + section_ids (S,) for per-section breakdown",
    )
    ap.add_argument(
        "--paths", nargs="+", default=None,
        help="Override score-matrix paths as cond=path tokens "
             "(e.g. a01=outputs/.../score_matrices.npz). Defaults to the "
             "hardcoded step1000 lineup.",
    )
    args = ap.parse_args()

    section_idx = None; section_ids = None
    if args.section_assignments and Path(args.section_assignments).exists():
        sa = np.load(args.section_assignments, allow_pickle=True)
        section_idx = sa["section_idx"]
        section_ids = sa["section_ids"]
        print(f"[sections] loaded {len(section_ids)} sections, "
              f"{len(section_idx)} query assignments", flush=True)

    if args.paths:
        paths = {}
        for tok in args.paths:
            cond, _, p = tok.partition("=")
            if not p:
                raise ValueError(f"--paths token must be cond=path, got {tok!r}")
            paths[cond] = p
    else:
        paths = {
            "a00":      "outputs/retrieval/gex_sweep_step1000/a00_26881/score_matrices.npz",
            "a01":      "outputs/retrieval/gex_sweep_step1000/a01_698685/score_matrices.npz",
            "a02":      "outputs/retrieval/gex_sweep_step1000/a02_698685/score_matrices.npz",
            "a04":      "outputs/retrieval/gex_sweep_step1000/a04_698685/score_matrices.npz",
            "bleep":    "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_bleep_uni_finetuned_step1000/score_matrices.npz",
            "clip":     "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_clip_uni_finetuned_step1000/score_matrices.npz",
            "plip":     "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_plip_uni_finetuned_step1000/score_matrices.npz",
            "omiclip":  "outputs/retrieval/baselines_uni2h_step1000/in-house_retrieval_omiclip_uni_finetuned_step1000/score_matrices.npz",
        }
    out = {}
    for cond, p in paths.items():
        if not Path(p).exists():
            print(f"[skip] {cond}: missing  ({p})", flush=True)
            continue
        out[cond] = {}
        for direction, mat in [("image", "scores_img2spatial.npy"),
                               ("spatial", "scores_spatial2img.npy")]:
            print(f"[run] {cond:8s} {direction:7s} ...", flush=True)
            stats = compute_rank_stats(
                p, mat, chunk_rows=args.chunk_rows,
                section_idx=section_idx, section_ids=section_ids,
            )
            out[cond][direction] = stats
            ps = stats.get("per_section_mean_std", {})
            mrr_meanstd = (f"{ps['mrr']['mean']:.4f}±{ps['mrr']['std']:.4f}"
                           if ps else f"{stats['mrr']:.4f}")
            r1_meanstd = (f"{ps['recall_at_1']['mean']:.4f}±{ps['recall_at_1']['std']:.4f}"
                          if ps else f"{stats['recall_at_1']:.4f}")
            print(f"  N={stats['n_references']}  MRR={mrr_meanstd}  R@1={r1_meanstd}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
