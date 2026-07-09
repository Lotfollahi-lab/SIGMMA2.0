"""
Cross-slide niche label-transfer probe.

For each (label, embedding) pair, fit a logistic regression on cells from
``split == 'train'`` slides and evaluate on cells from ``split == 'test'``
slides — i.e. true section-level transfer using the same split SIGMMA was
trained with.

Embeddings (per-cell, from extract_cell_niche_features.py output):
  - z_uni_macro     (1024 d)  raw UNI @ 256 px crop — single UNI baseline.
  - z_sigmma_micro  (512  d)  per-scale LoRA adapter scale_0, scale_mlps proj.
  - z_sigmma_meso   (512  d)  per-scale LoRA adapter scale_1, scale_mlps proj.
  - z_sigmma_macro  (512  d)  per-scale LoRA adapter scale_2, scale_mlps proj.
  - z_sigmma_concat (1536 d)  concat of the three projected SIGMMA scales.
  - z_sigmma_<scale>_dense_noproj (1536 d)  RAW-GRID variant: mask-pooled
        LoRA-adapted UNI patch tokens, WITHOUT the scale_mlps projection head
        (parallels the raw 16x16 patch grid used by the EVA / GEX-2D probes).
  - z_sigmma_concat_dense_noproj (4608 d)  concat of the three raw-grid scales.

Labels: niche_category (7-class), niches_terra (17-class). Both are evaluated
in parallel.

Outputs (under ``--out_dir``):
  - probe_metrics_cross_slide.csv      5 emb x 2 labels = 10 rows.
  - probe_metrics_per_slide.csv        per test-slide breakdown.
  - niche_category_overlay.png         GT + 5 embeddings on one test slide.
  - niches_terra_overlay.png           same, for the fine-grained label.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


SCALES = ("micro", "meso", "macro")
# Baselines that may live in extra feature dirs (one per FT'd UNI variant).
# Short names match scripts/extract_cell_niche_features_baselines.py.
BASELINE_SHORTS = ("bleep", "clip", "plip", "omiclip")
EMBEDDINGS = (
    "z_uni_micro",                  # UNI dense pool, 1024d (64 px crop)
    "z_uni_meso",                   # UNI dense pool, 1024d (128 px crop)
    "z_uni_macro",                  # UNI dense pool, 1024d (256 px crop)
    "z_uni_macro_cls",              # UNI CLS, 1024d
    "z_sigmma_micro",               # SIGMMA dense + projection, 512d
    "z_sigmma_meso",                # SIGMMA dense + projection, 512d
    "z_sigmma_macro",               # SIGMMA dense + projection, 512d
    "z_sigmma_concat",              # SIGMMA dense + projection concat, 1536d
    "z_sigmma_micro_dense_noproj",  # SIGMMA dense, no projection, 1536d (64 px)
    "z_sigmma_meso_dense_noproj",   # SIGMMA dense, no projection, 1536d (128 px)
    "z_sigmma_macro_dense_noproj",  # SIGMMA dense, no projection, 1536d (256 px)
    "z_sigmma_concat_dense_noproj", # SIGMMA raw-grid concat (no projection), 4608d
    "z_sigmma_macro_cls",           # LoRA-UNI CLS, 1024d
) + tuple(
    f"z_{b}_{s}{suffix}"
    for b in BASELINE_SHORTS
    for s in SCALES
    for suffix in ("", "_dense_noproj", "_cls")
) + tuple(f"z_{b}_concat" for b in BASELINE_SHORTS)

EMB_TITLE = {
    "z_uni_micro": "UNI@64-dense",
    "z_uni_meso": "UNI@128-dense",
    "z_uni_macro": "UNI@256-dense",
    "z_uni_macro_cls": "UNI@256-CLS",
    "z_sigmma_micro": "SIGMMA-micro",
    "z_sigmma_meso": "SIGMMA-meso",
    "z_sigmma_macro": "SIGMMA-macro",
    "z_sigmma_concat": "SIGMMA-concat",
    "z_sigmma_micro_dense_noproj": "SIGMMA-micro-noproj",
    "z_sigmma_meso_dense_noproj": "SIGMMA-meso-noproj",
    "z_sigmma_macro_dense_noproj": "SIGMMA-macro-noproj",
    "z_sigmma_concat_dense_noproj": "SIGMMA-concat-noproj",
    "z_sigmma_macro_cls": "SIGMMA-UNI-FT",
}
for _b in BASELINE_SHORTS:
    _label = _b.upper() if _b != "omiclip" else "OmiCLIP"
    for _s, _crop in zip(SCALES, ("64", "128", "256")):
        EMB_TITLE[f"z_{_b}_{_s}"] = f"{_label}-{_s}"
        EMB_TITLE[f"z_{_b}_{_s}_dense_noproj"] = f"{_label}@{_crop}-dense"
        EMB_TITLE[f"z_{_b}_{_s}_cls"] = f"{_label}@{_crop}-CLS"
    EMB_TITLE[f"z_{_b}_concat"] = f"{_label}-concat"

LABEL_FIELDS = ("niche_category", "niches_terra")

# Fixed overlay colours for RCC level_2_cell_type: tumour = brand pink, the four
# stromal compartments = blue family, the rest (normal epithelium + immune) =
# distinct non-blue hues. Classes absent here fall back to tab20.
CELLTYPE_COLOURS = {
    "Tumour epithelium": "#ca7695",     # pink (brand)
    # stroma — blue family
    "Fibroblast": "#2171b5",            # strong blue
    "Blood vasculature": "#6baed6",     # light blue
    "Perivascular cell": "#08306b",     # navy
    "Lymphatic vasculature": "#9ecae1", # pale blue
    # normal epithelium + immune — kept off-blue to contrast with stroma
    "Nephron epithelium": "#4daf4a",    # green
    "Macrophage": "#ff7f00",            # orange
    "T lymphocyte": "#984ea3",          # purple
    "B lymphocyte": "#a65628",          # brown
    "Dendritic cell": "#ffd92f",        # yellow
}


def _load_slide_features(
    *,
    data_h5: Path,
    uni_h5: Path | None,
    sigmma_h5: Path | None,
    baseline_h5s: list[Path] | None = None,
    wanted_raw_keys: set[str] | None = None,
) -> dict:
    """Join cell_id-aligned datasets from up to three per-slide HDF5s.

      * data_h5    — required. Provides cell_id + labels.
      * uni_h5     — optional. Provides z_uni_* (raw UNI features).
      * sigmma_h5  — optional. Provides z_sigmma_* (LoRA-adapted features).
      * baseline_h5s — optional list of finetuned-UNI baseline H5s
        (BLEEP/CLIP/OmiCLIP/PLIP) keyed by z_<baseline>_<scale>.

    ``wanted_raw_keys`` — if given, only these raw z_ datasets are read into
    memory (the per-scale keys needed to also build any requested concat).
    Skips the wide 1024-d ``_cls`` / ``_dense_noproj`` variants and unused
    baseline scales, which otherwise blow past the memory limit on the
    whole-embryo cohort (1M cells × ~30 datasets). ``None`` loads every z_ key.

    cell_id ordering is asserted equal across every loaded HDF5 — drift is
    caught here, not silently passed to LogReg.
    """
    with h5py.File(data_h5, "r") as f:
        out = {
            "cell_id": np.array([c.decode() for c in f["cell_id"][:]]),
            "sample": f.attrs["sample_name"],
        }
        # Label datasets are discovered from the attr written by the extractor;
        # fall back to the historical skin pair for caches predating the attr.
        label_names = [n for n in f.attrs.get(
            "label_names", ["niche_category", "niches_terra"])]
        for name in label_names:
            out[name] = np.array([c.decode() for c in f[name][:]])

    def _merge(h5_path: Path, label: str) -> None:
        with h5py.File(h5_path, "r") as fb:
            cid_b = np.array([c.decode() for c in fb["cell_id"][:]])
            if cid_b.shape != out["cell_id"].shape or not np.array_equal(cid_b, out["cell_id"]):
                raise SystemExit(
                    f"cell_id mismatch between data ({data_h5.name}) and "
                    f"{label} ({h5_path}): re-extract with matching evaluable manifest"
                )
            for k in fb.keys():
                if k == "cell_id":
                    continue
                if not k.startswith("z_") or k in out:
                    continue
                if wanted_raw_keys is not None and k not in wanted_raw_keys:
                    continue
                out[k] = fb[k][:]

    if uni_h5 is not None:
        _merge(uni_h5, "uni")
    if sigmma_h5 is not None:
        _merge(sigmma_h5, "sigmma")
    if baseline_h5s:
        for extra in baseline_h5s:
            if not extra.exists():
                print(f"  [skip] no baseline h5: {extra}")
                continue
            _merge(extra, "baseline")

    # Build SIGMMA concat lazily (projected + raw-grid / no-projection).
    if all(f"z_sigmma_{s}" in out for s in SCALES):
        out["z_sigmma_concat"] = np.concatenate(
            [out[f"z_sigmma_{s}"] for s in SCALES], axis=1
        )
    if all(f"z_sigmma_{s}_dense_noproj" in out for s in SCALES):
        out["z_sigmma_concat_dense_noproj"] = np.concatenate(
            [out[f"z_sigmma_{s}_dense_noproj"] for s in SCALES], axis=1
        )
    # And baseline-FT concats.
    for b in BASELINE_SHORTS:
        keys = [f"z_{b}_{s}" for s in SCALES]
        if all(k in out for k in keys):
            out[f"z_{b}_concat"] = np.concatenate([out[k] for k in keys], axis=1)
    return out


def _gather_split(
    *,
    data_dir: Path,
    uni_dir: Path | None,
    sigmma_dir: Path | None,
    manifest: pd.DataFrame,
    split_value: str,
    baseline_dirs: list[Path] | None = None,
    wanted_raw_keys: set[str] | None = None,
):
    slides = []
    rows = manifest[manifest["split"] == split_value]
    for sample in rows["sample_name"]:
        data_h5 = data_dir / f"{sample}.h5"
        if not data_h5.exists():
            print(f"  [skip] {sample}: no data h5 at {data_h5}")
            continue
        uni_h5 = (uni_dir / f"{sample}.h5") if uni_dir else None
        if uni_h5 and not uni_h5.exists():
            print(f"  [warn] {sample}: uni h5 missing at {uni_h5}; uni features unavailable")
            uni_h5 = None
        sigmma_h5 = (sigmma_dir / f"{sample}.h5") if sigmma_dir else None
        if sigmma_h5 and not sigmma_h5.exists():
            print(f"  [warn] {sample}: sigmma h5 missing at {sigmma_h5}; sigmma features unavailable")
            sigmma_h5 = None
        extras = (
            [d / f"{sample}.h5" for d in baseline_dirs] if baseline_dirs else None
        )
        slides.append(_load_slide_features(
            data_h5=data_h5, uni_h5=uni_h5, sigmma_h5=sigmma_h5,
            baseline_h5s=extras, wanted_raw_keys=wanted_raw_keys,
        ))
    return slides


def _stack(slides: list[dict], key: str) -> np.ndarray:
    return np.concatenate([s[key] for s in slides], axis=0)


def _slide_offsets(slides: list[dict]) -> list[tuple[str, int, int]]:
    offsets = []
    cursor = 0
    for s in slides:
        n = len(s["cell_id"])
        offsets.append((s["sample"], cursor, cursor + n))
        cursor += n
    return offsets


def _probe_train_eval(X_tr, y_tr, eval_sets):
    """LogReg fit on (X_tr, y_tr) once; predict on each (name, X, y) in
    ``eval_sets``. Cells whose label is not present in the train fold are
    dropped — LR cannot predict unseen classes; their count is reported
    per eval split.

    Returns ``(out, scaler, clf)`` where ``out`` is a dict keyed by eval-set
    name (each entry has the same shape as the legacy ``_probe_train_test``
    return value, plus ``split`` for clarity), and ``scaler``/``clf`` are the
    fitted ``StandardScaler`` + ``LogisticRegression`` so the caller can persist
    a reusable predictor.
    """
    train_classes = set(y_tr)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)

    clf = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0, n_jobs=-1,
    ).fit(Xtr_s, y_tr)

    out = {}
    for name, X_ev, y_ev in eval_sets:
        keep = np.array([c in train_classes for c in y_ev])
        X_ev_kept, y_ev_kept = X_ev[keep], y_ev[keep]
        n_dropped = int((~keep).sum())
        Xev_s = scaler.transform(X_ev_kept)
        preds = clf.predict(Xev_s)
        full_preds_kept = np.array([""] * len(y_ev), dtype=object)
        full_preds_kept[keep] = preds
        out[name] = {
            "split": name,
            "preds_aligned_to_y_eval": full_preds_kept,
            "keep_mask": keep,
            "n_dropped": n_dropped,
            "n_train_cells": int(len(y_tr)),
            "n_eval_cells_kept": int(len(y_ev_kept)),
            "n_train_classes": int(len(train_classes)),
            "n_eval_classes_kept": int(len(set(y_ev_kept))),
            "accuracy": (
                float(accuracy_score(y_ev_kept, preds)) if len(y_ev_kept) else float("nan")
            ),
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_ev_kept, preds)) if len(y_ev_kept) else float("nan")
            ),
            "macro_f1": (
                float(f1_score(y_ev_kept, preds, average="macro")) if len(y_ev_kept) else float("nan")
            ),
        }
    return out, scaler, clf


def _read_thumbnail(zarr_path: Path, target_long_side: int = 1500):
    z = zarr.open(str(zarr_path), mode="r")
    C, H, W = z.shape
    step = max(1, int(np.ceil(max(H, W) / target_long_side)))
    thumb = z[:, ::step, ::step][:]
    if thumb.dtype != np.uint8:
        thumb = thumb.astype(np.uint8)
    return np.transpose(thumb, (1, 2, 0)), step


def _load_centroids(preproc_path: Path, um_per_px: float = 0.5) -> pd.DataFrame:
    # Overlay-only helper. Mirrors extract._load_preproc_centroids: prefer pixel
    # columns, else derive from µm centroids (post-05-29 data has µm only).
    with h5py.File(preproc_path, "r") as f:
        obs = f["obs"]
        cell_ids = [c.decode() if isinstance(c, bytes) else c
                    for c in obs["cell_id"][:]]
        if "x_pixel" in obs and "y_pixel" in obs:
            x = obs["x_pixel"][:]
            y = obs["y_pixel"][:]
        else:
            x = obs["x_centroid"][:] / um_per_px
            y = obs["y_centroid"][:] / um_per_px
    return pd.DataFrame(
        {"cell_id": cell_ids, "x_pixel": x, "y_pixel": y}
    ).set_index("cell_id")


def _plot_overlay(thumb, step, centroids_xy, gt, preds_per_emb, emb_titles,
                  out_path, sample_name, label_field):
    # Predictions can include train-only classes absent from this test slide's
    # GT, so the colour map covers the union.
    classes = set(c for c in gt if c)
    for vec in preds_per_emb:
        classes.update(c for c in vec if c)
    classes = sorted(classes)
    cmap = plt.get_cmap("tab20", max(len(classes), 1))
    colour = {c: CELLTYPE_COLOURS.get(c, cmap(i)) for i, c in enumerate(classes)}

    panels = [(f"Ground truth (n={len(gt)})", gt)]
    panels += list(zip(emb_titles, preds_per_emb))
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]
    xs = centroids_xy[:, 0] / step
    ys = centroids_xy[:, 1] / step
    # Draw abundant classes first so the rarest sit on top (reinforced by
    # zorder); faint H&E background. Mirrors scripts/plot_celltype_overlay.py.
    from collections import Counter as _Counter
    _counts = _Counter()
    for vec in [gt] + list(preds_per_emb):
        _counts.update(c for c in vec if c)
    draw_order = sorted(classes, key=lambda c: _counts.get(c, 0), reverse=True)
    for ax, (title, vec) in zip(axes, panels):
        ax.imshow(thumb, alpha=0.2)  # faint H&E background
        vec = np.asarray(vec)
        for z, c in enumerate(draw_order):
            m = vec == c
            if not m.any():
                continue
            ax.scatter(xs[m], ys[m], c=[colour[c]], s=3, linewidths=0,
                       alpha=0.85, zorder=2 + z)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Legend order: Tumour -> stroma -> rest (CELLTYPE_COLOURS insertion order),
    # unknown classes appended last. Independent of the scatter draw order.
    legend_order = ([c for c in CELLTYPE_COLOURS if c in classes]
                    + [c for c in classes if c not in CELLTYPE_COLOURS])
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=colour[c], markeredgecolor="none",
                          markersize=8, label=c) for c in legend_order]
    # One cell type per row, placed to the right of the rightmost panel.
    fig.legend(handles=handles, loc="center left", ncol=1,
               frameon=False, bbox_to_anchor=(1.0, 0.5),
               title=label_field, fontsize=9)
    fig.suptitle(
        f"Cross-slide niche transfer ({label_field}) — test slide {sample_name}",
        y=1.02, fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] wrote {out_path}")


def main():
    global LABEL_FIELDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Per-slide DATA HDF5 dir (kind=shared_data) with "
                         "cell_id + niche labels. Always required.")
    ap.add_argument("--uni_dir", default=None,
                    help="Per-slide UNI feature HDF5 dir (kind=uni_features). "
                         "Required for any z_uni_* embedding to be evaluated.")
    ap.add_argument("--sigmma_dir", default=None,
                    help="Per-slide SIGMMA feature HDF5 dir (kind=sigmma_features). "
                         "Required for any z_sigmma_* embedding to be evaluated.")
    ap.add_argument("--manifest", required=True,
                    help="cell_niche_evaluable_slides.csv with split column.")
    ap.add_argument("--out_dir", required=True,
                    help="Dir for DATA artifacts (metrics CSVs, predictions.npz, "
                         "models/). Figures go to --fig_dir.")
    ap.add_argument("--fig_dir", default=None,
                    help="Dir for FIGURE outputs (overlays). Defaults to --out_dir "
                         "for back-compat; pass a path under figures/ to keep all "
                         "plots there.")
    ap.add_argument(
        "--baseline_features_dirs", nargs="*", default=None,
        help="Optional list of per-baseline feature dirs (one per FT'd UNI "
             "variant). Each must contain <sample>.h5 files with cell_id "
             "ordering matching --data_dir.",
    )
    ap.add_argument(
        "--embeddings", nargs="*", default=None,
        help="If set, restrict probing to these embedding keys (e.g. "
             "z_uni_macro z_bleep_macro). Subset of the EMBEDDINGS constant.",
    )
    ap.add_argument(
        "--label_fields", default=",".join(LABEL_FIELDS),
        help="Comma-separated niche label columns to probe (must exist in the "
             "data H5s). Default = skin's 'niche_category,niches_terra'; for "
             "whole_embryo pass 'niche40'.",
    )
    args = ap.parse_args()

    LABEL_FIELDS = tuple(s for s in args.label_fields.split(",") if s)

    from sigmma.caches import validate_cache_meta
    data_dir = Path(args.data_dir)
    validate_cache_meta(data_dir, required={"kind": "shared_data", "task": "cell_niche"})
    uni_dir = Path(args.uni_dir) if args.uni_dir else None
    if uni_dir is not None:
        validate_cache_meta(uni_dir, required={"kind": "uni_features", "task": "cell_niche"})
    sigmma_dir = Path(args.sigmma_dir) if args.sigmma_dir else None
    if sigmma_dir is not None:
        validate_cache_meta(sigmma_dir, required={"kind": "sigmma_features", "task": "cell_niche"})

    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    baseline_dirs = (
        [Path(d) for d in args.baseline_features_dirs]
        if args.baseline_features_dirs else None
    )
    if baseline_dirs:
        print(f"[load] also merging baseline feature dirs: "
              f"{[str(d) for d in baseline_dirs]}")

    print(f"[load] data_dir={data_dir} uni_dir={uni_dir} sigmma_dir={sigmma_dir}")
    # Restrict what's read into RAM to exactly the requested embeddings,
    # expanding any *_concat into its per-scale source keys. Without this every
    # z_ dataset (incl. wide 1024-d _cls/_dense_noproj) is loaded for every
    # slide — fine for skin, OOM on whole-embryo (1M cells). None => load all.
    wanted_raw_keys: set[str] | None = None
    if args.embeddings:
        wanted_raw_keys = set()
        for e in args.embeddings:
            if e == "z_sigmma_concat":
                wanted_raw_keys.update(f"z_sigmma_{s}" for s in SCALES)
            elif e == "z_sigmma_concat_dense_noproj":
                wanted_raw_keys.update(f"z_sigmma_{s}_dense_noproj" for s in SCALES)
            elif e.endswith("_concat"):
                b = e[len("z_"):-len("_concat")]
                wanted_raw_keys.update(f"z_{b}_{s}" for s in SCALES)
            else:
                wanted_raw_keys.add(e)
    gather_kwargs = dict(
        data_dir=data_dir, uni_dir=uni_dir, sigmma_dir=sigmma_dir,
        manifest=manifest, baseline_dirs=baseline_dirs,
        wanted_raw_keys=wanted_raw_keys,
    )
    train_slides = _gather_split(split_value="train", **gather_kwargs)
    val_slides = _gather_split(split_value="val", **gather_kwargs)
    test_slides = _gather_split(split_value="test", **gather_kwargs)
    print(f"  loaded: train={len(train_slides)}  val={len(val_slides)}  "
          f"test={len(test_slides)}")
    if not train_slides or not test_slides:
        raise SystemExit("Insufficient slides in train or test split.")

    # Skip embeddings that aren't present in every loaded slide (e.g. CLS
    # variants only exist in the re-extracted run; baseline features may be
    # missing for slides whose baseline H5 was not produced). Intersect across
    # ALL slides so a later _stack() doesn't KeyError.
    all_slides = train_slides + val_slides + test_slides
    available = set(all_slides[0])
    for s in all_slides[1:]:
        available &= set(s)
    if args.embeddings:
        emb_pool = tuple(e for e in EMBEDDINGS if e in args.embeddings)
        unknown = [e for e in args.embeddings if e not in EMBEDDINGS]
        if unknown:
            raise SystemExit(f"Unknown embeddings (not in EMBEDDINGS): {unknown}")
    else:
        emb_pool = EMBEDDINGS
    embeddings_present = tuple(e for e in emb_pool if e in available)
    skipped = [e for e in emb_pool if e not in available]
    if skipped:
        print(f"  [skip] embeddings not in H5: {skipped}")
    if not embeddings_present:
        raise SystemExit("No requested embeddings found in the loaded H5s.")

    test_offsets = _slide_offsets(test_slides)
    val_offsets = _slide_offsets(val_slides) if val_slides else []

    # Pre-stack labels (small) once per split.
    y_train_all = {lf: _stack(train_slides, lf) for lf in LABEL_FIELDS}
    y_test_all  = {lf: _stack(test_slides, lf) for lf in LABEL_FIELDS}
    y_val_all   = {lf: _stack(val_slides, lf) for lf in LABEL_FIELDS} if val_slides else {}

    # Pick the test slide with the most labeled cells across both labels for viz.
    best_sample, best_n = None, 0
    for s in test_slides:
        n = sum(int((s[lf] != "").sum()) for lf in LABEL_FIELDS)
        if n > best_n:
            best_n, best_sample = n, s["sample"]
    print(f"  [viz] picked test slide for overlays: {best_sample} ({best_n} labels)")

    main_rows = []
    per_slide_rows = []
    per_class_rows = []
    pred_arrays = {}  # for npz: keys "{label}/{embedding}/{split}/{kind}" → array
    test_preds_full = {lf: {} for lf in LABEL_FIELDS}

    # Iterate embeddings outer (memory-friendly: only one X stacked at a time).
    for emb_key in embeddings_present:
        print(f"\n[stack] {emb_key}")
        X_train = _stack(train_slides, emb_key)
        X_test  = _stack(test_slides, emb_key)
        X_val   = _stack(val_slides, emb_key) if val_slides else None
        print(f"  X_train={X_train.shape}  X_test={X_test.shape}"
              f"{' X_val=' + str(X_val.shape) if X_val is not None else ''}")

        for label_field in LABEL_FIELDS:
            y_train = y_train_all[label_field]
            y_test  = y_test_all[label_field]
            y_val   = y_val_all.get(label_field)

            train_keep = y_train != ""
            test_keep  = y_test != ""
            val_keep   = (y_val != "") if y_val is not None else None
            X_tr = X_train[train_keep]; y_tr = y_train[train_keep]
            X_te = X_test[test_keep];   y_te = y_test[test_keep]
            X_va, y_va = (None, None)
            if X_val is not None:
                X_va = X_val[val_keep]; y_va = y_val[val_keep]

            print(f"  [{label_field}] probing {emb_key}: "
                  f"X_tr={X_tr.shape} X_te={X_te.shape}"
                  f"{' X_va=' + str(X_va.shape) if X_va is not None else ''} "
                  f"classes_tr={len(set(y_tr))}")
            eval_sets = [("test", X_te, y_te)]
            if X_va is not None and len(y_va) > 0:
                eval_sets.append(("val", X_va, y_va))
            probe_out, scaler, clf = _probe_train_eval(X_tr, y_tr, eval_sets)

            # Persist the reusable predictor (scaler + LogReg) so it can be
            # re-applied to new cells without re-fitting. One bundle per
            # (label, embedding); ``classes_`` is the LogReg label order.
            models_dir = out_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"{label_field}__{emb_key}.joblib"
            joblib.dump(
                {
                    "scaler": scaler,
                    "clf": clf,
                    "label_field": label_field,
                    "embedding": emb_key,
                    "classes_": clf.classes_,
                    "n_features": int(X_tr.shape[1]),
                    "n_train_cells": int(len(y_tr)),
                },
                model_path,
            )
            print(f"    [model] wrote {model_path}")

            for split_name, res in probe_out.items():
                print(f"    [{split_name}] acc={res['accuracy']:.4f} "
                      f"bal={res['balanced_accuracy']:.4f} "
                      f"macro_f1={res['macro_f1']:.4f} dropped={res['n_dropped']}")
                main_rows.append({
                    "label": label_field,
                    "embedding": emb_key,
                    "split": split_name,
                    "n_train_cells": res["n_train_cells"],
                    "n_eval_cells_kept": res["n_eval_cells_kept"],
                    "n_eval_dropped": res["n_dropped"],
                    "n_train_classes": res["n_train_classes"],
                    "n_eval_classes_kept": res["n_eval_classes_kept"],
                    "accuracy": res["accuracy"],
                    "balanced_accuracy": res["balanced_accuracy"],
                    "macro_f1": res["macro_f1"],
                })

            # Per-class metrics on test (kept consistent with previous output).
            res_test = probe_out["test"]
            keep_mask = res_test["keep_mask"]
            preds_kept = res_test["preds_aligned_to_y_eval"][keep_mask]
            y_kept = y_te[keep_mask]
            class_names = sorted(set(y_kept) | set(preds_kept))
            prec, rec, f1, support = precision_recall_fscore_support(
                y_kept, preds_kept, labels=class_names, zero_division=0,
            )
            for c, p, r, f, s in zip(class_names, prec, rec, f1, support):
                per_class_rows.append({
                    "label": label_field,
                    "embedding": emb_key,
                    "class": c,
                    "support_test": int(s),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                })
            pred_arrays[f"{label_field}/{emb_key}/test/y_true"] = y_kept.astype("U")
            pred_arrays[f"{label_field}/{emb_key}/test/y_pred"] = preds_kept.astype("U")
            if "val" in probe_out:
                rv = probe_out["val"]
                vkeep = rv["keep_mask"]
                pred_arrays[f"{label_field}/{emb_key}/val/y_true"] = y_va[vkeep].astype("U")
                pred_arrays[f"{label_field}/{emb_key}/val/y_pred"] = \
                    rv["preds_aligned_to_y_eval"][vkeep].astype("U")

            # Re-expand TEST predictions to full stacked test (incl. unlabeled)
            # for per-slide breakdown + overlay plots.
            preds_full_stacked = np.array([""] * len(y_test), dtype=object)
            test_keep_idx = np.flatnonzero(test_keep)
            preds_full_stacked[test_keep_idx[keep_mask]] = \
                res_test["preds_aligned_to_y_eval"][keep_mask]
            test_preds_full[label_field][emb_key] = preds_full_stacked

            for sample, start, end in test_offsets:
                gt_slide = y_test[start:end]
                pred_slide = preds_full_stacked[start:end]
                valid = (gt_slide != "") & (pred_slide != "")
                if not valid.any():
                    continue
                gt_v = gt_slide[valid]; pr_v = pred_slide[valid]
                per_slide_rows.append({
                    "label": label_field,
                    "embedding": emb_key,
                    "sample": sample,
                    "split": "test",
                    "n_cells_scored": int(valid.sum()),
                    "accuracy": float(accuracy_score(gt_v, pr_v)),
                    "balanced_accuracy": float(balanced_accuracy_score(gt_v, pr_v)),
                    "macro_f1": float(f1_score(gt_v, pr_v, average="macro")),
                    "weighted_f1": float(f1_score(gt_v, pr_v, average="weighted")),
                    "mcc": float(matthews_corrcoef(gt_v, pr_v)),
                })

            # Per-slide breakdown on VAL too.
            if "val" in probe_out and val_offsets:
                rv = probe_out["val"]
                val_full = np.array([""] * len(y_val), dtype=object)
                val_keep_idx = np.flatnonzero(val_keep)
                val_full[val_keep_idx[rv["keep_mask"]]] = \
                    rv["preds_aligned_to_y_eval"][rv["keep_mask"]]
                for sample, start, end in val_offsets:
                    gt_slide = y_val[start:end]
                    pred_slide = val_full[start:end]
                    valid = (gt_slide != "") & (pred_slide != "")
                    if not valid.any():
                        continue
                    gt_v = gt_slide[valid]; pr_v = pred_slide[valid]
                    per_slide_rows.append({
                        "label": label_field,
                        "embedding": emb_key,
                        "sample": sample,
                        "split": "val",
                        "n_cells_scored": int(valid.sum()),
                        "accuracy": float(accuracy_score(gt_v, pr_v)),
                        "balanced_accuracy": float(balanced_accuracy_score(gt_v, pr_v)),
                        "macro_f1": float(f1_score(gt_v, pr_v, average="macro")),
                        "weighted_f1": float(f1_score(gt_v, pr_v, average="weighted")),
                        "mcc": float(matthews_corrcoef(gt_v, pr_v)),
                    })

        del X_train, X_test  # release before next embedding
        if X_val is not None:
            del X_val

    main_path = out_dir / "probe_metrics_cross_slide.csv"
    pd.DataFrame(main_rows).to_csv(main_path, index=False)
    print(f"\n[main metrics] wrote {main_path}")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(pd.DataFrame(main_rows).to_string(index=False))

    per_slide_df = pd.DataFrame(per_slide_rows)
    per_slide_path = out_dir / "probe_metrics_per_slide.csv"
    per_slide_df.to_csv(per_slide_path, index=False)
    print(f"[per-slide metrics] wrote {per_slide_path}")

    # Section-level aggregate: each slide is one section, treat slides as
    # independent samples and report mean ± std across them. This is the
    # right framing when section variability matters more than pooled cell
    # counts (a single big slide can otherwise dominate the cross-slide row).
    if not per_slide_df.empty:
        agg_metrics = ["accuracy", "balanced_accuracy", "macro_f1",
                       "weighted_f1", "mcc"]
        section_agg = (
            per_slide_df
            .groupby(["label", "embedding", "split"], sort=False)[agg_metrics]
            .agg(["mean", "std", "count"])
        )
        section_agg.columns = [f"{m}_{stat}" for m, stat in section_agg.columns]
        section_agg = section_agg.reset_index().rename(
            columns={"accuracy_count": "n_sections"}
        )
        # Drop duplicate count cols from the other two metrics.
        section_agg = section_agg.drop(
            columns=[c for c in section_agg.columns
                     if c.endswith("_count") and c != "n_sections"]
        )
        section_path = out_dir / "probe_metrics_section_summary.csv"
        section_agg.to_csv(section_path, index=False)
        print(f"[section summary] wrote {section_path}")
        # Test-split summary to stdout for quick eyeballing.
        test_agg = section_agg[section_agg.split == "test"].copy()
        for m in agg_metrics:
            test_agg[f"{m} mean±std"] = (
                test_agg[f"{m}_mean"].map("{:.3f}".format)
                + " ± "
                + test_agg[f"{m}_std"].map("{:.3f}".format)
            )
        cols = ["label", "embedding", "n_sections"] + [
            f"{m} mean±std" for m in agg_metrics
        ]
        print("\n[section summary | test]")
        print(test_agg[cols].to_string(index=False))

    per_class_path = out_dir / "probe_metrics_per_class.csv"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    print(f"[per-class metrics] wrote {per_class_path}")

    np.savez_compressed(out_dir / "predictions.npz", **pred_arrays)
    print(f"[predictions] wrote {out_dir / 'predictions.npz'}")

    target = next((s for s in test_slides if s["sample"] == best_sample), None)
    if target is None:
        print("[viz] no viz target — skipping overlays")
        return

    target_row = manifest[manifest["sample_name"] == best_sample].iloc[0]
    he_zarr = Path(target_row["hematoxylin_eosin_image"])
    preproc_h5 = Path(target_row["spatial_preprocessed"])
    print(f"\n[viz] reading H&E + centroids for {best_sample}")
    thumb, step = _read_thumbnail(he_zarr)
    centroids = _load_centroids(preproc_h5)
    cents_xy = np.stack([
        centroids.loc[c, ["x_pixel", "y_pixel"]].to_numpy()
        if c in centroids.index else (np.nan, np.nan)
        for c in target["cell_id"]
    ])
    have_centroid = ~np.isnan(cents_xy[:, 0])

    target_offset = next(o for o in test_offsets if o[0] == best_sample)
    _, ts, te = target_offset

    for label_field in LABEL_FIELDS:
        y_full_test = y_test_all[label_field]
        gt_target = y_full_test[ts:te]
        preds_target = [test_preds_full[label_field][e][ts:te] for e in embeddings_present]
        titles = [EMB_TITLE[e] for e in embeddings_present]

        keep = have_centroid & (gt_target != "")
        out_png = fig_dir / f"{label_field}_overlay.png"
        _plot_overlay(
            thumb, step,
            cents_xy[keep],
            gt_target[keep],
            [p[keep] for p in preds_target],
            titles,
            out_png,
            sample_name=best_sample,
            label_field=label_field,
        )

    print("\n[done]")


if __name__ == "__main__":
    main()
