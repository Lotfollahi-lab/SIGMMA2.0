"""Phase A0 audit: per-tissue annotation availability for cross-tissue retrieval.

For each in-house Sanger tissue, report:
  - presence of labels.h5ad (curated niche annotation)
  - presence of path_to_cell_type_anotations/ (Tangram per-sample h5ad files)
  - cell-type-like obs columns in test-split spatial_data_processed.h5
  - test split slide count
  - which case-study role (source/target) the tissue can serve

Output: outputs/cross_tissue_retrieval/annotation_audit.md
"""
from __future__ import annotations
from pathlib import Path
import re
import sys

import h5py
import pandas as pd

DATA_ROOT = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium")
OUT_PATH = Path("/nfs/team361/dj16/projects/SIGMMA/outputs/cross_tissue_retrieval/annotation_audit.md")

TISSUES = [
    ("skin", "sanger_beacon"),
    ("glioblastoma", "sanger_glioblastoma"),
    ("liver", "sanger_liver"),
    ("lung", "sanger_lung"),
    ("pancreas", "sanger_pancreas"),
    ("pediatric_ovary", "sanger_pediatric_ovary"),
    ("renal_cell_carcinoma", "sanger_renal_cell_carcinoma"),
    ("whole_embryo", "sanger_whole_embryo"),
]

CELL_TYPE_PATTERNS = [
    re.compile(r"cell.?type", re.I),
    re.compile(r"annotation", re.I),
    re.compile(r"label", re.I),
    re.compile(r"niche", re.I),
    re.compile(r"cluster", re.I),
]


def find_manifest(prefix: str, split: str) -> Path | None:
    candidate = DATA_ROOT / f"{prefix}_MANIFEST_{split}.csv"
    return candidate if candidate.exists() else None


def manifest_slide_count(manifest_path: Path | None) -> int:
    if manifest_path is None:
        return 0
    return max(0, sum(1 for _ in open(manifest_path)) - 1)


def first_test_slide(prefix: str) -> dict | None:
    test_path = find_manifest(prefix, "test")
    if test_path is None:
        return None
    df = pd.read_csv(test_path)
    if df.empty:
        return None
    row = df.iloc[0]
    return {col: row[col] for col in df.columns}


def cell_type_obs_columns(spatial_h5: str) -> list[str]:
    if not Path(spatial_h5).exists():
        return []
    with h5py.File(spatial_h5, "r") as f:
        if "obs" not in f:
            return []
        cols = list(f["obs"].keys())
    return [c for c in cols if any(p.search(c) for p in CELL_TYPE_PATTERNS)]


def labels_h5ad_path(prefix: str) -> Path | None:
    candidate = DATA_ROOT / prefix / "labels.h5ad"
    return candidate if candidate.exists() else None


def cell_type_annotations_dir(prefix: str) -> tuple[Path | None, int]:
    """Returns (dir, file_count) for Tangram per-sample annotation dir."""
    candidate = DATA_ROOT / prefix / "path_to_cell_type_anotations"
    if not candidate.is_dir():
        return None, 0
    n_files = sum(1 for p in candidate.glob("*.h5ad"))
    return candidate, n_files


def audit_tissue(tissue: str, prefix: str) -> dict:
    train_path = find_manifest(prefix, "train")
    val_path = find_manifest(prefix, "val")
    test_path = find_manifest(prefix, "test")
    n_train = manifest_slide_count(train_path)
    n_val = manifest_slide_count(val_path)
    n_test = manifest_slide_count(test_path)

    labels = labels_h5ad_path(prefix)
    ct_dir, ct_files = cell_type_annotations_dir(prefix)

    test_row = first_test_slide(prefix)
    if test_row is not None and isinstance(test_row.get("spatial_preprocessed"), str):
        ct_obs_cols = cell_type_obs_columns(test_row["spatial_preprocessed"])
        sample_name = test_row.get("sample_name", "")
    else:
        ct_obs_cols = []
        sample_name = ""

    return {
        "tissue": tissue,
        "prefix": prefix,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "labels_h5ad": str(labels) if labels else None,
        "cell_type_dir": str(ct_dir) if ct_dir else None,
        "cell_type_files": ct_files,
        "cell_type_obs_cols": ct_obs_cols,
        "test_first_sample": sample_name,
    }


def render_markdown(rows: list[dict]) -> str:
    lines = []
    lines.append("# Cross-tissue annotation availability audit")
    lines.append("")
    lines.append(f"Source: `{DATA_ROOT}`")
    lines.append("")
    lines.append("## Per-tissue summary")
    lines.append("")
    lines.append(
        "| tissue | train | val | test | labels.h5ad | cell-type dir (n files) | obs cell-type-like cols |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        labels_cell = "✓" if r["labels_h5ad"] else "—"
        ct_cell = f"✓ ({r['cell_type_files']})" if r["cell_type_dir"] else "—"
        cols_cell = ", ".join(r["cell_type_obs_cols"]) or "—"
        lines.append(
            f"| {r['tissue']} | {r['n_train']} | {r['n_val']} | {r['n_test']} "
            f"| {labels_cell} | {ct_cell} | {cols_cell} |"
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['tissue']} (`{r['prefix']}`)")
        lines.append(f"- train/val/test slides: {r['n_train']}/{r['n_val']}/{r['n_test']}")
        lines.append(f"- labels.h5ad: `{r['labels_h5ad'] or '—'}`")
        lines.append(
            f"- cell-type annotation dir: `{r['cell_type_dir'] or '—'}`"
            f" ({r['cell_type_files']} files)"
        )
        cols = ", ".join(r["cell_type_obs_cols"]) or "—"
        lines.append(f"- test first sample: `{r['test_first_sample']}`")
        lines.append(f"- obs cell-type-like columns in that sample: {cols}")
        lines.append("")
    lines.append("## Case-study readiness")
    lines.append("")
    lines.append(
        "Curated niche labels (`labels.h5ad`) are required for clean source-niche "
        "definitions. Tissues without `labels.h5ad` need a proxy (cell-type composition "
        "cluster, DEG-based) to define a source niche."
    )
    lines.append("")
    lines.append("| tissue | source-ready? | reason |")
    lines.append("|---|---|---|")
    for r in rows:
        if r["labels_h5ad"]:
            ready = "✓ (curated)"
            reason = "labels.h5ad present"
        elif r["cell_type_dir"]:
            ready = "△ (proxy)"
            reason = f"per-sample cell-type h5ad ({r['cell_type_files']} files)"
        elif r["cell_type_obs_cols"]:
            ready = "△ (proxy)"
            reason = "obs columns: " + ", ".join(r["cell_type_obs_cols"])
        else:
            ready = "✗"
            reason = "no curated labels and no cell-type annotation found"
        lines.append(f"| {r['tissue']} | {ready} | {reason} |")
    lines.append("")
    lines.append("All tissues with non-empty test split are valid **target** tissues; "
                 "embedding extraction (Phase A1) covers all 8.")
    return "\n".join(lines)


def main() -> None:
    rows = [audit_tissue(t, prefix) for t, prefix in TISSUES]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render_markdown(rows))
    print(f"wrote {OUT_PATH}")
    for r in rows:
        print(
            f"  {r['tissue']:24s}  test={r['n_test']:2d}  "
            f"labels={'yes' if r['labels_h5ad'] else 'no '}  "
            f"ct_dir={'yes' if r['cell_type_dir'] else 'no '}  "
            f"obs_cols={r['cell_type_obs_cols']}"
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
