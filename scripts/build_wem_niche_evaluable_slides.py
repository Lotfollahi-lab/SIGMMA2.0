"""Build the whole_embryo evaluable-slides manifest for the niche label-transfer task.

Mirrors the skin manifest `configs/cell_niche_evaluable_slides.csv` schema so the same
extractor / probe consume it. Slides come from the per-split WEM manifests; the `norm`
column is the section suffix (e.g. `S203`) used to match `labels.h5ad` `obs/sample`, and
`n_cells_labelled` counts cells with polygon ∧ centroid ∧ a non-null `niche40` label.
Zero-coverage slides are dropped.

Run once:  python scripts/build_wem_niche_evaluable_slides.py
Writes:    configs/cell_niche_evaluable_slides_whole_embryo.csv
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium")
LABELS = ROOT / "sanger_whole_embryo/labels.h5ad"
NICHE_COL = "niche40"
OUT = Path("/nfs/team361/dj16/projects/SIGMMA/configs/"
           "cell_niche_evaluable_slides_whole_embryo.csv")


def suffix(sample_name: str) -> str:
    """SOB26-WEM-FFPE-1-S203 -> S203 (matches labels.h5ad obs/sample)."""
    return sample_name.split("-")[-1].upper()


def resolve_raw_dir(spatial_preprocessed: str) -> Path:
    # .../sanger_whole_embryo/<run>/<sample>/spatial_data_processed.h5
    p = Path(spatial_preprocessed)
    run, sample = p.parent.parent.name, p.parent.name
    return ROOT / "raw_data/sanger_whole_embryo" / run / sample


def main() -> None:
    # Read label categoricals once (sample, cell_id, niche40).
    with h5py.File(LABELS, "r") as f:
        samp_cats = [c.decode() if isinstance(c, bytes) else c
                     for c in f["obs/sample/categories"][:]]
        samp_codes = f["obs/sample/codes"][:]
        cid_cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in f["obs/cell_id/categories"][:]])
        cid_codes = f["obs/cell_id/codes"][:]
        niche_codes = f[f"obs/{NICHE_COL}/codes"][:]
    samp_cats_u = [s.upper() for s in samp_cats]

    def labelled_ids(suf: str) -> set:
        if suf not in samp_cats_u:
            return set()
        mask = samp_codes == samp_cats_u.index(suf)
        ids = cid_cats[cid_codes[mask]]
        keep = niche_codes[mask] >= 0
        return set(ids[keep])

    def polygon_ids(raw_dir: Path) -> set:
        cands = sorted(Path(raw_dir).glob("output-XETG*"))
        if not cands:
            return set()
        bnd = cands[0] / "cell_boundaries.parquet"
        if not bnd.exists():
            return set()
        return set(pd.read_parquet(bnd, columns=["cell_id"])["cell_id"].unique())

    def centroid_ids(spatial: str) -> set:
        with h5py.File(spatial, "r") as f:
            return set(c.decode() if isinstance(c, bytes) else c
                       for c in f["obs/cell_id"][:])

    rows = []
    for split in ("train", "val", "test"):
        man = pd.read_csv(ROOT / f"sanger_whole_embryo_MANIFEST_{split}.csv")
        for _, r in man.iterrows():
            sn = r["sample_name"]
            raw_dir = resolve_raw_dir(r["spatial_preprocessed"])
            inter = (labelled_ids(suffix(sn))
                     & polygon_ids(raw_dir)
                     & centroid_ids(r["spatial_preprocessed"]))
            n = len(inter)
            print(f"{sn:28s} {split:5s} n_cells_labelled={n}")
            if n == 0:
                print(f"  [drop] {sn}: zero coverage")
                continue
            rows.append({
                "sample_name": sn,
                "hematoxylin_eosin_image": r["hematoxylin_eosin_image"],
                "index": r["index"],
                "spatial_preprocessed": r["spatial_preprocessed"],
                "annotation": "",
                "split": split,
                "norm": suffix(sn),
                "raw_dir": str(raw_dir),
                "n_cells_labelled": n,
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    counts = df["split"].value_counts().to_dict()
    print(f"\n[wrote] {OUT} ({len(df)} slides; splits={counts})")


if __name__ == "__main__":
    main()
