#!/usr/bin/env python
"""Build a TCGA manifest CSV (sample_name, svs, hematoxylin_eosin_image,
data_h5) from converted *.he.zarr under a slides root. sample_name =
.svs/.he.zarr stem, which joins to the GDC survival table's file_name
("<stem>.svs"). The svs column is the sibling .svs path (may no longer exist if
deleted post-conversion); the seg jobscript only needs it when the zarr is
missing, so a stale path is fine once the zarr is present."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_root", required=True)
    ap.add_argument("--data_dir", required=True, help="where seg data_h5s will live")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    root = Path(args.slides_root)
    rows = []
    for zj in sorted(root.glob("*/*.he.zarr/zarr.json")):
        zarr_path = zj.parent
        sample = zarr_path.name[:-len(".he.zarr")]
        rows.append({
            "sample_name": sample,
            "svs": str(zarr_path.parent / f"{sample}.svs"),
            "hematoxylin_eosin_image": str(zarr_path),
            "data_h5": str(Path(args.data_dir) / f"{sample}.h5"),
        })
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["sample_name", "svs",
                                          "hematoxylin_eosin_image", "data_h5"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} converted slides -> {out}")


if __name__ == "__main__":
    main()
