#!/usr/bin/env python
"""Compare per-slide protocol TSR between two runs (e.g. 8000-cap vs full cells).

Joins two protocol-TSR csvs on sample_name and prints stroma_fraction (and
n_valid_fields, tumour/stroma area) side by side with the delta, plus summary
stats of |delta|. Used to test whether the --max_cells 8000 subsample biases the
central-window area TSR.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_csv", required=True, help="8000-cap TSR csv")
    ap.add_argument("--full_csv", required=True, help="full-cell TSR csv")
    ap.add_argument("--cols", nargs="+",
                    default=["stroma_fraction", "n_valid_fields",
                             "tumour_area_px", "stroma_area_px"])
    args = ap.parse_args()
    a = pd.read_csv(args.cap_csv).set_index("sample_name")
    b = pd.read_csv(args.full_csv).set_index("sample_name")
    common = [s for s in b.index if s in a.index]
    print(f"[compare] {len(common)} slides in both "
          f"(cap={len(a)} full={len(b)})\n")
    for s in common:
        print(f"=== {s.split('.')[0]} ===")
        for c in args.cols:
            va, vb = a.loc[s, c], b.loc[s, c]
            d = vb - va if pd.notna(va) and pd.notna(vb) else np.nan
            print(f"  {c:18s} cap={va!s:>10}  full={vb!s:>10}  Δ={d:+.4g}"
                  if isinstance(d, float) and not np.isnan(d)
                  else f"  {c:18s} cap={va!s:>10}  full={vb!s:>10}")
    sf = pd.DataFrame({"cap": a.loc[common, "stroma_fraction"],
                       "full": b.loc[common, "stroma_fraction"]}).dropna()
    if len(sf):
        d = (sf["full"] - sf["cap"]).abs()
        print(f"\n[stroma_fraction |Δ|] n={len(sf)} mean={d.mean():.4f} "
              f"median={d.median():.4f} max={d.max():.4f}")
        print(f"[stroma_fraction] cap mean={sf['cap'].mean():.4f} "
              f"full mean={sf['full'].mean():.4f} "
              f"corr={sf['cap'].corr(sf['full']):.4f}")


if __name__ == "__main__":
    main()
