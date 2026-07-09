"""Reduce per-section anchor-cell stats (emt_2d_score.py --dump_anchor_stats) into
cohort-pooled E-anchor (normal nephron-epithelium) and M-anchor (fibroblast) centroids.

Per-section centroids are unreliable where a compartment is scarce: a handful of
normal-epithelium cells gives a noise-dominated centroid that lands in a random
quadrant (see CV6-FT-4-S3 / CV9-FT-3-S3, nephron-epi <=10 cells). Pooling every
labeled RCC section's anchor cells gives one stable reference per pole. We pool the
SUMS, so the result is the exact cell-weighted centroid over all sections combined:

    E_anchor = sum_sections(sum_E over nephron-epi cells) / sum_sections(n nephron-epi)

The control-gene baseline makes the per-section scores comparable across slides
(0 = above matched-expression background, not above the slide mean), so a pooled
centroid in (E, M) score space is meaningful.
"""
import argparse, json, glob
from pathlib import Path


def pool(stat_list, key):
    sE = sum(s[key]["sum_E"] for s in stat_list)
    sM = sum(s[key]["sum_M"] for s in stat_list)
    n = sum(s[key]["n"] for s in stat_list)
    k = sum(1 for s in stat_list if s[key]["n"] > 0)
    if n == 0:
        return {"E": float("nan"), "M": float("nan"), "n_cells": 0, "n_sections": 0}
    return {"E": sE / n, "M": sM / n, "n_cells": int(n), "n_sections": int(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True,
                    help="directory of <section>_<gene_set>.json files from --dump_anchor_stats")
    ap.add_argument("--gene_set", default="union")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <dump_dir>/anchors_<gene_set>.json)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dump_dir) / f"*_{args.gene_set}.json")))
    if not files:
        raise SystemExit(f"no dump files matching *_{args.gene_set}.json in {args.dump_dir}")
    stats = [json.loads(Path(f).read_text()) for f in files]
    print(f"[reduce] {len(stats)} sections, gene_set={args.gene_set}")

    nep = pool(stats, "nephron_epi")
    fib = pool(stats, "fibroblast")

    # per-section breakdown for transparency (centroid + count per pole)
    print(f"{'section':22} {'neph-epi (E,M  n)':>26} {'fibro (E,M  n)':>26}")
    for s in sorted(stats, key=lambda s: s["section"]):
        ne, fb = s["nephron_epi"], s["fibroblast"]
        nec = (ne["sum_E"] / ne["n"], ne["sum_M"] / ne["n"]) if ne["n"] else (float("nan"),) * 2
        fbc = (fb["sum_E"] / fb["n"], fb["sum_M"] / fb["n"]) if fb["n"] else (float("nan"),) * 2
        print(f"{s['section']:22} ({nec[0]:+.3f},{nec[1]:+.3f}  {ne['n']:>7,}) "
              f"({fbc[0]:+.3f},{fbc[1]:+.3f}  {fb['n']:>7,})")

    out = {"gene_set": args.gene_set, "nephron_epi": nep, "fibroblast": fib,
           "n_sections_used": len(stats)}
    print("\n[pooled anchors]")
    print(f"  E-anchor (nephron-epi): E={nep['E']:+.4f}  M={nep['M']:+.4f}  "
          f"(n_cells={nep['n_cells']:,}, sections={nep['n_sections']})")
    print(f"  M-anchor (fibroblast):  E={fib['E']:+.4f}  M={fib['M']:+.4f}  "
          f"(n_cells={fib['n_cells']:,}, sections={fib['n_sections']})")

    outp = Path(args.out) if args.out else Path(args.dump_dir) / f"anchors_{args.gene_set}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {outp}")


if __name__ == "__main__":
    main()
