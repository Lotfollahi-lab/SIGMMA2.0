"""Format nemo-harmonize within-section retrieval results into a table.

Re-derives public vs in-house from the section ID (TENX*/NCBI* = HEST-1k public)
because the job's is_public flag was set from the dir-name tissue_type and is
unreliable. Prints per-section MRR/R@1%/R@1 per variant + public tile-weighted
pooled, and the verdict on whether harmonization lifts public above chance.
"""
import json
import numpy as np

JSON = "outputs/retrieval/nemo_harmonize_within_section.json"
VARIANTS = ["H0", "H1", "H2"]
VLABEL = {"H0": "H0 raw", "H1": "H1 mean-shift", "H2": "H2 z-match"}


def is_public(sid):
    return sid.startswith("TENX") or sid.startswith("NCBI")


def main():
    d = json.load(open(JSON))
    have = [v for v in VARIANTS if v in d]
    secs = sorted(d[have[0]].keys(), key=lambda s: (not is_public(s), s))

    # per-section MRR across variants
    print("=== per-section MRR (within-section)  [pub=HEST-1k public] ===")
    hdr = f"{'section':14}{'grp':>5}{'n':>7}{'chance':>9}" + "".join(f"{VLABEL[v]:>15}" for v in have)
    print(hdr)
    for s in secs:
        grp = "pub" if is_public(s) else "in"
        n = d[have[0]][s]["n"]
        ch = d[have[0]][s]["chance_mrr"]
        row = f"{s[:14]:14}{grp:>5}{n:>7}{ch:>9.4f}"
        for v in have:
            row += f"{d[v][s]['mrr']:>15.4f}"
        print(row)

    # tile-weighted pooled per group per metric
    print("\n=== tile-weighted pooled (public vs in-house) ===")
    for metric in ["mrr", "recall_at_1pct", "recall_at_1"]:
        print(f"\n-- {metric} --")
        print(f"{'group':10}" + "".join(f"{VLABEL[v]:>15}" for v in have) + f"{'chance':>10}")
        for grp, want in [("public", True), ("in-house", False)]:
            sl = [s for s in secs if is_public(s) == want]
            if not sl:
                continue
            w = np.array([d[have[0]][s]["n"] for s in sl], float)
            line = f"{grp:10}"
            for v in have:
                x = np.array([d[v][s][metric] for s in sl], float)
                line += f"{(x*w).sum()/w.sum():>15.4f}"
            if metric == "mrr":
                ch = np.array([d[have[0]][s]["chance_mrr"] for s in sl], float)
                line += f"{(ch*w).sum()/w.sum():>10.4f}"
            print(line)

    # verdict
    if set(have) >= {"H0", "H1"}:
        pub = [s for s in secs if is_public(s)]
        w = np.array([d["H0"][s]["n"] for s in pub], float)
        def pooled(v):
            x = np.array([d[v][s]["mrr"] for s in pub], float)
            return (x*w).sum()/w.sum()
        ch = np.array([d["H0"][s]["chance_mrr"] for s in pub], float)
        chp = (ch*w).sum()/w.sum()
        print(f"\n=== VERDICT (public pooled MRR) ===")
        for v in have:
            print(f"  {VLABEL[v]:16} {pooled(v):.4f}  ({pooled(v)/chp:.1f}x chance)")
        print(f"  chance           {chp:.4f}")
        best = max(have, key=pooled)
        if pooled(best) > 3*chp:
            print(f"  -> harmonization RECOVERS signal ({best} = {pooled(best)/chp:.1f}x chance): "
                  "OOD failure is a correctable nemo batch shift.")
        else:
            print("  -> harmonization does NOT recover (all near chance): "
                  "offset removal insufficient; deeper gap (shape / learned correspondence).")


if __name__ == "__main__":
    main()
