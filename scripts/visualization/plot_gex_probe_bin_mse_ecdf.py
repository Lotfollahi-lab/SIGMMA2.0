import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS = Path("outputs/gex_2d/disp_g14/results_linear")
DEFAULT_OUTPUT = Path("outputs/gex_2d/disp_g14/figures/bin_mse_ecdf.png")

REPRS = ["uni", "micro", "meso", "macro", "concat"]

# Paul Tol muted (subset) — keep consistent with the manifest plot.
REPR_PALETTE = {
    "uni":    "#332288",   # indigo
    "micro":  "#88CCEE",   # light blue
    "meso":   "#44AA99",   # teal
    "macro":  "#DDCC77",   # sand
    "concat": "#CC6677",   # rose
}

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
}


def load_mse(results_dir: Path) -> dict[str, np.ndarray]:
    data = {}
    for repr in REPRS:
        path = results_dir / f"{repr}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        j = json.loads(path.read_text())
        if "mse_per_bin" not in j:
            raise KeyError(f"{path} has no 'mse_per_bin' — re-run eval after summarize() update")
        data[repr] = np.asarray(j["mse_per_bin"], dtype=np.float64)
    return data


def plot_ecdf(data: dict[str, np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 2.8))
        for repr in REPRS:
            arr = data[repr]
            xs = np.sort(arr)
            ys = np.arange(1, xs.size + 1) / xs.size
            median = float(np.median(arr))
            ax.plot(
                xs, ys,
                color=REPR_PALETTE[repr],
                linewidth=1.4,
                label=f"{repr} (median={median:.3g})",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Bin MSE (mean over G=50 genes)")
        ax.set_ylabel("Cumulative fraction of bins")
        ax.set_ylim(0, 1.0)
        ax.grid(False)
        ax.legend(loc="lower right", frameon=False)

        fig.tight_layout()
        fig.savefig(output)
        fig.savefig(output.with_suffix(".pdf"))
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    data = load_mse(args.results_dir)
    plot_ecdf(data, args.output)

    print(f"{'repr':<8} {'n_bins':>8} {'median':>12} {'p75':>12} {'p95':>12}")
    for repr in REPRS:
        a = data[repr]
        print(f"{repr:<8} {a.size:>8d} {np.median(a):>12.4g} "
              f"{np.percentile(a, 75):>12.4g} {np.percentile(a, 95):>12.4g}")
    print(f"\nSaved figure: {args.output}")
    print(f"Saved vector: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
