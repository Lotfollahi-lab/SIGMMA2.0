"""Plot LoRA weight delta magnitudes from diff_uni_weights CSV.

Per (scale, block, qkv/proj), shows rel = ||delta_W|| / ||W_base|| across
ViT depth. Three lines per panel = micro/meso/macro adapters.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCALE_LABEL = {"scale_0": "micro", "scale_1": "meso", "scale_2": "macro"}
SCALE_COLOR = {"scale_0": "#d62728", "scale_1": "#2ca02c", "scale_2": "#1f77b4"}


def _block_idx(param: str) -> int | None:
    m = re.match(r"blocks\.(\d+)\.attn\.(qkv|proj)$", param)
    return int(m.group(1)) if m else None


def _ptype(param: str) -> str | None:
    if param == "patch_embed.proj":
        return "patch_embed"
    m = re.match(r"blocks\.\d+\.attn\.(qkv|proj)$", param)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output path stem (no ext)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["block"] = df["param"].map(_block_idx)
    df["ptype"] = df["param"].map(_ptype)
    df["rel_pct"] = df["rel"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, ptype, title in zip(axes, ["qkv", "proj"], ["attn.qkv", "attn.proj"]):
        sub = df[df["ptype"] == ptype].sort_values(["adapter", "block"])
        for adapter in ["scale_0", "scale_1", "scale_2"]:
            s = sub[sub["adapter"] == adapter]
            ax.plot(
                s["block"], s["rel_pct"],
                marker="o", markersize=3.5, linewidth=1.4,
                color=SCALE_COLOR[adapter], label=SCALE_LABEL[adapter],
            )
        ax.set_title(title)
        ax.set_xlabel("ViT block index")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xticks(range(0, 24, 4))
    axes[0].set_ylabel(r"$\|\Delta W\| / \|W_{\mathrm{base}}\|$  (%)")
    axes[0].legend(title="LoRA scale", frameon=False, loc="upper right", fontsize=9)

    # Annotate patch_embed values in a side note
    pe = df[df["ptype"] == "patch_embed"].set_index("adapter")["rel_pct"]
    pe_txt = "patch_embed.proj: " + ", ".join(
        f"{SCALE_LABEL[a]}={pe[a]:.2f}%" for a in ["scale_0", "scale_1", "scale_2"] if a in pe.index
    )
    fig.text(0.5, -0.02, pe_txt, ha="center", fontsize=8, color="0.35")

    fig.suptitle("LoRA weight delta magnitude across ViT depth", y=1.02, fontsize=11)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), dpi=200, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
