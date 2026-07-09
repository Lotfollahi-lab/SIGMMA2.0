"""Generalize scripts/plot_gex_cossim_gene_panels.py to EVERY merged SVG panel
in the panel-aware cohort (svg_panels_2026_05_29), not just the 5K panel.

For each panel with enough sections to form cross-section pairs, reconstruct the
per-FOV raw-GEX vectors (sum of in-FOV cell counts; dataset.py:537) and plot the
GEXAwareSoftContrastiveLoss cosine similarity under ALL genes / HVG-50 / that
panel's own SVG, split within-section vs cross-section. Also writes a summary
bar chart of within-cross separation across panels.

Output:
  figures/diagnostics/gex_cossim_panels/<panel_id>.{png,pdf}
  figures/diagnostics/gex_cossim_panels/_separation_summary.{png,pdf}
  figures/diagnostics/gex_cossim_panels/_stats.json
Run: .venv/bin/python scripts/plot_gex_cossim_all_panels.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# make the repo root importable so `scripts.` resolves when run as a file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# reuse the validated single-panel helpers
from scripts.plot_gex_cossim_gene_panels import (
    preprocess_gex, cosine_offdiag, collect_fov_gex,
)

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
SVD = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels_2026_05_29")
MANIFEST = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST.csv")
METACSV = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sample_metadata.csv")
HVG_NPY = ROOT / "data/in_house_5k_disp_hvg_50.npy"
OUT = ROOT / "figures/diagnostics/gex_cossim_panels"

MIN_SECTIONS = 4          # need >=2 sections for cross pairs; >=4 for a real distribution
MAX_SECTIONS = 10         # cap sampled sections per panel (cost)
SEED = 0


def panel_svg(pid: str, k_max: int = 100) -> list[str]:
    npys = sorted((SVD / pid).glob("svg_top*.npy"))
    chosen = None
    for p in reversed(npys):
        arr = np.load(p, allow_pickle=True)
        if len(arr) <= k_max:
            chosen = arr
            break
    if chosen is None:
        chosen = np.load(npys[-1], allow_pickle=True)[:k_max]
    return [str(g) for g in chosen]


def build_reps(raw_blocks, hvg, svg):
    """common-axis ALL matrix + HVG/SVG column subsets + section labels."""
    common = None
    for _, _, vidx in raw_blocks:
        common = set(vidx) if common is None else (common & set(vidx))
    common = sorted(common)
    cpos = {g: j for j, g in enumerate(common)}
    allmat, labels = [], []
    for name, V, vidx in raw_blocks:
        pos = {g: j for j, g in enumerate(vidx)}
        cols = np.array([pos[g] for g in common])
        allmat.append(V[:, cols])
        labels += [name] * V.shape[0]
    A = np.vstack(allmat)
    labels = np.asarray(labels)
    hvg_cols = [cpos[g] for g in hvg if g in cpos]
    svg_cols = [cpos[g] for g in svg if g in cpos]
    reps = {
        f"ALL genes (n={len(common)})": A,
        f"HVG-50 (n={len(hvg_cols)})": A[:, hvg_cols] if hvg_cols else None,
        f"SVG (n={len(svg_cols)})": A[:, svg_cols] if svg_cols else None,
    }
    return reps, labels, len(common), len(hvg_cols), len(svg_cols)


def plot_panel(pid, reps, labels, out_prefix, sec2tissue, tissue_colors):
    same = labels[:, None] == labels[None, :]
    F = len(labels)
    iu = np.triu_indices(F, k=1)
    same_u = same[iu]
    colors = {"within": "#d62728", "cross": "#4477AA"}
    # per-FOV tissue color (FOVs are already ordered tissue-then-section)
    fov_tissue = np.array([sec2tissue.get(s, "??") for s in labels])
    fov_rgb = np.array([tissue_colors[t] for t in fov_tissue])  # (F, 3)
    # per-FOV section color (cycling palette in order of appearance)
    uniq_sec = list(dict.fromkeys(labels.tolist()))
    cmap20 = plt.get_cmap("tab20")
    sec_color = {s: cmap20(i % 20)[:3] for i, s in enumerate(uniq_sec)}
    sec_rgb = np.array([sec_color[s] for s in labels])  # (F, 3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    stat = {}
    for c, (title, mat) in enumerate(reps.items()):
        ax, ax2 = axes[0, c], axes[1, c]
        if mat is None or mat.shape[1] == 0:
            ax.set_title(f"{title}\n(no genes on axis)", fontsize=9)
            ax.axis("off"); ax2.axis("off")
            continue
        S = cosine_offdiag(preprocess_gex(mat))
        im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        # annotation strips on both axes: tissue (inner) + section (outer)
        div = make_axes_locatable(ax)
        # tissue strips — closest to the heatmap
        tax = div.append_axes("top", size="4%", pad=0.03, sharex=ax)
        tax.imshow(fov_rgb[None, :, :], aspect="auto", extent=[-0.5, F - 0.5, 0, 1])
        tax.set_xticks([]); tax.set_yticks([])
        lax = div.append_axes("left", size="4%", pad=0.03, sharey=ax)
        lax.imshow(fov_rgb[:, None, :], aspect="auto", extent=[0, 1, F - 0.5, -0.5])
        lax.set_xticks([]); lax.set_yticks([])
        # section strips — outer
        sax = div.append_axes("top", size="4%", pad=0.03, sharex=ax)
        sax.imshow(sec_rgb[None, :, :], aspect="auto", extent=[-0.5, F - 0.5, 0, 1])
        sax.set_xticks([]); sax.set_yticks([])
        sax.set_title(title, fontsize=10)
        slax = div.append_axes("left", size="4%", pad=0.03, sharey=ax)
        slax.imshow(sec_rgb[:, None, :], aspect="auto", extent=[0, 1, F - 0.5, -0.5])
        slax.set_xticks([]); slax.set_yticks([])
        if c == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine sim")
        sv = S[iu]; win = sv[same_u]; crs = sv[~same_u]
        bins = np.linspace(-1, 1, 80)
        if crs.size:
            ax2.hist(crs, bins=bins, density=True, alpha=0.55, color=colors["cross"],
                     label=f"cross (med {np.median(crs):.2f})")
            ax2.axvline(np.median(crs), color=colors["cross"], ls="--", lw=1)
        ax2.hist(win, bins=bins, density=True, alpha=0.55, color=colors["within"],
                 label=f"within (med {np.median(win):.2f})")
        ax2.axvline(np.median(win), color=colors["within"], ls="--", lw=1)
        ax2.set_xlabel("FOV–FOV cosine similarity")
        if c == 0:
            ax2.set_ylabel("density")
        sep = (np.median(win) - np.median(crs)) if crs.size else float("nan")
        ax2.set_title(f"separation (within−cross) = {sep:+.3f}", fontsize=9)
        ax2.legend(fontsize=7, loc="upper left")
        stat[title.split(" (")[0]] = {
            "within_med": float(np.median(win)),
            "cross_med": float(np.median(crs)) if crs.size else None,
            "separation_med": float(sep) if crs.size else None,
        }
    # tissue legend (inner strip) + section legend (outer strip) along the bottom
    present, cnts = np.unique(fov_tissue, return_counts=True)
    t_handles = [plt.Rectangle((0, 0), 1, 1, color=tissue_colors[t]) for t in present]
    t_txt = [f"{t} (n={n})" for t, n in zip(present, cnts)]
    leg1 = fig.legend(t_handles, t_txt, loc="lower center", ncol=min(len(present), 6),
                      fontsize=8, frameon=False,
                      title="FOV tissue (inner strip)", bbox_to_anchor=(0.5, -0.01))
    fig.add_artist(leg1)
    s_cnt = {s: int((labels == s).sum()) for s in uniq_sec}
    s_handles = [plt.Rectangle((0, 0), 1, 1, color=sec_color[s]) for s in uniq_sec]
    s_txt = [f"{s} (n={s_cnt[s]})" for s in uniq_sec]
    fig.legend(s_handles, s_txt, loc="lower center", ncol=min(len(uniq_sec), 5),
               fontsize=6, frameon=False,
               title="FOV section (outer strip)", bbox_to_anchor=(0.5, -0.11))
    fig.suptitle(f"GEX cosine similarity — panel {pid}\n"
                 f"{len(set(labels))} sections, {F} FOVs; log1p + per-FOV z-score",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stat


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hvg = [str(g) for g in np.load(HVG_NPY, allow_pickle=True)]
    pi = json.loads((SVD / "panel_index.json").read_text())
    counts = Counter(pi.values())
    man = pd.read_csv(MANIFEST)
    name2paths = {str(r["sample_name"]): (r["spatial_preprocessed"], r["index"])
                  for _, r in man.iterrows()}

    # authoritative section -> tissue map + a stable color per tissue (tab20)
    meta = pd.read_csv(METACSV)
    sec2tissue = {str(k): str(v) for k, v in zip(meta["sample_id"], meta["tissue_type"])}
    all_tis = sorted({sec2tissue.get(s, "??") for s in pi if s in name2paths})
    cmap = plt.get_cmap("tab20")
    tissue_colors = {t: cmap(i % 20)[:3] for i, t in enumerate(all_tis)}
    tissue_colors["??"] = (0.6, 0.6, 0.6)

    # merged panels with >= MIN_SECTIONS, plus singletons that still have
    # >= 2 sections (need >=2 to form any cross-section pair; the 1-section
    # singletons have no cross distribution so are not plottable here).
    panels = [pid for pid, n in counts.most_common()
              if (n >= MIN_SECTIONS and not pid.endswith("singleton"))
              or (pid.endswith("singleton") and n >= 2)]
    print(f"[plan] {len(panels)} panels (incl. multi-section singletons): {panels}", flush=True)

    all_stats = {}
    for pid in panels:
        secs = [s for s, p in pi.items() if p == pid and s in name2paths]
        if len(secs) > MAX_SECTIONS:
            secs = [secs[i] for i in rng.choice(len(secs), MAX_SECTIONS, replace=False)]
        svg = panel_svg(pid)
        raw_blocks = []
        for s in secs:
            sf, idx = name2paths[s]
            try:
                V, vidx = collect_fov_gex(sf, idx, rng)
            except Exception as e:
                print(f"  [{pid}] {s}: ERR {type(e).__name__}: {e}", flush=True)
                continue
            if V.shape[0]:
                raw_blocks.append((s, V, vidx))
        if len(raw_blocks) < 2:
            print(f"  [{pid}] only {len(raw_blocks)} sections with FOVs — skipped", flush=True)
            continue
        # order FOV blocks by tissue (then section) so the strip shows clean blocks
        raw_blocks.sort(key=lambda b: (sec2tissue.get(b[0], "zz"), b[0]))
        reps, labels, n_all, n_hvg, n_svg = build_reps(raw_blocks, hvg, svg)
        print(f"  [{pid}] {len(raw_blocks)} sec, {len(labels)} FOVs | "
              f"ALL={n_all} HVG={n_hvg} SVG={n_svg}", flush=True)
        all_stats[pid] = {
            "n_sections": len(raw_blocks), "n_fovs": int(len(labels)),
            "n_all": n_all, "n_hvg_on_axis": n_hvg, "n_svg_on_axis": n_svg,
            "reps": plot_panel(pid, reps, labels, str(OUT / pid),
                               sec2tissue, tissue_colors),
        }

    # summary bar chart of separation across panels
    if all_stats:
        keys = ["ALL genes", "HVG-50", "SVG"]
        cols = {"ALL genes": "#888888", "HVG-50": "#DDCC77", "SVG": "#d62728"}
        pids = list(all_stats.keys())
        x = np.arange(len(pids)); w = 0.26
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(pids)), 4.5))
        for j, k in enumerate(keys):
            vals = [(all_stats[p]["reps"].get(k) or {}).get("separation_med") for p in pids]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(x + (j - 1) * w, vals, w, label=k, color=cols[k])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p.replace('panel_','')}\n(n={all_stats[p]['n_sections']})"
                            for p in pids], fontsize=7, rotation=0)
        ax.set_ylabel("within−cross median separation")
        ax.set_title("GEX cosine separation by gene set, per panel")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"_separation_summary.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    (OUT / "_stats.json").write_text(json.dumps(all_stats, indent=2))
    print("\n[done] wrote", OUT)
    print(json.dumps({p: s["reps"] for p, s in all_stats.items()}, indent=2))


if __name__ == "__main__":
    main()
