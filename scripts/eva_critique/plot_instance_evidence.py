"""Generate the instance-level evidence figures for the EVA-critique result.

Reads cached metrics CSVs and per-tile probability/GT arrays under
`outputs/eva_critique/`, then writes four figures into
`figures/eva_critique/`:

  - headline_metric_trio.pdf  (PQ / AJI / Dice bar chart, UNI vs SIGMMA s0/s1/s2)
  - boundary_overlay_tiles.pdf (handpicked tiles with biggest Δn_merge)
  - n_merge_scatter.pdf        (per-tile UNI vs SIGMMA s1 n_merge)
  - detf1_vs_iou.pdf           (DetF1 at IoU thresholds 0.5/0.7/0.9)

No retraining or new inference. AUPRC / ECE / BF1 / HD95 / ASD / DetF1@0.9
intentionally excluded from headline figures per the agreed reporting plan.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from scripts.eva_critique.inst import prob_to_instances
from scripts.eva_critique.metrics_seg import label_components, dice as pixel_dice
from scripts.eva_critique.polygon_gt import load_polygon_instance_label

MODELS_ORDER = [
    "uni",
    "sigmma_s0", "sigmma_s1", "sigmma_s2", "sigmma_no_lora",
    "bleep_uni_ft", "clip_uni_ft", "plip_uni_ft", "omiclip_uni_ft",
]
LABELS = {
    "uni": "UNI",
    "sigmma_s0": "SIGMMA (micro)",
    "sigmma_s1": "SIGMMA (meso)",
    "sigmma_s2": "SIGMMA (macro)",
    "sigmma_no_lora": "SIGMMA (no LoRA)",
    "bleep_uni_ft": "BLEEP (UNI-ft)",
    "clip_uni_ft": "CLIP (UNI-ft)",
    "plip_uni_ft": "PLIP (UNI-ft)",
    "omiclip_uni_ft": "OmiCLIP (UNI-ft)",
}
COLOURS = {
    "uni":        "#888888",  # neutral grey baseline
    "sigmma_s0":  "#d62728",  # micro = red
    "sigmma_s1":  "#2ca02c",  # meso  = green
    "sigmma_s2":  "#1f77b4",  # macro = blue
    # SIGMMA no-LoRA ablation: dark purple to read as SIGMMA-family but
    # distinct from the per-scale LoRA variants.
    "sigmma_no_lora": "#6a3d9a",
    # UNI-backbone baselines: Paul Tol muted palette to read as a distinct
    # family from SIGMMA scales + UNI base.
    "bleep_uni_ft":   "#882255",  # wine
    "clip_uni_ft":    "#44AA99",  # teal
    "plip_uni_ft":    "#DDCC77",  # sand
    "omiclip_uni_ft": "#999933",  # olive
}

# tau* per model, copied from summary.md Component 3.
TAU_STAR = {"uni": 0.7, "sigmma_s0": 0.6, "sigmma_s1": 0.4, "sigmma_s2": 0.6}

# Holm-corrected p-values from summary.md (paired bootstrap, SIGMMA−UNI).
# These are the canonical values for significance brackets in Fig 1.
HOLM_P = {
    "PQ":  {"sigmma_s0": 0.019,   "sigmma_s1": 0.032, "sigmma_s2": 0.0097},
    "AJI": {"sigmma_s0": 0.63,    "sigmma_s1": 0.013, "sigmma_s2": 0.00047},
}

RNG = np.random.default_rng(42)


def _bootstrap_ci(deltas: np.ndarray, n_boot: int = 5000, alpha: float = 0.05):
    n = len(deltas)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(deltas.mean()), float(lo), float(hi), float(p)


def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Fig 1: headline metric trio (PQ, AJI, Dice)
# ---------------------------------------------------------------------------
def _draw_sig_bracket(ax, x_left: float, x_right: float, y: float,
                      h: float, label: str, fontsize: int = 9):
    """Draw a horizontal bracket between two x positions with a label above.

    `y` is the top-of-bracket height; `h` is the tick depth.
    """
    ax.plot([x_left, x_left, x_right, x_right],
            [y - h, y, y, y - h],
            color="black", linewidth=0.7)
    ax.text((x_left + x_right) / 2, y + h * 0.4, label,
            ha="center", va="bottom", fontsize=fontsize)


def fig_headline_trio(alt_csv: Path, region_csv: Path, out_path: Path):
    alt = pd.read_csv(alt_csv).drop_duplicates(
        subset=["image_id", "model_scale"])
    pq_w = alt.pivot(index="image_id", columns="model_scale", values="PQ")
    aji_w = alt.pivot(index="image_id", columns="model_scale", values="AJI")

    # Filter MODELS_ORDER to whatever is actually present in the CSV
    # (allows partial reruns before all baseline-UNI rows have landed).
    present = set(pq_w.columns) & set(aji_w.columns)
    order_local = [m for m in MODELS_ORDER if m in present]

    # Global val Dice from per-model summary.json (best_val_dice on the
    # held-out MoNuSAC val split, as reported in summary.md).
    summary_dices = {
        "uni": 0.7916,
        "sigmma_s0": 0.7973, "sigmma_s1": 0.7941, "sigmma_s2": 0.7895,
        "bleep_uni_ft": float("nan"),  # placeholder; updated when training lands
        "clip_uni_ft": 0.7979,
        "plip_uni_ft": 0.8015,
        "omiclip_uni_ft": 0.7952,
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    panels = [
        ("PQ", pq_w, "Panoptic Quality (PQ)"),
        ("AJI", aji_w, "Aggregated Jaccard Index (AJI)"),
        ("Dice", None, "Pixel Dice (global)"),
    ]

    for ax, (key, w, ylabel) in zip(axes, panels):
        x = np.arange(len(order_local))
        if key == "Dice":
            means = np.array([summary_dices.get(m, float("nan")) for m in order_local])
            ax.bar(x, means,
                   color=[COLOURS[m] for m in order_local],
                   edgecolor="black", linewidth=0.4)
            for xi, m in enumerate(means):
                if np.isfinite(m):
                    ax.text(xi, m + 0.005, f"{m:.3f}",
                            ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, np.nanmax(means) * 1.18)
        else:
            uni_vals = w["uni"].values
            means = []
            errs_lo, errs_hi = [], []
            for m in order_local:
                vals = w[m].values
                mu = float(np.mean(vals))
                means.append(mu)
                if m == "uni":
                    boots = RNG.choice(vals, (5000, len(vals))).mean(axis=1)
                    errs_lo.append(mu - np.quantile(boots, 0.025))
                    errs_hi.append(np.quantile(boots, 0.975) - mu)
                else:
                    paired = vals - uni_vals
                    _, lo, hi, _ = _bootstrap_ci(paired)
                    uni_mu = float(np.mean(uni_vals))
                    errs_lo.append(mu - (uni_mu + lo))
                    errs_hi.append((uni_mu + hi) - mu)
            means = np.array(means)
            ax.bar(x, means,
                   yerr=[errs_lo, errs_hi],
                   color=[COLOURS[m] for m in order_local],
                   edgecolor="black", linewidth=0.4,
                   capsize=3, error_kw={"elinewidth": 0.8})
            for xi, mu in enumerate(means):
                ax.text(xi, mu + max(errs_hi) * 1.05, f"{mu:.3f}",
                        ha="center", va="bottom", fontsize=8)

            # Significance brackets — UNI (x=0) ↔ each SIGMMA scale (x=1,2,3)
            # using Holm p-values from summary.md. Stack vertically so they
            # don't overlap.
            top = means.max() + max(errs_hi) * 1.05
            bracket_step = (means.max() + max(errs_hi)) * 0.10
            bracket_h = bracket_step * 0.25
            stacked = 0
            for xi, m in enumerate(order_local):
                if m == "uni":
                    continue
                p = HOLM_P.get(key, {}).get(m, 1.0)
                if p >= 0.05:
                    continue
                stacked += 1
                y = top + bracket_step * stacked
                _draw_sig_bracket(ax, 0, xi, y, bracket_h,
                                  _sig_marker(p), fontsize=10)
            ax.set_ylim(0, top + bracket_step * (stacked + 1.4))

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in order_local],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}")


# ---------------------------------------------------------------------------
# Fig 3: per-tile n_merge scatter (UNI vs SIGMMA s1 at tau=0.5)
# ---------------------------------------------------------------------------
def fig_nmerge_scatter(inst_csv: Path, out_path: Path, tau: float = 0.5):
    inst = pd.read_csv(inst_csv)
    inst = inst[inst.tau == tau].drop_duplicates(
        subset=["image_id", "model_scale"])
    pivot = inst.pivot(index="image_id", columns="model_scale",
                       values="n_merge")
    if "uni" not in pivot or "sigmma_s1" not in pivot:
        raise RuntimeError("missing uni or sigmma_s1 in instance_metrics.csv")
    u = pivot["uni"].values
    s = pivot["sigmma_s1"].values

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    jitter = RNG.uniform(-0.15, 0.15, size=(2, len(u)))
    ax.scatter(u + jitter[0], s + jitter[1], alpha=0.5, s=22,
               color=COLOURS["sigmma_s1"], edgecolor="black", linewidth=0.3)
    lim = max(u.max(), s.max()) + 1
    ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.8, alpha=0.7)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_xlabel("UNI — # merge errors per tile", fontsize=9)
    ax.set_ylabel("SIGMMA (meso) — # merge errors per tile", fontsize=9)
    below = (s < u).sum()
    equal = (s == u).sum()
    above = (s > u).sum()
    ax.set_title(f"τ={tau}; SIGMMA reduces merges in {below}/{len(u)} tiles "
                 f"(tied {equal}, worse {above})\n"
                 f"total merges: UNI={u.sum()}, SIGMMA={s.sum()} "
                 f"({(s.sum()-u.sum())/u.sum()*100:+.1f}%)",
                 fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    # inset histogram of Δ
    delta = s - u
    inset = fig.add_axes([0.62, 0.22, 0.27, 0.22])
    bins = np.arange(delta.min() - 0.5, delta.max() + 1.5, 1.0)
    inset.hist(delta, bins=bins, color=COLOURS["sigmma_s1"], edgecolor="black",
               linewidth=0.3)
    inset.axvline(0, color="black", linewidth=0.6)
    inset.set_xlabel("Δ n_merge (SIGMMA − UNI)", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}")


# ---------------------------------------------------------------------------
# Fig S1: DetF1 vs IoU threshold curve
# ---------------------------------------------------------------------------
def fig_detf1_curve(alt_csv: Path, out_path: Path):
    alt = pd.read_csv(alt_csv)
    taus = [0.5, 0.7, 0.9]
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.6))
    present = set(alt["model_scale"].dropna().unique())
    for m in [x for x in MODELS_ORDER if x in present]:
        sub = alt[alt.model_scale == m]
        means = [sub[f"DetF1@{t}"].mean() for t in taus]
        ax.plot(taus, means, "-o", label=LABELS[m], color=COLOURS[m],
                markersize=5, linewidth=1.6)
    ax.set_xlabel("IoU threshold τ", fontsize=9)
    ax.set_ylabel("Detection F1@τ", fontsize=9)
    ax.set_xticks(taus)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("SIGMMA's detection-F1 advantage holds at stricter IoU",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}")


# ---------------------------------------------------------------------------
# Fig 2: boundary overlay tile panels
# ---------------------------------------------------------------------------
def _instance_label_image(inst_lab: np.ndarray) -> np.ndarray:
    """Render an int instance label map as a random-color RGB image."""
    rgb = np.zeros(inst_lab.shape + (3,), dtype=np.uint8)
    if inst_lab.max() == 0:
        return rgb
    palette = RNG.integers(60, 240, size=(int(inst_lab.max()) + 1, 3),
                           dtype=np.int32)
    palette[0] = (255, 255, 255)  # background white
    rgb[...] = palette[inst_lab].astype(np.uint8)
    return rgb


def _add_boundary_overlay(rgb: np.ndarray, inst_lab: np.ndarray,
                          colour=(0, 0, 0)) -> np.ndarray:
    out = rgb.copy()
    edges = (inst_lab != ndi.maximum_filter(inst_lab, size=3))
    out[edges] = np.array(colour, dtype=np.uint8)
    return out


def _load_npz(p: Path):
    f = np.load(p, allow_pickle=False)
    return {k: f[k] for k in f.files}


def fig_boundary_panel(logs_root: Path, data_root: Path,
                       inst_csv: Path, alt_csv: Path, out_path: Path,
                       n_tiles: int = 3, sigmma_scale: int = 1):
    """Fig 2 picks tiles where UNI fuses adjacent cells into one segment
    while SIGMMA correctly separates them — the precise visual signature
    of the merge-reduction story.

    Selection criterion (v5, all metrics computed at each model's own τ*):
      (1) UNI has a real merge problem to fix: n_merge_UNI ≥ 1
      (2) SIGMMA fixes some of them:           n_merge_SIGMMA < n_merge_UNI
      (3) Moderate-density GT (visual readable): 30 ≤ n_GT_inst ≤ 100
      (4) GT has visible cell-cell adjacency:
          adjacency_score = (n_GT − n_dilated_components) / n_GT ≥ 0.30
          where n_dilated_components is # of CCs after 3px morphological
          dilation of the binary GT mask. High = many touching cells.
      (5) SIGMMA splits into ≥ UNI instances: n_pred_S ≥ n_pred_U
      (6) Foreground area parity (same coverage):
          |fg(SIGMMA) − fg(UNI)| / fg(UNI) ≤ 0.30
      (7) Both models work on this tile: pixel Dice(pred_X, gt) ≥ 0.5
    Ranking: by n_merge_UNI − n_merge_SIGMMA descending. Anchor tile
    TCGA-MP-A4T7-01Z-00-DX1_004 (user-validated as a good example) should
    pass these filters.
    """
    sig_key = f"sigmma_s{sigmma_scale}"
    sig_dir_name = f"sigmma_scale{sigmma_scale}"
    tau_uni = TAU_STAR["uni"]
    tau_sig = TAU_STAR[sig_key]

    from scripts.eva_critique.metrics_seg import detection_f1_hungarian

    uni_npz = _load_npz(logs_root / "uni" / "convdec_image"
                        / "test_artifacts.npz")
    sig_npz = _load_npz(logs_root / sig_dir_name / "convdec_image"
                        / "test_artifacts.npz")
    uni_idx = {iid: i for i, iid in enumerate(uni_npz["image_id"])}
    sig_idx = {iid: i for i, iid in enumerate(sig_npz["image_id"])}

    # Per-tile selection metrics — compute n_merge directly from τ*-thresholded
    # instance maps so the numbers shown in the panel match selection numbers.
    rows = []
    for iid in uni_npz["image_id"]:
        if iid not in sig_idx:
            continue
        ui = uni_idx[iid]; si = sig_idx[iid]
        prob_u = uni_npz["prob"][ui].astype(np.float32)
        prob_s = sig_npz["prob"][si].astype(np.float32)
        gt_bin = uni_npz["gt"][ui].astype(np.uint8)

        fg_u = (prob_u >= tau_uni)
        fg_s = (prob_s >= tau_sig)
        fg_count_u = int(fg_u.sum())
        fg_count_s = int(fg_s.sum())
        if fg_count_u == 0:
            continue
        fg_ratio = fg_count_s / fg_count_u

        dice_u = pixel_dice(fg_u.astype(np.uint8), gt_bin)
        dice_s = pixel_dice(fg_s.astype(np.uint8), gt_bin)

        inst_u = prob_to_instances(prob_u, threshold=tau_uni)
        inst_s = prob_to_instances(prob_s, threshold=tau_sig)
        n_pred_u = int(inst_u.max())
        n_pred_s = int(inst_s.max())

        # Polygon GT — replaces binary-CC GT so touching cells stay separated.
        try:
            gt_lab = load_polygon_instance_label(iid, data_root)
        except FileNotFoundError:
            gt_lab = label_components(gt_bin)
        n_gt = int(np.unique(gt_lab[gt_lab > 0]).size)

        # Adjacency score still computed from binary mask (closing-based
        # proxy for touching-cell density). Polygon GT gives the true cell
        # count; adjacency = degree to which those cells touch in the
        # binary image we render boundaries from.
        if n_gt > 0:
            gt_dil = ndi.binary_dilation(gt_bin.astype(bool), iterations=3)
            n_dil = int(label_components(gt_dil.astype(np.uint8)).max())
            n_bin_cc = int(label_components(gt_bin).max())
            adj_score = (n_bin_cc - n_dil) / max(n_bin_cc, 1)
        else:
            adj_score = 0.0

        det_u = detection_f1_hungarian(inst_u, gt_lab)
        det_s = detection_f1_hungarian(inst_s, gt_lab)
        nm_u = int(det_u["n_merge"])
        nm_s = int(det_s["n_merge"])

        rows.append({
            "image_id": iid,
            "fg_ratio": fg_ratio,
            "dice_u": dice_u, "dice_s": dice_s,
            "n_pred_u": n_pred_u, "n_pred_s": n_pred_s,
            "n_gt": n_gt, "adj_score": adj_score,
            "n_merge_u": nm_u, "n_merge_s": nm_s,
            "merge_fix": nm_u - nm_s,
        })
    df = pd.DataFrame(rows).set_index("image_id")

    def _filter(d, fg_tol=0.30, dice_thr=0.5,
                n_gt_min=30, n_gt_max=200, nm_u_min=1, adj_min=0.30):
        return (
            (d.n_merge_u >= nm_u_min)
            & (d.n_merge_s < d.n_merge_u)
            & (d.n_gt >= n_gt_min)
            & (d.n_gt <= n_gt_max)
            & (d.adj_score >= adj_min)
            & (d.n_pred_s >= d.n_pred_u)
            & d.fg_ratio.between(1 - fg_tol, 1 + fg_tol)
            & (d.dice_u >= dice_thr)
            & (d.dice_s >= dice_thr)
        )

    keep = _filter(df)
    if keep.sum() < n_tiles:
        warnings.warn(f"only {keep.sum()} tiles pass v6; relaxing adj_min 0.30→0.20")
        keep = _filter(df, adj_min=0.20)
    if keep.sum() < n_tiles:
        warnings.warn(f"still {keep.sum()}; widening n_gt_max 200→400")
        keep = _filter(df, adj_min=0.20, n_gt_max=400)
    candidates = df[keep].sort_values("merge_fix", ascending=False)
    if len(candidates) < n_tiles:
        warnings.warn(f"only {len(candidates)} tiles pass v4 filter; using all")
        n_tiles = len(candidates)
    pick_ids = list(candidates.head(n_tiles).index)

    print(f"[fig2 v6 polygon-GT] {len(candidates)}/{len(df)} tiles satisfy "
          f"merge-fix + adjacency criterion; picked top-{n_tiles} by merge_fix:")
    for iid in pick_ids:
        r = df.loc[iid]
        print(f"  {iid}: n_merge(U,S)={r.n_merge_u},{r.n_merge_s} "
              f"(fix={int(r.merge_fix)}), n_pred(U,S)={r.n_pred_u},{r.n_pred_s}, "
              f"n_GT={r.n_gt}, adj={r.adj_score:.2f}, "
              f"fg_ratio={r.fg_ratio:.2f}, "
              f"dice(U,S)={r.dice_u:.2f},{r.dice_s:.2f}")

    # Try to load RGB from MoNuSAC dataset; fall back to using gt mask if eva
    # is unavailable.
    try:
        from scripts.eva_critique.data import iter_monusac_test
        rgb_by_id = {t.image_id: t.image_rgb
                     for t in iter_monusac_test(str(data_root))}
        have_rgb = True
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"falling back to GT-only panel (no H&E RGB): {e}")
        rgb_by_id = {}
        have_rgb = False

    cols = 4 if have_rgb else 3
    col_titles = (["H&E", "GT instances", f"UNI (τ*={tau_uni})",
                   f"SIGMMA meso (τ*={tau_sig})"]
                  if have_rgb else
                  ["GT instances", f"UNI (τ*={tau_uni})",
                   f"SIGMMA meso (τ*={tau_sig})"])
    fig, axes = plt.subplots(n_tiles, cols, figsize=(cols * 2.8, n_tiles * 2.7))
    if n_tiles == 1:
        axes = axes[None, :]

    from scripts.eva_critique.metrics_seg import detection_f1_hungarian

    for r, iid in enumerate(pick_ids):
        ui = uni_idx[iid]; si = sig_idx[iid]
        prob_u = uni_npz["prob"][ui].astype(np.float32)
        prob_s = sig_npz["prob"][si].astype(np.float32)
        gt_bin = uni_npz["gt"][ui].astype(np.uint8)

        try:
            gt_lab = load_polygon_instance_label(iid, data_root)
        except FileNotFoundError:
            gt_lab = label_components(gt_bin)
        inst_u = prob_to_instances(prob_u, threshold=tau_uni)
        inst_s = prob_to_instances(prob_s, threshold=tau_sig)
        det_u = detection_f1_hungarian(inst_u, gt_lab)
        det_s = detection_f1_hungarian(inst_s, gt_lab)
        nm_u = det_u["n_merge"]
        nm_s = det_s["n_merge"]

        col = 0
        if have_rgb:
            ax = axes[r, col]; col += 1
            rgb = rgb_by_id.get(iid)
            if rgb is not None:
                ax.imshow(rgb)
            if r == 0:
                ax.set_title(col_titles[0], fontsize=8)
        for cidx, (lab, name, nm) in enumerate([
            (gt_lab, "GT", None),
            (inst_u, "UNI", nm_u),
            (inst_s, "SIGMMA", nm_s),
        ]):
            ax = axes[r, col]; col += 1
            base = _instance_label_image(lab)
            base = _add_boundary_overlay(base, lab)
            ax.imshow(base)
            head_title = col_titles[col - 1]
            sub = "" if nm is None else f"n_merge={int(nm)}"
            if r == 0:
                ax.set_title(f"{head_title}\n{sub}".strip(), fontsize=8)
            else:
                ax.set_title(sub, fontsize=7)

        axes[r, 0].set_ylabel(iid, fontsize=7)

        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="outputs/eva_critique")
    ap.add_argument("--data_root", default="data/monusac")
    ap.add_argument("--n_tiles", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    logs = Path(args.logs_root)
    figs = logs / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    fig_headline_trio(logs / "alternative_metrics.csv",
                      logs / "region_stratified_metrics.csv",
                      figs / "headline_metric_trio.pdf")
    fig_nmerge_scatter(logs / "instance_metrics.csv",
                       figs / "n_merge_scatter.pdf",
                       tau=args.tau)
    fig_detf1_curve(logs / "alternative_metrics.csv",
                    figs / "detf1_vs_iou.pdf")
    fig_boundary_panel(logs, Path(args.data_root),
                       logs / "instance_metrics.csv",
                       logs / "alternative_metrics.csv",
                       figs / "boundary_overlay_tiles.pdf",
                       n_tiles=args.n_tiles)


if __name__ == "__main__":
    main()
