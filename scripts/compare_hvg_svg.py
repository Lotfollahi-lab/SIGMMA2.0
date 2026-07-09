import numpy as np, glob, os, json

HVG = "data/in_house_5k_disp_hvg_50.npy"
SVG_DIR = "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels_2026_05_29"


def load_genes(p):
    a = np.load(p, allow_pickle=True)
    return [str(x) for x in a.ravel().tolist()]


hvg = load_genes(HVG)
hvg_set = set(hvg)
print(f"HVG (previous, in_house_5k_disp_hvg_50): n={len(hvg)} (unique {len(hvg_set)})")
print("  genes:", ", ".join(hvg))

panels = sorted(glob.glob(os.path.join(SVG_DIR, "panel_*")))
union = set()
print(f"\n{'panel':<34}{'nSVG':>6}{'∩HVG':>6}{'Jacc':>8}{'%HVG':>7}  examples_overlap")
for pdir in panels:
    npy = os.path.join(pdir, "svg_top100.npy")
    if not os.path.exists(npy):
        continue
    svg = set(load_genes(npy))
    union |= svg
    inter = hvg_set & svg
    jac = len(inter) / len(hvg_set | svg)
    pct = len(inter) / len(hvg_set) * 100
    name = os.path.basename(pdir)
    ex = ", ".join(sorted(inter)[:8])
    print(f"{name:<34}{len(svg):>6}{len(inter):>6}{jac:>8.3f}{pct:>6.0f}%  {ex}")

inter_u = hvg_set & union
print(f"\n=== UNION of all SVG panels: n={len(union)} ===")
print(f"HVG ∩ SVG-union: {len(inter_u)}/{len(hvg_set)} = {len(inter_u)/len(hvg_set)*100:.0f}% of HVG covered")
print(f"Jaccard(HVG, SVG-union) = {len(inter_u)/len(hvg_set | union):.4f}")
print(f"\nHVG genes ALSO spatially variable (in some SVG panel): {sorted(inter_u)}")
print(f"\nHVG genes NOT in ANY SVG panel ({len(hvg_set-union)}): {sorted(hvg_set-union)}")

# main panel detail
main = os.path.join(SVG_DIR, "panel_35a7bd78801c", "svg_top100.npy")
if os.path.exists(main):
    msvg = set(load_genes(main))
    mi = hvg_set & msvg
    print(f"\n=== MAIN panel 35a7bd78801c (247 sec): nSVG={len(msvg)}, ∩HVG={len(mi)} ({len(mi)/len(hvg_set)*100:.0f}% of HVG) ===")
    print("  overlap:", sorted(mi))
