"""Build an assay-based 'full-gene' panel grouping for GEX-aware training.

Groups sections by their assay panel (panel_design_id + num_targets_custom from
gene_panel_info/section_to_gene_panel.csv) and defines each panel's gene axis as
the INTERSECTION of measured var.index (gene_name) across the panel's sections.
Because the intersection is by construction present in every member section, the
panel-aware SVG loader path maps it with all-valid masks (no use_full_panel
strict-identity requirement). Output mirrors svg_panel_dir layout so it plugs
into the existing dataset code.

Leaves svg_panels_2026_05_29 untouched. Writes:
  <out>/panel_index.json                 section_id -> panel_id
  <out>/<panel_id>/svg_top<K>.npy        intersection gene names (object array)
"""
from __future__ import annotations
import json, collections
from pathlib import Path
import numpy as np, pandas as pd, h5py

D = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium")
OUT = Path("/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/full_panels_2026_06_06")
# train+val DEFINE each panel's gene axis (intersection). Test sections are added
# to panel_index but do NOT shrink an axis already defined by train+val (keeps the
# trained model's axis fixed). A panel seen ONLY in test gets its axis from test.
AXIS_MANIFESTS = [D/"sanger_MANIFEST_train.csv", D/"sanger_MANIFEST_val.csv"]
INDEX_ONLY_MANIFESTS = [D/"sanger_MANIFEST_test.csv"]

def read_genes(h5path):
    with h5py.File(h5path, "r") as h:
        idx = h["var"].attrs.get("_index", "gene_name")
        return [g.decode() if isinstance(g, bytes) else g for g in h["var"][idx][:]]

# 1. sections + spatial paths (train+val, deduped)
sec_path = {}
for mp in MANIFESTS:
    m = pd.read_csv(mp).drop_duplicates("sample_name")
    for _, r in m.iterrows():
        sec_path.setdefault(r["sample_name"], r["spatial_preprocessed"])

# 2. assay assignment
cs = (pd.read_csv(D/"gene_panel_info"/"section_to_gene_panel.csv")
        .drop_duplicates("sample_name", keep="last").set_index("sample_name"))
def panel_id(sid):
    if sid not in cs.index:
        return None
    return f"assay_{cs.loc[sid,'panel_design_id']}_c{int(cs.loc[sid,'num_targets_custom'])}"

# 3. read measured genes, group by panel
panel_secs = collections.defaultdict(list)
panel_geneset = {}
skipped = []
for sid, sp in sec_path.items():
    pid = panel_id(sid)
    if pid is None:
        skipped.append(sid); continue
    try:
        genes = set(read_genes(sp))
    except Exception as e:
        skipped.append(f"{sid}(read-err)"); continue
    panel_secs[pid].append(sid)
    panel_geneset[pid] = panel_geneset.get(pid, genes) & genes if pid in panel_geneset else genes

# 4. write artifacts
OUT.mkdir(parents=True, exist_ok=True)
section_to_panel = {}
summary = []
maxK = 0
for pid, secs in sorted(panel_secs.items()):
    inter = sorted(panel_geneset[pid])
    if len(inter) == 0:
        print(f"  WARN panel {pid} empty intersection ({len(secs)} sec) -> skip")
        continue
    pdir = OUT / pid
    pdir.mkdir(exist_ok=True)
    np.save(pdir / f"svg_top{len(inter)}.npy", np.array(inter, dtype=object))
    for s in secs:
        section_to_panel[s] = pid
    maxK = max(maxK, len(inter))
    summary.append((pid, len(secs), len(inter)))

(OUT / "panel_index.json").write_text(json.dumps(section_to_panel, indent=0))

print(f"\nwrote {len(section_to_panel)} sections across {len(summary)} panels -> {OUT}")
print(f"skipped {len(skipped)} sections (not in csv / unreadable)")
print(f"\n{'panel_id':40s} {'n_sec':>6s} {'n_genes(intersection)':>22s}")
for pid, ns, ng in sorted(summary, key=lambda x:-x[1]):
    print(f"{pid:40s} {ns:6d} {ng:22d}")
print(f"\n==> set config k_max >= {maxK}  (largest panel intersection)")
