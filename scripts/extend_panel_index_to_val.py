"""Extend an existing ``panel_index.json`` (built from the train manifest) so
that validation/eval sections are also mapped to a merged panel id, without
re-running the SVG aggregation pipeline.

For each val section we read ``var.index`` from its ``spatial_preprocessed`` h5,
compute its ``panel_hash`` (sha1 of the gene-name index, matching the convention
in scripts/diagnose_panel_inventory.py), and look up which merged panel covers
that hash via the per-panel ``svg_top*.meta.json`` files (``merged_from_hashes``
list). Val sections whose panel_hash matches an existing merged panel are added
to ``panel_index.json``. Sections whose panel is novel (no train coverage) are
reported and skipped — they would have no SVG target anyway.

Idempotent: if a val sid is already in panel_index.json the script reports it
and leaves it alone. Always writes a timestamped backup before modifying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd


def _read_var_index(spatial_h5_path: str) -> list[str]:
    with h5py.File(spatial_h5_path, "r") as f:
        var_grp = f["var"]
        idx_attr = var_grp.attrs.get("_index", "_index")
        if isinstance(idx_attr, bytes):
            idx_attr = idx_attr.decode()
        if idx_attr not in var_grp:
            for k in ("gene_name", "feature_name", "gene_symbol", "_index"):
                if k in var_grp:
                    idx_attr = k
                    break
            else:
                raise KeyError(f"No gene-name dataset in var/ of {spatial_h5_path}")
        raw = var_grp[idx_attr][:]
    return [g.decode() if isinstance(g, (bytes, bytearray)) else str(g) for g in raw]


def _panel_hash(var_index: list[str]) -> str:
    return hashlib.sha1("|".join(var_index).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--panel_root",
        default="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels",
        help="Directory containing panel_index.json and panel_<hash>/ subdirs.",
    )
    ap.add_argument(
        "--val_manifest",
        default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_val.csv",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Without this flag, the script is a dry-run (reports diff only).",
    )
    args = ap.parse_args()

    root = Path(args.panel_root)
    panel_index_path = root / "panel_index.json"
    panel_index = json.loads(panel_index_path.read_text())
    print(f"[load] panel_index.json: {len(panel_index)} existing entries")

    # Build (original) panel_hash -> merged panel_id from each panel's meta.
    hash_to_panel_id: dict[str, str] = {}
    for panel_dir in sorted(root.iterdir()):
        if not panel_dir.is_dir() or not panel_dir.name.startswith("panel_"):
            continue
        meta_files = sorted(panel_dir.glob("svg_top*.meta.json"))
        if not meta_files:
            continue
        meta = json.loads(meta_files[0].read_text())
        for h in meta.get("merged_from_hashes", []):
            if h in hash_to_panel_id and hash_to_panel_id[h] != meta["panel_id"]:
                print(
                    f"[WARN] hash {h} appears in multiple panels: "
                    f"{hash_to_panel_id[h]} and {meta['panel_id']}"
                )
            hash_to_panel_id[h] = meta["panel_id"]
    print(f"[load] {len(hash_to_panel_id)} distinct original panel_hashes covered "
          f"by {len({v for v in hash_to_panel_id.values()})} merged panels")

    # Iterate val manifest, compute panel_hash, look up.
    val_df = pd.read_csv(args.val_manifest).drop_duplicates("sample_name", keep="last")
    print(f"[load] val manifest: {len(val_df)} sections")

    matched: list[tuple[str, str, str]] = []   # (sid, hash, panel_id)
    novel: list[tuple[str, str]] = []          # (sid, hash) — hash not in any merged panel
    already: list[str] = []                    # already in panel_index
    failed: list[tuple[str, str]] = []         # (sid, error)

    for _, row in val_df.iterrows():
        sid = row["sample_name"]
        if sid in panel_index:
            already.append(sid)
            continue
        try:
            var_idx = _read_var_index(row["spatial_preprocessed"])
        except Exception as e:
            failed.append((sid, f"{type(e).__name__}: {e}"))
            continue
        h = _panel_hash(var_idx)
        pid = hash_to_panel_id.get(h)
        if pid is None:
            novel.append((sid, h))
        else:
            matched.append((sid, h, pid))

    print("\n========== Summary ==========")
    print(f"  already in panel_index: {len(already)}")
    print(f"  matched (will add):     {len(matched)}")
    print(f"  novel panel (skip):     {len(novel)}")
    print(f"  read failed:            {len(failed)}")

    if matched:
        print("\n[matched] (sid -> panel_id):")
        # group by target panel for readability
        by_panel = Counter(pid for _, _, pid in matched)
        for pid, n in by_panel.most_common():
            print(f"  {pid}: {n}")
        print("  Examples:")
        for sid, h, pid in matched[:5]:
            print(f"    {sid:<40s} hash={h}  -> {pid}")

    if novel:
        print("\n[novel] (panel_hash not in any merged panel — section will be filtered out):")
        for sid, h in novel:
            print(f"  {sid:<40s} hash={h}")

    if failed:
        print("\n[failed]:")
        for sid, err in failed:
            print(f"  {sid}: {err}")

    if not args.apply:
        print("\n[dry-run] re-run with --apply to write panel_index.json")
        return

    # Apply.
    backup = root / f"panel_index.json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(panel_index_path, backup)
    print(f"\n[backup] {backup}")

    for sid, _, pid in matched:
        panel_index[sid] = pid

    panel_index_path.write_text(json.dumps(panel_index, indent=2))
    print(f"[write] {panel_index_path}: {len(panel_index)} entries "
          f"(added {len(matched)} val sections)")


if __name__ == "__main__":
    main()
