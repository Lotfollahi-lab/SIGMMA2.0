#!/usr/bin/env python
"""Fetch TCGA-KIRC tumour grade (G1-G4) per case from the GDC clinical API.

Grade is absent from the local TCGA-KIRC_clinical.tsv (which has stage +
survival only), but it is the classic RCC severity axis, so we pull
diagnoses.tumor_grade for every TCGA-KIRC case from the public GDC API.

Output: CSV case_submitter_id, tumor_grade, grade_ord (G1..G4 -> 1..4; GX/NA ->
blank). Joined to per-patient predictions by case_submitter_id downstream.

Uses only stdlib urllib (no requests dependency). Needs outbound HTTPS.
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

GDC_CASES = "https://api.gdc.cancer.gov/cases"
GRADE_ORD = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}


def fetch(project: str, size: int) -> list[dict]:
    filters = {"op": "in",
               "content": {"field": "project.project_id", "value": [project]}}
    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,diagnoses.tumor_grade",
        "format": "JSON",
        "size": str(size),
    }
    from urllib.parse import urlencode
    url = f"{GDC_CASES}?{urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    return payload["data"]["hits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-KIRC")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--size", type=int, default=2000)
    args = ap.parse_args()

    hits = fetch(args.project, args.size)
    print(f"[gdc] {len(hits)} cases for {args.project}")
    rows = []
    for h in hits:
        sub = h.get("submitter_id")
        diags = h.get("diagnoses", []) or []
        grade = None
        for d in diags:
            g = d.get("tumor_grade")
            if g and g.upper() in GRADE_ORD:
                grade = g.upper()
                break
        rows.append({"case_submitter_id": sub, "tumor_grade": grade,
                     "grade_ord": GRADE_ORD.get(grade)})

    df = pd.DataFrame(rows).drop_duplicates("case_submitter_id")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df["tumor_grade"].value_counts(dropna=False).to_string())
    print(f"\n[out] {len(df)} cases -> {args.out_csv}")


if __name__ == "__main__":
    main()
