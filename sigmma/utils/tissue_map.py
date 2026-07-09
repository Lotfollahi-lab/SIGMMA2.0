"""Authoritative section -> tissue mapping.

Single source of truth: ``sample_metadata.csv`` (cols sample_id, tissue_type,
run_id). ALWAYS use this instead of filename ``split('-')`` parsing -- the old
parser dumped public/PX-PD samples into '?'/'s3' junk buckets and lost the
adrenocortical_carcinoma cohort. User-confirmed standing rule (2026-06-06).

Typical use in a figure script::

    from sigmma.utils.tissue_map import tissue_of, GROUP_OF, GROUP_LABEL, PUBLIC_TT
    t = tissue_of(section_id)              # raw tissue_type (e.g. 'beacon')
    grp = GROUP_OF.get(t)                  # display group (e.g. 'skin')

``tissue_of`` returns the raw ``tissue_type``. Use ``GROUP_OF`` to collapse the
two skin cohorts (beacon + CTCL) into a single 'skin' group, and ``PUBLIC_TT``
to drop external HEST-1k public sections (genuine OOD, no Sanger panel).
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Canonical path; fall back to the lustre_dabin symlink if the scratch mount
# is not visible from this host.
_CANDIDATES = [
    Path("/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/"
         "sample_metadata.csv"),
    Path("/nfs/users/nfs_d/dj16/lustre_dabin/data/sanger_in_house_xenium/"
         "sample_metadata.csv"),
]

# tissue_type -> display group. beacon + CTCL (two skin cohorts) collapse to one.
GROUP_OF = {
    "beacon": "skin",
    "CTCL": "skin",
    "pediatric_ovary": "pediatric_ovary",
    "renal_cell_carcinoma": "renal_cell_carcinoma",
    "pancreas": "pancreas",
    "liver": "liver",
    "lung": "lung",
    "whole_embryo": "whole_embryo",
    "adrenocortical_carcinoma": "adrenocortical_carcinoma",
    "glioblastoma": "glioblastoma",
}
GROUP_LABEL = {
    "skin": "Skin",
    "pediatric_ovary": "Ovary",
    "renal_cell_carcinoma": "Kidney (RCC)",
    "pancreas": "Pancreas",
    "liver": "Liver",
    "lung": "Lung",
    "whole_embryo": "Whole embryo",
    "adrenocortical_carcinoma": "Adrenocortical",
    "glioblastoma": "Glioblastoma",
}
# External HEST-1k public tissue_types: no Sanger panel -> genuine OOD.
PUBLIC_TT = {"IDC", "PAAD", "LUAD", "SKCM"}


def metadata_path() -> Path:
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "sample_metadata.csv not found in any known location: "
        + ", ".join(str(p) for p in _CANDIDATES)
    )


@lru_cache(maxsize=1)
def load_tissue_map() -> dict[str, str]:
    """sample_id -> tissue_type (authoritative). Cached per process."""
    df = pd.read_csv(metadata_path())
    return dict(zip(df["sample_id"].astype(str), df["tissue_type"].astype(str)))


def tissue_of(section_id: str, default: str | None = None) -> str | None:
    """Authoritative tissue_type for a section_id. Returns ``default`` if the
    section is not in sample_metadata.csv (e.g. an unexpected public id)."""
    return load_tissue_map().get(str(section_id), default)


def is_public(section_id: str) -> bool:
    """True if the section is an external HEST-1k public panel (genuine OOD)."""
    return tissue_of(section_id) in PUBLIC_TT
