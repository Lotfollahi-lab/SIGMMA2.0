"""Helpers for the cross-tissue SIGMMA atlas analysis.

Pure-array operations only — composition normalisation, Shannon entropy,
and the H_tissue − H_niche prioritisation score. No I/O.
"""
from __future__ import annotations

import numpy as np


def safe_l1_normalize(counts: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L1-normalize ``counts`` along ``axis``; rows that sum to zero stay zero."""
    counts = np.asarray(counts, dtype=np.float64)
    denom = counts.sum(axis=axis, keepdims=True)
    out = np.zeros_like(counts)
    nz = denom > eps
    np.divide(counts, denom, out=out, where=nz)
    return out


def shannon_entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy of a probability distribution (natural log).

    Inputs that do not sum to one are L1-normalized first. An all-zero
    distribution returns 0 (degenerate cluster).
    """
    p = np.asarray(p, dtype=np.float64)
    p = safe_l1_normalize(p, axis=axis, eps=eps)
    safe = np.where(p > eps, p, 1.0)
    return -np.sum(p * np.log(safe), axis=axis)


def cluster_label_counts(cluster_ids: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Per-cluster integer-label histogram.

    ``cluster_ids`` and ``labels`` are 1-D arrays of equal length.
    Negative labels (e.g. -1 NA) are dropped before counting.
    Returns (n_clusters, n_labels) where n_clusters = cluster_ids.max() + 1.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if cluster_ids.shape != labels.shape:
        raise ValueError(
            f"cluster_ids ({cluster_ids.shape}) and labels ({labels.shape}) must match"
        )
    keep = labels >= 0
    cluster_ids = cluster_ids[keep]
    labels = labels[keep]
    n_clusters = int(cluster_ids.max()) + 1 if cluster_ids.size else 0
    out = np.zeros((n_clusters, n_labels), dtype=np.int64)
    if cluster_ids.size:
        np.add.at(out, (cluster_ids, labels), 1)
    return out


def cluster_composition_sum(cluster_ids: np.ndarray, composition: np.ndarray) -> np.ndarray:
    """Sum a per-patch composition matrix into per-cluster totals.

    ``composition`` is (N, C); ``cluster_ids`` is (N,). Returns (n_clusters, C).
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    composition = np.asarray(composition, dtype=np.float64)
    if cluster_ids.shape[0] != composition.shape[0]:
        raise ValueError("cluster_ids and composition must align on axis 0")
    n_clusters = int(cluster_ids.max()) + 1 if cluster_ids.size else 0
    out = np.zeros((n_clusters, composition.shape[1]), dtype=np.float64)
    if cluster_ids.size:
        np.add.at(out, cluster_ids, composition)
    return out


def prioritization_score(h_tissue: np.ndarray, h_niche: np.ndarray) -> np.ndarray:
    """Cluster priority: high cross-tissue mixing, low niche mixing.

    NaN inputs propagate (used when a cluster lacks the minimum labelled-patch
    coverage to estimate H_niche).
    """
    return np.asarray(h_tissue, dtype=np.float64) - np.asarray(h_niche, dtype=np.float64)
