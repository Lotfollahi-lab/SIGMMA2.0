import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import f1_score, r2_score
from scipy.spatial.distance import cdist
from sigmma.evaluation.linear_probing_cell_level import label_prediction_cell_level


def build_cell_type_compositions_onehot(labels: np.ndarray, n_classes: int = None):
    """
    labels: (N,) integer cell type labels for each cell
    returns: (N, C) one-hot matrix; row i is the one-hot vector of cell i's type
    """
    labels = np.asarray(labels, dtype=np.int64)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    comps = np.zeros((len(labels), n_classes), dtype=np.float64)
    comps[np.arange(len(labels)), labels] = 1.0
    return comps


def compute_distance_bins(coordinates: np.ndarray, bin_edges: list = None):
    if bin_edges is None:
        sample_idx = np.random.choice(
            len(coordinates), min(500, len(coordinates)), replace=False
        )
        sample_coords = coordinates[sample_idx]
        dists = cdist(sample_coords, sample_coords).ravel()
        dists = dists[dists > 0]
        # 5 bins from 0 to 100th percentile
        bin_edges = np.percentile(dists, [0, 20, 40, 60, 80, 100]).tolist()
    return bin_edges


def safe_r2(y_true, y_pred, var_threshold: float = 1e-6):
    """
    Robust R^2 for multi-output regression.
    - Drops columns with near-zero variance in y_true.
    - Falls back to flattened R^2 if no columns pass the filter.
    - Returns np.nan only if truly degenerate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        if y_true.var() < var_threshold:
            return np.nan
        return r2_score(y_true, y_pred)

    col_var = y_true.var(axis=0)
    valid_cols = col_var > var_threshold

    if valid_cols.sum() == 0:
        if y_true.var() < var_threshold:
            return np.nan
        return r2_score(y_true.ravel(), y_pred.ravel())

    return r2_score(
        y_true[:, valid_cols],
        y_pred[:, valid_cols],
        multioutput="uniform_average",
    )


def neighbor_prediction_by_distance(
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
    cell_type_compositions: np.ndarray = None,
    bin_edges: list = None,
    random_state: int = 42,
    min_samples_per_bin: int = 20,
    auto_onehot_composition: bool = True,
    n_classes: int = None,
    verbose: bool = False,
):
    """
    Linear probing of embeddings against neighborhood composition / majority label,
    stratified by distance shells.

    If `cell_type_compositions` is None and `auto_onehot_composition` is True,
    a one-hot composition matrix is built from `labels`. The regression target
    then becomes the average one-hot of neighbors within each distance shell,
    i.e. the local cell-type composition vector (sums to 1).
    """
    N = len(embeddings)
    bin_edges = compute_distance_bins(coordinates, bin_edges)
    n_bins = len(bin_edges) - 1

    # --- Auto-build one-hot composition from labels if not provided ---
    if cell_type_compositions is None and auto_onehot_composition:
        if n_classes is None:
            n_classes = int(np.max(labels)) + 1
        cell_type_compositions = build_cell_type_compositions_onehot(
            labels, n_classes=n_classes
        )

    dist_matrix = cdist(coordinates, coordinates)

    clf_f1_per_bin = []
    reg_r2_per_bin = []
    bin_ranges = []
    n_kept_cols_per_bin = []  # diagnostic: how many cell-type columns survived

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_ranges.append((lo, hi))

        X_center_list, y_cls_list, y_reg_list = [], [], []

        for i in range(N):
            neighbor_mask = (dist_matrix[i] >= lo) & (dist_matrix[i] < hi)
            neighbor_mask[i] = False
            neighbor_idx = np.where(neighbor_mask)[0]

            if len(neighbor_idx) == 0:
                continue

            neighbor_labels = labels[neighbor_idx]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]

            X_center_list.append(embeddings[i])
            y_cls_list.append(majority_label)

            if cell_type_compositions is not None:
                mean_comp = cell_type_compositions[neighbor_idx].mean(axis=0)
                y_reg_list.append(mean_comp)

        if len(X_center_list) < min_samples_per_bin:
            clf_f1_per_bin.append(np.nan)
            reg_r2_per_bin.append(np.nan)
            n_kept_cols_per_bin.append(0)
            continue

        X = np.array(X_center_list)
        y_cls = np.array(y_cls_list)

        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(X))
        split = int(0.7 * len(X))
        tr, te = idx[:split], idx[split:]

        # --- Classification ---
        if len(np.unique(y_cls[tr])) < 2:
            clf_f1_per_bin.append(np.nan)
        else:
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X[tr], y_cls[tr])
            y_pred_cls = clf.predict(X[te])
            f1 = f1_score(y_cls[te], y_pred_cls, average="macro", zero_division=0)
            clf_f1_per_bin.append(f1)

        # --- Regression: predict neighborhood cell-type composition ---
        if cell_type_compositions is not None and len(y_reg_list) == len(X_center_list):
            y_reg = np.array(y_reg_list, dtype=np.float64)

            # Replace any NaN/Inf with column means
            if not np.all(np.isfinite(y_reg)):
                col_means = np.nanmean(y_reg, axis=0)
                col_means = np.where(np.isfinite(col_means), col_means, 0.0)
                bad = ~np.isfinite(y_reg)
                y_reg[bad] = np.take(col_means, np.where(bad)[1])

            # Drop near-constant columns globally to stabilize Ridge
            global_var = y_reg.var(axis=0)
            keep_cols = global_var > 1e-10
            n_kept = int(keep_cols.sum())
            n_kept_cols_per_bin.append(n_kept)

            if n_kept == 0:
                reg_r2_per_bin.append(np.nan)
                continue

            y_reg_fit = y_reg[:, keep_cols]

            reg = Ridge(alpha=1.0)
            reg.fit(X[tr], y_reg_fit[tr])
            y_pred_reg = reg.predict(X[te])

            r2 = safe_r2(y_reg_fit[te], y_pred_reg)
            reg_r2_per_bin.append(r2)

            if verbose:
                print(
                    f"  bin {b} [{lo:.1f}, {hi:.1f}): "
                    f"n_samples={len(X)}, kept_cols={n_kept}/{y_reg.shape[1]}, "
                    f"f1={clf_f1_per_bin[-1]:.3f}, r2={r2:.3f}"
                )
        else:
            reg_r2_per_bin.append(np.nan)
            n_kept_cols_per_bin.append(0)

    return {
        "bin_ranges": bin_ranges,
        "classification": {"f1_macro": clf_f1_per_bin},
        "regression": {"r2": reg_r2_per_bin},
        "n_kept_cols": n_kept_cols_per_bin,
    }


def linear_probing_neighbor_prediction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config,
    device: torch.device,
    cell_ids_to_labels: dict,
    cell_type_compositions: dict = None,
    bin_edges: list = None,
    multi_scale: bool = False,
    auto_onehot_composition: bool = True,
    verbose: bool = False,
):
    """
    Wrapper that runs `neighbor_prediction_by_distance` on model embeddings.

    By default, the regression target is the per-shell one-hot cell-type
    composition built automatically from `labels` (the integer labels produced
    by `label_prediction_cell_level` via `cell_ids_to_labels`).

    The `cell_type_compositions` argument is kept for backward compatibility:
    if provided as a dict {cell_id: vector}, it overrides the auto one-hot.
    """
    output = label_prediction_cell_level(
        dataloader=dataloader,
        model=model,
        config=config,
        device=device,
        cell_ids_to_labels=cell_ids_to_labels,
        train_mode=True,
        multi_scale=multi_scale,
    )

    # Global n_classes from the label mapping (so all scales use the same width)
    n_classes_global = (
        len(cell_ids_to_labels) if isinstance(cell_ids_to_labels, dict) else None
    )

    def _build_comps_from_dict(cell_ids):
        """Only used if user explicitly passes a dict of compositions."""
        if cell_type_compositions is None:
            return None
        ref_shape = next(iter(cell_type_compositions.values())).shape
        comps = np.stack(
            [
                cell_type_compositions.get(cid, np.zeros(ref_shape))
                for cid in cell_ids
            ]
        ).astype(np.float64)
        comps = np.nan_to_num(comps, nan=0.0, posinf=0.0, neginf=0.0)

        missing = sum(1 for cid in cell_ids if cid not in cell_type_compositions)
        if missing > 0:
            print(
                f"[warn] {missing}/{len(cell_ids)} "
                f"({100 * missing / len(cell_ids):.1f}%) cell_ids missing in "
                f"cell_type_compositions; filled with zeros."
            )
        return comps

    def _run(scale_output):
        # Prefer user-provided dict; otherwise let neighbor_prediction_by_distance
        # build a one-hot composition from labels.
        comps = _build_comps_from_dict(scale_output["cell_ids"])
        return neighbor_prediction_by_distance(
            embeddings=scale_output["embeddings"],
            coordinates=scale_output["coordinates"],
            labels=scale_output["labels"],
            cell_type_compositions=comps,
            bin_edges=bin_edges,
            auto_onehot_composition=auto_onehot_composition,
            n_classes=n_classes_global,
            verbose=verbose,
        )

    if multi_scale:
        return {k: _run(v) for k, v in output.items()}
    else:
        return _run(output)