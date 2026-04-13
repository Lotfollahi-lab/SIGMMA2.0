import numpy as np
import xgboost as xgb
import torch
import dgl
import ml_collections
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from sigmma.misc.extract_embeddings import extract_graphs, extract_cell_embeddings_and_coordinates


def evaluate_section_level_prediction(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _make_predictor(method, random_state, num_classes):
    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
    if method == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=random_state,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss"
        )
    raise ValueError(f"Unknown method: {method}")


def _extract_cell_embeddings_with_section(
    dataloader, model, config, device, labels_dict, multi_scale=False
):
    """
    Extract cell embeddings from graphs, following the same approach as cell_level probing.
    Each cell is assigned the section_id and section-level label from labels_dict.

    Returns:
        result: dict with keys per scale, each containing:
            - embeddings: (N, D)
            - section_labels: (N,) - section-level label for each cell
            - section_ids: (N,) - section_id for each cell
    """
    graph_across_resolution, section_ids_flat = extract_graphs(
        dataloader, model, config=config, device=device,
        group_to_id=None,
        multi_scale=multi_scale
    )

    num_resolutions = int(config.model.num_resolutions)
    scales = [f"scale_{k}" for k in range(num_resolutions)] if multi_scale else [None]

    result = {}

    for scale in scales:
        graphs_li = graph_across_resolution[scale] if multi_scale else graph_across_resolution

        cell_ids, embeddings, _ = extract_cell_embeddings_and_coordinates(graphs_li=graphs_li)

        # section_ids_flat is already flat (one sid per graph)
        # unbatch all batched graphs to get per-graph node counts
        cell_section_ids = []
        for batched_g, sid in zip(graphs_li, section_ids_flat):
            graphs = dgl.unbatch(batched_g)
            for g in graphs:
                cell_section_ids.extend([sid] * g.num_nodes())

        cell_section_ids = np.array(cell_section_ids)
        embeddings = np.nan_to_num(embeddings)

        # Sanity check
        assert len(embeddings) == len(cell_section_ids), \
            f"Mismatch: embeddings {len(embeddings)} vs section_ids {len(cell_section_ids)}"

        # Filter out cells whose section_id is not in labels_dict
        valid_mask = np.array([sid in labels_dict for sid in cell_section_ids])
        embeddings = embeddings[valid_mask]
        cell_section_ids = cell_section_ids[valid_mask]

        # Map section_id to section-level label for each cell
        section_labels = np.array([labels_dict[sid] for sid in cell_section_ids])

        key = scale if multi_scale else "single"
        result[key] = {
            "embeddings": embeddings,
            "section_labels": section_labels,
            "section_ids": cell_section_ids,
        }

    return result


def _soft_voting_mil(proba_instance, section_ids):
    """
    Aggregate instance-level probabilities to section level via soft voting.
    Averages instance probabilities within each section.
    """
    unique_sections = np.unique(section_ids)
    bag_proba = []
    for sid in unique_sections:
        idx = section_ids == sid
        bag_proba.append(proba_instance[idx].mean(axis=0))
    return np.vstack(bag_proba), unique_sections


def linear_probing_section_level(
    model,
    train_dataloader,
    test_dataloader,
    config,
    device,
    labels_dict,
    multi_scale=False,
    method="xgboost",
    random_state=42,
):
    """
    Section-level classification using multi-scale cell embeddings with soft voting MIL.
    Predictor is trained at cell level and evaluated at section level.
    """
    # ----- Training -----
    train_data = _extract_cell_embeddings_with_section(
        train_dataloader, model, config, device, labels_dict, multi_scale=multi_scale
    )
    scales = list(train_data.keys())
    trained_predictors = {}
    label_encoders = {}
    for scale in scales:
        X_train = train_data[scale]["embeddings"]
        y_train_raw = train_data[scale]["section_labels"]

        # Sanity check: each section must have a single label
        section_ids_train = train_data[scale]["section_ids"]
        for sid in np.unique(section_ids_train):
            labels_in_section = y_train_raw[section_ids_train == sid]
            assert len(np.unique(labels_in_section)) == 1, (
                f"[Train] Section {sid} has mixed labels: {np.unique(labels_in_section)}"
            )

        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        num_classes = len(np.unique(y_train))
        predictor = _make_predictor(method, random_state, num_classes)
        predictor.fit(X_train, y_train)
        trained_predictors[scale] = predictor
        label_encoders[scale] = le

    # ----- Testing -----
    test_data = _extract_cell_embeddings_with_section(
        test_dataloader, model, config, device, labels_dict, multi_scale=multi_scale
    )
    metrics = {}
    for scale in scales:
        X_test = test_data[scale]["embeddings"]
        y_test_raw = test_data[scale]["section_labels"]
        section_test = test_data[scale]["section_ids"]
        predictor = trained_predictors[scale]
        le = label_encoders[scale]

        # Sanity check: each section must have a single label
        for sid in np.unique(section_test):
            labels_in_section = y_test_raw[section_test == sid]
            assert len(np.unique(labels_in_section)) == 1, (
                f"[Test] Section {sid} has mixed labels: {np.unique(labels_in_section)}"
            )

        # Predict instance-level probabilities and aggregate via soft voting
        # section_order: the order of sections as aggregated inside _soft_voting_mil
        proba_instance = predictor.predict_proba(X_test)
        proba_section, section_order = _soft_voting_mil(proba_instance, section_test)

        # Align ground truth to match section_order from _soft_voting_mil
        section_to_true_raw = {
            sid: y_test_raw[section_test == sid][0]
            for sid in np.unique(section_test)
        }
        y_section_true_raw = np.array([section_to_true_raw[sid] for sid in section_order])
        y_section_true = le.transform(y_section_true_raw)

        y_section_pred = np.argmax(proba_section, axis=1)
        metrics[scale] = evaluate_section_level_prediction(y_section_true, y_section_pred)

    return metrics