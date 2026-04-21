import ml_collections
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import torch
import dgl

from sigmma.misc.extract_embeddings import extract_graphs, extract_cell_embeddings_and_coordinates

def linear_probing_cell_level(model: torch.nn.Module,
                              train_dataloader: torch.utils.data.DataLoader,
                              test_dataloader: torch.utils.data.DataLoader,
                              cell_ids_to_labels: dict,
                              config: ml_collections.config_dict.ConfigDict,
                              device: torch.device,
                              multi_scale: bool=False,
                              annotation_lookup: dict=None):
    """Performs linear probing at the cell level using the embeddings extracted from the SIGMMA model.
    ------------------------------------------------------------------------------
    Args:
    - model: The SIGMMA model used to extract cell embeddings.
    - train_dataloader: DataLoader for the training dataset, yielding batches of data with cell embeddings.
    - test_dataloader: DataLoader for the test dataset, yielding batches of data with cell embeddings.
    - config: Configuration dictionary for the model and training settings.
    - labels: Dictionary mapping cell IDs to their corresponding labels.
    - device: The device (CPU or GPU) on which to perform computations.
    - multi_scale: Whether to use multi-scale embeddings.

    Returns:
    - eval: A dictionary containing the evaluation metrics for the multi-modal predictions.
    """
    output_predictor_train = label_prediction_cell_level(
            train_dataloader, model, config, device, cell_ids_to_labels,
            multi_scale=multi_scale, annotation_lookup=annotation_lookup
    )

    if multi_scale:
        if output_predictor_train is None or all(v is None for v in output_predictor_train.values()):
            return None
        # multi_scale: {'scale_0': {'predictor': ..., 'label_encoder': ...}, ...}
        trained_predictor = {
            k: (v["predictor"] if v is not None else None)
            for k, v in output_predictor_train.items()
        }
    else:
        if output_predictor_train is None:
            return None
        # single scale: {'predictor': {'method': clf, 'label_encoder': le}, ...}
        trained_predictor = output_predictor_train['predictor']

    output_predictor_test = label_prediction_cell_level(
        test_dataloader, model, config, device, cell_ids_to_labels,
        predictor=trained_predictor, multi_scale=multi_scale, train_mode=False,
        annotation_lookup=annotation_lookup
    )

    if multi_scale:
        if output_predictor_test is None:
            return None
        eval = {}
        for k, v in output_predictor_test.items():
            if v is None:
                eval[k] = None
                continue
            eval[k] = evaluate_cell_level_prediction(v['labels'], v['labels_pred'])
    else:
        if output_predictor_test is None:
            return None
        y = output_predictor_test['labels']
        y_pred = output_predictor_test['labels_pred']
        eval = evaluate_cell_level_prediction(y, y_pred)

    return eval


def build_predictor(method: str,
                    alpha: float,
                    num_classes: int,
                    max_iter: int,
                    random_state: int):
    if method == "ridge":
        return RidgeClassifier(
            alpha=alpha,
            fit_intercept=True,
            max_iter=max_iter,
        )
    elif method == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    elif method == "xgboost":
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
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

def label_prediction_cell_level(dataloader: torch.utils.data.DataLoader,
                                model: torch.nn.Module,
                                config: ml_collections.config_dict.ConfigDict,
                                device: torch.device,
                                cell_ids_to_labels: dict,
                                train_mode: bool=True,
                                method: str='ridge',
                                predictor=None,
                                max_iter: int=1000,
                                random_state: int=42,
                                pipe=None,
                                multi_scale=False,
                                annotation_lookup=None):
    graph_across_resolution, section_ids = extract_graphs(
        dataloader, model, config=config, device=device,
        group_to_id=cell_ids_to_labels, multi_scale=multi_scale,
        annotation_lookup=annotation_lookup,
    )

    num_resolutions = int(config.model.num_resolutions)

    if multi_scale:
        output = {}
        for k in range(num_resolutions):
            cell_ids, embeddings, coordinates = extract_cell_embeddings_and_coordinates(
                graphs_li=graph_across_resolution[f"scale_{k}"]
            )

            cell_ids = np.asarray(cell_ids)
            embeddings = np.nan_to_num(embeddings)
            coordinates = np.nan_to_num(coordinates)

            all_graphs = []
            for batched_g in graph_across_resolution[f"scale_{k}"]:
                all_graphs.extend(dgl.unbatch(batched_g))

            if len(all_graphs) == 0:
                output[f"scale_{k}"] = None
                continue

            labels_k = np.concatenate([g.ndata["group"].cpu().numpy() for g in all_graphs])

            # Filter out cells with unknown labels (group == -1)
            valid_mask = labels_k != -1
            if not valid_mask.any():
                output[f"scale_{k}"] = None
                continue
            labels_k = labels_k[valid_mask]
            embeddings = embeddings[valid_mask]
            cell_ids = cell_ids[valid_mask]
            coordinates = coordinates[valid_mask]

            if pipe is not None:
                if train_mode:
                    embeddings = pipe.fit_transform(embeddings)
                else:
                    embeddings = pipe.transform(embeddings)

            if train_mode:
                le = LabelEncoder()
                y = le.fit_transform(labels_k)
                num_classes = int(np.unique(y).shape[0])

                alpha = 1.0
                clf = build_predictor(method, alpha, num_classes, max_iter, random_state)
                clf.fit(embeddings, y)

                output[f"scale_{k}"] = {
                    "cell_ids": cell_ids,
                    "embeddings": embeddings,
                    "coordinates": coordinates,
                    "labels": labels_k,
                    "predictor": {"method": clf, "label_encoder": le},
                    "label_encoder": le,
                    "preprocessing_pipeline": pipe,
                }
            else:
                assert predictor is not None, "Predictor must be provided in evaluation mode"
                if f"scale_{k}" not in predictor or predictor[f"scale_{k}"] is None:
                    output[f"scale_{k}"] = None
                    continue

                clf = predictor[f"scale_{k}"]["method"]
                le = predictor[f"scale_{k}"]["label_encoder"]

                y_pred_encoded = clf.predict(embeddings)
                y_pred = le.inverse_transform(y_pred_encoded)

                output[f"scale_{k}"] = {
                    "cell_ids": cell_ids,
                    "embeddings": embeddings,
                    "coordinates": coordinates,
                    "labels": labels_k,
                    "labels_pred": y_pred,
                    "predictor": {"method": clf, "label_encoder": le},
                    "label_encoder": le,
                }

        return output

    else:
        cell_ids, embeddings, coordinates = extract_cell_embeddings_and_coordinates(graphs_li=graph_across_resolution)

        cell_ids = np.asarray(cell_ids)
        embeddings = np.nan_to_num(embeddings)
        coordinates = np.nan_to_num(coordinates)
        
        all_graphs = []
        for batched_g in graph_across_resolution:
            all_graphs.extend(dgl.unbatch(batched_g))

        if len(all_graphs) == 0:
            return None

        labels_ = np.concatenate([g.ndata["group"].cpu().numpy() for g in all_graphs])

        # Filter out cells with unknown labels (group == -1)
        valid_mask = labels_ != -1
        if not valid_mask.any():
            return None
        labels_ = labels_[valid_mask]
        embeddings = embeddings[valid_mask]
        cell_ids = cell_ids[valid_mask]
        coordinates = coordinates[valid_mask]


        if pipe is not None:
            if train_mode:
                embeddings = pipe.fit_transform(embeddings)
            else:
                embeddings = pipe.transform(embeddings)

        if train_mode:
            le = LabelEncoder()
            y = le.fit_transform(labels_)
            num_classes = int(np.unique(y).shape[0])

            alpha = 1.0
            clf = build_predictor(method, alpha, num_classes, max_iter, random_state)
            clf.fit(embeddings, y)

            return {
                "cell_ids": cell_ids,
                "embeddings": embeddings,
                "coordinates": coordinates,
                "labels": labels_,
                "predictor": {"method": clf, "label_encoder": le},
                "preprocessing_pipeline": pipe,
            }
        else:
            assert predictor is not None, "Predictor must be provided in evaluation mode"
            le = predictor["label_encoder"]
            clf = predictor["method"]

            y_pred_encoded = clf.predict(embeddings)
            y_pred = le.inverse_transform(y_pred_encoded)

            return {
                "cell_ids": cell_ids,
                "embeddings": embeddings,
                "coordinates": coordinates,
                "labels": labels_,
                "labels_pred": y_pred,
                "predictor": clf,
                "label_encoder": le,
            }

def evaluate_cell_level_prediction(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

    return metrics