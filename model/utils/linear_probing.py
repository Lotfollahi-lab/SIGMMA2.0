# Code adapted from HEST-1k benchmarking codebase (https://github.com/mahmoodlab/HEST/blob/main/src/hest/bench/trainer.py)

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, SGDRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def linear_probing(train_dataloader, test_dataloader, model, device, predicted_label, norm, use_pca, method, seed=42):
    output_predictor_train = label_prediction(train_dataloader, model, device, train_mode=True, method=method, predictor=None, random_state=seed, predicted_label=predicted_label, norm=norm, use_pca=use_pca)
    trained_predictor = output_predictor_train["predictor"]

    output_predictor_test = label_prediction(test_dataloader, model, device, train_mode=False, predictor=trained_predictor, random_state=seed, predicted_label=predicted_label, norm=norm, use_pca=use_pca, pipe=output_predictor_train["preprocessing_pipeline"])
    y = output_predictor_test["labels"]
    y_pred = output_predictor_test['labels_pred']

    eval_multimodal = evaluate_gene_exp_prediction(y, y_pred)
    return eval_multimodal

def label_prediction(dataloader, model, device, train_mode=True, method='ridge', predictor=None, max_iter=1000, random_state=42, predicted_label='topK_exp', norm='log1p', use_pca=True, pipe=None):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            label = batch["label"][predicted_label]
            if norm == 'log1p':
                label = torch.log1p(label)
            elif norm == 'standard':
                label = label / label.sum(dim=1, keepdim=True)
            else:
                label = label
            with torch.amp.autocast(device.type, enabled=True):
                embeddings = model.get_image_features(image)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(label)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    if use_pca:
        n_feat = all_embeddings.shape[1]
        n_comp = min(256, n_feat)
        if train_mode:
            pipe = Pipeline([('scaler', StandardScaler()), (f'PCA', PCA(n_components=n_comp, random_state=42))])
            all_embeddings = pipe.fit_transform(all_embeddings)
        else:
            assert pipe is not None, "preprocessing pipeline trained on train dataset must be provided in test mode if you want to use PCA"
            all_embeddings = pipe.transform(all_embeddings)
    all_labels = np.concatenate(all_labels, axis=0)

    # If there are NaNs, replace them with 0
    all_embeddings = np.nan_to_num(all_embeddings)
    all_labels = np.nan_to_num(all_labels)

    alpha = 100 /(all_embeddings.shape[1] * all_labels.shape[1]) 
    if train_mode:
        if method == 'ridge':
            predictor = Ridge(solver='lsqr',
                                alpha=alpha,
                                random_state=random_state,
                                fit_intercept=True,
                                max_iter=max_iter)
            predictor.fit(all_embeddings, all_labels)
        elif method == 'random_forest':
            predictor = RandomForestRegressor(n_estimators=70,
                                               random_state=random_state)
            predictor.fit(all_embeddings, all_labels)
        elif method == 'xgboost':
            predictor = xgb.XGBRegressor(n_estimators=100,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   min_child_weight=1,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   gamma=0,
                                   reg_alpha=0,
                                   reg_lambda=1,
                                   random_state=random_state)
            predictor.fit(all_embeddings, all_labels)

        output = {"embeddings": all_embeddings,
                "labels": all_labels,
                "predictor": predictor,
                "preprocessing_pipeline": pipe,
                }
    else:
        assert predictor is not None, "Predictor model must be provided for evaluation mode"
        y_pred = predictor.predict(all_embeddings)
        output = {"embeddings": all_embeddings,
                "labels": all_labels,
                "labels_pred": y_pred,
                "predictor": predictor
                }
    return output


def evaluate_gene_exp_prediction(y, y_pred, genes=None):
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    i = 0
    for target in range(y.shape[1]):
        preds = y_pred[:, target]
        target_vals = y[:, target]
        var = np.var(target_vals)
        if var < 1e-8:
            continue
        l2_error = float(np.mean((preds - target_vals)**2))
        # compute r2 score
        r2_score = float(1 - np.sum((target_vals - preds)**2) / np.sum((target_vals - np.mean(target_vals))**2))
        pearson_corr, _ = pearsonr(target_vals, preds)
        if np.isnan(pearson_corr):
            print(target_vals)
            print(preds)
            pearson_corr = 0.0
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            'name': i, #genes[i], # TODO: add gene names
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
        i += 1
        

    results = {'l2_errors': list(errors), 
               'r2_scores': list(r2_scores),
               'l2_error_mean': float(np.mean(errors)),
               'l2_error_std': float(np.std(errors)),
               'pearson_corrs': pearson_genes,
               'pearson_mean': float(np.mean(pearson_corrs)),
               'pearson_std': float(np.std(pearson_corrs)),
               'l2_error_q1': float(np.percentile(errors, 25)),
               'l2_error_q2': float(np.median(errors)),
               'l2_error_q3': float(np.percentile(errors, 75)),
               'r2_score_q1': float(np.percentile(r2_scores, 25)),
               'r2_score_q2': float(np.median(r2_scores)),
               'r2_score_q3': float(np.percentile(r2_scores, 75)),}
    
    return results
    return results