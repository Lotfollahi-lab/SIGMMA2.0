import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, SGDRegressor

def linear_probing(train_dataloader, test_dataloader, model, device, seed=42):
    output_predictor_train = gene_exp_prediction(train_dataloader, model, device, train_mode=True, random_state=seed)
    trained_predictor = output_predictor_train["predictor"]

    output_predictor_test = gene_exp_prediction(test_dataloader, model, device, train_mode=False, predictor=trained_predictor,random_state=seed)
    y = output_predictor_test["labels"]
    y_pred = output_predictor_test['labels_pred']

    eval_multimodal = evaluate_gene_exp_prediction(y, y_pred)
    return eval_multimodal

def gene_exp_prediction(dataloader, model, device, train_mode=True, predictor=None, max_iter=1000, random_state=42):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            label = batch["spatial"].to(device) # TODO: adjust to accept multiple labels
            embeddings = model.get_image_features(image)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(label.cpu())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    alpha = 100 /(all_embeddings.shape[1] * all_labels.shape[1]) 
    if train_mode:
        predictor = Ridge(solver='lsqr',
                            alpha=alpha,
                            random_state=random_state,
                            fit_intercept=False,
                            max_iter=max_iter)
        predictor.fit(all_embeddings, all_labels)
        output = {"embeddings": all_embeddings,
                "labels": all_labels,
                "predictor": predictor
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
        l2_error = float(np.mean((preds - target_vals)**2))
        # compute r2 score
        r2_score = float(1 - np.sum((target_vals - preds)**2) / np.sum((target_vals - np.mean(target_vals))**2))
        pearson_corr, _ = pearsonr(target_vals, preds)
        if np.isnan(pearson_corr):
            print(target_vals)
            print(preds)
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            # 'name': genes[i], # TODO: add gene names
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
        i += 1
        

    results = {'l2_errors': list(errors), 
               'r2_scores': list(r2_scores),
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