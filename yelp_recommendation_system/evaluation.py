import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Calculate precision and recall at k for a set of predictions.

    Precision@K measures what proportion of top k recommendations are relevant.
    Recall@K measures what proportion of all relevant items are in top k.

    Args:
        predictions: List of Surprise Prediction objects
        k: Number of top recommendations to consider
        threshold: Rating threshold for considering an item as relevant

    Returns:
        tuple: (average_precision, average_recall) across all users
    """
    # organize preds by user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        # sort by estimated rating in desc orser
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # count relevant restaurants
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # count recommended restaurants (predicted rating >= threshold) in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # count relevant items that are also recommended in top K
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        # calc precision and recall
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

    return np.mean(precisions), np.mean(recalls)


def evaluate_model(predictions):
    """
    Computes multiple metrics to assess recommendation quality:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - Precision@K and Recall@K for K in [5, 10, 20, 30, 50]

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        dict: Dictionary containing all computed metrics
    """
    # extract true and predicted ratings
    y_true = [pred.r_ui for pred in predictions]
    y_pred = [pred.est for pred in predictions]

    # calc basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # initialize result dictionary
    result_dict = {
        'RMSE': rmse, 'MAE': mae,
        'Precision@5': None, 'Recall@5': None,
        'Precision@10': None, 'Recall@10': None,
        'Precision@20': None, 'Recall@20': None,
        'Precision@30': None, 'Recall@30': None,
        'Precision@50': None, 'Recall@50': None,}

    # calc precision and recall at different k vals
    for k in [5, 10, 20, 30, 50]:
        precision, recall = precision_recall_at_k(predictions, k=k)
        result_dict[f'Precision@{k}'] = precision
        result_dict[f'Recall@{k}'] = recall

    return result_dict


def print_evaluation_results(metrics):
    """
    Pretty-print evaluation metrics.

    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)

    # basic metrics
    print(f"\nPrediction Error:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")

    # precision and recall
    print(f"\nRanking Metrics:")
    for k in [5, 10, 20, 30, 50]:
        if f'Precision@{k}' in metrics and metrics[f'Precision@{k}'] is not None:
            print(f"  Precision@{k:2d}: {metrics[f'Precision@{k}']:.4f}  |  "
                  f"Recall@{k:2d}: {metrics[f'Recall@{k}']:.4f}")

    print("="*50 + "\n")

