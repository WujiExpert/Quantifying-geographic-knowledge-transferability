import numpy as np
import pandas as pd
from typing import Union, List


def remove_outliers(data_vector: Union[np.ndarray, pd.Series, List], std_multiplier: float = 2.5) -> List[int]:
    """
    Remove outliers.

    Parameters:
        data_vector: Data vector that needs outlier detection
        std_multiplier: Multiplier of standard deviation, used to determine outlier threshold. Default is 2.5.

    Returns:
        List of indices of outliers in the data vector
    """
    # Convert to numpy array for calculation
    if isinstance(data_vector, pd.Series):
        data_array = data_vector.values
    else:
        data_array = np.array(data_vector)

    # Calculate mean and standard deviation (ignore NaN values)
    mean_value = np.nanmean(data_array)
    std_value = np.nanstd(data_array)

    # Calculate upper and lower thresholds
    upper_threshold = mean_value + std_multiplier * std_value
    lower_threshold = mean_value - std_multiplier * std_value

    # Find indices of outliers
    outlier_indices = np.where(
        np.isnan(data_array) |
        (data_array > upper_threshold) |
        (data_array < lower_threshold)
    )[0].tolist()

    # Output message
    if outlier_indices:
        print(f"Remove {len(outlier_indices)} outlier(s)")
    else:
        print("No outlier.")

    return outlier_indices


def calculate_metrics(observed, predicted):
    """
    Calculate multiple evaluation metrics

    Parameters:
        observed: Observed values
        predicted: Predicted values

    Returns:
        Dictionary containing multiple evaluation metrics
    """
    # Remove rows containing NaN
    valid_mask = ~(np.isnan(observed) | np.isnan(predicted))
    observed = observed[valid_mask]
    predicted = predicted[valid_mask]

    # Calculate various metrics
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    mae = np.mean(np.abs(observed - predicted))
    correlation = np.corrcoef(observed, predicted)[0, 1]

    # Calculate R²
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate relative error metrics
    mape = np.mean(np.abs((observed - predicted) / observed)) * 100  # Mean absolute percentage error

    # Calculate prediction accuracy interval proportions
    within_10_percent = np.mean(np.abs((observed - predicted) / observed) <= 0.1) * 100
    within_20_percent = np.mean(np.abs((observed - predicted) / observed) <= 0.2) * 100

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": correlation,
        "mape": mape,
        "within_10_percent": within_10_percent,
        "within_20_percent": within_20_percent,
        "sample_size": len(observed)
    }


def evaluate_multiclass_prediction(actual_classes: List[List], predicted_classes: List[List], k_value=5):
    """
    Evaluate multiclass classification prediction results, supporting TOP-K evaluation

    Parameters:
        actual_classes: List of actual classes, each element is a list containing all classes for that sample
        predicted_classes: List of predicted classes, each element is a list containing predicted classes for that sample
        k_values: List of TOP-K values to evaluate, default is [1, 3, 5]

    Returns:
        Dictionary containing multiple evaluation metrics
    """
    # Remove empty prediction results
    valid_indices = [i for i, pred in enumerate(predicted_classes) if pred]
    
    valid_actual = [actual_classes[i] for i in valid_indices]
    valid_predicted = [predicted_classes[i] for i in valid_indices]
    
    results = {
        "sample_size": len(valid_actual),
        "empty_predictions": len(predicted_classes) - len(valid_indices)
    }
    
    # Calculate evaluation metrics
    # Calculate TOP-K accuracy
    top_k_acc = top_k_accuracy(valid_actual, valid_predicted, k_value)
    results[f"top_{k_value}_accuracy"] = top_k_acc

    # Calculate Jaccard similarity at TOP-K
    jaccard_k = jaccard_similarity_at_k(valid_actual, valid_predicted, k_value)
    results[f"jaccard_at_{k_value}"] = jaccard_k

    # Calculate precision, recall and F1 score at TOP-K
    metrics_k = precision_recall_f1_at_k(valid_actual, valid_predicted, k_value)
    results.update(metrics_k)
    
    # Calculate proportion of at least one correct prediction
    at_least_one_correct = sum(1 for act, pred in zip(valid_actual, valid_predicted) 
                              if len(set(act).intersection(set(pred[:k_value]))) > 0)
    results["at_least_one_correct_rate"] = at_least_one_correct / len(valid_actual) if valid_actual else 0
    
    return results


def top_k_accuracy(y_true, y_pred, k=5):
    """
    Calculate TOP-K accuracy
    
    Parameters:
        y_true: List of true labels
        y_pred: List of predicted labels
        k: TOP-K value
        
    Returns:
        TOP-K accuracy
    """
    hits = 0
    total = 0
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        # Ensure predicted labels do not exceed k
        pred_k = pred_labels[:k] if len(pred_labels) > k else pred_labels
        # Calculate hits
        hit = len(set(true_labels) & set(pred_k))
        hits += hit
        total += len(true_labels)
    
    return hits / total if total > 0 else 0


def jaccard_similarity_at_k(y_true, y_pred, k=5):
    """
    Calculate Jaccard similarity at TOP-K
    
    Parameters:
        y_true: List of true labels
        y_pred: List of predicted labels
        k: TOP-K value
        
    Returns:
        Jaccard similarity at TOP-K
    """
    scores = []
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        # Limit predicted labels to k
        pred_k = pred_labels[:k] if len(pred_labels) > k else pred_labels
        
        if not true_labels and not pred_k:  # If both are empty
            scores.append(1.0)
        elif not true_labels or not pred_k:  # If one is empty
            scores.append(0.0)
        else:
            # Calculate intersection and union
            intersection = len(set(true_labels) & set(pred_k))
            union = len(set(true_labels) | set(pred_k))
            # Calculate Jaccard similarity
            score = intersection / union if union > 0 else 0
            scores.append(score)
    
    return np.mean(scores)


def precision_recall_f1_at_k(y_true, y_pred, k=5):
    """
    Calculate precision, recall and F1 score at TOP-K
    
    Parameters:
        y_true: List of true labels
        y_pred: List of predicted labels
        k: TOP-K value
        
    Returns:
        Dictionary containing precision, recall and F1 score at TOP-K
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        # Limit predicted labels to k
        pred_k = pred_labels[:k] if len(pred_labels) > k else pred_labels
        
        if not pred_k:  # If prediction is empty
            precision_scores.append(0.0)
            recall_scores.append(0.0 if true_labels else 1.0)
            f1_scores.append(0.0)
            continue
            
        # Calculate intersection
        intersection = len(set(true_labels) & set(pred_k))
        
        # Precision: number of correctly predicted labels / total number of predicted labels
        precision = intersection / len(pred_k)
        precision_scores.append(precision)
        
        if not true_labels:  # If true labels are empty
            recall_scores.append(1.0 if not pred_k else 0.0)
            f1_scores.append(0.0)
            continue
            
        # Recall: number of correctly predicted labels / total number of true labels
        recall = intersection / len(true_labels)
        recall_scores.append(recall)
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        f"precision@{k}": np.mean(precision_scores),
        f"recall@{k}": np.mean(recall_scores),
        f"f1@{k}": np.mean(f1_scores)
    }


def calculate_random_f1_baseline(actual_classes, k=5):
    """
    Calculate theoretical F1 value for random prediction in multiclass classification problem
    
    Parameters:
        actual_classes: List of actual classes, each element is a list containing all classes for that sample
        k: TOP-K value
        
    Returns:
        Theoretical F1 value for random prediction
    """
    # Calculate average number of labels
    avg_labels_per_sample = np.mean([len(labels) for labels in actual_classes])
    
    # Get label space size (total number of possible labels)
    all_labels = set()
    for labels in actual_classes:
        all_labels.update(labels)
    label_space_size = len(all_labels)
    
    # Calculate theoretical random F1 value
    random_precision = avg_labels_per_sample / label_space_size
    random_recall = k / label_space_size
    
    if random_precision + random_recall > 0:
        random_f1 = 2 * random_precision * random_recall / (random_precision + random_recall)
    else:
        random_f1 = 0
    
    return {
        "random_precision": random_precision,
        "random_recall": random_recall,
        "random_f1": random_f1,
        "avg_labels_per_sample": avg_labels_per_sample,
        "label_space_size": label_space_size
    }