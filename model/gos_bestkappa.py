import multiprocessing
from typing import Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gos import geographic_optimal_similarity


def calculate_rmse(observed, predicted):
    return np.sqrt(np.nanmean((observed - predicted) ** 2))


def calculate_cv_rmse(args):
    params, data, target_variable, feature_variables, categorical_features, train_test_split_ratio = args
    _, seed, threshold = params
    np.random.seed(seed)

    num_samples = len(data)
    # Randomly split training and test sets
    train_indices = np.random.choice(range(num_samples),
                                     size=int(train_test_split_ratio * num_samples),
                                     replace=False)
    train_data = data.iloc[train_indices]
    test_data = data.drop(train_indices)

    # Use GOS model for prediction
    gos_results = geographic_optimal_similarity(
        target_variable=target_variable,
        feature_variables=feature_variables,
        training_data=train_data,
        prediction_data=test_data,
        categorical_features=categorical_features,
        similarity_threshold=threshold
    )
    predictions = gos_results['prediction'].values

    # Calculate RMSE
    rmse_value = calculate_rmse(test_data[target_variable].values, predictions)
    return {'threshold': threshold, 'rmse': rmse_value}

def find_optimal_similarity_threshold(target_variable: str,
                                      feature_variables: List[str],
                                      data: pd.DataFrame,
                                      categorical_features: List[str] = None,
                                      threshold_candidates: Union[List[float], np.ndarray] = None,
                                      num_repetitions: int = 5,
                                      train_test_split_ratio: float = 0.5,
                                      num_cores: int = 1) -> Dict:
    """
    Computational optimization function to determine the optimal similarity threshold parameter

    Parameters:
        target_variable: Target variable name
        feature_variables: List of feature variable names
        data: DataFrame of observation data
        categorical_features: List of categorical feature variable names, default is None
        threshold_candidates: List of candidate similarity thresholds. Default is sequence from 0.05 to 1 with step 0.05.
        num_repetitions: Number of cross-validation repetitions. Default is 10.
        train_test_split_ratio: Training set ratio, in range (0,1). Default is 0.5.
        num_cores: Number of CPU cores for parallel computation. Default is 1.

    Returns:
        Dictionary containing the following:
        - optimal_threshold: Optimal similarity threshold
        - cross_validation_rmse: All RMSE calculation results during cross-validation
        - mean_rmse_by_threshold: Average RMSE corresponding to different thresholds
        - visualization: RMSE change plot corresponding to different thresholds
    """
    # Set default threshold candidates
    if threshold_candidates is None:
        threshold_candidates = np.arange(0.05, 1.05, 0.05)

    # Set up parallel processing
    pool = None
    use_parallel = False
    if num_cores > 1:
        use_parallel = True
        pool = multiprocessing.Pool(processes=num_cores)

    num_samples = len(data)

    # Create parameter list
    param_list = []
    for seed in range(1, num_repetitions + 1):
        for threshold in threshold_candidates:
            param_list.append((len(param_list) + 1, seed, threshold))

    # Create complete parameter tuples for each parameter
    args_list = [(params, data, target_variable, feature_variables, categorical_features, train_test_split_ratio)
                 for params in param_list]

    # Parallel or serial computation
    if use_parallel and pool is not None:
        rmse_results_list = pool.map(calculate_cv_rmse, args_list)
        pool.close()
        pool.join()
    else:
        rmse_results_list = [calculate_cv_rmse(args) for args in args_list]

    # Convert results to DataFrame
    rmse_results = pd.DataFrame(rmse_results_list)

    # Calculate average RMSE for each threshold
    mean_rmse_by_threshold = rmse_results.groupby('threshold')['rmse'].mean().reset_index()

    # Find optimal threshold
    best_idx = mean_rmse_by_threshold['rmse'].idxmin()
    optimal_threshold = mean_rmse_by_threshold.loc[best_idx, 'threshold']

    # Create visualization
    margin = (mean_rmse_by_threshold['rmse'].max() - mean_rmse_by_threshold['rmse'].min()) * 0.1
    best_x = optimal_threshold
    best_y = mean_rmse_by_threshold.loc[best_idx, 'rmse']

    plt.figure(figsize=(10, 6))
    plt.plot(mean_rmse_by_threshold['threshold'], mean_rmse_by_threshold['rmse'], 'o-')
    plt.annotate(f"{optimal_threshold:.2f}",
                 xy=(best_x, best_y),
                 xytext=(best_x + 0.05, best_y),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.ylim(mean_rmse_by_threshold['rmse'].min() - margin, mean_rmse_by_threshold['rmse'].max())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Similarity Threshold')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Similarity Threshold')

    # Return results
    results = {
        "optimal_threshold": optimal_threshold,
        "cross_validation_rmse": rmse_results,
        "mean_rmse_by_threshold": mean_rmse_by_threshold,
        "visualization": plt
    }

    return results