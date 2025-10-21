import multiprocessing
from typing import List
import numpy as np
import pandas as pd


def calculate_continuous_similarity(obs_feature_vector, pred_feature_value, num_samples, feature_std):
    """Continuous variable similarity calculation function"""
    # Calculate standardization factor
    std_factor = np.sqrt(np.sum((pred_feature_value - obs_feature_vector) ** 2) / num_samples)
    # Calculate similarity
    similarity = np.exp(-(pred_feature_value - obs_feature_vector) ** 2 / (2 * (feature_std * feature_std / std_factor) ** 2))
    return similarity


def calculate_categorical_similarity(obs_category, pred_category):
    """Categorical variable similarity calculation function"""
    return np.where(obs_category == pred_category, 1.0, 0.0)


class GOSCalculator:
    def __init__(self, target_values, continuous_features, categorical_features,
                 train_continuous, pred_continuous, continuous_std_values,
                 training_data, prediction_data, quantile_threshold, num_samples, min_similarity):
        self.target_values = target_values
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.train_continuous = train_continuous
        self.pred_continuous = pred_continuous
        self.continuous_std_values = continuous_std_values
        self.training_data = training_data
        self.prediction_data = prediction_data
        self.quantile_threshold = quantile_threshold
        self.num_samples = num_samples
        self.min_similarity = min_similarity
        # Check target variable type
        self.is_multiclass = isinstance(target_values[0], list)

    def calculate_gos_for_sample(self, sample_idx):
        # Initialize feature similarity list
        feature_similarities = []

        # Process continuous features
        if self.continuous_features:
            pred_sample_continuous = self.pred_continuous[sample_idx, :]
            for feature_idx, feature_name in enumerate(self.continuous_features):
                feature_similarity = calculate_continuous_similarity(
                    self.train_continuous[:, feature_idx],
                    pred_sample_continuous[feature_idx],
                    self.num_samples,
                    self.continuous_std_values[feature_idx]
                )
                feature_similarities.append(feature_similarity)

        # Process categorical features
        for feature_name in self.categorical_features:
            pred_category = self.prediction_data.iloc[sample_idx][feature_name]
            obs_categories = self.training_data[feature_name].values
            feature_similarity = calculate_categorical_similarity(obs_categories, pred_category)
            feature_similarities.append(feature_similarity)

        # Take minimum value as combined similarity measure
        combined_similarity = np.minimum.reduce(feature_similarities)

        # Select observations with high similarity
        quantile_threshold = np.quantile(combined_similarity, self.quantile_threshold)
        high_similarity_indices = np.where(
            (combined_similarity >= quantile_threshold) &
            (combined_similarity >= self.min_similarity)
        )[0]
        high_similarities = combined_similarity[high_similarity_indices]

        # Calculate prediction value (weighted average)
        if len(high_similarities) == 0:
            if self.is_multiclass:
                return [[], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]  # Return empty list and nan uncertainty for multiclass case
            else:
                return [np.nan] * 7  # Return NaN and uncertainty for continuous variable case

        # Choose different prediction logic based on target variable type
        if self.is_multiclass:
            # Multiclass case: calculate weighted sum for each category
            # First get all high similarity sample categories
            selected_targets = [self.target_values[i] for i in high_similarity_indices]
            
            # Create a dictionary to store weighted sum for each category
            class_weights = {}
            for target, similarity in zip(selected_targets, high_similarities):
                for cls in target:
                    if cls in class_weights:
                        class_weights[cls] += similarity
                    else:
                        class_weights[cls] = similarity
            
            # Sort categories by weighted sum
            sorted_classes = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
            prediction = [cls for cls, _ in sorted_classes]
        else:
            # Continuous variable case: use original logic
            prediction = np.sum(self.target_values[high_similarity_indices] * high_similarities) / (
                        np.sum(high_similarities) + 1e-10)

        # Calculate uncertainty metrics
        uncertainty_metrics = [
            1 - np.quantile(high_similarities, 0.9),
            1 - np.quantile(high_similarities, 0.95),
            1 - np.quantile(high_similarities, 0.99),
            1 - np.quantile(high_similarities, 0.995),
            1 - np.quantile(high_similarities, 0.999),
            1 - np.quantile(high_similarities, 1.0)
        ]

        return [prediction] + uncertainty_metrics

def process_batch(args):
    calculator, batch_range = args
    start_idx, end_idx = batch_range
    return [calculator.calculate_gos_for_sample(idx)
            for idx in range(start_idx, end_idx)]


def geographic_optimal_similarity(target_variable: str,
                                  feature_variables: List[str],
                                  training_data: pd.DataFrame,
                                  prediction_data: pd.DataFrame,
                                  categorical_features: List[str] = None,
                                  similarity_threshold: float = 0.25,
                                  min_similarity=0.5) -> pd.DataFrame:
    quantile_threshold = 1 - similarity_threshold
    if categorical_features is None:
        categorical_features = []

    target_values = training_data[target_variable].values
    num_pred_samples = len(prediction_data)

    # Distinguish between continuous features and categorical features
    continuous_features = [f for f in feature_variables if f not in categorical_features]

    # Prepare continuous feature data
    train_continuous = None
    pred_continuous = None
    continuous_std_values = None
    if continuous_features:
        train_continuous = training_data[continuous_features].values
        pred_continuous = prediction_data[continuous_features].values
        all_continuous = np.vstack((train_continuous, pred_continuous))
        continuous_std_values = np.array([np.std(all_continuous[:, i], ddof=1) for i in range(len(continuous_features))])

    # Create calculator instance
    calculator = GOSCalculator(
        target_values=target_values,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        train_continuous=train_continuous,
        pred_continuous=pred_continuous,
        continuous_std_values=continuous_std_values,
        training_data=training_data,
        prediction_data=prediction_data,
        quantile_threshold=quantile_threshold,
        num_samples=len(training_data),
        min_similarity = min_similarity
    )

    results = [calculator.calculate_gos_for_sample(idx)
                   for idx in range(num_pred_samples)]

    # Convert results to DataFrame
    column_names = ['prediction', 'uncertainty90', 'uncertainty95', 'uncertainty99',
                    'uncertainty99.5', 'uncertainty99.9', 'uncertainty100']
    results_df = pd.DataFrame(results, columns=column_names)

    return results_df