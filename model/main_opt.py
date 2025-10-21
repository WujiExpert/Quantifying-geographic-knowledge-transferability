"""
Geographic Transfer Analysis Main Program

This module provides a comprehensive framework for analyzing geographic knowledge transferability
across different domains including soil analysis, crime analysis, and sustainable development pathways recommendations.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from load_data import load_multiclass_data
from gos import geographic_optimal_similarity
from model.load_data import loadData
from model.gos_bestkappa import find_optimal_similarity_threshold
from significance_test import run_significance_tests
from utils import (
    remove_outliers, 
    calculate_metrics, 
    evaluate_multiclass_prediction, 
    calculate_random_f1_baseline
)

# Configure matplotlib to use system fonts
matplotlib.rcParams['font.family'] = 'Hiragino Sans GB'


@dataclass
class AnalysisConfig:
    """Configuration class for analysis parameters"""
    data_path: str
    target_variable: str
    feature_variables: List[str]
    categorical_features: List[str]
    similarity_threshold: float = 0.01
    min_similarity: float = 0.01
    std_multiplier: float = 2.5
    min_samples: int = 5
    output_dir: str = "../results"
    # New fields for multiclass support
    is_multiclass: bool = False
    multiclass_data_paths: Optional[List[str]] = None
    top_k: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.is_multiclass:
            # For single file data
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
        else:
            # For multiclass data with multiple files
            if self.multiclass_data_paths:
                for path in self.multiclass_data_paths:
                    if not Path(path).exists():
                        raise FileNotFoundError(f"Multiclass data file not found: {path}")
        
        if not self.feature_variables:
            raise ValueError("Feature variables list cannot be empty")


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    predictions: pd.DataFrame
    metrics: Dict[str, Any]
    processing_time: float
    config: AnalysisConfig


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def load_data(config: AnalysisConfig) -> pd.DataFrame:
        """Load data based on file extension and multiclass support"""
        if config.is_multiclass:
            print(f"Loading multiclass data from: {config.multiclass_data_paths}")
            if config.multiclass_data_paths and len(config.multiclass_data_paths) == 2:
                return load_multiclass_data(*config.multiclass_data_paths)
            else:
                raise ValueError("Multiclass data requires 2 file paths")
        else:
            print(f"Loading data from: {config.data_path}")
            file_ext = Path(config.data_path).suffix.lower()
            
            if file_ext == '.csv':
                return loadData(config.data_path)
            elif file_ext == '.xlsx':
                return pd.read_excel(config.data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def preprocess_data(data: pd.DataFrame, config: AnalysisConfig, 
                       apply_log_transform: bool = True) -> pd.DataFrame:
        """Preprocess data including outlier removal and optional log transformation"""
        print("Preprocessing data...")
        
        # Create a copy to avoid modifying original data
        processed_data = data.copy()

        if not config.is_multiclass:
            # Apply log transformation if specified
            if apply_log_transform and config.target_variable in processed_data.columns:
                print(f"Applying logarithmic transformation to {config.target_variable}")
                processed_data[f'{config.target_variable}_original'] = processed_data[config.target_variable]
                processed_data[config.target_variable] = np.log(processed_data[config.target_variable])
                # Visualize distribution
                DataProcessor._plot_distribution(processed_data, config.target_variable)

            # Remove outliers
            print("Removing outliers...")
            outlier_indices = remove_outliers(
                processed_data[config.target_variable],
                std_multiplier=config.std_multiplier
            )
            processed_data = processed_data.drop(outlier_indices).reset_index(drop=True)
        
        print(f"Data preprocessing completed. Original samples: {len(data)}, "
              f"Cleaned samples: {len(processed_data)}")
        
        return processed_data
    
    @staticmethod
    def _plot_distribution(data: pd.DataFrame, target_var: str):
        """Plot distribution of target variable"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(data[f'{target_var}_original'], bins=30)
        plt.title(f'Original {target_var} Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(data[target_var], bins=30)
        plt.title(f'{target_var} Distribution After Log Transformation')
        
        plt.tight_layout()
        plt.show() 


class ModelPredictor:
    """Handles model prediction using Geographic Optimal Similarity"""
    
    @staticmethod
    def leave_one_out_prediction(data: pd.DataFrame, config: AnalysisConfig) -> Tuple[pd.DataFrame, float]:
        """Perform leave-one-out prediction using GOS model"""
        print("Starting leave-one-out prediction...")
        start_time = time.time()
        
        # Initialize prediction columns based on problem type
        result_data = data.copy()
        if config.is_multiclass:
            result_data['predicted'] = result_data.apply(lambda x: [], axis=1)
            result_data['uncertainty'] = np.nan
        else:
            result_data['predicted'] = np.nan
            result_data['uncertainty'] = np.nan
        
        # Perform prediction for each sample
        for i in tqdm(range(len(result_data)), desc="Leave-one-out prediction", ncols=100):
            # Current sample as test sample
            test_sample = result_data.iloc[[i]]
            
            # Remaining samples as training samples
            train_samples = result_data.drop(i)
            
            # Use GOS model for prediction
            gos_result = geographic_optimal_similarity(
                target_variable=config.target_variable,
                feature_variables=config.feature_variables,
                training_data=train_samples,
                prediction_data=test_sample,
                categorical_features=config.categorical_features,
                similarity_threshold=config.similarity_threshold,
                min_similarity=config.min_similarity
            )
            
            # Save prediction results
            if config.is_multiclass:
                result_data.at[i, 'predicted'] = gos_result['prediction'].iloc[0]
                result_data.at[i, 'uncertainty'] = gos_result['uncertainty90'].iloc[0]
            else:
                result_data.loc[i, 'predicted'] = gos_result['prediction'].iloc[0]
                result_data.loc[i, 'uncertainty'] = gos_result['uncertainty90'].iloc[0]
        
        processing_time = time.time() - start_time
        print(f"Leave-one-out prediction completed in {processing_time:.2f} seconds")
        
        return result_data, processing_time
    
    @staticmethod
    def group_based_prediction(data: pd.DataFrame, config: AnalysisConfig, 
                             key_factor: str, factor_type: str = 'categorical',
                             num_classes: int = 3, group_method: str = 'kmeans') -> pd.DataFrame:
        """
        Perform unified group-based prediction analysis for both regression and multiclass.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for analysis
        config : AnalysisConfig
            Configuration object containing all analysis parameters
        key_factor : str
            Name of the factor to use for grouping (must exist in dataset)
        factor_type : str, default='categorical'
            Type of the key factor. Options: 'categorical', 'numerical'
        num_classes : int, default=3
            Number of groups to create (only used for numerical factors)
        group_method : str, default='kmeans'
            Method for grouping numerical factors. Options:
            - 'kmeans': K-means clustering (requires scikit-learn)
            - 'quantile': Quantile-based grouping
            - 'jenks': Jenks Natural Breaks (requires jenkspy)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with within-group and cross-group predictions
            
        Notes:
        ------
        - Automatically handles both regression and multiclass based on config.is_multiclass
        - Performs both within-group and cross-group predictions
        - Uses appropriate uncertainty metrics (uncertainty90 for regression, uncertainty95 for multiclass)
        """
        print(f"Starting unified group-based prediction analysis for factor: {key_factor}")
        
        # Check if key factor exists in the dataset
        if key_factor not in data.columns:
            raise ValueError(f"Key factor '{key_factor}' not found in dataset")
        
        # Process numerical key factors
        processed_data = data.copy()
        if factor_type.lower() == 'numerical':
            print(f"Processing numerical key factor: {key_factor}")
            processed_data, key_factor = ModelPredictor._group_numerical_factor(
                processed_data, key_factor, num_classes, group_method
            )
            if processed_data is None or key_factor is None:
                raise ValueError("Grouping failed, cannot continue analysis")
        
        # Get unique factor values
        factor_types = processed_data[key_factor].unique()
        print(f"Dataset contains {len(factor_types)} types of {key_factor}")
        
        # Initialize prediction columns based on problem type
        if config.is_multiclass:
            processed_data['predicted_within'] = processed_data.apply(lambda x: [], axis=1)
            processed_data['predicted_cross'] = processed_data.apply(lambda x: [], axis=1)
        else:
            processed_data['predicted_within'] = np.nan
            processed_data['predicted_cross'] = np.nan
        
        processed_data['uncertainty_within'] = np.nan
        processed_data['uncertainty_cross'] = np.nan
        
        # 1. Perform within-group prediction
        print(f"Performing within-group predictions for {key_factor}...")
        for factor_value in factor_types:
            print(f"Processing {key_factor}: {factor_value}")
            group_data = processed_data[processed_data[key_factor] == factor_value].copy()
            
            if len(group_data) < config.min_samples:
                print(f"Warning: {key_factor} = {factor_value} has insufficient samples, skipping")
                continue
            
            for i in tqdm(group_data.index, desc=f"Within-group: {factor_value}", ncols=100):
                test_sample = processed_data.loc[[i]]
                train_indices = group_data.index.difference([i])
                train_samples = processed_data.loc[train_indices]
                
                dynamic_threshold = max(0.001, min(1.0, config.min_samples / len(train_samples)))
                
                gos_result = geographic_optimal_similarity(
                    target_variable=config.target_variable,
                    feature_variables=config.feature_variables,
                    training_data=train_samples,
                    prediction_data=test_sample,
                    categorical_features=config.categorical_features,
                    similarity_threshold=dynamic_threshold,
                    min_similarity=config.min_similarity
                )
                
                if config.is_multiclass:
                    processed_data.at[i, 'predicted_within'] = gos_result['prediction'].iloc[0]
                    processed_data.at[i, 'uncertainty_within'] = gos_result['uncertainty95'].iloc[0]
                else:
                    processed_data.loc[i, 'predicted_within'] = gos_result['prediction'].iloc[0]
                    processed_data.loc[i, 'uncertainty_within'] = gos_result['uncertainty90'].iloc[0]
        
        # 2. Perform cross-group prediction
        print(f"Performing cross-group predictions for {key_factor}...")
        for i in tqdm(range(len(processed_data)), desc="Cross-group prediction", ncols=100):
            test_sample = processed_data.iloc[[i]]
            current_factor = test_sample[key_factor].iloc[0]
            
            train_samples = processed_data[processed_data[key_factor] != current_factor]
            
            if len(train_samples) < config.min_samples:
                continue
            
            dynamic_threshold = max(0.001, min(1.0, config.min_samples / len(train_samples)))
            
            gos_result = geographic_optimal_similarity(
                target_variable=config.target_variable,
                feature_variables=config.feature_variables,
                training_data=train_samples,
                prediction_data=test_sample,
                categorical_features=config.categorical_features,
                similarity_threshold=dynamic_threshold,
                min_similarity=config.min_similarity
            )
            
            if config.is_multiclass:
                processed_data.at[i, 'predicted_cross'] = gos_result['prediction'].iloc[0]
                processed_data.at[i, 'uncertainty_cross'] = gos_result['uncertainty95'].iloc[0]
            else:
                processed_data.loc[i, 'predicted_cross'] = gos_result['prediction'].iloc[0]
                processed_data.loc[i, 'uncertainty_cross'] = gos_result['uncertainty90'].iloc[0]
        
        return processed_data
    
    @staticmethod
    def _group_numerical_factor(data: pd.DataFrame, factor_name: str, 
                               num_classes: int, method: str) -> Tuple[pd.DataFrame, str]:
        """
        Group numerical factors into categorical groups using various methods.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data containing the numerical factor to be grouped
        factor_name : str
            Name of the numerical factor column to be grouped
        num_classes : int
            Number of groups to create
        method : str
            Grouping method to use. Options:
            - 'kmeans': K-means clustering (requires scikit-learn)
            - 'quantile': Quantile-based grouping
            - 'jenks': Jenks Natural Breaks (requires jenkspy)
        
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            Tuple containing (grouped_data, new_factor_name)
        
        Raises:
        -------
        ImportError
            If required libraries are not installed
        ValueError
            If method is not supported or grouping fails
        """
        print(f"Grouping numerical factor '{factor_name}' using {method} method")
        
        # Check input data
        if factor_name not in data.columns:
            raise ValueError(f"Factor '{factor_name}' not found in dataset")
        
        # Ensure factor is numerical
        if not np.issubdtype(data[factor_name].dtype, np.number):
            print(f"Warning: Factor '{factor_name}' is not numerical, attempting conversion...")
            try:
                data[factor_name] = pd.to_numeric(data[factor_name])
            except:
                raise ValueError(f"Cannot convert '{factor_name}' to numerical")
        
        result_data = data.copy()
        grouped_factor_name = f'{factor_name}_group'
        
        method = method.lower()
        
        if method == 'kmeans':
            # Import K-means
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError("scikit-learn is required for K-means grouping. Install with: pip install scikit-learn")
            
            # K-means clustering
            X = result_data[factor_name].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_classes, random_state=42)
            result_data[grouped_factor_name] = kmeans.fit_predict(X)
            
            # Sort by mean values for better interpretability
            group_means = result_data.groupby(grouped_factor_name)[factor_name].mean().reset_index()
            group_means = group_means.sort_values(factor_name)
            
            mapping = {old_group: f'Group_{i+1}' for i, old_group in enumerate(group_means[grouped_factor_name])}
            result_data[grouped_factor_name] = result_data[grouped_factor_name].map(mapping)
            
        elif method == 'quantile':
            # Quantile-based grouping
            quantiles = np.linspace(0, 1, num_classes + 1)
            breaks = np.quantile(result_data[factor_name].dropna(), quantiles)
            breaks = np.unique(breaks)
            
            if len(breaks) < num_classes + 1:
                print(f"Warning: Cannot create {num_classes} unique quantiles. Using {len(breaks) - 1} groups.")
            
            labels = [f'Group_{i+1}' for i in range(len(breaks) - 1)]
            result_data[grouped_factor_name] = pd.cut(
                result_data[factor_name],
                bins=breaks,
                labels=labels,
                include_lowest=True
            )
            
        elif method == 'jenks':
            # Jenks Natural Breaks
            try:
                import jenkspy
            except ImportError:
                raise ImportError("jenkspy is required for Jenks Natural Breaks. Install with: pip install jenkspy")
            
            # Extract values and remove NaN
            values = result_data[factor_name].dropna().tolist()
            if not values:
                raise ValueError(f"Factor '{factor_name}' has no valid numerical values")
            
            try:
                # jenkspy.jenks_breaks returns num_classes + 1 breakpoint values
                breaks = jenkspy.jenks_breaks(values, n_classes=num_classes)
                
                # Use pd.cut to group according to breakpoints
                labels = [f'Group_{i+1}' for i in range(len(breaks) - 1)]
                result_data[grouped_factor_name] = pd.cut(
                    result_data[factor_name],
                    bins=breaks,
                    labels=labels,
                    include_lowest=True,
                    right=True
                )
            except Exception as e:
                print(f"Warning: Jenks Natural Breaks failed: {e}")
                print("Falling back to K-means method...")
                return ModelPredictor._group_numerical_factor(data, factor_name, num_classes, 'kmeans')
        
        else:
            raise ValueError(f"Unsupported grouping method: '{method}'. Supported methods: 'kmeans', 'quantile', 'jenks'")
        
        # Print statistics for each group
        print(f"\n{method.capitalize()} grouping results:")
        for group in sorted(result_data[grouped_factor_name].unique()):
            group_data = result_data[result_data[grouped_factor_name] == group]
            if not group_data.empty:
                print(f"  {group}: {len(group_data)} samples, mean = {group_data[factor_name].mean():.2f}, "
                      f"min = {group_data[factor_name].min():.2f}, max = {group_data[factor_name].max():.2f}")
            else:
                print(f"  {group}: 0 samples (empty group)")
        
        return result_data, grouped_factor_name
    



class ResultEvaluator:
    """Handles result evaluation and metrics calculation"""
    
    @staticmethod
    def evaluate_regression_results(data: pd.DataFrame, target_var: str, 
                                  prediction_col: str = 'predicted') -> Dict[str, Any]:
        """Evaluate regression prediction results"""
        valid_data = data[data[prediction_col].notna()]
        
        if len(valid_data) == 0:
            return {"error": "No valid predictions found"}
        
        metrics = calculate_metrics(
            valid_data[target_var].to_numpy(),
            valid_data[prediction_col].to_numpy()
        )
        
        return metrics
    
    @staticmethod
    def evaluate_multiclass_results(data: pd.DataFrame, target_var: str,
                                  prediction_col: str = 'predicted', top_k: int = 5) -> Dict[str, Any]:
        """Evaluate multiclass prediction results"""
        valid_data = data[data[prediction_col].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        
        if len(valid_data) == 0:
            return {"error": "No valid predictions found"}
        
        evaluation_results = evaluate_multiclass_prediction(
            actual_classes=valid_data[target_var].tolist(),
            predicted_classes=valid_data[prediction_col].tolist(),
            k_value=top_k
        )
        
        # Calculate random baseline
        random_baseline = calculate_random_f1_baseline(
            actual_classes=valid_data[target_var].tolist(),
            k=top_k
        )
        
        evaluation_results.update(random_baseline)
        return evaluation_results
    
    @staticmethod
    def print_evaluation_results(metrics: Dict[str, Any], analysis_type: str = "Regression"):
        """Print evaluation results in a formatted way"""
        print(f"\n{analysis_type} Evaluation Results:")
        print("=" * 50)
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


class ResultSaver:
    """Handles saving results to files"""
    
    @staticmethod
    def save_results(data: pd.DataFrame, config: AnalysisConfig, 
                    analysis_name: str, file_format: str = 'csv') -> str:
        """Save analysis results to file"""
        os.makedirs(config.output_dir, exist_ok=True)
        
        if file_format.lower() == 'csv':
            output_path = os.path.join(config.output_dir, f"{analysis_name}_results.csv")
            data.to_csv(output_path, index=False)
        elif file_format.lower() == 'excel':
            output_path = os.path.join(config.output_dir, f"{analysis_name}_results.xlsx")
            data.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    @staticmethod
    def save_significance_results(test_results: Dict[str, Any], key_factor: str, 
                                output_dir: str) -> str:
        """Save significance test results"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"significance_test_{key_factor}.json")
        
        # Convert non-serializable objects to strings
        serializable_results = {}
        for test_type, results in test_results.items():
            if isinstance(results, dict):
                serializable_results[test_type] = {
                    k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v
                    for k, v in results.items()
                }
            else:
                serializable_results[test_type] = str(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        
        print(f"Significance test results saved to: {output_path}")
        return output_path 


class GeographicTransferAnalyzer:
    """Main analyzer class that orchestrates the entire analysis process"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_predictor = ModelPredictor()
        self.result_evaluator = ResultEvaluator()
        self.result_saver = ResultSaver()
    
    def run_basic_analysis(self, config: AnalysisConfig, 
                          apply_log_transform: bool = True) -> AnalysisResult:
        """
        Run basic leave-one-out analysis using the unified framework.
        
        Parameters:
        -----------
        config : AnalysisConfig
            Configuration object containing all analysis parameters
        apply_log_transform : bool, default=True
            Whether to apply logarithmic transformation to the target variable
            (only applies to regression problems)
        
        Returns:
        --------
        AnalysisResult
            Object containing predictions, metrics, processing time, and config
        
        Notes:
        ------
        - For regression problems: applies log transformation and uses regression metrics
        - For multiclass problems: uses multiclass metrics with TOP-K evaluation
        - Automatically saves results to files
        """
        print(f"Starting basic analysis for {config.target_variable}")
        
        # Load and preprocess data
        data = self.data_processor.load_data(config)
        processed_data = self.data_processor.preprocess_data(
            data, config, apply_log_transform
        )
        
        # Perform prediction
        result_data, processing_time = self.model_predictor.leave_one_out_prediction(
            processed_data, config
        )
        
        # Evaluate results based on problem type
        if config.is_multiclass:
            metrics = self.result_evaluator.evaluate_multiclass_results(
                result_data, config.target_variable, top_k=config.top_k
            )
        else:
            metrics = self.result_evaluator.evaluate_regression_results(
                result_data, config.target_variable
            )
        
        # Save results
        self.result_saver.save_results(result_data, config, "basic_analysis")
        
        return AnalysisResult(
            predictions=result_data,
            metrics=metrics,
            processing_time=processing_time,
            config=config
        )
    
    def run_group_analysis(self, config: AnalysisConfig, key_factor: str,
                          factor_type: str = 'categorical', num_classes: int = 3,
                          group_method: str = 'kmeans') -> AnalysisResult:
        """
        Run unified group-based analysis to compare within-group vs cross-group performance.
        
        This function automatically handles both regression and multiclass problems
        based on the configuration.
        
        Parameters:
        -----------
        config : AnalysisConfig
            Configuration object containing all analysis parameters
        key_factor : str
            Name of the factor to use for grouping (must exist in dataset)
        factor_type : str, default='categorical'
            Type of the key factor. Options: 'categorical', 'numerical'
        num_classes : int, default=3
            Number of groups to create (only used for numerical factors)
        group_method : str, default='kmeans'
            Method for grouping numerical factors. Options:
            - 'kmeans': K-means clustering (requires scikit-learn)
            - 'quantile': Quantile-based grouping
            - 'jenks': Jenks Natural Breaks (requires jenkspy)
        
        Returns:
        --------
        AnalysisResult
            Object containing predictions, metrics, processing time, and config
        
        Notes:
        ------
        - Automatically detects problem type (regression vs multiclass) from config
        - Performs both within-group and cross-group predictions
        - Uses appropriate evaluation metrics based on problem type
        - Compares performance between same-group and different-group predictions
        - Automatically runs significance tests if available
        - Saves results to files with group-specific naming
        """
        print(f"Starting unified group analysis for factor: {key_factor}")
        
        # Load and preprocess data
        data = self.data_processor.load_data(config)
        processed_data = self.data_processor.preprocess_data(data, config)
        
        # Perform group-based prediction using ModelPredictor
        result_data = self.model_predictor.group_based_prediction(
            processed_data, config, key_factor, factor_type, num_classes, group_method
        )
        
        # 3. Evaluate results based on problem type
        if config.is_multiclass:
            # Multiclass evaluation
            valid_within_indices = result_data[result_data['predicted_within'].apply(lambda x: len(x) > 0)].index
            valid_cross_indices = result_data[result_data['predicted_cross'].apply(lambda x: len(x) > 0)].index
            
            within_metrics = None
            cross_metrics = None
            
            if len(valid_within_indices) > 0:
                within_metrics = self.result_evaluator.evaluate_multiclass_results(
                    result_data.loc[valid_within_indices], config.target_variable, 'predicted_within', config.top_k
                )
            
            if len(valid_cross_indices) > 0:
                cross_metrics = self.result_evaluator.evaluate_multiclass_results(
                    result_data.loc[valid_cross_indices], config.target_variable, 'predicted_cross', config.top_k
                )
        else:
            # Regression evaluation
            within_metrics = self.result_evaluator.evaluate_regression_results(
                result_data, config.target_variable, 'predicted_within'
            )
            cross_metrics = self.result_evaluator.evaluate_regression_results(
                result_data, config.target_variable, 'predicted_cross'
            )
        
        # Combine metrics
        metrics = {
            'within_group': within_metrics,
            'cross_group': cross_metrics
        }
        
        # Save results
        self.result_saver.save_results(result_data, config, f"{key_factor}_group_analysis")
        
        # Run significance tests if available
        try:
            if config.is_multiclass:
                test_results = run_significance_tests(
                    result_data, key_factor, target_variable=config.target_variable, k=config.top_k
                )
            else:
                test_results = run_significance_tests(
                    result_data, key_factor, target_variable=config.target_variable, regression=True
                )
            
            if test_results:
                self.result_saver.save_significance_results(
                    test_results, key_factor, config.output_dir
                )
        except Exception as e:
            print(f"Warning: Could not run significance tests: {e}")
        
        return AnalysisResult(
            predictions=result_data,
            metrics=metrics,
            processing_time=0.0,  # Not tracked for group analysis
            config=config
        )
    



# Predefined configurations for different analysis types
SOIL_CONFIG = AnalysisConfig(
    data_path="../data/soil_train.csv",
    target_variable="OC",
    feature_variables=[
        'LU', 'clay_S', 'clay_D', 'sand_S', 'sand_D', 'Elevation', 'Slope', 
        'Avg_Tem', 'Avg_Pec', 'Hig_Tem', 'Low_Tem', 'Dif_Tem'
    ],
    categorical_features=['LU'],
    similarity_threshold=0.001,
    min_similarity=0.1,
    min_samples=20
)

CRIME_CONFIG = AnalysisConfig(
    data_path="../data/community_crime_stats.xlsx",
    target_variable="CrimeRate",
    feature_variables=[
        "MedInc", "LowIncPct", "HousCostBurden", "SingleParentPct", "YoungAdultPct",
        "YouthPct", "NoHSPct", "RentPct", "VacancyPct", "UnempRate", "NotInLF",
        "ForeignBornPct", "NonEngPct", "ComLandPct", "IndLandPct", "NoCarPct", "PubTransPct"
    ],
    categorical_features=[],
    similarity_threshold=0.1,
    min_similarity=0.01,
    min_samples=5
)

SDP_CONFIG = AnalysisConfig(
    data_path="../data/region.xlsx",  # Not used for multiclass
    target_variable="patterns",
    feature_variables=[
        'precipitation', 'average_altitude', 'cultivated_area', 'grass_coverage',
        'water_per', 'forest_coverage', 'gdp_total', 'gdp_per', 'saving_per',
        'second_output', 'third_output', 'water_quality', 'soil_erosion',
        'all_habitats', 'nature_area', 'nature_risk', 'density_high',
        'number_health', 'number_three', 'number_chl'
    ],
    categorical_features=[],
    similarity_threshold=0.01,
    min_similarity=0.01,
    min_samples=10,
    # Multiclass configuration
    is_multiclass=True,
    multiclass_data_paths=['../data/region.xlsx', '../data/ecpr_case.xlsx'],
    top_k=5
)


# Example of how to create a custom multiclass configuration
def create_multiclass_config(data_paths: List[str], target_variable: str, 
                           feature_variables: List[str], top_k: int = 5) -> AnalysisConfig:
    """Create a custom multiclass configuration"""
    return AnalysisConfig(
        data_path=data_paths[0],  # Not used for multiclass
        target_variable=target_variable,
        feature_variables=feature_variables,
        categorical_features=[],
        similarity_threshold=0.01,
        min_similarity=0.01,
        min_samples=10,
        # Multiclass configuration
        is_multiclass=True,
        multiclass_data_paths=data_paths,
        top_k=top_k
    )


def run_soil_analysis():
    """
    Run soil analysis using the optimized framework.
    
    Returns:
    --------
    AnalysisResult
        Object containing soil analysis results with regression metrics
        
    Notes:
    ------
    - Uses predefined SOIL_CONFIG
    - Applies logarithmic transformation to OC values
    - Performs leave-one-out prediction
    - Saves results to '../results/basic_analysis_results.csv'
    """
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_basic_analysis(SOIL_CONFIG)
    
    # Print results
    ResultEvaluator.print_evaluation_results(result.metrics, "Soil Analysis")
    print(
        f"Prediction success, please run visualization.py for viewing and evaluate_transferbility.py to get the evaluation results...")
    return result


def run_crime_analysis():
    """
    Run crime analysis using the optimized framework.
    
    Returns:
    --------
    AnalysisResult
        Object containing crime analysis results with regression metrics
        
    Notes:
    ------
    - Uses predefined CRIME_CONFIG
    - Applies logarithmic transformation to CrimeRate values
    - Performs leave-one-out prediction
    - Saves results to '../results/basic_analysis_results.csv'
    """
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_basic_analysis(CRIME_CONFIG)
    
    # Print results
    ResultEvaluator.print_evaluation_results(result.metrics, "Crime Analysis")
    print(
        f"Prediction success, please run visualization.py for viewing and evaluate_transferbility.py to get the evaluation results...")
    return result


def run_sdp_analysis():
    """
    Run sustainable development pathways recommendation analysis using the optimized framework.
    
    Returns:
    --------
    AnalysisResult
        Object containing SDP analysis results with multiclass metrics
        
    Notes:
    ------
    - Uses predefined SDP_CONFIG (multiclass configuration)
    - No log transformation applied (multiclass problem)
    - Performs leave-one-out prediction with TOP-K evaluation
    - Saves results to '../results/basic_analysis_results.csv'
    """
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_basic_analysis(SDP_CONFIG, apply_log_transform=False)
    
    # Print results
    ResultEvaluator.print_evaluation_results(result.metrics, "SDP Analysis")
    print(
        f"Prediction success, please run evaluate_transferbility.py to get the evaluation results...")
    return result


def run_soil_key_factor_analysis(key_factor='clay_S', factor_type='numerical', 
                                num_classes=3, group_method='kmeans'):
    """
    Run soil analysis with key factor grouping.
    
    Parameters:
    -----------
    key_factor : str, default='clay_S'
        Name of the factor to use for grouping (must exist in soil dataset)
    factor_type : str, default='numerical'
        Type of the key factor. Options: 'categorical', 'numerical'
    num_classes : int, default=3
        Number of groups to create (only used for numerical factors)
    group_method : str, default='kmeans'
        Method for grouping numerical factors. Options:
        - 'kmeans': K-means clustering (requires scikit-learn)
        - 'quantile': Quantile-based grouping
        - 'jenks': Jenks Natural Breaks (requires jenkspy)
    
    Returns:
    --------
    AnalysisResult
        Object containing soil analysis results with group comparison metrics
        
    Notes:
    ------
    - Uses predefined SOIL_CONFIG
    - Compares within-group vs cross-group performance
    - Prints comparison results (within - cross differences)
    - Saves results to '../results/{key_factor}_group_analysis_results.csv'
    """
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_group_analysis(
        SOIL_CONFIG, key_factor, factor_type, num_classes, group_method
    )
    
    # Print comparison results
    if result.metrics['within_group'] and result.metrics['cross_group']:
        print("\nComparison Results (Within - Cross):")
        within = result.metrics['within_group']
        cross = result.metrics['cross_group']
        
        for metric in ['rmse', 'mae', 'r2', 'correlation', 'mape']:
            if metric in within and metric in cross:
                diff = within[metric] - cross[metric]
                print(f"  {metric.upper()} difference: {diff:.4f}")
    
    return result


def run_crime_key_factor_analysis(key_factor='RentPct', factor_type='numerical',
                                 num_classes=3, group_method='kmeans'):
    """
    Run crime analysis with key factor grouping.
    
    Parameters:
    -----------
    key_factor : str, default='RentPct'
        Name of the factor to use for grouping (must exist in crime dataset)
    factor_type : str, default='numerical'
        Type of the key factor. Options: 'categorical', 'numerical'
    num_classes : int, default=3
        Number of groups to create (only used for numerical factors)
    group_method : str, default='kmeans'
        Method for grouping numerical factors. Options:
        - 'kmeans': K-means clustering (requires scikit-learn)
        - 'quantile': Quantile-based grouping
        - 'jenks': Jenks Natural Breaks (requires jenkspy)
    
    Returns:
    --------
    AnalysisResult
        Object containing crime analysis results with group comparison metrics
        
    Notes:
    ------
    - Uses predefined CRIME_CONFIG
    - Compares within-group vs cross-group performance
    - Prints comparison results (within - cross differences)
    - Saves results to '../results/{key_factor}_group_analysis_results.csv'
    """
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_group_analysis(
        CRIME_CONFIG, key_factor, factor_type, num_classes, group_method
    )
    
    # Print comparison results
    if result.metrics['within_group'] and result.metrics['cross_group']:
        print("\nComparison Results (Within - Cross):")
        within = result.metrics['within_group']
        cross = result.metrics['cross_group']
        
        for metric in ['rmse', 'mae', 'r2', 'correlation', 'mape']:
            if metric in within and metric in cross:
                diff = within[metric] - cross[metric]
                print(f"  {metric.upper()} difference: {diff:.4f}")
    
    return result


def run_sdp_key_factor_analysis(key_factor='climate_type_final', factor_type='categorical',
                                num_classes=4, group_method='kmeans', top_k=5):
    """
    Run SDP analysis with key factor grouping for multiclass classification.
    
    Parameters:
    -----------
    key_factor : str, default='climate_type_final'
        Name of the factor to use for grouping (must exist in SDP dataset)
    factor_type : str, default='categorical'
        Type of the key factor. Options: 'categorical', 'numerical'
    num_classes : int, default=4
        Number of groups to create (only used for numerical factors)
    group_method : str, default='kmeans'
        Method for grouping numerical factors. Options:
        - 'kmeans': K-means clustering (requires scikit-learn)
        - 'quantile': Quantile-based grouping
        - 'jenks': Jenks Natural Breaks (requires jenkspy)
    top_k : int, default=5
        Number of top predictions to evaluate for multiclass metrics
    
    Returns:
    --------
    AnalysisResult
        Object containing SDP analysis results with group comparison metrics
        
    Notes:
    ------
    - Uses predefined SDP_CONFIG (multiclass configuration)
    - Compares within-group vs cross-group performance using TOP-K metrics
    - Prints comparison results (within - cross differences)
    - Saves results to '../results/prediction_multiclass_with_groups_{key_factor}.xlsx'
    """
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_group_analysis(
        SDP_CONFIG, key_factor, factor_type, num_classes, group_method
    )
    
    # Print comparison results
    if result.metrics['within_group'] and result.metrics['cross_group']:
        print(f"\nSDP Comparison Results (Within - Cross) for {key_factor}:")
        within = result.metrics['within_group']
        cross = result.metrics['cross_group']
        
        for metric in [f'top_{top_k}_accuracy', f'jaccard_at_{top_k}', 
                      f'precision@{top_k}', f'recall@{top_k}', f'f1@{top_k}',
                      'at_least_one_correct_rate']:
            if metric in within and metric in cross:
                diff = within[metric] - cross[metric]
                print(f"  {metric} difference: {diff:.4f}")
    
    return result


def find_optimal_kappa():
    """Find optimal similarity threshold for soil analysis"""
    print("Finding optimal similarity threshold...")

    # Load and preprocess data
    soil_data = loadData('../data/soil_train.csv')
    soil_data['OC'] = np.log(soil_data['OC'])

    outlier_indices = remove_outliers(soil_data['OC'], std_multiplier=2.5)
    cleaned_data = soil_data.drop(outlier_indices).reset_index(drop=True)

    # Find optimal threshold
    optimal_results = find_optimal_similarity_threshold(
        target_variable='OC',
        feature_variables=['clay_S', 'clay_D', 'sand_S', 'sand_D', 'silt_S', 'silt_D',
                           'Elevation', 'Slope', 'Aspect', 'Avg_Tem', 'Avg_Pec',
                           'Hig_Tem', 'Low_Tem', 'Dif_Tem'],
        data=cleaned_data,
        categorical_features=[],
        threshold_candidates=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
        num_cores=6
    )

    print(f"Optimal similarity threshold: {optimal_results['optimal_threshold']:.4f}")
    optimal_results['visualization'].show()

    return optimal_results


def run_multiclass_key_factor_analysis(config: AnalysisConfig, key_factor: str,
                                      factor_type: str = 'categorical', num_classes: int = 3,
                                      group_method: str = 'kmeans'):
    """
    Run multiclass analysis with key factor grouping using any multiclass configuration.
    
    Parameters:
    -----------
    config : AnalysisConfig
        Multiclass configuration object (must have is_multiclass=True)
    key_factor : str
        Name of the factor to use for grouping (must exist in dataset)
    factor_type : str, default='categorical'
        Type of the key factor. Options: 'categorical', 'numerical'
    num_classes : int, default=3
        Number of groups to create (only used for numerical factors)
    group_method : str, default='kmeans'
        Method for grouping numerical factors. Options:
        - 'kmeans': K-means clustering (requires scikit-learn)
        - 'quantile': Quantile-based grouping
        - 'jenks': Jenks Natural Breaks (requires jenkspy)
    
    Returns:
    --------
    AnalysisResult
        Object containing multiclass analysis results with group comparison metrics
        
    Notes:
    ------
    - Generic function for any multiclass configuration
    - Compares within-group vs cross-group performance using TOP-K metrics
    - Prints comparison results (within - cross differences)
    - Saves results to '../results/prediction_multiclass_with_groups_{key_factor}.xlsx'
    """
    if not config.is_multiclass:
        raise ValueError("Configuration must be set for multiclass analysis (is_multiclass=True)")
    
    analyzer = GeographicTransferAnalyzer()
    result = analyzer.run_group_analysis(
        config, key_factor, factor_type, num_classes, group_method
    )
    
    # Print comparison results
    if result.metrics['within_group'] and result.metrics['cross_group']:
        print(f"\nMulticlass Comparison Results (Within - Cross) for {key_factor}:")
        within = result.metrics['within_group']
        cross = result.metrics['cross_group']
        
        top_k = config.top_k
        for metric in [f'top_{top_k}_accuracy', f'jaccard_at_{top_k}', 
                      f'precision@{top_k}', f'recall@{top_k}', f'f1@{top_k}',
                      'at_least_one_correct_rate']:
            if metric in within and metric in cross:
                diff = within[metric] - cross[metric]
                print(f"  {metric} difference: {diff:.4f}")
    
    return result


if __name__ == "__main__":
    # Run basic analyses
    print("Running basic analyses...")
    soil_result = run_soil_analysis()
    # crime_result = run_crime_analysis()
    # sdp_result = run_sdp_analysis()
    
    # Run key factor analyses using unified framework
    # print("\nRunning key factor analyses...")
    # soil_key_result = run_soil_key_factor_analysis('climate_type', 'categorical')
    # crime_key_result = run_crime_key_factor_analysis('TotIncome', 'numerical', 3, 'kmeans')
    # sdp_key_result = run_sdp_key_factor_analysis('climate_type_final', 'categorical', top_k=5)
    
    # Example of using unified framework for any multiclass configuration
    # custom_multiclass_config = create_multiclass_config(
    #     data_paths=['../data/region.xlsx', '../data/ecpr_case.xlsx'],
    #     target_variable='patterns',
    #     feature_variables=['precipitation', 'average_altitude'],
    #     top_k=3
    # )
    # custom_multiclass_result = run_multiclass_key_factor_analysis(
    #     custom_multiclass_config, 'climate_type_final', 'categorical', 3, 'jenks'
    # )
    
    print("\nAll analyses completed successfully!") 