import pandas as pd
import os
import numpy as np


def loadData(csv_path):
    df = pd.read_csv(csv_path)

    return df


def load_multiclass_data(features_xlsx_path, classification_xlsx_path):
    """
    Load multiclass prediction data

    Parameters:
        features_xlsx_path (str): Path to feature data xlsx file
        classification_xlsx_path (str): Path to classification records xlsx file

    Returns:
        DataFrame: DataFrame containing features and target variable (patterns), where each element in the patterns column is a list,
                  containing all unique pattern_ids corresponding to that region_id
    """
    # Load feature data
    print(f"Loading feature data: {features_xlsx_path}")
    features_df = pd.read_excel(features_xlsx_path)

    # Load classification record data
    print(f"Loading classification record data: {classification_xlsx_path}")
    patterns_df = pd.read_excel(classification_xlsx_path)

    # Check data
    if 'id' not in features_df.columns:
        raise ValueError("Feature data missing 'id' column")

    if 'pattern_id' not in patterns_df.columns or 'region_id' not in patterns_df.columns:
        raise ValueError("Classification record data missing 'pattern_id' or 'region_id' column")

    # Group classification records by region_id and merge pattern_ids for each region_id into a list
    # Use set to remove duplicate pattern_ids
    pattern_groups = patterns_df.groupby('region_id')['pattern_id'].apply(
        lambda x: list(set(x))  # Use set to remove duplicates
    ).reset_index()
    
    pattern_groups.rename(columns={'pattern_id': 'patterns'}, inplace=True)

    # Merge feature data with classification records
    merged_data = pd.merge(features_df, pattern_groups, left_on='id', right_on='region_id', how='left')

    # Handle feature data without corresponding classification records
    if merged_data['patterns'].isna().any():
        print(f"Warning: {merged_data['patterns'].isna().sum()} feature data entries have no corresponding classification records")
        # Set missing classification records to empty list
        merged_data['patterns'] = merged_data['patterns'].apply(
            lambda x: [] if isinstance(x, float) and np.isnan(x) else x)

    # Extract feature data and target variable
    X = merged_data.drop(['region_id'], axis=1, errors='ignore')

    # Count duplicate categories
    total_patterns_before = sum(patterns_df.groupby('region_id')['pattern_id'].apply(len))
    total_patterns_after = sum(X['patterns'].apply(len))
    if total_patterns_before > total_patterns_after:
        print(f"Note: {total_patterns_before - total_patterns_after} duplicate categories have been removed")

    print(f"Loading completed: Data has {X.shape[0]} rows, {X.shape[1]} columns, average {X['patterns'].apply(len).mean():.2f} classifications per sample")

    return X