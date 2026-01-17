"""
Utility functions for UIDAI Hackathon
Shared across all scripts for consistent data handling
"""

import pandas as pd
import numpy as np


def get_numeric_features(df, exclude_patterns=None):
    """
    Extract only numeric features from dataframe
    Automatically excludes IDs and target columns

    Args:
        df: pandas DataFrame
        exclude_patterns: list of strings to exclude from column names

    Returns:
        list of column names (numeric features only)
    """

    # Default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = [
            'id', 'ID', 'Id',
            '_id', '_ID',
            'enrolment_id', 'operator_id', 'officer_id',
            'fraud', 'anomaly', 'risk',
            'label', 'target', 'class',
            'iso_forest', 'autoencoder', 'combined',
            'xgboost', 'random_forest', 'hybrid',
            'prediction', 'score', 'probability'
        ]

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter out excluded columns
    feature_cols = []
    for col in numeric_cols:
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern.lower() in col.lower():
                should_exclude = True
                break

        if not should_exclude:
            feature_cols.append(col)

    return feature_cols


def clean_numeric_data(X):
    """
    Clean numeric data for ML models
    - Replace infinity with NaN
    - Fill missing values
    - Ensure all values are float

    Args:
        X: pandas DataFrame or numpy array

    Returns:
        Cleaned DataFrame
    """

    # Convert to DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Replace infinity
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values with median
    X = X.fillna(X.median())

    # Fill any remaining NaN with 0
    X = X.fillna(0)

    # Convert to float
    X = X.astype(float)

    return X


def validate_numeric_data(X, name="Data"):
    """
    Validate that data is ready for ML models
    Raises errors if problems found

    Args:
        X: pandas DataFrame or numpy array
        name: Name for error messages
    """

    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Check for non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        raise ValueError(f"{name} contains non-numeric columns: {non_numeric}")

    # Check for NaN
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"{name} contains NaN in columns: {nan_cols}")

    # Check for infinity
    if np.isinf(X.values).any():
        inf_cols = X.columns[np.isinf(X.values).any(axis=0)].tolist()
        raise ValueError(f"{name} contains infinity in columns: {inf_cols}")

    print(f"✅ {name} validation passed: {X.shape[0]:,} × {X.shape[1]}")
