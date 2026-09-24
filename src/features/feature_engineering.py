import pandas as pd
from typing import Tuple

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the feature dataset from the validated stock data.
    Adds simple, explainable lag features.
    
    Args:
        df: Validated Pandas DataFrame containing stock data.
        
    Returns:
        DataFrame containing engineered features.
    """
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Basic lag features (prevent future data leakage)
    features['Prev_Close'] = features['Close'].shift(1)
    features['Prev_Open'] = features['Open'].shift(1)
    features['Prev_High'] = features['High'].shift(1)
    features['Prev_Low'] = features['Low'].shift(1)
    features['Prev_Volume'] = features['Volume'].shift(1)
    
    return features

def prepare_target(df: pd.DataFrame) -> pd.Series:
    """
    Prepares the prediction target.
    Target = Next Trading Day Close
    
    Args:
        df: Validated Pandas DataFrame.
        
    Returns:
        Series containing the target variable.
    """
    # Shift -1 aligns tomorrow's close with today's features
    target = df['Close'].shift(-1)
    target.name = 'Target_Next_Close'
    return target

def prepare_ml_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Coordinates feature engineering and target creation.
    Cleans up missing values caused by shift operations.
    
    Args:
        df: Validated Pandas DataFrame.
        
    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series.
    """
    X = prepare_features(df)
    y = prepare_target(df)
    
    # Combine temporarily to drop rows with NaNs in either features or target
    # Lagging (shift 1) creates a NaN on the very first row.
    # The target (shift -1) creates a NaN on the very last row.
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    
    X_clean = combined[X.columns]
    y_clean = combined[y.name]
    
    return X_clean, y_clean
