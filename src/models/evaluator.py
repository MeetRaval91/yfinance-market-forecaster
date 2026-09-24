import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates evaluation metrics (RSS, RMSE, R²).
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        
    Returns:
        Dictionary of computed metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Residual Sum of Squares
    rss = np.sum((np.array(y_true) - y_pred) ** 2)
    
    return {
        'RSS': float(rss),
        'RMSE': float(rmse),
        'R2': float(r2)
    }

def evaluate_models(trained_pipeline: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates all trained models on the test set.
    
    Args:
        trained_pipeline: Dictionary containing the trained models and scaler.
        X_test: Test features.
        y_test: Test targets.
        
    Returns:
        Dictionary mapping model names to their calculated metrics.
    """
    evaluation_results = {}
    
    for name, pipeline_components in trained_pipeline.items():
        model = pipeline_components['model']
        scaler = pipeline_components['scaler']
        
        # Scale test data using the already fitted scaler (no data leakage)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        evaluation_results[name] = metrics
        
    return evaluation_results
