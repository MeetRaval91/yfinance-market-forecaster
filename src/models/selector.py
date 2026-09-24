import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import logging
from src.config.settings import CV_SPLITS
from src.models.evaluator import calculate_metrics

logger = logging.getLogger(__name__)

def select_best_model(evaluation_results: Dict[str, Any]) -> str:
    """
    Selects the best model based on the lowest RMSE on the test set.
    This rule provides a transparent selection mechanism.
    
    Args:
        evaluation_results: Dictionary mapping model names to their test metrics.
        
    Returns:
        The string name of the selected best model.
    """
    # Transparent selection rule: lowest RMSE wins
    best_model_name = min(evaluation_results, key=lambda k: evaluation_results[k]['RMSE'])
    
    logger.info(f"Selected best model: {best_model_name} with RMSE: {evaluation_results[best_model_name]['RMSE']:.4f}")
    return best_model_name

def perform_cross_validation(model_pipeline: Dict[str, Any], X: pd.DataFrame, y: pd.Series, splits: int = CV_SPLITS) -> List[Dict[str, Any]]:
    """
    Performs TimeSeriesSplit cross-validation on the selected model.
    Ensures chronological order is strictly preserved during splits.
    
    Args:
        model_pipeline: Dictionary with 'model' and 'scaler' for the selected model.
        X: Full feature dataset (pre-split).
        y: Full target dataset (pre-split).
        splits: Number of time-series splits (default 5).
        
    Returns:
        List of metric dictionaries for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=splits)
    cv_results = []
    
    model = model_pipeline['model']
    scaler = model_pipeline['scaler']
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        
        # Fit scaler ONLY on the training fold to prevent future-data leakage
        X_train_scaled = scaler.fit_transform(X_train_cv)
        X_test_scaled = scaler.transform(X_test_cv)
        
        # Clone model to avoid bleeding learned parameters between folds
        cloned_model = clone(model)
        
        cloned_model.fit(X_train_scaled, y_train_cv)
        y_pred_cv = cloned_model.predict(X_test_scaled)
        
        metrics = calculate_metrics(y_test_cv, y_pred_cv)
        metrics['fold'] = fold + 1
        cv_results.append(metrics)
        
    return cv_results
