import pandas as pd
from typing import Tuple, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler
from src.config.settings import TRAIN_RATIO

logger = logging.getLogger(__name__)

def split_data_chronologically(X: pd.DataFrame, y: pd.Series, train_ratio: float = TRAIN_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into train and test sets while preserving chronological order.
    Does not randomly shuffle the dataset.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        train_ratio: Proportion of data to use for training (default 0.70).
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    split_index = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    
    return X_train, X_test, y_train, y_test

def train_models(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Trains a dictionary of models on the training dataset.
    Applies StandardScaler to features since SVR and Ridge are sensitive to feature scales.
    
    Args:
        models: Dictionary of un-trained scikit-learn models from model_factory.
        X_train: Training features.
        y_train: Training targets.
        
    Returns:
        Dictionary containing trained models and the fitted scaler.
    """
    trained_pipeline = {}
    
    # SVR and Ridge require scaled features. HistGBR doesn't strictly need it, 
    # but applying it consistently ensures transparency and easier tracking.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    for name, model in models.items():
        logger.info(f"Training model: {name}")
        try:
            model.fit(X_train_scaled, y_train)
            trained_pipeline[name] = {
                'model': model,
                'scaler': scaler
            }
        except Exception as e:
            logger.exception(f"Failed to train model {name}")
            raise RuntimeError(f"Model training failed for {name}: {str(e)}")
            
    return trained_pipeline
