from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Dict, Any
from src.config.settings import RANDOM_STATE

def get_models() -> Dict[str, Any]:
    """
    Returns an instantiated dictionary of the strictly required 
    machine learning models for this academic project.
    
    Models included:
    1. SVR
    2. Ridge
    3. HistGradientBoostingRegressor
    
    No other models should be added unless explicitly approved.
    """
    return {
        'SVR': SVR(),
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    }
