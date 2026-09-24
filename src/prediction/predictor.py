import pandas as pd
from typing import Dict, Any, Tuple

def extract_latest_features_for_prediction(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Extracts the latest available row of features to predict tomorrow.
    """
    from src.features.feature_engineering import prepare_features
    
    all_features = prepare_features(df_raw)
    latest_features = all_features.iloc[[-1]].copy()
    latest_close = float(df_raw['Close'].iloc[-1])
    
    return latest_features, latest_close

def predict_next_day(model_pipeline: Dict[str, Any], latest_features: pd.DataFrame, latest_close: float, native_currency: str = "Unknown") -> Dict[str, Any]:
    """
    Predicts the next trading day's closing price.
    Calculates expected direction in the native currency.
    """
    model = model_pipeline['model']
    scaler = model_pipeline['scaler']
    
    latest_features_scaled = pd.DataFrame(
        scaler.transform(latest_features), 
        columns=latest_features.columns, 
        index=latest_features.index
    )
    
    predicted_close = float(model.predict(latest_features_scaled)[0])
    
    predicted_change = predicted_close - latest_close
    predicted_change_pct = (predicted_change / latest_close) * 100
    expected_direction = "UP" if predicted_close > latest_close else "DOWN"
    
    return {
        'Latest_Close': latest_close,
        'Predicted_Close': predicted_close,
        'Predicted_Change': predicted_change,
        'Predicted_Change_Pct': predicted_change_pct,
        'Expected_Direction': expected_direction,
        'Currency': native_currency
    }

