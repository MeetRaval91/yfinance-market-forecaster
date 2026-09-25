import pandas as pd
import logging
logger = logging.getLogger(__name__)
class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass
def validate_stock_data(df: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    """
    Validates the downloaded stock data.
    Args:
        df: Pandas DataFrame containing stock data.
        min_rows: Minimum number of required historical observations.
    Returns:
        Validated Pandas DataFrame.
    Raises:
        DataValidationError: If data is invalid, empty, or missing columns.
    """
    if df is None or df.empty:
        raise DataValidationError("The retrieved dataset is empty. Check if the ticker is valid.")
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_cols)}")
    if len(df) < min_rows:
        raise DataValidationError(f"Insufficient data: {len(df)} rows. Minimum required is {min_rows}.")
    if df[required_columns].isnull().values.any():
        logger.warning("Missing values detected in stock data. Applying forward fill.")
        df[required_columns] = df[required_columns].ffill()
        df = df.dropna(subset=required_columns)
        if df.empty or len(df) < min_rows:
            raise DataValidationError("Insufficient data after handling missing values.")
    return df
