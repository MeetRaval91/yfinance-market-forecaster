import yfinance as yf
import streamlit as st
import logging
from src.config.settings import CACHE_TTL

logger = logging.getLogger(__name__)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_exchange_rate(from_currency: str, to_currency: str = "INR") -> float:
    """
    Retrieves the exchange rate between two currencies using yfinance.
    Results are cached using Streamlit's cache_data.
    
    Args:
        from_currency: Source currency code (e.g., 'USD').
        to_currency: Target currency code (e.g., 'INR').
        
    Returns:
        The exchange rate as a float.
        
    Raises:
        ValueError: If exchange rate is not found or invalid.
    """
    if from_currency.upper() == to_currency.upper():
        return 1.0
        
    symbol = f"{from_currency.upper()}{to_currency.upper()}=X"
    try:
        logger.info(f"Fetching exchange rate for {symbol}")
        fx = yf.Ticker(symbol)
        df = fx.history(period="1d")
        
        if df.empty:
            raise ValueError(f"No exchange rate data found for {symbol}")
        
        rate = float(df['Close'].iloc[-1])
        if rate <= 0:
            raise ValueError(f"Invalid exchange rate retrieved: {rate}")
            
        return rate
    except Exception as e:
        logger.error(f"Failed to fetch exchange rate for {symbol}: {e}")
        raise
