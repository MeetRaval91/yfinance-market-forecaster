import yfinance as yf
import pandas as pd
import streamlit as st
import logging
from typing import Tuple
from dataclasses import dataclass
from src.config.settings import DEFAULT_DATA_PERIOD, CACHE_TTL
from src.data.data_validator import validate_stock_data, DataValidationError
logger = logging.getLogger(__name__)
@dataclass
class StockInfo:
    ticker: str
    company_name: str
    exchange: str
    native_currency: str
    display_currency: str = "INR"
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = DEFAULT_DATA_PERIOD) -> Tuple[pd.DataFrame, StockInfo]:
    """
    Fetches historical stock data from Yahoo Finance and its metadata.
    Args:
        ticker: The stock ticker symbol (e.g., 'RELIANCE.NS').
        period: The time period to fetch (e.g., '5y', '1y').
    Returns:
        Tuple of (Validated Pandas DataFrame, StockInfo object).
    """
    try:
        logger.info(f"Fetching data for {ticker} over period {period}")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        validated_df = validate_stock_data(df)
        info = stock.info
        currency = info.get('currency', 'USD')
        exchange = info.get('exchange', 'Unknown')
        company_name = info.get('longName', info.get('shortName', ticker))
        stock_info = StockInfo(
            ticker=ticker,
            company_name=company_name,
            exchange=exchange,
            native_currency=currency,
            display_currency="INR"
        )
        return validated_df, stock_info
    except DataValidationError as e:
        logger.error(f"Validation error for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Failed to fetch data for {ticker}")
        raise Exception(f"Failed to fetch data from Yahoo Finance for {ticker}. Check network connection or ticker symbol.")
