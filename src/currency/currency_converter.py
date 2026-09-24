from typing import Optional
import logging
from src.currency.exchange_rate import get_exchange_rate

logger = logging.getLogger(__name__)

def convert_currency(amount: float, from_currency: str, to_currency: str = "INR") -> Optional[float]:
    """
    Converts an amount from one currency to another.
    Returns None if conversion fails instead of throwing an exception,
    so the UI can gracefully fallback to native values.
    
    Args:
        amount: The monetary value to convert.
        from_currency: Source currency code.
        to_currency: Target currency code.
        
    Returns:
        The converted amount, or None if the exchange rate could not be fetched.
    """
    if from_currency.upper() == to_currency.upper():
        return amount
        
    try:
        rate = get_exchange_rate(from_currency, to_currency)
        return amount * rate
    except Exception as e:
        logger.error(f"Currency conversion failed from {from_currency} to {to_currency}: {e}")
        return None
