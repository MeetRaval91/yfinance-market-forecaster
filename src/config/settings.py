"""
Global configuration settings for the Stock Prediction System.
"""

# ML Split settings
TRAIN_RATIO = 0.70
TEST_RATIO = 0.30

# Cross-validation
CV_SPLITS = 5

# Reproducibility
RANDOM_STATE = 42

# Default Data Settings
DEFAULT_DATA_PERIOD = "5y"

# Streamlit Caching
CACHE_TTL = 3600  # 1 hour
