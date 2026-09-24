import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the Project")

st.markdown("""
### Project Purpose
This application is an academic machine learning project designed to demonstrate a complete, end-to-end supervised learning workflow using historical stock market data. The primary objective is to predict the **Next Trading Day's Closing Price** and derive the expected market direction (UP/DOWN).

### Architecture & Methodology
The project utilizes a **Modular Monolithic Architecture** built entirely in Python, using Streamlit for the application layer.

**1. Data Retrieval & Validation**
Market data is sourced from Yahoo Finance (`yfinance`). The dataset is strictly validated to ensure required features are present and missing values are handled without inducing data leakage.

**2. Feature Engineering**
The system uses a simple, explainable feature set consisting of `Open`, `High`, `Low`, `Close`, `Volume`, and their respective 1-day lags. To strictly prevent future data leakage, the target variable (Next Day Close) is created by shifting the close price backwards, ensuring that a model predicting "tomorrow" only ever has access to data from "today" and before.

**3. Machine Learning Models**
As per the academic requirements, this project evaluates exactly three regression models:
* **Support Vector Regression (SVR)**
* **Ridge Regression**
* **HistGradientBoostingRegressor**

**4. Training & Evaluation**
The dataset undergoes a strict chronological `70/30` Train/Test split. Random shuffling is strictly prohibited to respect the time-series nature of financial data. Models are evaluated on the test set using `RMSE`, `RSS`, and `R²`.

**5. Cross-Validation**
The best-performing model on the test set (lowest RMSE) is selected automatically and undergoes a rigorous **5-Fold TimeSeriesSplit** cross-validation to prove its stability across different historical timeframes.
""")
