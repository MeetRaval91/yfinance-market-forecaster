import streamlit as st

st.set_page_config(
    page_title="Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Welcome to the Stock Prediction System")

st.markdown("""
This is an academic machine-learning application designed to forecast the **Next Trading Day's Closing Price** using historical market data.

Please use the sidebar to navigate through the application:
- **Dashboard**: Search for stocks, view latest predictions and charts.
- **Stock Analysis**: Deep dive into historical price action.
- **Model Analysis**: Review machine learning metrics and cross-validation results.
- **About**: Learn about the project architecture and ML methodology.
- **Disclaimer / Terms**: Read the terms of use.
""")
