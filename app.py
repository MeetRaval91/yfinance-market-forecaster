import streamlit as st
from src.ui.sidebar import render_sidebar
st.set_page_config(
    page_title="Stock Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)
render_sidebar()
pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon="📊"),
    st.Page("pages/stock_analysis.py", title="Stock Analysis", icon="📉"),
    st.Page("pages/model_analysis.py", title="Model Analysis", icon="🤖"),
    st.Page("pages/about.py", title="About", icon="ℹ️"),
    st.Page("pages/terms.py", title="Terms & Disclaimer", icon="⚖️"),
]
pg = st.navigation(pages)
pg.run()
