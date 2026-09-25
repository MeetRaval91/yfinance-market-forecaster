import streamlit as st
import logging
from src.utils.stock_mapping import SUPPORTED_STOCKS
from src.pipeline.main_pipeline import execute_prediction_pipeline
logger = logging.getLogger(__name__)
def render_sidebar():
    """Renders the global application sidebar and handles state selection."""
    with st.sidebar:
        st.markdown("### Stock Selection")
        company_names = list(SUPPORTED_STOCKS.keys())
        current_ticker = st.session_state.get('ticker', list(SUPPORTED_STOCKS.values())[0])
        default_index = 0
        for i, name in enumerate(company_names):
            if SUPPORTED_STOCKS[name] == current_ticker:
                default_index = i
                break
        selected_company = st.selectbox(
            "Company / Ticker",
            options=company_names,
            index=default_index,
            label_visibility="collapsed"
        )
        ticker = SUPPORTED_STOCKS[selected_company]
        st.caption(f"`{ticker}`")
        st.write("")
        if st.button("Run Prediction Pipeline", type="primary", use_container_width=True):
            try:
                execute_prediction_pipeline(ticker)
            except Exception as e:
                st.error(f"An error occurred while running the pipeline: {str(e)}")
                logger.exception("Pipeline failure")
