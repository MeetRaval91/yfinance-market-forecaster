import streamlit as st
import pandas as pd
import altair as alt
from src.models.selector import perform_cross_validation
from src.utils.ui_helpers import render_empty_state, apply_chart_theme
st.title("Model Analysis")
if 'evaluation_results' not in st.session_state or 'best_model_name' not in st.session_state:
    render_empty_state()
else:
    evaluation_results = st.session_state['evaluation_results']
    best_model_name = st.session_state['best_model_name']
    trained_pipeline = st.session_state['trained_pipeline']
    X = st.session_state['X']
    y = st.session_state['y']
    stock_info = st.session_state.get('stock_info')
    st.markdown("Machine learning model performance and cross-validation metrics.")
    st.write("")
    if stock_info:
        with st.container(border=True):
            st.markdown(f"#### **{stock_info.company_name}**")
            st.caption(f"{stock_info.ticker} · {stock_info.exchange} · Native Currency: **{stock_info.native_currency}**")
        st.write("")
    st.markdown("#### 1. Test Set Evaluation Metrics")
    st.write("The models were trained on the first 70% of chronological data and evaluated on the remaining 30%.")
    results_df = pd.DataFrame(evaluation_results).T
    col_chart, col_table = st.columns([2, 1])
    with col_table:
        with st.container(border=True):
            st.markdown(f"**Best Model:** `{best_model_name}`")
            st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='#86efac', axis=0), use_container_width=True)
    with col_chart:
        results_df['Model'] = results_df.index
        rmse_chart = alt.Chart(results_df).mark_bar(color='#1E3A8A').encode(
            x=alt.X('Model:N', sort='-y', title='Model'),
            y=alt.Y('RMSE:Q', title='Root Mean Squared Error'),
            tooltip=['Model', 'RMSE', 'R2']
        ).properties(height=300)
        st.altair_chart(apply_chart_theme(rmse_chart, "Model Comparison (RMSE - Lower is Better)"), use_container_width=True)
    st.write("")
    st.markdown(f"#### 2. Time-Series Cross Validation: `{best_model_name}`")
    st.write("To ensure robustness and avoid overfitting, the selected best model undergoes a 5-fold TimeSeriesSplit. This preserves the chronological order of the financial data.")
    with st.spinner("Running 5-Fold Time-Series Cross Validation..."):
        cv_results = perform_cross_validation(trained_pipeline[best_model_name], X, y)
        cv_df = pd.DataFrame(cv_results)
    col_cv_table, col_cv_chart = st.columns([1, 2])
    with col_cv_table:
        with st.container(border=True):
            st.caption("**5-Fold Chronological Results**")
            st.dataframe(cv_df[['fold', 'RMSE', 'R2', 'RSS']].set_index('fold'), use_container_width=True)
            avg_rmse = cv_df['RMSE'].mean()
            st.metric("Average CV RMSE", f"{avg_rmse:.4f}")
    with col_cv_chart:
        cv_chart = alt.Chart(cv_df).mark_line(point=True, color='#0F172A').encode(
            x=alt.X('fold:O', title='Fold (Chronological)'),
            y=alt.Y('RMSE:Q', title='RMSE', scale=alt.Scale(zero=False)),
            tooltip=['fold', 'RMSE', 'R2']
        ).properties(height=280)
        st.altair_chart(apply_chart_theme(cv_chart, "RMSE Across Chronological Folds"), use_container_width=True)
