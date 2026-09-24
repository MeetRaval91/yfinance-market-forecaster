import streamlit as st
import pandas as pd
import altair as alt

from src.models.selector import perform_cross_validation

st.set_page_config(page_title="Model Analysis", page_icon="🤖", layout="wide")

st.title("🤖 Model Analysis")

if 'evaluation_results' not in st.session_state or 'best_model_name' not in st.session_state:
    st.info("Please select a stock and run the prediction pipeline on the **Dashboard** first to view model metrics.")
else:
    evaluation_results = st.session_state['evaluation_results']
    best_model_name = st.session_state['best_model_name']
    trained_pipeline = st.session_state['trained_pipeline']
    X = st.session_state['X']
    y = st.session_state['y']
    
    st.markdown("### 1. Test Set Evaluation Metrics")
    st.write("The models were trained on the first 70% of chronological data and evaluated on the remaining 30%.")
    
    # Format the evaluation results into a DataFrame
    results_df = pd.DataFrame(evaluation_results).T
    
    # Highlight the best model visually
    st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='lightgreen', axis=0), use_container_width=True)
    
    # Bar chart comparing RMSE
    results_df['Model'] = results_df.index
    rmse_chart = alt.Chart(results_df).mark_bar(color='#2ca02c').encode(
        x=alt.X('Model:N', sort='-y', title='Model'),
        y=alt.Y('RMSE:Q', title='Root Mean Squared Error'),
        tooltip=['Model', 'RMSE', 'R2']
    ).properties(
        height=300,
        title="Model Comparison (RMSE - Lower is Better)"
    ).interactive()
    
    st.altair_chart(rmse_chart, use_container_width=True)
    
    st.divider()
    
    st.markdown(f"### 2. Time-Series Cross Validation: `{best_model_name}`")
    st.write("To ensure robustness and avoid overfitting, the selected best model undergoes a 5-fold TimeSeriesSplit. This preserves the chronological order of the financial data.")
    
    with st.spinner("Running 5-Fold Time-Series Cross Validation..."):
        # We run the cross-validation dynamically here
        cv_results = perform_cross_validation(trained_pipeline[best_model_name], X, y)
        cv_df = pd.DataFrame(cv_results)
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(cv_df[['fold', 'RMSE', 'R2', 'RSS']].set_index('fold'), use_container_width=True)
        
        avg_rmse = cv_df['RMSE'].mean()
        st.metric("Average CV RMSE", f"{avg_rmse:.4f}")
        
    with col2:
        cv_chart = alt.Chart(cv_df).mark_line(point=True, color='#d62728').encode(
            x=alt.X('fold:O', title='Fold (Chronological)'),
            y=alt.Y('RMSE:Q', title='RMSE', scale=alt.Scale(zero=False)),
            tooltip=['fold', 'RMSE', 'R2']
        ).properties(
            height=250,
            title="RMSE Across Chronological Folds"
        )
        st.altair_chart(cv_chart, use_container_width=True)
