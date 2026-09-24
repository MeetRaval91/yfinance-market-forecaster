import streamlit as st
import pandas as pd
import logging

from src.data.stock_data import fetch_stock_data
from src.features.feature_engineering import prepare_ml_data
from src.models.model_factory import get_models
from src.models.trainer import split_data_chronologically, train_models
from src.models.evaluator import evaluate_models
from src.models.selector import select_best_model
from src.prediction.predictor import extract_latest_features_for_prediction, predict_next_day
from src.utils.stock_mapping import SUPPORTED_STOCKS
from src.currency.currency_converter import convert_currency

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Dashboard - Stock Prediction", page_icon="📊", layout="wide")

st.title("📊 Stock Prediction Dashboard")
st.markdown("Forecast the **Next Trading Day's Closing Price** using Machine Learning.")

# --- Stock Selection ---
col1, col2 = st.columns([1, 2])
with col1:
    selected_company = st.selectbox(
        "Select a Company / Ticker",
        options=list(SUPPORTED_STOCKS.keys())
    )
    ticker = SUPPORTED_STOCKS[selected_company]
    st.caption(f"Selected Ticker: `{ticker}`")

with col2:
    st.write("") # spacing
    st.write("")
    if st.button("Run Prediction Pipeline", type="primary"):
        st.session_state['run_pipeline'] = True
        st.session_state['ticker'] = ticker

if st.session_state.get('run_pipeline', False):
    ticker = st.session_state['ticker']
    
    try:
        with st.spinner("Fetching Market Data..."):
            df_raw, stock_info = fetch_stock_data(ticker)
            
        with st.spinner("Engineering Features..."):
            X, y = prepare_ml_data(df_raw)
            
        with st.spinner("Training ML Models (SVR, Ridge, HistGBR)..."):
            X_train, X_test, y_train, y_test = split_data_chronologically(X, y)
            models = get_models()
            trained_pipeline = train_models(models, X_train, y_train)
            
        with st.spinner("Evaluating Models..."):
            evaluation_results = evaluate_models(trained_pipeline, X_test, y_test)
            best_model_name = select_best_model(evaluation_results)
            
        with st.spinner("Generating Prediction..."):
            latest_features, latest_close = extract_latest_features_for_prediction(df_raw)
            prediction = predict_next_day(trained_pipeline[best_model_name], latest_features, latest_close, stock_info.native_currency)
            
        # --- UI DISPLAY ---
        st.divider()
        
        st.markdown(f"### {stock_info.company_name} ({stock_info.ticker})")
        st.caption(f"Exchange: **{stock_info.exchange}** | Native Currency: **{stock_info.native_currency}** | Display Currency: **{stock_info.display_currency}**")
        
        # Native metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Latest Close ({stock_info.native_currency})", f"{prediction['Latest_Close']:.2f}")
        
        direction_color = "normal" if prediction['Expected_Direction'] == "UP" else "inverse"
        m2.metric(
            f"Predicted Next Close ({stock_info.native_currency})", 
            f"{prediction['Predicted_Close']:.2f}",
            f"{prediction['Predicted_Change']:.2f} ({prediction['Predicted_Change_Pct']:.2f}%)",
            delta_color=direction_color
        )
        m3.metric("Expected Direction", prediction['Expected_Direction'])
        m4.metric("Selected Best Model", best_model_name)
        
        # Currency Conversion Section
        is_foreign = stock_info.native_currency.upper() != stock_info.display_currency.upper()
        if is_foreign:
            st.divider()
            st.subheader(f"Currency Conversion ({stock_info.display_currency} Equivalent)")
            
            inr_latest = convert_currency(prediction['Latest_Close'], stock_info.native_currency, stock_info.display_currency)
            inr_predicted = convert_currency(prediction['Predicted_Close'], stock_info.native_currency, stock_info.display_currency)
            
            if inr_latest is not None and inr_predicted is not None:
                inr_change = inr_predicted - inr_latest
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Latest Close ({stock_info.display_currency})", f"₹{inr_latest:,.2f}")
                c2.metric(
                    f"Predicted Next Close ({stock_info.display_currency})", 
                    f"₹{inr_predicted:,.2f}",
                    f"₹{inr_change:,.2f} ({prediction['Predicted_Change_Pct']:.2f}%)",
                    delta_color=direction_color
                )
                rate = inr_latest / prediction['Latest_Close']
                c3.info(f"**Exchange Rate Used:** 1 {stock_info.native_currency} = ₹{rate:.4f}")
            else:
                st.warning(f"⚠️ {stock_info.display_currency} conversion is temporarily unavailable. Showing the stock price in its native currency ({stock_info.native_currency}).")
        
        st.divider()
        
        # Historical Chart
        st.subheader(f"Historical Closing Price ({stock_info.native_currency})")
        st.line_chart(df_raw['Close'])
        
        # Model Comparison Expander
        with st.expander("View Model Comparison (Test Set)"):
            results_df = pd.DataFrame(evaluation_results).T
            st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='lightgreen', axis=0))
            
        # Save results to session state
        st.session_state['df_raw'] = df_raw
        st.session_state['stock_info'] = stock_info
        st.session_state['evaluation_results'] = evaluation_results
        st.session_state['trained_pipeline'] = trained_pipeline
        st.session_state['best_model_name'] = best_model_name
        st.session_state['prediction'] = prediction
        st.session_state['X'] = X
        st.session_state['y'] = y
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Pipeline failure")

