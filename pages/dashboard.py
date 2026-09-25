import streamlit as st
import pandas as pd
import altair as alt
import logging

from src.currency.currency_converter import convert_currency
from src.utils.ui_helpers import render_empty_state, apply_chart_theme, render_prediction_metric, render_direction_metric

logger = logging.getLogger(__name__)

# --- Header ---
st.title("Stock Prediction Dashboard")
st.markdown("An academic machine-learning application predicting the next trading day's closing price.")

if not st.session_state.get('run_pipeline', False):
    render_empty_state()
else:
    ticker = st.session_state['ticker']
    df_raw = st.session_state['df_raw']
    stock_info = st.session_state['stock_info']
    evaluation_results = st.session_state['evaluation_results']
    best_model_name = st.session_state['best_model_name']
    prediction = st.session_state['prediction']
    
    try:
        # --- UI DISPLAY ---
        st.write("")
        
        # Selected Stock Summary
        with st.container(border=True):
            st.markdown(f"#### **{stock_info.company_name}**")
            st.caption(f"{stock_info.ticker} · {stock_info.exchange} · Native Currency: **{stock_info.native_currency}**")
            
        # Key Prediction Metrics
        st.markdown("#### Key Prediction Metrics")
        
        is_foreign = stock_info.native_currency.upper() != stock_info.display_currency.upper()
        
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            with m1:
                render_prediction_metric(
                    label=f"Latest Close ({stock_info.native_currency})",
                    value=prediction['Latest_Close']
                )
            with m2:
                render_prediction_metric(
                    label=f"Predicted Next Close ({stock_info.native_currency})", 
                    value=prediction['Predicted_Close'],
                    change=prediction['Predicted_Change'],
                    pct_change=prediction['Predicted_Change_Pct']
                )
            with m3:
                render_direction_metric("Expected Direction", prediction['Expected_Direction'])
            
        # Secondary INR Equivalent (No excessive boxing)
        if is_foreign:
            inr_latest = convert_currency(prediction['Latest_Close'], stock_info.native_currency, stock_info.display_currency)
            inr_predicted = convert_currency(prediction['Predicted_Close'], stock_info.native_currency, stock_info.display_currency)
            
            if inr_latest is not None and inr_predicted is not None:
                inr_change = inr_predicted - inr_latest
                st.write("")
                st.caption(f"**{stock_info.display_currency} Equivalent**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    render_prediction_metric(
                        label=f"Latest Close ({stock_info.display_currency})",
                        value=inr_latest,
                        symbol="₹"
                    )
                with c2:
                    render_prediction_metric(
                        label=f"Predicted Next Close ({stock_info.display_currency})", 
                        value=inr_predicted,
                        change=inr_change,
                        pct_change=prediction['Predicted_Change_Pct'],
                        symbol="₹"
                    )
                with c3:
                    st.write("") # spacing
                    st.write("")
                    rate = inr_latest / prediction['Latest_Close']
                    st.caption(f"FX rate used (latest available): 1 {stock_info.native_currency} = ₹{rate:.4f}")
            else:
                st.warning(f"{stock_info.display_currency} conversion is temporarily unavailable. Showing the stock price in its native currency ({stock_info.native_currency}).")
        
        st.write("")
        
        # Price Analysis & Model Summary
        col_chart, col_model = st.columns([2, 1])
        
        with col_chart:
            st.markdown("#### Price Analysis")
            df_chart = df_raw.reset_index()
            # Unified Altair Chart with subtle area fill for professional aesthetic
            base = alt.Chart(df_chart).encode(
                x=alt.X('Date:T', title='Date')
            )
            
            line = base.mark_line(color='#1E3A8A').encode(
                y=alt.Y('Close:Q', title=f'Closing Price ({stock_info.native_currency})', scale=alt.Scale(zero=False)),
                tooltip=['Date:T', 'Close:Q']
            )
            
            area = base.mark_area(
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color='#1E3A8A', offset=0.0),
                        alt.GradientStop(color='white', offset=1.0)
                    ],
                    x1=1, x2=1, y1=1, y2=0
                ),
                opacity=0.2
            ).encode(
                y=alt.Y('Close:Q')
            )
            
            raw_chart = (area + line).properties(height=320)
            
            st.altair_chart(apply_chart_theme(raw_chart, f"Historical Closing Price ({stock_info.native_currency})"), use_container_width=True)
            
        with col_model:
            st.markdown("#### Model Summary")
            with st.container(border=True):
                st.markdown(f"**Best Model Selected:** `{best_model_name}`")
                
                results_df = pd.DataFrame(evaluation_results).T
                best_metrics = results_df.loc[best_model_name]
                
                st.caption("Test Set Performance:")
                st.metric("RMSE", f"{best_metrics['RMSE']:.4f}")
                st.metric("R² Score", f"{best_metrics['R2']:.4f}")
                
                with st.expander("View All Models"):
                    st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='#86efac', axis=0), use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred while rendering the dashboard: {str(e)}")
        logger.exception("Dashboard rendering failure")

