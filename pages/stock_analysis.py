import streamlit as st
import pandas as pd
import altair as alt

from src.utils.ui_helpers import render_empty_state, apply_chart_theme

st.title("Stock Analysis")

if 'df_raw' not in st.session_state or 'ticker' not in st.session_state:
    render_empty_state()
else:
    df_raw = st.session_state['df_raw']
    ticker = st.session_state['ticker']
    stock_info = st.session_state['stock_info']
    
    st.markdown("Detailed historical price and volume action analysis.")
    st.write("")
    
    # Selected Stock Header
    with st.container(border=True):
        st.markdown(f"#### **{stock_info.company_name}**")
        st.caption(f"{stock_info.ticker} · {stock_info.exchange} · Native Currency: **{stock_info.native_currency}**")
    
    st.write("")
    
    # Reset index to make 'Date' a column for Altair
    df_chart = df_raw.reset_index()
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        st.markdown(f"#### Price & Volume History")
        
        # Closing Price Chart
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
        
        close_chart = (area + line).properties(height=350)
        st.altair_chart(apply_chart_theme(close_chart, f"Closing Price Over Time ({stock_info.native_currency})"), use_container_width=True)
        
        st.write("")
        
        # Trading Volume Chart
        volume_chart = alt.Chart(df_chart).mark_bar(color='#64748B', opacity=0.7).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Volume:Q', title='Volume'),
            tooltip=['Date:T', 'Volume:Q']
        ).properties(height=200)
        st.altair_chart(apply_chart_theme(volume_chart, "Trading Volume Over Time"), use_container_width=True)
        
    with col_stats:
        st.markdown("#### Market Statistics")
        with st.container(border=True):
            st.caption(f"**Descriptive Statistics ({stock_info.native_currency})**")
            st.dataframe(df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].describe(), use_container_width=True)
            
        with st.container(border=True):
            st.caption(f"**Recent Raw Data ({stock_info.native_currency})**")
            # Show the last 10 rows (most recent first)
            st.dataframe(df_raw[['Close', 'Volume']].tail(10).sort_index(ascending=False), use_container_width=True)
