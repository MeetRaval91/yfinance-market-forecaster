import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Stock Analysis", page_icon="📉", layout="wide")

st.title("📉 Stock Analysis")

if 'df_raw' not in st.session_state or 'ticker' not in st.session_state:
    st.info("Please select a stock and run the prediction pipeline on the **Dashboard** first to load data.")
else:
    df_raw = st.session_state['df_raw']
    ticker = st.session_state['ticker']
    stock_info = st.session_state['stock_info']
    
    st.markdown(f"### Historical Data for `{ticker}`")
    st.caption(f"**{stock_info.company_name}** | Native Currency: **{stock_info.native_currency}**")
    
    # Reset index to make 'Date' a column for Altair
    df_chart = df_raw.reset_index()
    
    # Closing Price Chart
    close_chart = alt.Chart(df_chart).mark_line(color='#1f77b4').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title=f'Closing Price ({stock_info.native_currency})', scale=alt.Scale(zero=False)),
        tooltip=['Date:T', 'Close:Q']
    ).properties(
        height=350,
        title=f"Closing Price Over Time ({stock_info.native_currency})"
    ).interactive()
    
    st.altair_chart(close_chart, use_container_width=True)
    
    # Trading Volume Chart
    volume_chart = alt.Chart(df_chart).mark_bar(color='#ff7f0e', opacity=0.7).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Volume:Q', title='Volume'),
        tooltip=['Date:T', 'Volume:Q']
    ).properties(
        height=200,
        title="Trading Volume Over Time"
    ).interactive()
    
    st.altair_chart(volume_chart, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Recent Raw Data ({stock_info.native_currency})")
        # Show the last 10 rows (most recent first)
        st.dataframe(df_raw.tail(10).sort_index(ascending=False), use_container_width=True)
        
    with col2:
        st.subheader("Descriptive Statistics")
        st.dataframe(df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].describe(), use_container_width=True)

