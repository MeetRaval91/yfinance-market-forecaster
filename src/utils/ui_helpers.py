import streamlit as st
import altair as alt

def render_empty_state(message: str = "No Stock Selected", sub_message: str = "Search for a company or ticker in the sidebar to begin analysis."):
    """Renders a visually balanced empty state when no data is active."""
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 2rem; background-color: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 8px; margin-top: 2rem;">
        <h3 style="color: #334155; margin-bottom: 0.5rem; font-weight: 600;">{message}</h3>
        <p style="color: #64748B;">{sub_message}</p>
    </div>
    """, unsafe_allow_html=True)

def apply_chart_theme(chart: alt.Chart, title: str = "") -> alt.Chart:
    """
    Applies consistent styling to Altair charts to match the global UI aesthetic.
    """
    props = {"width": "container"}
    if title:
        props["title"] = title
        
    return chart.properties(
        **props
    ).configure_title(
        fontSize=16,
        color='#1E293B',
        anchor='start',
        offset=15,
        fontWeight=600
    ).configure_axis(
        labelColor='#64748B',
        titleColor='#475569',
        gridColor='#F1F5F9',
        domainColor='#E2E8F0',
        tickColor='#E2E8F0'
    ).configure_view(
        strokeWidth=0
    )

def render_prediction_metric(label: str, value: float, change: float = None, pct_change: float = None, symbol: str = ""):
    """
    Renders a custom metric card that strictly controls the change indicator styling
    without relying on Streamlit's automatic delta parsing.
    """
    value_str = f"{symbol}{value:,.2f}"
    
    if change is None or pct_change is None:
        delta_str = "&nbsp;" # Empty space to maintain vertical alignment
        color = "transparent"
    else:
        abs_change = abs(change)
        
        if change > 0:
            color = "#16a34a" # green
            arrow = "↑"
            pct_str = f"(+{pct_change:.2f}%)"
        elif change < 0:
            color = "#dc2626" # red
            arrow = "↓"
            pct_str = f"({pct_change:.2f}%)"
        else:
            color = "#64748b" # neutral
            arrow = "→"
            pct_str = "(0.00%)"
            
        delta_str = f"{arrow} {symbol}{abs_change:,.2f} {pct_str}"
    
    st.markdown(f"""
    <div style="padding: 0.5rem 0;">
        <div style="font-size: 0.875rem; color: #64748B; margin-bottom: 0.25rem;">{label}</div>
        <div style="font-size: 1.875rem; color: #0F172A; font-weight: 600; line-height: 1.2;">{value_str}</div>
        <div style="font-size: 0.875rem; font-weight: 500; color: {color}; margin-top: 0.25rem;">{delta_str}</div>
    </div>
    """, unsafe_allow_html=True)

def render_direction_metric(label: str, direction: str):
    """
    Renders a custom metric card for Expected Direction with appropriate semantic coloring.
    """
    if direction.upper() == "UP":
        color = "#16a34a" # green
        arrow = "↑"
    elif direction.upper() == "DOWN":
        color = "#dc2626" # red
        arrow = "↓"
    else:
        color = "#64748b" # neutral
        arrow = "→"
        
    st.markdown(f"""
    <div style="padding: 0.5rem 0;">
        <div style="font-size: 0.875rem; color: #64748B; margin-bottom: 0.25rem;">{label}</div>
        <div style="font-size: 1.875rem; color: {color}; font-weight: 600; line-height: 1.2;">{direction} {arrow}</div>
        <div style="font-size: 0.875rem; font-weight: 500; color: {color}; margin-top: 0.25rem;">Model Estimate</div>
    </div>
    """, unsafe_allow_html=True)
