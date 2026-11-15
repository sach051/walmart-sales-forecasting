"""
Streamlit Interactive Dashboard for Walmart Sales
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from interactive_visualizations import (
    load_and_prepare_data,
    create_sales_trend_chart,
    create_sales_by_store_chart,
    create_monthly_sales_boxplot,
    create_forecast_visualization,
    create_year_comparison_chart,
    create_seasonal_decomposition_chart,
    create_heatmap_sales_by_store_time,
    create_statistics_dashboard
)

# Page config
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data with caching."""
    df, df_sales = load_and_prepare_data()
    return df, df_sales

# Title and description
st.title("📊 Walmart Sales Interactive Dashboard")
st.markdown("""
This dashboard provides comprehensive interactive visualizations for Walmart sales data,
including trend analysis, forecasting, and seasonal decomposition.
""")

# Load data
with st.spinner('Loading data...'):
    df, df_sales = load_data()

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select a visualization:", [
    "📈 Overview",
    "📊 Sales Trends",
    "🏪 Store Analysis",
    "📅 Seasonal Analysis",
    "🔮 Sales Forecast",
    "🔍 Detailed Insights"
])

# Page 1: Overview
if page == "📈 Overview":
    st.header("Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Sales",
            f"${df_sales['Weekly_Sales'].sum():,.0f}",
            delta=f"{(df_sales['Weekly_Sales'].iloc[-1] / df_sales['Weekly_Sales'].iloc[0] - 1) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Weekly Sales",
            f"${df_sales['Weekly_Sales'].mean():,.0f}",
            delta=f"{df_sales['Weekly_Sales'].std():,.0f}"
        )
    
    with col3:
        st.metric(
            "Peak Sales",
            f"${df_sales['Weekly_Sales'].max():,.0f}",
            delta=f"${df_sales['Weekly_Sales'].max() - df_sales['Weekly_Sales'].mean():,.0f}"
        )
    
    with col4:
        st.metric(
            "Total Stores",
            f"{df['Store'].nunique()}",
            delta="Active"
        )
    
    st.divider()
    
    # Statistics dashboard
    st.subheader("Key Statistics")
    stats_fig = create_statistics_dashboard(df_sales, df)
    st.plotly_chart(stats_fig, use_container_width=True)
    
    # Sales trend
    st.subheader("Sales Trend Over Time")
    trend_fig = create_sales_trend_chart(df_sales)
    st.plotly_chart(trend_fig, use_container_width=True)

# Page 2: Sales Trends
elif page == "📊 Sales Trends":
    st.header("Sales Trend Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Weekly Sales Trend")
        trend_fig = create_sales_trend_chart(df_sales)
        st.plotly_chart(trend_fig, use_container_width=True)
    
    with col2:
        st.subheader("Statistics")
        st.write(f"""
        - **Total Records:** {len(df_sales):,}
        - **Date Range:** {df_sales['Date'].min().strftime('%Y-%m-%d')} to {df_sales['Date'].max().strftime('%Y-%m-%d')}
        - **Mean Sales:** ${df_sales['Weekly_Sales'].mean():,.0f}
        - **Median Sales:** ${df_sales['Weekly_Sales'].median():,.0f}
        - **Std Dev:** ${df_sales['Weekly_Sales'].std():,.0f}
        - **Min Sales:** ${df_sales['Weekly_Sales'].min():,.0f}
        - **Max Sales:** ${df_sales['Weekly_Sales'].max():,.0f}
        """)
    
    st.divider()
    
    st.subheader("Monthly Distribution")
    box_fig = create_monthly_sales_boxplot(df_sales)
    st.plotly_chart(box_fig, use_container_width=True)
    
    st.subheader("Year-over-Year Comparison")
    year_fig = create_year_comparison_chart(df_sales)
    st.plotly_chart(year_fig, use_container_width=True)

# Page 3: Store Analysis
elif page == "🏪 Store Analysis":
    st.header("Store Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Store")
        store_fig = create_sales_by_store_chart(df)
        st.plotly_chart(store_fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Stores")
        top_stores = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(10)
        st.dataframe(
            top_stores.reset_index().rename(columns={'Store': 'Store ID', 'Weekly_Sales': 'Total Sales'}),
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    st.subheader("Sales Heatmap by Store and Month")
    heatmap_fig = create_heatmap_sales_by_store_time(df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Page 4: Seasonal Analysis
elif page == "📅 Seasonal Analysis":
    st.header("Seasonal Decomposition Analysis")
    
    st.subheader("Trend, Seasonality, and Residuals")
    seasonal_fig = create_seasonal_decomposition_chart(df_sales)
    st.plotly_chart(seasonal_fig, use_container_width=True)
    
    st.info("""
    📌 **What you're seeing:**
    - **Trend:** Overall direction of sales over time
    - **Yearly Seasonality:** Patterns that repeat annually
    - **Residuals:** Unexplained variations (errors)
    """)

# Page 5: Sales Forecast
elif page == "🔮 Sales Forecast":
    st.header("Sales Forecast (Next 26 Weeks)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.spinner('Generating forecast...'):
            forecast_fig, forecast_df = create_forecast_visualization(df_sales)
        st.plotly_chart(forecast_fig, use_container_width=True)
    
    with col2:
        st.subheader("Forecast Stats")
        future_forecast = forecast_df[forecast_df['ds'] > df_sales['Date'].max()]
        st.write(f"""
        - **Forecast Period:** 26 weeks
        - **Avg Forecast:** ${future_forecast['yhat'].mean():,.0f}
        - **Upper Bound:** ${future_forecast['yhat_upper'].max():,.0f}
        - **Lower Bound:** ${future_forecast['yhat_lower'].min():,.0f}
        - **Confidence Level:** 95%
        """)
    
    st.divider()
    
    st.subheader("Forecast Details")
    forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound (95%)', 'Upper Bound (95%)']
    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
    forecast_display = forecast_display.round(0)
    
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)

# Page 6: Detailed Insights
elif page == "🔍 Detailed Insights":
    st.header("Detailed Data Insights")
    
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Data Summary", "Download"])
    
    with tab1:
        st.subheader("Sales Data Preview")
        st.dataframe(df_sales.head(20), use_container_width=True)
    
    with tab2:
        st.subheader("Data Summary Statistics")
        st.write(df_sales['Weekly_Sales'].describe())
        
        st.divider()
        
        st.subheader("Data Quality Check")
        st.write(f"""
        - **Total Records:** {len(df_sales):,}
        - **Missing Values:** {df_sales.isnull().sum().sum()}
        - **Date Range:** {(df_sales['Date'].max() - df_sales['Date'].min()).days} days
        - **Number of Unique Stores:** {df['Store'].nunique()}
        """)
    
    with tab3:
        st.subheader("Export Data")
        
        csv = df_sales.to_csv(index=False)
        st.download_button(
            label="📥 Download Sales Data (CSV)",
            data=csv,
            file_name="walmart_sales.csv",
            mime="text/csv"
        )

# Footer
st.divider()
st.markdown("""
---
**Walmart Sales Interactive Dashboard** | Powered by Plotly & Streamlit
- 📊 Data: Real Walmart sales data
- 🤖 Forecast: Facebook Prophet
- 📈 Last Updated: 2026-05-22
""")
