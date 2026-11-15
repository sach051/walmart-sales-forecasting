import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")

st.title("📊 Walmart Sales Forecasting Dashboard")

# Load data
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "NoteBook", "Walmart.csv")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

try:
    df = load_data()
    
    # Data preprocessing
    df_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    df_prophet = df_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    forecast_periods = st.sidebar.slider("Forecast Periods (weeks):", 1, 52, 13)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Historical Sales Trend")
        fig_history = px.line(
            df_prophet,
            x='ds',
            y='y',
            title="Historical Walmart Sales",
            template="plotly_dark",
            markers=True
        )
        fig_history.update_layout(height=500)
        st.plotly_chart(fig_history, use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Summary")
        st.metric("Total Records", len(df_prophet))
        st.metric("Avg Weekly Sales", f"${df_prophet['y'].mean():,.0f}")
        st.metric("Max Weekly Sales", f"${df_prophet['y'].max():,.0f}")
        st.metric("Min Weekly Sales", f"${df_prophet['y'].min():,.0f}")
    
    # Train model
    st.sidebar.header("🔧 Model Training")
    with st.spinner("Training Prophet model..."):
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(df_prophet)
    
    # Generate forecast
    future = model.make_future_dataframe(periods=forecast_periods, freq='W')
    forecast = model.predict(future)
    
    # Plot forecast
    st.subheader("🔮 Sales Forecast")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_prophet['ds'],
        y=df_prophet['y'],
        mode='lines',
        name='Past Sales',
        line=dict(color='deepskyblue')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.25)',
        line=dict(width=0),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title="Walmart Sales: Past vs Future Forecast",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        height=600,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Components
    st.subheader("📉 Model Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)
    
    # Model validation
    st.subheader("✅ Model Validation")
    train = df_prophet.iloc[:-13]
    test = df_prophet.iloc[-13:]
    
    model_test = Prophet(yearly_seasonality=True)
    model_test.fit(train)
    
    future_test = model_test.make_future_dataframe(periods=13, freq='W')
    forecast_test = model_test.predict(future_test)
    
    mae = mean_absolute_error(test['y'], forecast_test['yhat'][-13:])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat'][-13:]))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"${mae:,.0f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.0f}")
    
    # Forecast table
    st.subheader("📋 Forecast Details")
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    forecast_display['Forecast'] = forecast_display['Forecast'].apply(lambda x: f"${x:,.0f}")
    forecast_display['Lower Bound'] = forecast_display['Lower Bound'].apply(lambda x: f"${x:,.0f}")
    forecast_display['Upper Bound'] = forecast_display['Upper Bound'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(forecast_display, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Data file not found. Please ensure 'Walmart.csv' is in the NoteBook folder.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
