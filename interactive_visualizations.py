"""
Interactive Plotly Visualizations for Walmart Sales Forecasting
Provides comprehensive interactive dashboard components
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta


def load_and_prepare_data(csv_path="NoteBook/Walmart.csv"):
    """Load and prepare data for visualization."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    return df, df_sales


def create_sales_trend_chart(df_sales):
    """Interactive line chart showing sales trend over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sales['Date'],
        y=df_sales['Weekly_Sales'],
        mode='lines',
        name='Weekly Sales',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>',
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title='Walmart Weekly Sales Trend',
        xaxis_title='Date',
        yaxis_title='Weekly Sales ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_sales_by_store_chart(df):
    """Interactive bar chart showing sales by store."""
    store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=store_sales.index,
        y=store_sales.values,
        marker=dict(
            color=store_sales.values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Total Sales ($)")
        ),
        hovertemplate='<b>Store:</b> %{x}<br><b>Total Sales:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Total Sales by Store',
        xaxis_title='Store ID',
        yaxis_title='Total Sales ($)',
        hovermode='x',
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_monthly_sales_boxplot(df_sales):
    """Interactive boxplot showing monthly sales distribution."""
    df_sales['Month'] = df_sales['Date'].dt.strftime('%Y-%m')
    
    fig = px.box(
        df_sales,
        x='Month',
        y='Weekly_Sales',
        title='Monthly Sales Distribution',
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Month': 'Month'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        hovermode='closest',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_forecast_visualization(df_sales):
    """Interactive forecast chart with Prophet predictions."""
    # Prepare data for Prophet
    df_prophet = df_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    # Train model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    
    # Generate forecast
    future = model.make_future_dataframe(periods=26, freq='W')
    forecast = model.predict(future)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df_sales['Date'],
        y=df_sales['Weekly_Sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add forecast
    forecast_df = forecast[forecast['ds'] > df_sales['Date'].max()]
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hovertemplate='<b>Upper Bound:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(255, 127, 14, 0.2)',
        hovertemplate='<b>Lower Bound:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Sales Forecast with 95% Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Weekly Sales ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig, forecast


def create_year_comparison_chart(df_sales):
    """Interactive chart comparing sales across years."""
    df_sales['Year'] = df_sales['Date'].dt.year
    df_sales['DayOfYear'] = df_sales['Date'].dt.dayofyear
    
    fig = px.line(
        df_sales,
        x='DayOfYear',
        y='Weekly_Sales',
        color='Year',
        title='Sales Comparison Across Years',
        labels={'DayOfYear': 'Day of Year', 'Weekly_Sales': 'Weekly Sales ($)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_seasonal_decomposition_chart(df_sales):
    """Interactive chart showing seasonal patterns."""
    df_prophet = df_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    
    forecast = model.predict(df_prophet)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Trend', 'Yearly Seasonality', 'Residuals'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Yearly seasonality
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Yearly Seasonality', line=dict(color='#ff7f0e')),
        row=2, col=1
    )
    
    # Residuals
    residuals = df_prophet['y'].values - forecast['yhat'].values
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=residuals, mode='markers', name='Residuals', marker=dict(color='#d62728')),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text='Sales ($)', row=1, col=1)
    fig.update_yaxes(title_text='Seasonality', row=2, col=1)
    fig.update_yaxes(title_text='Residuals ($)', row=3, col=1)
    
    fig.update_layout(
        title_text='Seasonal Decomposition Analysis',
        height=800,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_statistics_dashboard(df_sales, df):
    """Create a dashboard with key statistics."""
    # Calculate statistics
    total_sales = df_sales['Weekly_Sales'].sum()
    avg_weekly_sales = df_sales['Weekly_Sales'].mean()
    max_sales = df_sales['Weekly_Sales'].max()
    min_sales = df_sales['Weekly_Sales'].min()
    total_stores = df['Store'].nunique()
    
    # Create indicator figures
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=total_sales,
        title={'text': "Total Sales"},
        domain={'x': [0, 0.25], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=avg_weekly_sales,
        title={'text': "Avg Weekly Sales"},
        domain={'x': [0.25, 0.5], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=max_sales,
        title={'text': "Max Weekly Sales"},
        domain={'x': [0.5, 0.75], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=total_stores,
        title={'text': "Total Stores"},
        domain={'x': [0.75, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        title_text='Key Statistics',
        height=200,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_heatmap_sales_by_store_time(df):
    """Interactive heatmap showing sales patterns by store over time."""
    df['Month'] = df['Date'].dt.to_period('M')
    
    # Pivot data for heatmap
    heatmap_data = df.groupby(['Store', 'Month'])['Weekly_Sales'].sum().unstack()
    heatmap_data.columns = heatmap_data.columns.astype(str)
    
    # Limit to first 20 stores for readability
    heatmap_data = heatmap_data.head(20)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlOrRd',
        colorbar=dict(title="Sales ($)")
    ))
    
    fig.update_layout(
        title='Sales Heatmap by Store and Month (Top 20 Stores)',
        xaxis_title='Month',
        yaxis_title='Store ID',
        height=600,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def export_interactive_html(figures_dict, filename='walmart_dashboard.html'):
    """Export all visualizations to a single interactive HTML file."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create HTML with all figures
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Walmart Sales Interactive Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .chart-section {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h2 {
                color: #1f77b4;
                margin-top: 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Walmart Sales Interactive Dashboard</h1>
    """
    
    # Save individual figures to HTML strings
    for title, fig in figures_dict.items():
        html_content += f"""
        <div class="chart-section">
            <h2>{title}</h2>
            {fig.to_html(include_plotlyjs=False, div_id=title.replace(" ", "_"))}
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard exported to {filename}")


def main():
    """Generate all interactive visualizations."""
    print("📊 Loading data...")
    df, df_sales = load_and_prepare_data()
    
    print("🎨 Creating interactive visualizations...")
    
    # Create all figures
    figures = {
        'Sales Trend': create_sales_trend_chart(df_sales),
        'Sales by Store': create_sales_by_store_chart(df),
        'Monthly Distribution': create_monthly_sales_boxplot(df_sales),
        'Year Comparison': create_year_comparison_chart(df_sales),
        'Seasonal Decomposition': create_seasonal_decomposition_chart(df_sales),
        'Store Sales Heatmap': create_heatmap_sales_by_store_time(df),
    }
    
    # Create and add forecast
    forecast_fig, forecast_df = create_forecast_visualization(df_sales)
    figures['26-Week Forecast'] = forecast_fig
    
    # Create statistics
    stats_fig = create_statistics_dashboard(df_sales, df)
    figures['Statistics'] = stats_fig
    
    # Export to HTML
    export_interactive_html(figures, 'walmart_dashboard.html')
    
    # Show individual figures (when run with streamlit or jupyter)
    print("✅ Visualization generation completed!")
    print("\n📈 Available visualizations:")
    for title in figures.keys():
        print(f"   • {title}")
    
    return figures


if __name__ == "__main__":
    figures = main()
    
    # To view a specific figure:
    # figures['Sales Trend'].show()
    # figures['26-Week Forecast'].show()
    # figures['Seasonal Decomposition'].show()
