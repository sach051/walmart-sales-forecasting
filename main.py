"""
Walmart Sales Forecasting Application
Main entry point for the forecasting system
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_and_prepare_data(csv_path="NoteBook/Walmart.csv"):
    """Load and prepare data for forecasting."""
    print("📂 Loading data...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    df_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    df_prophet = df_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    print(f"✅ Data loaded successfully. Records: {len(df_prophet)}")
    return df_prophet

def train_forecast_model(df, forecast_weeks=13):
    """Train Prophet model and generate forecast."""
    print("\n🔧 Training Prophet model...")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(df)
    
    print("✅ Model training completed")
    
    # Generate forecast
    future = model.make_future_dataframe(periods=forecast_weeks, freq='W')
    forecast = model.predict(future)
    
    print(f"📈 Forecast generated for {forecast_weeks} weeks")
    
    return model, forecast

def validate_model(df, forecast_periods=13):
    """Validate model performance using train-test split."""
    print("\n✅ Validating model...")
    
    train = df.iloc[:-forecast_periods]
    test = df.iloc[-forecast_periods:]
    
    model = Prophet(yearly_seasonality=True)
    model.fit(train)
    
    future = model.make_future_dataframe(periods=forecast_periods, freq='W')
    forecast = model.predict(future)
    
    mae = mean_absolute_error(test['y'], forecast['yhat'][-forecast_periods:])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat'][-forecast_periods:]))
    
    print(f"\n📊 Model Validation Results:")
    print(f"   • Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"   • Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    
    return mae, rmse

def print_statistics(df):
    """Print data statistics."""
    print("\n📈 Data Statistics:")
    print(f"   • Total Records: {len(df)}")
    print(f"   • Date Range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print(f"   • Average Weekly Sales: ${df['y'].mean():,.2f}")
    print(f"   • Maximum Weekly Sales: ${df['y'].max():,.2f}")
    print(f"   • Minimum Weekly Sales: ${df['y'].min():,.2f}")
    print(f"   • Std Dev: ${df['y'].std():,.2f}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("🎯 Walmart Sales Forecasting Application")
    print("=" * 60)
    
    try:
        # Load data
        df = load_and_prepare_data()
        print_statistics(df)
        
        # Train model
        model, forecast = train_forecast_model(df, forecast_weeks=13)
        
        # Validate
        mae, rmse = validate_model(df)
        
        # Display forecast
        print("\n📋 Next 13 Weeks Forecast:")
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13)
        print(forecast_display.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✅ Forecasting completed successfully!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   • Run 'streamlit run forecast_gui.py' to launch the dashboard")
        print("   • Run 'python generate_pdf_report.py' to generate a PDF report")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please ensure Walmart.csv exists in the NoteBook folder")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
