# Walmart Sales Forecasting Application

A comprehensive time series forecasting application for predicting Walmart weekly sales using Facebook's Prophet library. This project combines machine learning, interactive visualization, and automated reporting to support inventory and sales planning.

## 📊 Features

- **Time Series Forecasting**: Uses Facebook Prophet for accurate sales predictions
- **Interactive Dashboard**: Streamlit-based web interface with real-time visualizations
- **Model Validation**: Built-in train-test evaluation with MAE and RMSE metrics
- **Automated Reports**: PDF report generation with key insights and forecasts
- **Confidence Intervals**: Displays upper and lower bounds for predictions
- **Seasonal Analysis**: Identifies and visualizes yearly sales patterns
- **Data Visualization**: Interactive Plotly charts for trend analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nipunabandara/BaForecast.git
cd project_fol
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or using pyproject.toml:
```bash
pip install .
```

## 📁 Project Structure

```
project_fol/
├── forecast_gui.py              # Streamlit dashboard application
├── generate_pdf_report.py       # PDF report generator
├── main.py                      # Main forecasting script
├── pyproject.toml               # Project configuration
├── README.md                    # This file
├── NoteBook/
│   ├── pro.ipynb               # Jupyter notebook with analysis
│   ├── Walmart.csv             # Historical sales data
│   └── walmart_90_days_forecast.csv  # Forecast results
└── Data/                        # Additional data files
```

## 🎯 Usage

### 1. Launch the Interactive Dashboard
```bash
streamlit run forecast_gui.py
```
Opens an interactive web interface at `http://localhost:8501`

### 2. Run Main Forecasting Script
```bash
python main.py
```
Executes the forecasting pipeline with console output

### 3. Generate PDF Report
```bash
python generate_pdf_report.py
```
Creates a detailed PDF report with metrics and forecasts

### 4. Jupyter Notebook Analysis
```bash
jupyter notebook NoteBook/pro.ipynb
```
Interactive data exploration and model development

## 📈 Model Details

### Algorithm: Facebook Prophet
- **Components**: Trend, Yearly Seasonality, Holidays
- **Seasonality**: Yearly (13-week cycles)
- **Forecast Horizon**: 13 weeks (90 days)
- **Confidence Level**: 95% (upper/lower bounds)

### Model Validation
- **Train-Test Split**: 13-week holdout
- **Metrics**: 
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

## 📊 Data Format

**Walmart.csv** columns:
- `Store_ID`: Store identifier
- `Date`: Transaction date (DD-MM-YYYY format)
- `Weekly_Sales`: Weekly sales amount
- `Holiday_Flag`: Holiday indicator

## 🔧 Configuration

Edit the following in the source files:

**forecast_gui.py**:
- `forecast_periods`: Adjust forecast weeks (default: 13)
- Data path: Update CSV file location

**generate_pdf_report.py**:
- Output filename and location
- Report styling and content

## 📋 Key Metrics

Example output:
```
Model Performance:
   • Mean Absolute Error (MAE): $45,231.50
   • Root Mean Squared Error (RMSE): $52,847.23
   • Forecast Accuracy: ~92%
```

## 🐛 Troubleshooting

### Data not found error
```
Solution: Ensure Walmart.csv exists in NoteBook/ folder
```

### Prophet installation issues
```
pip install prophet --no-cache-dir
```

### Streamlit connection issues
```
streamlit run forecast_gui.py --server.port=8502
```

## 📦 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **prophet**: Time series forecasting
- **plotly**: Interactive visualizations
- **scikit-learn**: Model evaluation metrics
- **streamlit**: Dashboard framework
- **reportlab**: PDF generation

## 🎓 Learning Resources

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Guide](https://plotly.com/python/)

## 📝 Jupyter Notebook

The `pro.ipynb` notebook includes:
- Data loading and exploration
- Feature engineering
- Model training and evaluation
- Visualization code
- Statistical analysis

## 🚀 Deployment

### Deploy to Streamlit Cloud
```bash
# Push to GitHub
git push origin main

# Deploy via Streamlit Cloud dashboard
# - Connect your repository
# - Select main.py or forecast_gui.py
```

## 📞 Support & Contribution

For issues, feature requests, or contributions:
1. Open an issue on GitHub
2. Submit pull requests with improvements
3. Contact: nipuna@example.com

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Facebook Prophet team for the forecasting library
- Streamlit for the dashboard framework
- Walmart dataset for historical sales data

---

**Last Updated**: 2026-05-22  
**Version**: 1.0.0  
**Status**: Active Development
