# 📊 Interactive Plotly Visualizations Guide

This project includes comprehensive interactive visualizations for Walmart sales data analysis using Plotly and Streamlit.

## 🎨 Visualization Components

### 1. **Sales Trend Chart**
   - Interactive line chart showing sales over time
   - Hover for exact values
   - Zoom and pan capabilities

### 2. **Sales by Store**
   - Bar chart with color scaling by sales volume
   - Top stores at a glance
   - Interactive filtering

### 3. **Monthly Sales Distribution**
   - Boxplot showing monthly variations
   - Identify outliers and trends
   - Statistical overview

### 4. **Sales Forecast (26-week)**
   - Prophet model predictions
   - 95% confidence intervals
   - Historical + forecast overlay

### 5. **Year-over-Year Comparison**
   - Compare sales patterns across years
   - Identify seasonal trends
   - Multi-year analysis

### 6. **Seasonal Decomposition**
   - Trend analysis
   - Yearly seasonality patterns
   - Residual analysis

### 7. **Store Sales Heatmap**
   - Sales intensity by store and month
   - Identify peak periods
   - Store performance patterns

### 8. **Key Statistics Dashboard**
   - Total sales metrics
   - Average weekly sales
   - Peak sales indicator
   - Store count

## 🚀 Quick Start

### Option 1: Generate HTML Dashboard (Recommended)
Run this command to generate a standalone HTML file that opens in your browser:

```bash
python view_visualizations.py
```

This creates `walmart_dashboard.html` - Open it directly in any web browser. No server needed!

### Option 2: View Individual Charts in Python
```python
from interactive_visualizations import *

df, df_sales = load_and_prepare_data()

# View specific chart
fig = create_sales_trend_chart(df_sales)
fig.show()  # Opens in browser

# Or view forecast
forecast_fig, forecast_df = create_forecast_visualization(df_sales)
forecast_fig.show()
```

### Option 3: Streamlit Interactive Dashboard (Full Experience)
Run this for a professional web dashboard with navigation:

```bash
streamlit run streamlit_dashboard.py
```

Then navigate to `http://localhost:8501` in your browser.

## 🎯 Features

✅ **Interactive Controls**
- Hover for tooltips
- Click legend to toggle series
- Zoom and pan
- Download as PNG

✅ **Dynamic Forecasting**
- Prophet model integration
- 26-week predictions
- Confidence intervals
- Trend analysis

✅ **Responsive Design**
- Works on desktop and mobile
- Auto-scaling charts
- Professional styling

✅ **Data Export**
- Download as CSV
- Export visualizations
- HTML dashboard

## 📦 Dependencies

All required packages are in your `pyproject.toml`:
- `plotly>=5.0.0` - Interactive charts
- `pandas>=1.5.0` - Data manipulation
- `prophet>=1.1` - Forecasting
- `streamlit>=1.20.0` - Web dashboard

## 💡 Usage Examples

### Example 1: Create and Export All Visualizations
```python
from interactive_visualizations import main, export_interactive_html

figures = main()
export_interactive_html(figures, 'my_dashboard.html')
```

### Example 2: Create Custom Combination
```python
from interactive_visualizations import *

df, df_sales = load_and_prepare_data()

fig1 = create_sales_trend_chart(df_sales)
fig2 = create_forecast_visualization(df_sales)[0]
fig3 = create_seasonal_decomposition_chart(df_sales)

# Save as HTML
from plotly.subplots import make_subplots
# Combine and export...
```

### Example 3: Filter and Visualize
```python
# Filter to specific date range
filtered_sales = df_sales[df_sales['Date'] > '2020-01-01']
fig = create_sales_trend_chart(filtered_sales)
fig.show()
```

## 🎨 Customization

### Change Chart Colors
Edit the color values in `interactive_visualizations.py`:
```python
line=dict(color='#1f77b4', width=2)  # Change hex color
```

### Adjust Forecast Period
In `create_forecast_visualization()`:
```python
future = model.make_future_dataframe(periods=52, freq='W')  # Change to 52 weeks
```

### Modify Confidence Intervals
```python
# Add different confidence levels in forecasting
```

## 📊 File Structure

```
project_fol/
├── interactive_visualizations.py    # Core visualization functions
├── view_visualizations.py           # Simple HTML export script
├── streamlit_dashboard.py           # Full Streamlit app
├── README.md                        # Original project README
└── data_processed/
    └── walmart_cleaned.csv
```

## 🔧 Troubleshooting

### Issue: Port 8501 already in use
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

### Issue: Module not found
```bash
# Make sure you're in the virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

pip install -e .
```

### Issue: Plotly not rendering
```bash
# Try installing with specific version
pip install plotly==5.14.0
```

## 📈 Performance Tips

- **Large Datasets**: Use `df_sales.head(1000)` for initial exploration
- **Multiple Charts**: Pre-calculate forecast once, reuse results
- **Export**: Use static images for presentations (dashboard has download option)

## 🎓 Learning Resources

- **Plotly Docs**: https://plotly.com/python/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Prophet Docs**: https://facebook.github.io/prophet/docs/quick_start.html

## ✨ Next Steps

1. **Start with HTML Dashboard**: `python view_visualizations.py`
2. **Explore Streamlit**: `streamlit run streamlit_dashboard.py`
3. **Customize colors/labels** in `interactive_visualizations.py`
4. **Add your own visualizations** following the same pattern

---

**Happy Visualizing! 📊✨**
