# Walmart Sales Forecasting Application 📈

A machine learning-based sales forecasting application developed as a group project for the Business Finance module at Uva Wellassa University.

The application uses Facebook Prophet for time-series forecasting and provides interactive data visualizations and reporting features to support sales analysis and business decision-making.

## 🚀 Key Features

- **Time-Series Forecasting:** Predicts future Walmart sales using Facebook Prophet.
- **Interactive Dashboard:** Explores sales data through a Streamlit dashboard.
- **Data Visualization:** Presents sales trends and patterns using Plotly.
- **Model Evaluation:** Uses Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Seasonal Analysis:** Explores sales trends and seasonal patterns.
- **PDF Reporting:** Generates reports using ReportLab.
- **Data Processing:** Cleans and prepares historical sales data for forecasting.

## 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Facebook Prophet | Time-series forecasting |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Model evaluation |
| ReportLab | PDF report generation |

## 📂 Project Structure

```text
walmart-sales-forecasting/
│
├── NoteBook/
│   └── Walmart.csv
│
├── data_processed/
│   └── walmart_cleaned.csv
│
├── README.md
├── INTERACTIVE_VISUALIZATIONS_README.md
├── data_cleaner.py
├── forecast_gui.py
├── generate_pdf_report.py
├── interactive_visualizations.py
├── main.py
├── streamlit_dashboard.py
├── view_visualizations.py
├── pyproject.toml
└── uv.lock
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd walmart-sales-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install .
```

The required packages are defined in `pyproject.toml`.

## ▶️ Running the Application

### Launch the Streamlit Dashboard

```bash
streamlit run streamlit_dashboard.py
```

### Run the Main Forecasting Script

```bash
python main.py
```

### Generate PDF Reports

```bash
python generate_pdf_report.py
```

### View Interactive Visualizations

```bash
python view_visualizations.py
```

## 📊 Model Evaluation

The project uses the following evaluation metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted sales.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.

## 🎓 Academic Background

**Module:** Business Finance  
**University:** Uva Wellassa University of Sri Lanka  
**Project Type:** Group Project

This project provided academic exposure to predictive analytics, sales forecasting, time-series analysis, and the application of machine learning techniques in business decision-making.

### Team Contributions and Acknowledgement

This was a collaborative academic project. The original application was developed and uploaded to GitHub by another group member.

**My contribution:** [Describe your actual contribution to the group project here.]

Original repository: https://github.com/nipunabandara/BaForecast

## 📚 Additional Documentation

For more information about the interactive Plotly visualizations, refer to:

`INTERACTIVE_VISUALIZATIONS_README.md`

## 🔗 References

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

*This repository acknowledges the original developer and the collaborative academic work behind the project.*
