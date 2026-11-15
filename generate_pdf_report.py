from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import os

def generate_walmart_forecast_report(output_filename="Walmart_Sales_Forecast_Report.pdf"):
    """Generate a comprehensive Walmart sales forecast report in PDF format."""
    
    # Load and prepare data
    df = pd.read_csv("NoteBook/Walmart.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    df_prophet = df_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    
    # Train model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    
    # Generate forecast
    future = model.make_future_dataframe(periods=13, freq='W')
    forecast = model.predict(future)
    
    # Validate model
    train = df_prophet.iloc[:-13]
    test = df_prophet.iloc[-13:]
    model_test = Prophet(yearly_seasonality=True)
    model_test.fit(train)
    future_test = model_test.make_future_dataframe(periods=13, freq='W')
    forecast_test = model_test.predict(future_test)
    
    mae = mean_absolute_error(test['y'], forecast_test['yhat'][-13:])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat'][-13:]))
    
    # Create PDF
    pdf_filename = output_filename.replace('.pdf', '') + f"_{datetime.now().strftime('%Y%m%d')}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph("📊 Walmart Sales Forecasting Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report info
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Historical Records', f"{len(df_prophet)}"],
        ['Average Weekly Sales', f"${df_prophet['y'].mean():,.0f}"],
        ['Maximum Weekly Sales', f"${df_prophet['y'].max():,.0f}"],
        ['Minimum Weekly Sales', f"${df_prophet['y'].min():,.0f}"],
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Model Performance
    story.append(Paragraph("Model Performance Metrics", styles['Heading2']))
    perf_data = [
        ['Metric', 'Value'],
        ['Mean Absolute Error (MAE)', f"${mae:,.2f}"],
        ['Root Mean Squared Error (RMSE)', f"${rmse:,.2f}"],
    ]
    perf_table = Table(perf_data, colWidths=[3*inch, 2*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Forecast Table
    story.append(Paragraph("13-Week Sales Forecast", styles['Heading2']))
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13)
    forecast_data = [['Date', 'Forecast', 'Lower Bound', 'Upper Bound']]
    for _, row in forecast_display.iterrows():
        forecast_data.append([
            row['ds'].strftime('%Y-%m-%d'),
            f"${row['yhat']:,.0f}",
            f"${row['yhat_lower']:,.0f}",
            f"${row['yhat_upper']:,.0f}"
        ])
    forecast_table = Table(forecast_data, colWidths=[1.2*inch, 1.3*inch, 1.3*inch, 1.2*inch])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(forecast_table)
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF Report generated successfully: {pdf_filename}")
    return pdf_filename

if __name__ == "__main__":
    generate_walmart_forecast_report()
