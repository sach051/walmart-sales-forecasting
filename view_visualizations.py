"""
Quick script to view interactive Plotly visualizations
Run this to open visualizations in your browser
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from interactive_visualizations import main as generate_visualizations

if __name__ == "__main__":
    print("\n🚀 Generating Interactive Walmart Sales Dashboard...\n")
    
    figures = generate_visualizations()
    
    print("\n" + "="*60)
    print("✅ Dashboard created successfully!")
    print("="*60)
    print("\n📂 Your dashboard has been saved as: walmart_dashboard.html")
    print("\n🌐 To view individual visualizations, uncomment the lines below:")
    print("""
# figures['Sales Trend'].show()
# figures['26-Week Forecast'].show()
# figures['Seasonal Decomposition'].show()
# figures['Sales by Store'].show()
# figures['Monthly Distribution'].show()
# figures['Year Comparison'].show()
# figures['Store Sales Heatmap'].show()
    """)
    
    # Uncomment any of the below to view individual charts
    # figures['Sales Trend'].show()
    # figures['26-Week Forecast'].show()
    # figures['Seasonal Decomposition'].show()
