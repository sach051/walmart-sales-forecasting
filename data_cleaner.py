"""
Data Cleaning and Preprocessing Script
Prepares raw Walmart sales data for Prophet forecasting model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess Walmart sales data for Prophet model."""
    
    def __init__(self, raw_data_path="NoteBook/Walmart.csv", output_dir="data_processed"):
        """Initialize data cleaner with file paths."""
        self.raw_data_path = raw_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load raw data from CSV."""
        try:
            logger.info(f"Loading data from {self.raw_data_path}...")
            self.df = pd.read_csv(self.raw_data_path)
            logger.info(f"✅ Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def inspect_data(self):
        """Display data information."""
        if self.df is None:
            logger.warning("Data not loaded. Call load_data() first.")
            return
        
        logger.info("\n" + "="*60)
        logger.info("DATA INSPECTION")
        logger.info("="*60)
        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"\nFirst few rows:\n{self.df.head()}")
        logger.info(f"\nData types:\n{self.df.dtypes}")
        logger.info(f"\nMissing values:\n{self.df.isnull().sum()}")
        logger.info(f"\nBasic statistics:\n{self.df.describe()}")
    
    def clean_data(self):
        """Clean data for Prophet model."""
        logger.info("\n" + "="*60)
        logger.info("DATA CLEANING")
        logger.info("="*60)
        
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            raise ValueError("Data not loaded")
        
        df = self.df.copy()
        
        # Step 1: Handle date column
        logger.info("\n1️⃣  Processing date column...")
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
                logger.info("   ✅ Date converted to datetime")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not parse with DD-MM-YYYY format: {e}")
                df['Date'] = pd.to_datetime(df['Date'])
                logger.info("   ✅ Date auto-detected and converted")
        else:
            logger.error("   ❌ 'Date' column not found!")
            raise ValueError("Date column not found")
        
        # Step 2: Aggregate sales by date
        logger.info("\n2️⃣  Aggregating sales by date...")
        initial_rows = len(df)
        df_agg = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        logger.info(f"   ✅ Aggregated {initial_rows} rows → {len(df_agg)} unique dates")
        
        # Step 3: Create Prophet-format dataframe
        logger.info("\n3️⃣  Creating Prophet format (ds, y)...")
        self.df_cleaned = df_agg.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        self.df_cleaned = self.df_cleaned.sort_values('ds').reset_index(drop=True)
        logger.info(f"   ✅ Columns renamed: {list(self.df_cleaned.columns)}")
        
        # Step 4: Handle missing values
        logger.info("\n4️⃣  Handling missing values...")
        missing_before = self.df_cleaned.isnull().sum().sum()
        
        # Remove rows with missing values
        self.df_cleaned = self.df_cleaned.dropna()
        
        missing_after = self.df_cleaned.isnull().sum().sum()
        logger.info(f"   ✅ Missing values: {missing_before} → {missing_after}")
        
        # Step 5: Data validation
        logger.info("\n5️⃣  Data validation...")
        
        # Check for negative values
        negative_count = (self.df_cleaned['y'] < 0).sum()
        if negative_count > 0:
            logger.warning(f"   ⚠️  Found {negative_count} negative sales values")
            self.df_cleaned = self.df_cleaned[self.df_cleaned['y'] >= 0]
            logger.info(f"   ✅ Removed negative values")
        
        # Check for duplicates
        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"   ⚠️  Found {duplicates} duplicate rows")
            self.df_cleaned = self.df_cleaned.drop_duplicates()
            logger.info(f"   ✅ Removed duplicates")
        
        # Check date continuity
        date_range = (self.df_cleaned['ds'].max() - self.df_cleaned['ds'].min()).days
        expected_rows = date_range // 7 + 1  # Weekly data
        actual_rows = len(self.df_cleaned)
        gap_percentage = ((expected_rows - actual_rows) / expected_rows * 100) if expected_rows > 0 else 0
        logger.info(f"   📅 Date range: {self.df_cleaned['ds'].min().date()} to {self.df_cleaned['ds'].max().date()}")
        logger.info(f"   📊 Records: {actual_rows} (expected ~{expected_rows}, gap: {gap_percentage:.1f}%)")
        
        logger.info(f"\n✅ Data cleaning completed!")
        logger.info(f"Final shape: {self.df_cleaned.shape}")
        logger.info(f"Date range: {self.df_cleaned['ds'].min().date()} to {self.df_cleaned['ds'].max().date()}")
        
        return self.df_cleaned
    
    def save_cleaned_data(self, filename="walmart_cleaned.csv"):
        """Save cleaned data to CSV."""
        if self.df_cleaned is None:
            logger.error("No cleaned data to save. Call clean_data() first.")
            raise ValueError("No cleaned data")
        
        output_path = self.output_dir / filename
        try:
            self.df_cleaned.to_csv(output_path, index=False)
            logger.info(f"\n💾 Cleaned data saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise
    
    def generate_summary_report(self):
        """Generate a summary report of cleaned data."""
        if self.df_cleaned is None:
            logger.error("No cleaned data. Call clean_data() first.")
            return
        
        logger.info("\n" + "="*60)
        logger.info("CLEANED DATA SUMMARY")
        logger.info("="*60)
        logger.info(f"\nShape: {self.df_cleaned.shape}")
        logger.info(f"\nColumn Names: {list(self.df_cleaned.columns)}")
        logger.info(f"\nData Types:\n{self.df_cleaned.dtypes}")
        logger.info(f"\nFirst 5 rows:\n{self.df_cleaned.head()}")
        logger.info(f"\nLast 5 rows:\n{self.df_cleaned.tail()}")
        logger.info(f"\nStatistics:\n{self.df_cleaned.describe()}")
        logger.info(f"\nMissing Values:\n{self.df_cleaned.isnull().sum()}")
        logger.info("\n" + "="*60)
    
    def run_full_pipeline(self):
        """Execute complete data cleaning pipeline."""
        logger.info("\n" + "🔄 "*10)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("🔄 "*10 + "\n")
        
        try:
            self.load_data()
            self.inspect_data()
            self.clean_data()
            self.generate_summary_report()
            self.save_cleaned_data()
            
            logger.info("\n" + "✅ "*10)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("✅ "*10)
            
            return self.df_cleaned
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize cleaner
        cleaner = DataCleaner(
            raw_data_path="NoteBook/Walmart.csv",
            output_dir="data_processed"
        )
        
        # Run pipeline
        cleaned_data = cleaner.run_full_pipeline()
        
        logger.info("\n📁 Output Location: data_processed/walmart_cleaned.csv")
        logger.info("🚀 Use this cleaned data with the forecasting model")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
