"""
Data Loader Module for Product Review Intelligence System
Handles loading Amazon reviews dataset from Kaggle and other data sources
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
import zipfile
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json
import hashlib

from config import DATABASE_URL, DATA_DIR, KAGGLE_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and initial processing of review datasets
    """
    
    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = DATABASE_URL.replace('sqlite:///', '')
        
        # Kaggle dataset info
        self.kaggle_dataset = "snap/amazon-fine-food-reviews"  # Example dataset
        self.dataset_file = "Reviews.csv"
        
    def setup_kaggle_api(self) -> bool:
        """
        Setup Kaggle API credentials
        Returns True if successful, False otherwise
        """
        try:
            import kaggle
            
            # Check if kaggle.json exists
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_json = kaggle_dir / 'kaggle.json'
            
            if not kaggle_json.exists():
                logger.warning("Kaggle API credentials not found. Please set up kaggle.json")
                logger.info("1. Go to https://www.kaggle.com/account")
                logger.info("2. Create New API Token")
                logger.info("3. Save kaggle.json to ~/.kaggle/")
                return False
                
            # Test API connection
            kaggle.api.authenticate()
            logger.info("Kaggle API authenticated successfully")
            return True
            
        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Kaggle API setup failed: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_name: str = None) -> bool:
        """
        Download dataset from Kaggle
        """
        if not dataset_name:
            dataset_name = self.kaggle_dataset
            
        try:
            import kaggle
            
            logger.info(f"Downloading dataset: {dataset_name}")
            
            # Download to data directory
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(self.data_dir),
                unzip=True
            )
            
            logger.info(f"Dataset downloaded to: {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def load_csv_dataset(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load CSV dataset with error handling and sampling
        """
        try:
            full_path = self.data_dir / file_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {full_path}")
            
            logger.info(f"Loading dataset from: {full_path}")
            
            # Read CSV with chunking for large files
            if sample_size:
                df = pd.read_csv(full_path, nrows=sample_size)
                logger.info(f"Loaded {len(df)} rows (sampled)")
            else:
                df = pd.read_csv(full_path)
                logger.info(f"Loaded {len(df)} rows (full dataset)")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV dataset: {e}")
            raise
    
    def load_amazon_reviews(self, sample_size: int = 50000) -> pd.DataFrame:
        """
        Load Amazon reviews dataset with standardized column names
        """
        try:
            # Try multiple possible filenames
            possible_files = [
                "Reviews.csv",
                "amazon_reviews.csv", 
                "amazon-fine-food-reviews.csv",
                "reviews.csv"
            ]
            
            df = None
            for filename in possible_files:
                try:
                    df = self.load_csv_dataset(filename, sample_size)
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                raise FileNotFoundError("No Amazon reviews dataset found")
            
            # Standardize column names
            df = self._standardize_amazon_columns(df)
            
            # Basic validation
            required_cols = ['product_id', 'user_id', 'rating', 'review_text']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            
            logger.info(f"Amazon reviews loaded: {len(df)} rows")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Amazon reviews: {e}")
            raise
    
    def _standardize_amazon_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different Amazon review datasets
        """
        # Common column mappings
        column_mappings = {
            'ProductId': 'product_id',
            'UserId': 'user_id', 
            'Score': 'rating',
            'Text': 'review_text',
            'Summary': 'review_title',
            'Time': 'timestamp',
            'HelpfulnessNumerator': 'helpful_votes',
            'HelpfulnessDenominator': 'total_votes',
            
            # Alternative names
            'asin': 'product_id',
            'reviewerID': 'user_id',
            'overall': 'rating',
            'reviewText': 'review_text',
            'summary': 'review_title',
            'unixReviewTime': 'timestamp',
            'helpful': 'helpful_votes'
        }
        
        # Rename columns
        df = df.rename(columns=column_mappings)
        
        # Convert timestamp if needed
        if 'timestamp' in df.columns:
            try:
                # Try unix timestamp first
                if df['timestamp'].dtype in ['int64', 'float64']:
                    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    df['date'] = pd.to_datetime(df['timestamp'])
            except:
                # If conversion fails, use current date
                df['date'] = datetime.now()
        else:
            df['date'] = datetime.now()
        
        # Add missing columns with defaults
        if 'helpful_votes' not in df.columns:
            df['helpful_votes'] = 0
            
        if 'total_votes' not in df.columns:
            df['total_votes'] = 0
            
        if 'review_title' not in df.columns:
            df['review_title'] = ''
        
        # Ensure required columns exist
        required_defaults = {
            'product_id': lambda: df.index.astype(str),
            'user_id': lambda: df.index.astype(str), 
            'rating': lambda: np.random.randint(1, 6, len(df)),
            'review_text': lambda: ['Sample review'] * len(df)
        }
        
        for col, default_func in required_defaults.items():
            if col not in df.columns:
                logger.warning(f"Creating default {col} column")
                df[col] = default_func()
        
        return df
    
    def save_to_database(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        Save DataFrame to SQLite database
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='append', index=False)
                logger.info(f"Saved {len(df)} rows to {table_name} table")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            return False
    
    def load_sample_dataset(self) -> pd.DataFrame:
        """
        Load a sample dataset for testing (creates synthetic data if no real data available)
        """
        try:
            # Try to load real dataset first
            return self.load_amazon_reviews(sample_size=1000)
            
        except Exception as e:
            logger.warning(f"Real dataset not available: {e}")
            logger.info("Creating synthetic sample dataset")
            
            # Create synthetic data for testing
            np.random.seed(42)
            n_samples = 1000
            
            synthetic_data = {
                'product_id': [f'B{str(i).zfill(8)}' for i in range(n_samples)],
                'user_id': [f'U{str(i).zfill(6)}' for i in np.random.randint(0, 100, n_samples)],
                'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
                'review_text': [f'Sample review text {i}' for i in range(n_samples)],
                'review_title': [f'Review title {i}' for i in range(n_samples)],
                'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
                'helpful_votes': np.random.randint(0, 10, n_samples),
                'total_votes': np.random.randint(0, 15, n_samples)
            }
            
            df = pd.DataFrame(synthetic_data)
            logger.info(f"Created synthetic dataset: {len(df)} rows")
            
            return df
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about loaded dataset
        """
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'rating_distribution': df['rating'].value_counts().to_dict() if 'rating' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_products': df['product_id'].nunique() if 'product_id' in df.columns else 0,
            'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0
        }
        
        return info
    
    def export_dataset_summary(self, df: pd.DataFrame, output_file: str = 'dataset_summary.json'):
        """
        Export dataset summary to JSON file
        """
        try:
            info = self.get_dataset_info(df)
            
            # Convert datetime objects to strings
            if info['date_range']['start']:
                info['date_range']['start'] = info['date_range']['start'].isoformat()
            if info['date_range']['end']:
                info['date_range']['end'] = info['date_range']['end'].isoformat()
            
            output_path = self.data_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"Dataset summary exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export dataset summary: {e}")


def main():
    """
    Main function for testing data loader
    """
    loader = DataLoader()
    
    try:
        # Setup Kaggle API
        if loader.setup_kaggle_api():
            # Try to download real dataset
            if loader.download_kaggle_dataset():
                df = loader.load_amazon_reviews(sample_size=5000)
            else:
                df = loader.load_sample_dataset()
        else:
            df = loader.load_sample_dataset()
        
        # Display dataset info
        info = loader.get_dataset_info(df)
        print("\n=== Dataset Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Export summary
        loader.export_dataset_summary(df)
        
        # Save sample to database for testing
        sample_df = df.head(100)
        loader.save_to_database(sample_df, 'reviews')
        
        print(f"\nSuccess! Loaded {len(df)} reviews")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()