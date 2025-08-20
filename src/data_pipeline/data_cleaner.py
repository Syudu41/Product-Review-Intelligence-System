"""
Data Cleaner Module for Product Review Intelligence System
Handles preprocessing, cleaning, and normalization of review data
"""

import pandas as pd
import numpy as np
import re
import string
import logging
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import html
import unicodedata

from config import DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing for review datasets
    """
    
    def __init__(self):
        self.db_path = DATABASE_URL.replace('sqlite:///', '')
        self.stemmer = PorterStemmer()
        self._setup_nltk()
        
        # Cleaning statistics
        self.cleaning_stats = {
            'original_rows': 0,
            'rows_after_cleaning': 0,
            'duplicates_removed': 0,
            'nulls_removed': 0,
            'invalid_ratings_fixed': 0,
            'text_normalized': 0,
            'outliers_removed': 0
        }
    
    def _setup_nltk(self):
        """
        Download required NLTK data
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            logger.warning("Could not load stopwords, using empty set")
            self.stop_words = set()
    
    def clean_dataset(self, df: pd.DataFrame, target_size: int = 20000) -> pd.DataFrame:
        """
        Main cleaning pipeline for review dataset
        """
        logger.info(f"Starting data cleaning pipeline for {len(df)} rows")
        self.cleaning_stats['original_rows'] = len(df)
        
        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 2: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 3: Validate and fix data types
        df = self._validate_data_types(df)
        
        # Step 4: Clean text fields
        df = self._clean_text_fields(df)
        
        # Step 5: Remove outliers
        df = self._remove_outliers(df)
        
        # Step 6: Sample to target size if needed
        if len(df) > target_size:
            df = self._smart_sample(df, target_size)
            logger.info(f"Sampled dataset to {target_size} rows")
        
        # Step 7: Add derived features
        df = self._add_derived_features(df)
        
        # Step 8: Final validation
        df = self._final_validation(df)
        
        self.cleaning_stats['rows_after_cleaning'] = len(df)
        
        logger.info("Data cleaning completed")
        logger.info(f"Cleaning statistics: {self.cleaning_stats}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate reviews based on content similarity
        """
        logger.info("Removing duplicates...")
        
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates(subset=['product_id', 'user_id', 'review_text'])
        
        # Remove near-duplicates (same user, product, similar text length)
        if 'review_text' in df.columns:
            df['text_length'] = df['review_text'].str.len()
            df = df.drop_duplicates(subset=['product_id', 'user_id', 'text_length'])
            df = df.drop(columns=['text_length'])
        
        duplicates_removed = initial_count - len(df)
        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies
        """
        logger.info("Handling missing values...")
        
        initial_count = len(df)
        
        # Required fields - drop rows if missing
        required_fields = ['product_id', 'user_id', 'rating', 'review_text']
        for field in required_fields:
            if field in df.columns:
                before_count = len(df)
                df = df.dropna(subset=[field])
                after_count = len(df)
                logger.info(f"Dropped {before_count - after_count} rows missing {field}")
        
        # Optional fields - fill with defaults
        if 'review_title' in df.columns:
            df['review_title'] = df['review_title'].fillna('')
        
        if 'helpful_votes' in df.columns:
            df['helpful_votes'] = df['helpful_votes'].fillna(0)
        
        if 'total_votes' in df.columns:
            df['total_votes'] = df['total_votes'].fillna(0)
        
        if 'date' in df.columns:
            df['date'] = df['date'].fillna(datetime.now())
        
        # Remove rows with empty review text
        if 'review_text' in df.columns:
            df = df[df['review_text'].str.strip() != '']
        
        nulls_removed = initial_count - len(df)
        self.cleaning_stats['nulls_removed'] = nulls_removed
        
        logger.info(f"Handled missing values, removed {nulls_removed} rows")
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix data types
        """
        logger.info("Validating data types...")
        
        # Fix rating values
        if 'rating' in df.columns:
            initial_invalid = len(df)
            
            # Convert to numeric
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            
            # Remove invalid ratings
            df = df.dropna(subset=['rating'])
            
            # Ensure ratings are in valid range (1-5)
            df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
            
            # Round to integers
            df['rating'] = df['rating'].round().astype(int)
            
            invalid_fixed = initial_invalid - len(df)
            self.cleaning_stats['invalid_ratings_fixed'] = invalid_fixed
            logger.info(f"Fixed {invalid_fixed} invalid ratings")
        
        # Fix helpful votes
        if 'helpful_votes' in df.columns:
            df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0)
            df['helpful_votes'] = df['helpful_votes'].astype(int)
        
        if 'total_votes' in df.columns:
            df['total_votes'] = pd.to_numeric(df['total_votes'], errors='coerce').fillna(0)
            df['total_votes'] = df['total_votes'].astype(int)
        
        # Ensure helpful_votes <= total_votes
        if 'helpful_votes' in df.columns and 'total_votes' in df.columns:
            mask = df['helpful_votes'] > df['total_votes']
            df.loc[mask, 'total_votes'] = df.loc[mask, 'helpful_votes']
        
        # Fix date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = df['date'].fillna(datetime.now())
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize text fields
        """
        logger.info("Cleaning text fields...")
        
        text_fields = ['review_text', 'review_title']
        
        for field in text_fields:
            if field in df.columns:
                logger.info(f"Cleaning {field}...")
                
                # Basic cleaning
                df[field] = df[field].astype(str)
                df[field] = df[field].apply(self._clean_text)
                
                # Remove very short reviews (less than 10 characters)
                if field == 'review_text':
                    initial_count = len(df)
                    df = df[df[field].str.len() >= 10]
                    removed = initial_count - len(df)
                    logger.info(f"Removed {removed} reviews with text < 10 characters")
        
        self.cleaning_stats['text_normalized'] = len(df)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Clean individual text string
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and decode HTML entities
        text = str(text)
        text = html.unescape(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Fix common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers
        """
        logger.info("Removing outliers...")
        
        initial_count = len(df)
        
        # Remove reviews with extreme lengths
        if 'review_text' in df.columns:
            df['text_length'] = df['review_text'].str.len()
            
            # Remove reviews longer than 99.5th percentile or shorter than 0.5th percentile
            lower_bound = df['text_length'].quantile(0.005)
            upper_bound = df['text_length'].quantile(0.995)
            
            df = df[(df['text_length'] >= lower_bound) & (df['text_length'] <= upper_bound)]
            df = df.drop(columns=['text_length'])
        
        # Remove users with excessive review counts (likely bots)
        if 'user_id' in df.columns:
            user_counts = df['user_id'].value_counts()
            # Remove users with more than 95th percentile of reviews
            max_reviews = user_counts.quantile(0.95)
            valid_users = user_counts[user_counts <= max_reviews].index
            df = df[df['user_id'].isin(valid_users)]
        
        outliers_removed = initial_count - len(df)
        self.cleaning_stats['outliers_removed'] = outliers_removed
        
        logger.info(f"Removed {outliers_removed} outlier rows")
        return df
    
    def _smart_sample(self, df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Intelligently sample dataset to target size while maintaining distribution
        """
        logger.info(f"Smart sampling to {target_size} rows...")
        
        if len(df) <= target_size:
            return df
        
        # Stratified sampling by rating to maintain distribution
        if 'rating' in df.columns:
            sample_dfs = []
            rating_counts = df['rating'].value_counts()
            
            for rating in rating_counts.index:
                rating_df = df[df['rating'] == rating]
                
                # Calculate proportional sample size
                proportion = len(rating_df) / len(df)
                sample_size = max(1, int(target_size * proportion))
                
                if len(rating_df) <= sample_size:
                    sample_dfs.append(rating_df)
                else:
                    sample_dfs.append(rating_df.sample(n=sample_size, random_state=42))
            
            sampled_df = pd.concat(sample_dfs, ignore_index=True)
            
            # If we're still over target, randomly sample remaining
            if len(sampled_df) > target_size:
                sampled_df = sampled_df.sample(n=target_size, random_state=42)
            
            return sampled_df
        else:
            # Simple random sampling if no rating column
            return df.sample(n=target_size, random_state=42)
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for analysis
        """
        logger.info("Adding derived features...")
        
        # Text-based features
        if 'review_text' in df.columns:
            df['review_length'] = df['review_text'].str.len()
            df['word_count'] = df['review_text'].str.split().str.len()
            df['char_count'] = df['review_text'].str.len()
            df['sentence_count'] = df['review_text'].str.count('[.!?]+') + 1
            
            # Readability metrics
            df['avg_word_length'] = df['review_text'].apply(self._avg_word_length)
            df['punctuation_ratio'] = df['review_text'].apply(self._punctuation_ratio)
            df['capital_ratio'] = df['review_text'].apply(self._capital_ratio)
        
        # Helpfulness ratio
        if 'helpful_votes' in df.columns and 'total_votes' in df.columns:
            df['helpfulness_ratio'] = np.where(
                df['total_votes'] > 0,
                df['helpful_votes'] / df['total_votes'],
                0
            )
        
        # Date-based features
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Rating deviation from product average
        if 'rating' in df.columns and 'product_id' in df.columns:
            product_avg_rating = df.groupby('product_id')['rating'].mean()
            df['rating_deviation'] = df.apply(
                lambda row: row['rating'] - product_avg_rating[row['product_id']], 
                axis=1
            )
        
        return df
    
    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length"""
        if not text or pd.isna(text):
            return 0
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def _punctuation_ratio(self, text: str) -> float:
        """Calculate ratio of punctuation characters"""
        if not text or pd.isna(text):
            return 0
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        return punctuation_count / len(text) if len(text) > 0 else 0
    
    def _capital_ratio(self, text: str) -> float:
        """Calculate ratio of capital letters"""
        if not text or pd.isna(text):
            return 0
        capital_count = sum(1 for char in text if char.isupper())
        letter_count = sum(1 for char in text if char.isalpha())
        return capital_count / letter_count if letter_count > 0 else 0
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation and cleanup
        """
        logger.info("Final validation...")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Ensure all required columns exist
        required_columns = ['product_id', 'user_id', 'rating', 'review_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Final data type validation
        df['rating'] = df['rating'].astype(int)
        df['product_id'] = df['product_id'].astype(str)
        df['user_id'] = df['user_id'].astype(str)
        df['review_text'] = df['review_text'].astype(str)
        
        # Remove any remaining empty reviews
        df = df[df['review_text'].str.strip() != '']
        
        logger.info(f"Final validation complete. Dataset has {len(df)} rows")
        
        return df
    
    def get_cleaning_summary(self) -> Dict:
        """
        Get summary of cleaning operations
        """
        summary = self.cleaning_stats.copy()
        
        if summary['original_rows'] > 0:
            summary['retention_rate'] = summary['rows_after_cleaning'] / summary['original_rows']
            summary['data_quality_score'] = (
                (summary['rows_after_cleaning'] - summary['nulls_removed'] - summary['duplicates_removed']) 
                / summary['original_rows']
            )
        else:
            summary['retention_rate'] = 0
            summary['data_quality_score'] = 0
        
        return summary
    
    def save_cleaned_data(self, df: pd.DataFrame, table_name: str = 'cleaned_reviews') -> bool:
        """
        Save cleaned data to database
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(df)} cleaned rows to {table_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to save cleaned data: {e}")
            return False


def main():
    """
    Main function for testing data cleaner
    """
    from data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    
    try:
        # Load sample dataset
        df = loader.load_sample_dataset()
        logger.info(f"Loaded {len(df)} rows for cleaning")
        
        # Clean the data
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataset(df, target_size=1000)
        
        # Display results
        summary = cleaner.get_cleaning_summary()
        print("\n=== Data Cleaning Summary ===")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
        print(f"\n=== Cleaned Dataset Info ===")
        print(f"Final rows: {len(cleaned_df)}")
        print(f"Columns: {list(cleaned_df.columns)}")
        print(f"Rating distribution: {cleaned_df['rating'].value_counts().to_dict()}")
        
        # Save cleaned data
        cleaner.save_cleaned_data(cleaned_df)
        
        print("\nData cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise


if __name__ == "__main__":
    main()