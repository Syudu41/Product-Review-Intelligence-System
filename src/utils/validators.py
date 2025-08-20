"""
Data validation utilities for Review Intelligence Engine
"""
import pandas as pd
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime, date
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataValidator:
    """Comprehensive data validation for reviews and products"""
    
    @staticmethod
    def validate_review_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate review DataFrame and return validation report"""
        report = {
            "total_rows": len(df),
            "valid_rows": 0,
            "errors": [],
            "warnings": [],
            "quality_metrics": {}
        }
        
        # Required columns
        required_columns = ['product_id', 'rating', 'review_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            report["errors"].append(f"Missing required columns: {missing_columns}")
            return report
        
        # Validate ratings
        invalid_ratings = df[~df['rating'].between(1, 5)]
        if len(invalid_ratings) > 0:
            report["errors"].append(f"Invalid ratings found: {len(invalid_ratings)} rows")
        
        # Validate product IDs
        missing_product_ids = df[df['product_id'].isna() | (df['product_id'] == '')]
        if len(missing_product_ids) > 0:
            report["warnings"].append(f"Missing product IDs: {len(missing_product_ids)} rows")
        
        # Validate review text
        empty_reviews = df[df['review_text'].isna() | (df['review_text'].str.strip() == '')]
        if len(empty_reviews) > 0:
            report["warnings"].append(f"Empty review texts: {len(empty_reviews)} rows")
        
        # Validate dates if present
        if 'review_date' in df.columns:
            invalid_dates = DataValidator._validate_dates(df['review_date'])
            if invalid_dates > 0:
                report["warnings"].append(f"Invalid dates: {invalid_dates} rows")
        
        # Quality metrics
        report["quality_metrics"] = {
            "avg_review_length": df['review_text'].str.len().mean() if 'review_text' in df.columns else 0,
            "rating_distribution": df['rating'].value_counts().to_dict() if 'rating' in df.columns else {},
            "missing_data_percentage": (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Calculate valid rows
        valid_mask = (
            df['rating'].between(1, 5) &
            df['product_id'].notna() &
            (df['product_id'] != '') &
            df['review_text'].notna() &
            (df['review_text'].str.strip() != '')
        )
        report["valid_rows"] = valid_mask.sum()
        
        return report
    
    @staticmethod
    def validate_product_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate product DataFrame"""
        report = {
            "total_rows": len(df),
            "valid_rows": 0,
            "errors": [],
            "warnings": []
        }
        
        # Required columns
        required_columns = ['product_id', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            report["errors"].append(f"Missing required columns: {missing_columns}")
            return report
        
        # Validate product IDs
        duplicate_products = df[df['product_id'].duplicated()]
        if len(duplicate_products) > 0:
            report["warnings"].append(f"Duplicate product IDs: {len(duplicate_products)} rows")
        
        # Validate product names
        missing_names = df[df['name'].isna() | (df['name'].str.strip() == '')]
        if len(missing_names) > 0:
            report["warnings"].append(f"Missing product names: {len(missing_names)} rows")
        
        # Calculate valid rows
        valid_mask = (
            df['product_id'].notna() &
            (df['product_id'] != '') &
            df['name'].notna() &
            (df['name'].str.strip() != '')
        )
        report["valid_rows"] = valid_mask.sum()
        
        return report
    
    @staticmethod
    def _validate_dates(date_series: pd.Series) -> int:
        """Validate date column and return number of invalid dates"""
        try:
            pd.to_datetime(date_series, errors='coerce')
            invalid_count = date_series.isna().sum()
            return invalid_count
        except Exception:
            return len(date_series)
    
    @staticmethod
    def validate_review_text_quality(text: str) -> Dict[str, Any]:
        """Validate individual review text quality"""
        if not isinstance(text, str):
            return {"valid": False, "issues": ["Not a string"]}
        
        issues = []
        quality_score = 100
        
        # Length checks
        if len(text.strip()) < 10:
            issues.append("Too short (< 10 characters)")
            quality_score -= 30
        elif len(text.strip()) > 5000:
            issues.append("Too long (> 5000 characters)")
            quality_score -= 10
        
        # Language checks
        if not re.search(r'[a-zA-Z]', text):
            issues.append("No alphabetic characters")
            quality_score -= 50
        
        # Spam indicators
        repeated_chars = re.findall(r'(.)\1{5,}', text.lower())
        if repeated_chars:
            issues.append("Excessive repeated characters")
            quality_score -= 20
        
        # Caps lock check
        if len(text) > 20 and text.isupper():
            issues.append("All caps text")
            quality_score -= 15
        
        # Special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?]', text)) / len(text)
        if special_char_ratio > 0.3:
            issues.append("Too many special characters")
            quality_score -= 25
        
        return {
            "valid": len(issues) == 0,
            "quality_score": max(0, quality_score),
            "issues": issues,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    @staticmethod
    def clean_review_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize review data"""
        logger.info(f"Cleaning review data: {len(df)} rows")
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'reviewText': 'review_text',
            'reviewerID': 'user_id',
            'asin': 'product_id',
            'overall': 'rating',
            'summary': 'review_title',
            'unixReviewTime': 'review_timestamp',
            'reviewTime': 'review_date',
            'helpful': 'helpful_votes'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in cleaned_df.columns and new_col not in cleaned_df.columns:
                cleaned_df[new_col] = cleaned_df[old_col]
        
        # Clean text fields
        text_columns = ['review_text', 'review_title']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str)
                cleaned_df[col] = cleaned_df[col].str.strip()
                cleaned_df[col] = cleaned_df[col].replace('nan', '')
        
        # Convert ratings to integer
        if 'rating' in cleaned_df.columns:
            cleaned_df['rating'] = pd.to_numeric(cleaned_df['rating'], errors='coerce')
            cleaned_df['rating'] = cleaned_df['rating'].fillna(0).astype(int)
        
        # Clean product IDs
        if 'product_id' in cleaned_df.columns:
            cleaned_df['product_id'] = cleaned_df['product_id'].astype(str).str.strip()
        
        # Convert dates
        if 'review_date' in cleaned_df.columns:
            cleaned_df['review_date'] = pd.to_datetime(cleaned_df['review_date'], errors='coerce')
        
        # Remove rows with critical missing data
        before_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['product_id', 'rating', 'review_text'])
        after_count = len(cleaned_df)
        
        logger.info(f"Removed {before_count - after_count} rows with missing critical data")
        logger.info(f"Cleaned data: {after_count} rows")
        
        return cleaned_df

def create_validation_report(df: pd.DataFrame, data_type: str = "review") -> Dict[str, Any]:
    """Create comprehensive validation report"""
    logger.info(f"Creating validation report for {data_type} data")
    
    if data_type == "review":
        report = DataValidator.validate_review_data(df)
    elif data_type == "product":
        report = DataValidator.validate_product_data(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Add general statistics
    report["validation_timestamp"] = datetime.now().isoformat()
    report["data_type"] = data_type
    report["columns"] = list(df.columns)
    report["data_types"] = df.dtypes.to_dict()
    
    # Log summary
    logger.info(f"Validation complete: {report['valid_rows']}/{report['total_rows']} valid rows")
    if report['errors']:
        logger.error(f"Validation errors: {report['errors']}")
    if report['warnings']:
        logger.warning(f"Validation warnings: {report['warnings']}")
    
    return report