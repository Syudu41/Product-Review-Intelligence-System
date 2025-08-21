"""
ETL Pipeline Module for Product Review Intelligence System
Main orchestrator for data loading, cleaning, and processing pipeline
Specialized for Amazon Fine Food Reviews dataset
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import json
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path to find config module
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import DATABASE_URL, DATA_DIR
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.data_cleaner import DataCleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    """
    Main ETL Pipeline orchestrator for Amazon Fine Food Reviews
    Manages the complete data processing workflow
    """
    
    def __init__(self):
        self.db_path = DATABASE_URL.replace('sqlite:///', '')
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'stages_completed': [],
            'total_records_processed': 0,
            'final_records': 0,
            'data_quality_score': 0,
            'errors': []
        }
        
        # Configuration for food reviews processing
        self.config = {
            'target_sample_size': 20000,
            'enable_kaggle_download': True,
            'enable_data_validation': True,
            'enable_quality_checks': True,
            'backup_synthetic_data': True,
            'dataset_type': 'Amazon Fine Food Reviews'
        }
    
    def run_full_pipeline(self, target_size: int = 20000) -> Dict:
        """
        Execute the complete ETL pipeline for Amazon Fine Food Reviews
        """
        logger.info("=" * 60)
        logger.info("STARTING ETL PIPELINE EXECUTION - AMAZON FINE FOOD REVIEWS")
        logger.info("=" * 60)
        
        self.pipeline_stats['start_time'] = datetime.now()
        
        try:
            # Stage 1: Data Loading
            logger.info("\n📄 STAGE 1: AMAZON FINE FOOD REVIEWS DATA LOADING")
            raw_data = self._execute_data_loading()
            self.pipeline_stats['stages_completed'].append('data_loading')
            
            # Stage 2: Data Cleaning
            logger.info("\n🧹 STAGE 2: FOOD REVIEWS DATA CLEANING")
            cleaned_data = self._execute_data_cleaning(raw_data, target_size)
            self.pipeline_stats['stages_completed'].append('data_cleaning')
            
            # Stage 3: Data Validation
            logger.info("\n✅ STAGE 3: FOOD REVIEWS DATA VALIDATION")
            validated_data = self._execute_data_validation(cleaned_data)
            self.pipeline_stats['stages_completed'].append('data_validation')
            
            # Stage 4: Database Loading
            logger.info("\n💾 STAGE 4: DATABASE LOADING")
            success = self._execute_database_loading(validated_data)
            self.pipeline_stats['stages_completed'].append('database_loading')
            
            # Stage 5: Quality Assessment
            logger.info("\n📊 STAGE 5: FOOD REVIEWS QUALITY ASSESSMENT")
            quality_report = self._execute_quality_assessment(validated_data)
            self.pipeline_stats['stages_completed'].append('quality_assessment')
            
            # Pipeline completion
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['total_duration'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            self.pipeline_stats['final_records'] = len(validated_data)
            self.pipeline_stats['data_quality_score'] = quality_report.get('overall_score', 0)
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ AMAZON FINE FOOD REVIEWS ETL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            # Generate final report
            final_report = self._generate_final_report(validated_data, quality_report)
            
            return {
                'success': True,
                'data': validated_data,
                'stats': self.pipeline_stats,
                'quality_report': quality_report,
                'final_report': final_report
            }
            
        except Exception as e:
            self.pipeline_stats['errors'].append(str(e))
            logger.error(f"❌ ETL PIPELINE FAILED: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'stats': self.pipeline_stats
            }
    
    def _execute_data_loading(self) -> pd.DataFrame:
        """
        Execute data loading stage for Amazon Fine Food Reviews
        """
        logger.info("Loading Amazon Fine Food Reviews dataset...")
        
        try:
            # Try Kaggle download first
            if self.config['enable_kaggle_download']:
                logger.info("Attempting Amazon Fine Food Reviews dataset download...")
                
                if self.loader.setup_kaggle_api():
                    if self.loader.download_kaggle_dataset():
                        logger.info("✅ Amazon Fine Food Reviews dataset downloaded successfully")
                        raw_data = self.loader.load_amazon_reviews(sample_size=50000)
                    else:
                        logger.warning("⚠️ Kaggle download failed, using synthetic food data")
                        raw_data = self.loader.load_sample_dataset()
                else:
                    logger.warning("⚠️ Kaggle API not available, using synthetic food data")
                    raw_data = self.loader.load_sample_dataset()
            else:
                logger.info("Using synthetic food sample data")
                raw_data = self.loader.load_sample_dataset()
            
            self.pipeline_stats['total_records_processed'] = len(raw_data)
            
            logger.info(f"✅ Food reviews data loading completed: {len(raw_data)} records")
            logger.info(f"Columns: {list(raw_data.columns)}")
            
            # Quick data preview
            self._log_data_preview(raw_data, "Raw Food Reviews Data")
            
            return raw_data
            
        except Exception as e:
            logger.error(f"❌ Food reviews data loading failed: {e}")
            raise
    
    def _execute_data_cleaning(self, data: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Execute data cleaning stage for food reviews
        """
        logger.info(f"Cleaning food reviews dataset (target size: {target_size})...")
        
        try:
            cleaned_data = self.cleaner.clean_dataset(data, target_size)
            
            # Get cleaning summary
            cleaning_summary = self.cleaner.get_cleaning_summary()
            
            logger.info("✅ Food reviews data cleaning completed")
            logger.info(f"Retention rate: {cleaning_summary['retention_rate']:.2%}")
            logger.info(f"Data quality score: {cleaning_summary['data_quality_score']:.2%}")
            
            # Log cleaning statistics
            for key, value in cleaning_summary.items():
                if isinstance(value, (int, float)) and key != 'retention_rate':
                    logger.info(f"{key}: {value}")
            
            self._log_data_preview(cleaned_data, "Cleaned Food Reviews Data")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"❌ Food reviews data cleaning failed: {e}")
            raise
    
    def _execute_data_validation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute data validation stage for food reviews
        """
        logger.info("Validating cleaned food reviews dataset...")
        
        try:
            validation_results = self._validate_dataset(data)
            
            if validation_results['is_valid']:
                logger.info("✅ Food reviews data validation passed")
                
                for check, result in validation_results['checks'].items():
                    status = "✅" if result['passed'] else "❌"
                    logger.info(f"{status} {check}: {result['message']}")
                
                return data
            else:
                failed_checks = [
                    check for check, result in validation_results['checks'].items() 
                    if not result['passed']
                ]
                logger.error(f"❌ Food reviews data validation failed: {failed_checks}")
                raise ValueError(f"Data validation failed: {failed_checks}")
                
        except Exception as e:
            logger.error(f"❌ Food reviews data validation failed: {e}")
            raise
    
    def _execute_database_loading(self, data: pd.DataFrame) -> bool:
        """
        Execute database loading stage for food reviews
        """
        logger.info("Loading food reviews data to database...")
        
        try:
            # Save to multiple tables
            success_count = 0
            
            # Main reviews table
            if self._save_to_table(data, 'reviews'):
                success_count += 1
                logger.info("✅ Saved to reviews table")
            
            # Products table (aggregated food products)
            products_data = self._create_food_products_table(data)
            if self._save_to_table(products_data, 'products'):
                success_count += 1
                logger.info("✅ Saved to food products table")
            
            # Users table (aggregated)
            users_data = self._create_users_table(data)
            if self._save_to_table(users_data, 'users'):
                success_count += 1
                logger.info("✅ Saved to users table")
            
            logger.info(f"✅ Database loading completed: {success_count}/3 tables saved")
            return success_count >= 2  # At least reviews and one other table
            
        except Exception as e:
            logger.error(f"❌ Database loading failed: {e}")
            return False
    
    def _execute_quality_assessment(self, data: pd.DataFrame) -> Dict:
        """
        Execute quality assessment stage for food reviews
        """
        logger.info("Assessing food reviews data quality...")
        
        try:
            quality_report = {
                'data_completeness': self._assess_completeness(data),
                'data_consistency': self._assess_consistency(data),
                'data_accuracy': self._assess_accuracy(data),
                'data_timeliness': self._assess_timeliness(data),
                'sample_statistics': self._get_food_sample_statistics(data)
            }
            
            # Calculate overall quality score
            scores = [
                quality_report['data_completeness']['score'],
                quality_report['data_consistency']['score'],
                quality_report['data_accuracy']['score'],
                quality_report['data_timeliness']['score']
            ]
            quality_report['overall_score'] = np.mean(scores)
            
            logger.info(f"✅ Food reviews quality assessment completed")
            logger.info(f"Overall quality score: {quality_report['overall_score']:.2f}/1.0")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return {'overall_score': 0, 'error': str(e)}
    
    def _validate_dataset(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive dataset validation for food reviews
        """
        checks = {}
        
        # Check required columns
        required_columns = ['product_id', 'user_id', 'rating', 'review_text']
        missing_columns = [col for col in required_columns if col not in data.columns]
        checks['required_columns'] = {
            'passed': len(missing_columns) == 0,
            'message': f"Missing columns: {missing_columns}" if missing_columns else "All required columns present"
        }
        
        # Check data types
        checks['data_types'] = {
            'passed': data['rating'].dtype in ['int64', 'int32'] and data['rating'].between(1, 5).all(),
            'message': "Rating column has valid integers 1-5" if data['rating'].dtype in ['int64', 'int32'] else "Invalid rating data type"
        }
        
        # Check for empty data
        checks['non_empty'] = {
            'passed': len(data) > 0,
            'message': f"Food reviews dataset has {len(data)} rows" if len(data) > 0 else "Dataset is empty"
        }
        
        # Check review text quality for food reviews
        if 'review_text' in data.columns:
            avg_length = data['review_text'].str.len().mean()
            checks['text_quality'] = {
                'passed': avg_length >= 20,
                'message': f"Average food review length: {avg_length:.1f} chars" if avg_length >= 20 else f"Food reviews too short: {avg_length:.1f} chars"
            }
        
        # Check rating distribution
        if 'rating' in data.columns:
            rating_counts = data['rating'].value_counts()
            checks['rating_distribution'] = {
                'passed': len(rating_counts) >= 3,
                'message': f"Rating distribution: {rating_counts.to_dict()}"
            }
        
        overall_passed = all(check['passed'] for check in checks.values())
        
        return {
            'is_valid': overall_passed,
            'checks': checks
        }
    
    def _assess_completeness(self, data: pd.DataFrame) -> Dict:
        """Assess data completeness for food reviews"""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells)
        
        return {
            'score': completeness_score,
            'missing_cells': missing_cells,
            'total_cells': total_cells,
            'details': data.isnull().sum().to_dict()
        }
    
    def _assess_consistency(self, data: pd.DataFrame) -> Dict:
        """Assess data consistency for food reviews"""
        consistency_checks = []
        
        # Rating consistency
        if 'rating' in data.columns:
            valid_ratings = data['rating'].between(1, 5).sum()
            consistency_checks.append(valid_ratings / len(data))
        
        # Text length consistency for food reviews
        if 'review_text' in data.columns:
            reasonable_length = data['review_text'].str.len().between(10, 5000).sum()
            consistency_checks.append(reasonable_length / len(data))
        
        consistency_score = np.mean(consistency_checks) if consistency_checks else 1.0
        
        return {
            'score': consistency_score,
            'checks_performed': len(consistency_checks),
            'details': consistency_checks
        }
    
    def _assess_accuracy(self, data: pd.DataFrame) -> Dict:
        """Assess data accuracy for food reviews"""
        accuracy_score = 0.8  # Placeholder - would need ground truth for real accuracy
        
        return {
            'score': accuracy_score,
            'method': 'estimated_based_on_cleaning',
            'note': 'Actual accuracy requires ground truth labels for food reviews'
        }
    
    def _assess_timeliness(self, data: pd.DataFrame) -> Dict:
        """Assess data timeliness for food reviews"""
        if 'date' in data.columns:
            latest_date = data['date'].max()
            days_old = (datetime.now() - latest_date).days
            timeliness_score = max(0, 1 - (days_old / 365))  # Decay over 1 year
        else:
            timeliness_score = 0.5  # No date info
        
        return {
            'score': timeliness_score,
            'latest_date': str(latest_date) if 'date' in data.columns else None,
            'days_old': days_old if 'date' in data.columns else None
        }
    
    def _get_food_sample_statistics(self, data: pd.DataFrame) -> Dict:
        """Get sample statistics specific to food reviews"""
        stats = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'dataset_type': 'Amazon Fine Food Reviews'
        }
        
        if 'rating' in data.columns:
            stats['rating_stats'] = {
                'mean': data['rating'].mean(),
                'std': data['rating'].std(),
                'distribution': data['rating'].value_counts().to_dict()
            }
        
        if 'review_text' in data.columns:
            stats['text_stats'] = {
                'avg_length': data['review_text'].str.len().mean(),
                'avg_words': data['review_text'].str.split().str.len().mean()
            }
        
        if 'product_id' in data.columns:
            stats['product_stats'] = {
                'unique_products': data['product_id'].nunique(),
                'top_products': data['product_id'].value_counts().head(5).to_dict()
            }
        
        return stats
    
    def _save_to_table(self, data: pd.DataFrame, table_name: str) -> bool:
        """Save data to database table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data.to_sql(table_name, conn, if_exists='replace', index=False)
                return True
        except Exception as e:
            logger.error(f"Failed to save to {table_name}: {e}")
            return False
    
    def _create_food_products_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create food products summary table"""
        if 'product_id' not in data.columns:
            return pd.DataFrame()
        
        products = data.groupby('product_id').agg({
            'rating': ['mean', 'count', 'std'],
            'review_text': 'count',
            'date': 'max' if 'date' in data.columns else lambda x: datetime.now()
        }).round(2)
        
        products.columns = ['avg_rating', 'total_reviews', 'rating_std', 'review_count', 'last_review_date']
        products = products.reset_index()
        products['product_name'] = 'Food Product ' + products['product_id'].astype(str)
        products['category'] = 'Food & Beverage'
        
        return products
    
    def _create_users_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create users summary table"""
        if 'user_id' not in data.columns:
            return pd.DataFrame()
        
        users = data.groupby('user_id').agg({
            'rating': ['mean', 'count'],
            'review_text': lambda x: x.str.len().mean(),
            'date': 'max' if 'date' in data.columns else lambda x: datetime.now()
        }).round(2)
        
        users.columns = ['avg_rating_given', 'review_count', 'avg_review_length', 'last_review_date']
        users = users.reset_index()
        users['username'] = 'User_' + users['user_id'].astype(str)
        
        return users
    
    def _log_data_preview(self, data: pd.DataFrame, stage_name: str):
        """Log data preview for debugging"""
        logger.info(f"\n--- {stage_name} Preview ---")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        
        if 'rating' in data.columns:
            logger.info(f"Rating distribution: {data['rating'].value_counts().to_dict()}")
        
        if 'review_text' in data.columns:
            logger.info(f"Avg review length: {data['review_text'].str.len().mean():.1f} chars")
    
    def _generate_final_report(self, data: pd.DataFrame, quality_report: Dict) -> Dict:
        """Generate comprehensive final report for food reviews"""
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': self.pipeline_stats['total_duration'],
                'stages_completed': self.pipeline_stats['stages_completed'],
                'success': len(self.pipeline_stats['errors']) == 0,
                'dataset_type': 'Amazon Fine Food Reviews'
            },
            'data_summary': {
                'final_records': len(data),
                'original_records': self.pipeline_stats['total_records_processed'],
                'retention_rate': len(data) / self.pipeline_stats['total_records_processed'] if self.pipeline_stats['total_records_processed'] > 0 else 0,
                'columns': list(data.columns)
            },
            'quality_metrics': quality_report,
            'business_insights': {
                'avg_rating': data['rating'].mean() if 'rating' in data.columns else None,
                'total_food_products': data['product_id'].nunique() if 'product_id' in data.columns else None,
                'total_users': data['user_id'].nunique() if 'user_id' in data.columns else None,
                'date_range': {
                    'start': data['date'].min().isoformat() if 'date' in data.columns else None,
                    'end': data['date'].max().isoformat() if 'date' in data.columns else None
                },
                'food_review_insights': {
                    'avg_review_length': data['review_text'].str.len().mean() if 'review_text' in data.columns else None,
                    'most_reviewed_products': data['product_id'].value_counts().head(5).to_dict() if 'product_id' in data.columns else None
                }
            }
        }
        
        return report
    
    def save_pipeline_report(self, report: Dict, filename: str = None):
        """Save pipeline execution report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_reviews_etl_report_{timestamp}.json"
        
        report_path = self.data_dir / filename
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Food reviews pipeline report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline report: {e}")


def main():
    """
    Main execution function for Amazon Fine Food Reviews ETL
    """
    print("🚀 STARTING AMAZON FINE FOOD REVIEWS ETL PIPELINE EXECUTION")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = ETLPipeline()
    
    # Configure pipeline for food reviews
    pipeline.config.update({
        'target_sample_size': 20000,
        'enable_kaggle_download': True,
        'enable_data_validation': True,
        'dataset_type': 'Amazon Fine Food Reviews'
    })
    
    # Execute pipeline
    result = pipeline.run_full_pipeline(target_size=20000)
    
    if result['success']:
        print("\n🎉 AMAZON FINE FOOD REVIEWS ETL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display summary
        stats = result['stats']
        print(f"⏱️  Duration: {stats['total_duration']:.1f} seconds")
        print(f"📊 Final records: {stats['final_records']:,}")
        print(f"🎯 Quality score: {stats['data_quality_score']:.2%}")
        print(f"✅ Stages completed: {len(stats['stages_completed'])}/5")
        
        # Save report
        pipeline.save_pipeline_report(result)
        
        print(f"\n📁 Food reviews data saved to database: {pipeline.db_path}")
        print(f"📋 Report saved to: {pipeline.data_dir}")
        
    else:
        print(f"\n❌ AMAZON FINE FOOD REVIEWS ETL PIPELINE FAILED!")
        print(f"Error: {result['error']}")
        print(f"Stages completed: {result['stats']['stages_completed']}")


if __name__ == "__main__":
    main()