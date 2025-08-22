"""
Data Drift Analyzer - Day 3 Validation System
Comprehensive drift detection and monitoring for Amazon Fine Food Reviews
Analyzes distribution changes, concept drift, and model performance degradation
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
# ks_2samp is in scipy, not sklearn
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import our ML models for drift analysis
from src.ml_models.sentiment_analyzer import SentimentAnalyzer
from src.ml_models.fake_detector import FakeReviewDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drift_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """Results from drift detection analysis"""
    feature_name: str
    drift_type: str  # 'distribution', 'concept', 'performance'
    drift_detected: bool
    drift_magnitude: float
    statistical_test: str
    p_value: float
    threshold: float
    timestamp: str
    detailed_analysis: Dict[str, Any]

@dataclass
class DriftReport:
    """Comprehensive drift analysis report"""
    analysis_id: str
    baseline_period: str
    comparison_period: str
    total_features_analyzed: int
    features_with_drift: int
    drift_severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    model_retrain_recommended: bool
    drift_results: List[DriftDetectionResult]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]

class DataDriftAnalyzer:
    """
    Advanced data drift detection and monitoring system
    Specialized for Amazon Fine Food Reviews temporal analysis
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.analysis_id = f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'ks_test': 0.05,           # Kolmogorov-Smirnov test
            'chi2_test': 0.05,         # Chi-square test
            'performance_drop': 0.10,   # 10% performance drop
            'distribution_shift': 0.15  # 15% distribution change
        }
        
        # Initialize models for performance drift detection
        self.sentiment_analyzer = None
        self.fake_detector = None
        
        # Data storage
        self.baseline_data = None
        self.comparison_data = None
        self.drift_results = []
        
        logger.info(f"SUCCESS: DataDriftAnalyzer initialized (analysis: {self.analysis_id})")
    
    def load_baseline_data(self, start_date: str = None, end_date: str = None, sample_size: int = 2000) -> bool:
        """
        Load baseline data (historical 2012 food reviews)
        """
        logger.info(f"LOADING: Baseline data for drift analysis (sample: {sample_size})")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load historical reviews as baseline
            baseline_query = """
                SELECT Id, product_id, user_id, rating, review_text, 
                       ProfileName, helpful_votes, total_votes,
                       LENGTH(review_text) as text_length,
                       date as review_date
                FROM reviews 
                WHERE review_text IS NOT NULL 
                AND LENGTH(review_text) > 10
                ORDER BY RANDOM()
                LIMIT ?
            """
            
            self.baseline_data = pd.read_sql_query(baseline_query, conn, params=[sample_size])
            
            # Add derived features for drift analysis
            self.baseline_data = self._extract_drift_features(self.baseline_data, dataset_name='baseline')
            
            conn.close()
            
            logger.info(f"SUCCESS: Loaded {len(self.baseline_data)} baseline reviews")
            logger.info(f"Date range: {self.baseline_data['review_date'].min()} to {self.baseline_data['review_date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to load baseline data: {e}")
            return False
    
    def load_comparison_data(self, data_source: str = "live_reviews", sample_size: int = 500) -> bool:
        """
        Load comparison data for drift detection
        """
        logger.info(f"LOADING: Comparison data from {data_source} (sample: {sample_size})")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            if data_source == "live_reviews":
                # Try to load from live_reviews table
                comparison_query = """
                    SELECT product_id, user_id, rating, review_text,
                           helpful_votes, total_votes,
                           LENGTH(review_text) as text_length,
                           date as review_date
                    FROM live_reviews 
                    WHERE review_text IS NOT NULL 
                    AND LENGTH(review_text) > 10
                    ORDER BY scrape_date DESC
                    LIMIT ?
                """
                
                comparison_data = pd.read_sql_query(comparison_query, conn, params=[sample_size])
                
                if len(comparison_data) == 0:
                    logger.warning("WARNING: No live reviews found, using recent historical data as comparison")
                    # Fallback: use more recent historical data
                    comparison_query = """
                        SELECT Id, product_id, user_id, rating, review_text, 
                               ProfileName, helpful_votes, total_votes,
                               LENGTH(review_text) as text_length,
                               date as review_date
                        FROM reviews 
                        WHERE review_text IS NOT NULL 
                        AND LENGTH(review_text) > 10
                        AND date >= '2012-01-01'  -- More recent historical data
                        ORDER BY date DESC
                        LIMIT ?
                    """
                    comparison_data = pd.read_sql_query(comparison_query, conn, params=[sample_size])
            
            else:
                # Load from regular reviews table with date filter
                comparison_query = """
                    SELECT Id, product_id, user_id, rating, review_text, 
                           ProfileName, helpful_votes, total_votes,
                           LENGTH(review_text) as text_length,
                           date as review_date
                    FROM reviews 
                    WHERE review_text IS NOT NULL 
                    AND LENGTH(review_text) > 10
                    AND date >= ?
                    ORDER BY date DESC
                    LIMIT ?
                """
                comparison_data = pd.read_sql_query(comparison_query, conn, params=[data_source, sample_size])
            
            self.comparison_data = self._extract_drift_features(comparison_data, dataset_name='comparison')
            
            conn.close()
            
            logger.info(f"SUCCESS: Loaded {len(self.comparison_data)} comparison reviews")
            if len(self.comparison_data) > 0:
                logger.info(f"Date range: {self.comparison_data['review_date'].min()} to {self.comparison_data['review_date'].max()}")
            
            return len(self.comparison_data) > 0
            
        except Exception as e:
            logger.error(f"ERROR: Failed to load comparison data: {e}")
            return False
    
    def _extract_drift_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Extract features for drift detection analysis
        """
        logger.info(f"EXTRACTING: Drift features for {dataset_name} dataset")
        
        try:
            # Text-based features
            df['word_count'] = df['review_text'].str.split().str.len()
            df['sentence_count'] = df['review_text'].str.count(r'\.') + 1  # Fixed regex
            df['exclamation_count'] = df['review_text'].str.count('!')
            df['question_count'] = df['review_text'].str.count(r'\?')  # Fixed regex
            df['caps_ratio'] = df['review_text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
            
            # Rating-based features
            df['helpful_ratio'] = df['helpful_votes'] / (df['total_votes'] + 1)  # Add 1 to avoid division by zero
            df['rating_deviation'] = abs(df['rating'] - df['rating'].mean())
            
            # Food-specific sentiment indicators
            food_positive_words = ['delicious', 'tasty', 'fresh', 'amazing', 'perfect', 'love', 'excellent', 'great']
            food_negative_words = ['terrible', 'awful', 'stale', 'bad', 'horrible', 'disgusting', 'waste', 'disappointed']
            
            df['positive_food_words'] = df['review_text'].str.lower().apply(
                lambda x: sum(1 for word in food_positive_words if word in str(x))
            )
            df['negative_food_words'] = df['review_text'].str.lower().apply(
                lambda x: sum(1 for word in food_negative_words if word in str(x))
            )
            
            # Temporal features (if date available)
            if 'review_date' in df.columns:
                df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
                df['year'] = df['review_date'].dt.year
                df['month'] = df['review_date'].dt.month
                df['day_of_week'] = df['review_date'].dt.dayofweek
            
            # Categorical features
            df['rating_category'] = pd.cut(df['rating'], bins=[0, 2, 3, 4, 5], labels=['low', 'medium', 'high', 'very_high'])
            df['text_length_category'] = pd.cut(df['text_length'], bins=[0, 50, 150, 500, float('inf')], 
                                               labels=['short', 'medium', 'long', 'very_long'])
            
            logger.info(f"SUCCESS: Extracted {len([col for col in df.columns if col not in ['Id', 'product_id', 'user_id', 'review_text', 'ProfileName']])} drift features")
            
            return df
            
        except Exception as e:
            logger.error(f"ERROR: Failed to extract drift features: {e}")
            return df
    
    def detect_distribution_drift(self, feature_name: str) -> DriftDetectionResult:
        """
        Detect distribution drift using statistical tests
        """
        try:
            baseline_values = self.baseline_data[feature_name].dropna()
            comparison_values = self.comparison_data[feature_name].dropna()
            
            if len(baseline_values) == 0 or len(comparison_values) == 0:
                return DriftDetectionResult(
                    feature_name=feature_name,
                    drift_type='distribution',
                    drift_detected=False,
                    drift_magnitude=0.0,
                    statistical_test='insufficient_data',
                    p_value=1.0,
                    threshold=self.drift_thresholds['ks_test'],
                    timestamp=datetime.now().isoformat(),
                    detailed_analysis={'error': 'Insufficient data for comparison'}
                )
            
            # Choose appropriate test based on data type
            if baseline_values.dtype in ['object', 'category']:
                # Categorical data - use Chi-square test
                baseline_counts = baseline_values.value_counts()
                comparison_counts = comparison_values.value_counts()
                
                # Align categories
                all_categories = set(baseline_counts.index) | set(comparison_counts.index)
                baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
                comparison_aligned = [comparison_counts.get(cat, 0) for cat in all_categories]
                
                try:
                    chi2_stat, p_value = stats.chi2_contingency([baseline_aligned, comparison_aligned])[:2]
                    test_name = 'chi2_test'
                    drift_magnitude = chi2_stat / (len(baseline_values) + len(comparison_values))
                except:
                    p_value = 1.0
                    drift_magnitude = 0.0
                    test_name = 'chi2_failed'
                    
            else:
                # Numerical data - use Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(baseline_values, comparison_values)
                test_name = 'ks_test'
                drift_magnitude = ks_stat
            
            # Determine if drift is detected
            drift_detected = p_value < self.drift_thresholds.get(test_name, 0.05)
            
            # Additional analysis
            baseline_mean = baseline_values.mean() if baseline_values.dtype != 'object' else None
            comparison_mean = comparison_values.mean() if comparison_values.dtype != 'object' else None
            
            detailed_analysis = {
                'baseline_samples': len(baseline_values),
                'comparison_samples': len(comparison_values),
                'baseline_mean': baseline_mean,
                'comparison_mean': comparison_mean,
                'baseline_std': baseline_values.std() if baseline_values.dtype != 'object' else None,
                'comparison_std': comparison_values.std() if comparison_values.dtype != 'object' else None,
                'mean_shift': abs(comparison_mean - baseline_mean) if baseline_mean and comparison_mean else None
            }
            
            return DriftDetectionResult(
                feature_name=feature_name,
                drift_type='distribution',
                drift_detected=drift_detected,
                drift_magnitude=drift_magnitude,
                statistical_test=test_name,
                p_value=p_value,
                threshold=self.drift_thresholds.get(test_name, 0.05),
                timestamp=datetime.now().isoformat(),
                detailed_analysis=detailed_analysis
            )
            
        except Exception as e:
            logger.error(f"ERROR: Distribution drift detection failed for {feature_name}: {e}")
            return DriftDetectionResult(
                feature_name=feature_name,
                drift_type='distribution',
                drift_detected=False,
                drift_magnitude=0.0,
                statistical_test='error',
                p_value=1.0,
                threshold=0.05,
                timestamp=datetime.now().isoformat(),
                detailed_analysis={'error': str(e)}
            )
    
    def detect_concept_drift(self) -> List[DriftDetectionResult]:
        """
        Detect concept drift by analyzing model performance on different time periods
        """
        logger.info("DETECTING: Concept drift through model performance analysis")
        
        concept_drift_results = []
        
        try:
            # Initialize models if not already loaded
            if not self.sentiment_analyzer:
                self.sentiment_analyzer = SentimentAnalyzer(self.db_path)
                
            if not self.fake_detector:
                self.fake_detector = FakeReviewDetector(self.db_path)
                try:
                    self.fake_detector.load_model()
                except:
                    logger.warning("WARNING: Could not load fake detector model for concept drift")
            
            # Test sentiment model performance on baseline vs comparison
            baseline_sentiment_accuracy = self._test_sentiment_performance(self.baseline_data, 'baseline')
            comparison_sentiment_accuracy = self._test_sentiment_performance(self.comparison_data, 'comparison')
            
            sentiment_performance_drop = baseline_sentiment_accuracy - comparison_sentiment_accuracy
            sentiment_drift_detected = abs(sentiment_performance_drop) > self.drift_thresholds['performance_drop']
            
            concept_drift_results.append(DriftDetectionResult(
                feature_name='sentiment_model_performance',
                drift_type='concept',
                drift_detected=sentiment_drift_detected,
                drift_magnitude=abs(sentiment_performance_drop),
                statistical_test='performance_comparison',
                p_value=0.05 if sentiment_drift_detected else 0.1,
                threshold=self.drift_thresholds['performance_drop'],
                timestamp=datetime.now().isoformat(),
                detailed_analysis={
                    'baseline_accuracy': baseline_sentiment_accuracy,
                    'comparison_accuracy': comparison_sentiment_accuracy,
                    'performance_change': sentiment_performance_drop,
                    'samples_tested_baseline': len(self.baseline_data),
                    'samples_tested_comparison': len(self.comparison_data)
                }
            ))
            
            # Analyze rating distribution changes (concept drift indicator)
            rating_drift = self.detect_distribution_drift('rating')
            if rating_drift.drift_detected:
                concept_drift_results.append(DriftDetectionResult(
                    feature_name='rating_distribution_concept',
                    drift_type='concept',
                    drift_detected=True,
                    drift_magnitude=rating_drift.drift_magnitude,
                    statistical_test='rating_shift_analysis',
                    p_value=rating_drift.p_value,
                    threshold=rating_drift.threshold,
                    timestamp=datetime.now().isoformat(),
                    detailed_analysis={
                        'indication': 'Rating distribution change may indicate concept drift',
                        'baseline_avg_rating': self.baseline_data['rating'].mean(),
                        'comparison_avg_rating': self.comparison_data['rating'].mean(),
                        'rating_shift': self.comparison_data['rating'].mean() - self.baseline_data['rating'].mean()
                    }
                ))
            
            logger.info(f"SUCCESS: Concept drift analysis complete - {len(concept_drift_results)} indicators analyzed")
            return concept_drift_results
            
        except Exception as e:
            logger.error(f"ERROR: Concept drift detection failed: {e}")
            return []
    
    def _test_sentiment_performance(self, data: pd.DataFrame, dataset_name: str) -> float:
        """
        Test sentiment model performance on a dataset
        """
        try:
            if len(data) == 0:
                return 0.0
            
            # Sample data for performance testing (limit for speed)
            test_sample = data.sample(min(100, len(data)), random_state=42)
            
            correct_predictions = 0
            total_predictions = 0
            
            for _, row in test_sample.iterrows():
                try:
                    # Get model prediction
                    result = self.sentiment_analyzer.analyze_review(row['review_text'])
                    predicted_sentiment = result.overall_sentiment
                    
                    # Create true label based on rating
                    if row['rating'] >= 4:
                        true_sentiment = 'POSITIVE'
                    elif row['rating'] <= 2:
                        true_sentiment = 'NEGATIVE'
                    else:
                        true_sentiment = 'NEUTRAL'
                    
                    if predicted_sentiment == true_sentiment:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"WARNING: Failed to test sentiment for review: {e}")
                    continue
            
            accuracy = correct_predictions / max(total_predictions, 1)
            logger.info(f"PERFORMANCE: {dataset_name} sentiment accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"ERROR: Sentiment performance test failed for {dataset_name}: {e}")
            return 0.0
    
    def analyze_temporal_patterns(self) -> DriftDetectionResult:
        """
        Analyze temporal patterns and seasonal drift
        """
        logger.info("ANALYZING: Temporal patterns and seasonal drift")
        
        try:
            # Combine datasets with labels
            baseline_temporal = self.baseline_data[['review_date', 'rating', 'text_length', 'year', 'month']].copy()
            baseline_temporal['dataset'] = 'baseline'
            
            comparison_temporal = self.comparison_data[['review_date', 'rating', 'text_length', 'year', 'month']].copy()
            comparison_temporal['dataset'] = 'comparison'
            
            combined_data = pd.concat([baseline_temporal, comparison_temporal])
            
            # Analyze monthly patterns
            monthly_baseline = baseline_temporal.groupby('month')['rating'].mean()
            monthly_comparison = comparison_temporal.groupby('month')['rating'].mean()
            
            # Calculate seasonal drift
            seasonal_correlation = monthly_baseline.corr(monthly_comparison) if len(monthly_comparison) > 1 else 0
            seasonal_drift_magnitude = 1 - abs(seasonal_correlation)
            
            # Temporal trends analysis
            baseline_trend = self._calculate_temporal_trend(baseline_temporal)
            comparison_trend = self._calculate_temporal_trend(comparison_temporal)
            
            trend_change = abs(comparison_trend - baseline_trend)
            
            drift_detected = seasonal_drift_magnitude > 0.3 or trend_change > 0.1
            
            detailed_analysis = {
                'seasonal_correlation': seasonal_correlation,
                'baseline_trend': baseline_trend,
                'comparison_trend': comparison_trend,
                'trend_change': trend_change,
                'baseline_date_range': f"{baseline_temporal['review_date'].min()} to {baseline_temporal['review_date'].max()}",
                'comparison_date_range': f"{comparison_temporal['review_date'].min()} to {comparison_temporal['review_date'].max()}",
                'monthly_pattern_baseline': monthly_baseline.to_dict(),
                'monthly_pattern_comparison': monthly_comparison.to_dict()
            }
            
            return DriftDetectionResult(
                feature_name='temporal_patterns',
                drift_type='temporal',
                drift_detected=drift_detected,
                drift_magnitude=seasonal_drift_magnitude,
                statistical_test='temporal_correlation_analysis',
                p_value=0.05 if drift_detected else 0.1,
                threshold=0.3,
                timestamp=datetime.now().isoformat(),
                detailed_analysis=detailed_analysis
            )
            
        except Exception as e:
            logger.error(f"ERROR: Temporal pattern analysis failed: {e}")
            return DriftDetectionResult(
                feature_name='temporal_patterns',
                drift_type='temporal',
                drift_detected=False,
                drift_magnitude=0.0,
                statistical_test='temporal_error',
                p_value=1.0,
                threshold=0.3,
                timestamp=datetime.now().isoformat(),
                detailed_analysis={'error': str(e)}
            )
    
    def _calculate_temporal_trend(self, data: pd.DataFrame) -> float:
        """
        Calculate temporal trend in ratings over time
        """
        try:
            if len(data) < 2:
                return 0.0
            
            # Sort by date and calculate trend
            data_sorted = data.sort_values('review_date')
            x = np.arange(len(data_sorted))
            y = data_sorted['rating'].values
            
            # Linear regression slope
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
            
        except Exception as e:
            logger.warning(f"WARNING: Trend calculation failed: {e}")
            return 0.0
    
    def run_comprehensive_drift_analysis(self) -> DriftReport:
        """
        Run comprehensive drift analysis across all features and types
        """
        logger.info("STARTING: Comprehensive drift analysis")
        
        analysis_start_time = time.time()
        
        try:
            # Load baseline data
            if not self.load_baseline_data():
                raise Exception("Failed to load baseline data")
            
            # Load comparison data
            if not self.load_comparison_data():
                raise Exception("Failed to load comparison data")
            
            logger.info("STEP 1: Analyzing distribution drift...")
            
            # Analyze distribution drift for key features
            key_features = [
                'rating', 'text_length', 'word_count', 'sentence_count',
                'helpful_votes', 'total_votes', 'helpful_ratio',
                'exclamation_count', 'caps_ratio', 'positive_food_words',
                'negative_food_words', 'rating_category', 'text_length_category'
            ]
            
            distribution_results = []
            for feature in key_features:
                if feature in self.baseline_data.columns and feature in self.comparison_data.columns:
                    result = self.detect_distribution_drift(feature)
                    distribution_results.append(result)
                    
                    if result.drift_detected:
                        logger.info(f"DRIFT DETECTED: {feature} - magnitude: {result.drift_magnitude:.3f}")
            
            logger.info("STEP 2: Analyzing concept drift...")
            concept_results = self.detect_concept_drift()
            
            logger.info("STEP 3: Analyzing temporal patterns...")
            temporal_result = self.analyze_temporal_patterns()
            
            # Combine all results
            all_results = distribution_results + concept_results + [temporal_result]
            self.drift_results = all_results
            
            # Calculate summary statistics
            features_with_drift = sum(1 for r in all_results if r.drift_detected)
            total_features = len(all_results)
            
            # Determine drift severity
            drift_percentage = features_with_drift / max(total_features, 1)
            if drift_percentage >= 0.7:
                severity = 'CRITICAL'
                retrain_recommended = True
            elif drift_percentage >= 0.4:
                severity = 'HIGH'
                retrain_recommended = True
            elif drift_percentage >= 0.2:
                severity = 'MEDIUM'
                retrain_recommended = False
            else:
                severity = 'LOW'
                retrain_recommended = False
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(all_results, severity)
            
            # Create summary statistics
            summary_stats = {
                'analysis_duration_seconds': time.time() - analysis_start_time,
                'baseline_samples': len(self.baseline_data),
                'comparison_samples': len(self.comparison_data),
                'drift_percentage': drift_percentage,
                'high_magnitude_drifts': sum(1 for r in all_results if r.drift_magnitude > 0.5),
                'significant_p_values': sum(1 for r in all_results if r.p_value < 0.01),
                'concept_drifts_detected': len(concept_results),
                'distribution_drifts_detected': len([r for r in distribution_results if r.drift_detected])
            }
            
            # Create drift report
            report = DriftReport(
                analysis_id=self.analysis_id,
                baseline_period="Historical 2012 Amazon Food Reviews",
                comparison_period="Current/Live Data",
                total_features_analyzed=total_features,
                features_with_drift=features_with_drift,
                drift_severity=severity,
                model_retrain_recommended=retrain_recommended,
                drift_results=all_results,
                summary_statistics=summary_stats,
                recommendations=recommendations
            )
            
            # Save results
            self._save_drift_analysis_results(report)
            
            logger.info("SUCCESS: Comprehensive drift analysis completed")
            logger.info(f"Features analyzed: {total_features}")
            logger.info(f"Drift detected in: {features_with_drift} features")
            logger.info(f"Severity: {severity}")
            logger.info(f"Retrain recommended: {retrain_recommended}")
            
            return report
            
        except Exception as e:
            logger.error(f"ERROR: Comprehensive drift analysis failed: {e}")
            return DriftReport(
                analysis_id=self.analysis_id,
                baseline_period="Error",
                comparison_period="Error",
                total_features_analyzed=0,
                features_with_drift=0,
                drift_severity="ERROR",
                model_retrain_recommended=False,
                drift_results=[],
                summary_statistics={'error': str(e)},
                recommendations=['Fix data loading and analysis pipeline']
            )
    
    def _generate_drift_recommendations(self, results: List[DriftDetectionResult], severity: str) -> List[str]:
        """
        Generate actionable recommendations based on drift analysis
        """
        recommendations = []
        
        # Severity-based recommendations
        if severity == 'CRITICAL':
            recommendations.extend([
                "IMMEDIATE ACTION: Retrain all models with recent data",
                "Implement emergency monitoring and rollback procedures",
                "Review data pipeline for quality issues"
            ])
        elif severity == 'HIGH':
            recommendations.extend([
                "Schedule model retraining within 1 week",
                "Increase monitoring frequency to daily",
                "Investigate root causes of drift"
            ])
        elif severity == 'MEDIUM':
            recommendations.extend([
                "Plan model retraining within 1 month",
                "Monitor key performance metrics closely",
                "Consider gradual model updates"
            ])
        else:
            recommendations.extend([
                "Continue regular monitoring schedule",
                "Document drift patterns for future reference"
            ])
        
        # Feature-specific recommendations
        concept_drifts = [r for r in results if r.drift_type == 'concept' and r.drift_detected]
        if concept_drifts:
            recommendations.append("Concept drift detected - review model assumptions and feature relevance")
        
        high_magnitude_drifts = [r for r in results if r.drift_magnitude > 0.7]
        if high_magnitude_drifts:
            features = [r.feature_name for r in high_magnitude_drifts]
            recommendations.append(f"High magnitude drift in: {', '.join(features)} - investigate data sources")
        
        # Food review specific recommendations
        rating_drift = next((r for r in results if r.feature_name == 'rating' and r.drift_detected), None)
        if rating_drift:
            recommendations.append("Rating distribution changed - review sentiment model calibration")
        
        text_drift = next((r for r in results if 'text' in r.feature_name and r.drift_detected), None)
        if text_drift:
            recommendations.append("Text patterns changed - consider updating text preprocessing and features")
        
        return recommendations
    
    def _save_drift_analysis_results(self, report: DriftReport):
        """
        Save drift analysis results to database
        """
        logger.info("SAVING: Drift analysis results to database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create drift_analysis table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE,
                    baseline_period TEXT,
                    comparison_period TEXT,
                    total_features_analyzed INTEGER,
                    features_with_drift INTEGER,
                    drift_severity TEXT,
                    model_retrain_recommended BOOLEAN,
                    summary_statistics TEXT,
                    recommendations TEXT,
                    created_at TEXT
                )
            """)
            
            # Insert main report
            conn.execute("""
                INSERT OR REPLACE INTO drift_analysis 
                (analysis_id, baseline_period, comparison_period, total_features_analyzed,
                 features_with_drift, drift_severity, model_retrain_recommended,
                 summary_statistics, recommendations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.analysis_id,
                report.baseline_period,
                report.comparison_period,
                report.total_features_analyzed,
                report.features_with_drift,
                report.drift_severity,
                report.model_retrain_recommended,
                json.dumps(report.summary_statistics),
                json.dumps(report.recommendations),
                datetime.now().isoformat()
            ))
            
            # Create drift_results table for detailed results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT,
                    feature_name TEXT,
                    drift_type TEXT,
                    drift_detected BOOLEAN,
                    drift_magnitude REAL,
                    statistical_test TEXT,
                    p_value REAL,
                    threshold REAL,
                    detailed_analysis TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES drift_analysis(analysis_id)
                )
            """)
            
            # Insert detailed results
            for result in report.drift_results:
                conn.execute("""
                    INSERT INTO drift_results 
                    (analysis_id, feature_name, drift_type, drift_detected, drift_magnitude,
                     statistical_test, p_value, threshold, detailed_analysis, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.analysis_id,
                    result.feature_name,
                    result.drift_type,
                    result.drift_detected,
                    result.drift_magnitude,
                    result.statistical_test,
                    result.p_value,
                    result.threshold,
                    json.dumps(result.detailed_analysis),
                    result.timestamp
                ))
            
            conn.commit()
            conn.close()
            
            logger.info("SUCCESS: Drift analysis results saved to database")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save drift analysis results: {e}")

def main():
    """
    Main function for testing drift analyzer
    """
    print("TESTING: Data Drift Analyzer - Day 3")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DataDriftAnalyzer()
    
    # Run comprehensive drift analysis
    report = analyzer.run_comprehensive_drift_analysis()
    
    # Display results
    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS RESULTS:")
    print(f"Analysis ID: {report.analysis_id}")
    print(f"Baseline: {report.baseline_period}")
    print(f"Comparison: {report.comparison_period}")
    print(f"Features Analyzed: {report.total_features_analyzed}")
    print(f"Drift Detected In: {report.features_with_drift} features")
    print(f"Drift Severity: {report.drift_severity}")
    print(f"Retrain Recommended: {report.model_retrain_recommended}")
    
    if report.drift_results:
        print(f"\nDRIFT DETECTION DETAILS:")
        for result in report.drift_results:
            if result.drift_detected:
                print(f"  ⚠️  {result.feature_name} ({result.drift_type})")
                print(f"      Magnitude: {result.drift_magnitude:.3f}")
                print(f"      P-value: {result.p_value:.3f}")
                print(f"      Test: {result.statistical_test}")
    
    if report.recommendations:
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print(f"\n✅ COMPLETE: Drift analysis finished!")
    print(f"Results saved to database for monitoring dashboard")

if __name__ == "__main__":
    main()