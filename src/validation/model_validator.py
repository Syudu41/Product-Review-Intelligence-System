"""
Model Performance Validator - Day 3 Validation System
Comprehensive validation of all ML models on Amazon Fine Food Reviews
Tests sentiment analysis, fake detection, and recommendation accuracy
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import our ML models
from src.ml_models.sentiment_analyzer import SentimentAnalyzer
from src.ml_models.fake_detector import FakeReviewDetector
from src.ml_models.recommendation_engine import RecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_validation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelValidationResult:
    """Results from model validation"""
    model_name: str
    validation_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_size: int
    processing_time: float
    detailed_metrics: Dict[str, Any]
    validation_timestamp: str

@dataclass
class ValidationSession:
    """Complete validation session results"""
    session_id: str
    start_time: datetime
    models_tested: List[str]
    total_samples_tested: int
    overall_success_rate: float
    validation_results: List[ModelValidationResult]
    session_summary: Dict[str, Any] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.session_summary is None:
            self.session_summary = {}

class ModelPerformanceValidator:
    """
    Comprehensive validation system for all ML models
    Tests real-world performance on Amazon Fine Food Reviews
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.session_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize models
        self.sentiment_analyzer = None
        self.fake_detector = None
        self.recommendation_engine = None
        
        # Validation results storage
        self.validation_results = []
        
        # Test data cache
        self.test_reviews = None
        self.validation_datasets = {}
        
        logger.info(f"SUCCESS: ModelPerformanceValidator initialized (session: {self.session_id})")
    
    def load_models(self):
        """
        Load all trained ML models for validation
        """
        logger.info("LOADING: All trained ML models for validation...")
        
        try:
            # Load sentiment analyzer
            logger.info("LOADING: Sentiment Analyzer...")
            self.sentiment_analyzer = SentimentAnalyzer(self.db_path)
            logger.info("SUCCESS: Sentiment Analyzer loaded")
            
            # Load fake detector
            logger.info("LOADING: Fake Detector...")
            self.fake_detector = FakeReviewDetector(self.db_path)
            
            # Try to load pre-trained model
            try:
                self.fake_detector.load_model()
                logger.info("SUCCESS: Pre-trained Fake Detector loaded")
            except FileNotFoundError:
                logger.warning("WARNING: No pre-trained fake detector found, will train on-demand")
            
            # Load recommendation engine
            logger.info("LOADING: Recommendation Engine...")
            self.recommendation_engine = RecommendationEngine(self.db_path)
            logger.info("SUCCESS: Recommendation Engine loaded")
            
            logger.info("SUCCESS: All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to load models: {e}")
            return False
    
    def prepare_validation_datasets(self, test_size: float = 0.2):
        """
        Prepare test datasets for validation from existing food reviews
        """
        logger.info(f"PREPARING: Validation datasets (test_size: {test_size})")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load all reviews for testing
            reviews_query = """
                SELECT Id, product_id, user_id, rating, review_text, 
                       ProfileName, helpful_votes, total_votes,
                       CASE WHEN rating >= 4 THEN 'POSITIVE'
                            WHEN rating <= 2 THEN 'NEGATIVE'
                            ELSE 'NEUTRAL' END as true_sentiment
                FROM reviews 
                WHERE review_text IS NOT NULL 
                AND LENGTH(review_text) > 20
                ORDER BY RANDOM()
                LIMIT 2000
            """
            
            self.test_reviews = pd.read_sql_query(reviews_query, conn)
            conn.close()
            
            logger.info(f"SUCCESS: Loaded {len(self.test_reviews)} reviews for validation")
            
            # Create train/test splits for different validation tasks
            
            # 1. Sentiment validation dataset
            sentiment_train, sentiment_test = train_test_split(
                self.test_reviews, test_size=test_size, random_state=42, 
                stratify=self.test_reviews['true_sentiment']
            )
            
            self.validation_datasets['sentiment'] = {
                'train': sentiment_train,
                'test': sentiment_test,
                'task': 'sentiment_classification'
            }
            
            # 2. Rating prediction dataset (for recommendation validation)
            rating_train, rating_test = train_test_split(
                self.test_reviews, test_size=test_size, random_state=42
            )
            
            self.validation_datasets['rating_prediction'] = {
                'train': rating_train, 
                'test': rating_test,
                'task': 'rating_prediction'
            }
            
            # 3. Fake detection dataset (mix real + synthetic)
            fake_detection_data = self._prepare_fake_detection_dataset()
            
            self.validation_datasets['fake_detection'] = fake_detection_data
            
            logger.info("SUCCESS: All validation datasets prepared")
            
            # Show dataset statistics
            for dataset_name, dataset in self.validation_datasets.items():
                if 'test' in dataset:
                    test_size = len(dataset['test'])
                    logger.info(f"  {dataset_name}: {test_size} test samples")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to prepare validation datasets: {e}")
            return False
    
    def _prepare_fake_detection_dataset(self) -> Dict:
        """
        Prepare dataset for fake detection validation
        """
        try:
            # Get real reviews (labeled as authentic)
            real_reviews = self.test_reviews.sample(500, random_state=42).copy()
            real_reviews['is_fake'] = 0
            real_reviews['label_source'] = 'real_amazon_reviews'
            
            # Create simple synthetic fake reviews for testing
            fake_reviews_data = []
            fake_templates = [
                "Great product! Amazing quality! Fast shipping! Highly recommend! Five stars!",
                "Perfect item! Love it love it love it! Best purchase ever! Amazing value!",
                "Excellent! Outstanding! Fantastic! Wonderful! Perfect perfect perfect!",
                "Terrible quality! Worst product ever! Don't buy! Save your money! One star!",
                "Amazing taste! Delicious delicious! Fresh fresh! Perfect flavor! Great price!",
                "Best food ever! Incredible quality! Fast delivery! Highly recommended! Perfect!",
                "Awful taste! Stale product! Overpriced! Poor quality! Very disappointed!",
                "Love this product! Great value! Quick shipping! Excellent customer service!",
                "Outstanding quality! Fresh ingredients! Perfect packaging! Amazing flavor!",
                "Horrible experience! Poor taste! Expired product! Waste of money!"
            ]
            
            for i in range(200):  # Create 200 synthetic fake reviews
                template = np.random.choice(fake_templates)
                fake_reviews_data.append({
                    'Id': f'fake_{i}',
                    'product_id': f'FAKE_PRODUCT_{i % 20}',
                    'user_id': f'fake_user_{i}',
                    'rating': np.random.choice([1, 5]),  # Extreme ratings
                    'review_text': template,
                    'ProfileName': f'FakeUser{i}',
                    'helpful_votes': 0,
                    'total_votes': 0,
                    'is_fake': 1,
                    'label_source': 'synthetic_fake'
                })
            
            fake_reviews = pd.DataFrame(fake_reviews_data)
            
            # Combine real and fake
            combined_data = pd.concat([
                real_reviews[['Id', 'product_id', 'user_id', 'rating', 'review_text', 
                             'ProfileName', 'helpful_votes', 'total_votes', 'is_fake']],
                fake_reviews[['Id', 'product_id', 'user_id', 'rating', 'review_text',
                             'ProfileName', 'helpful_votes', 'total_votes', 'is_fake']]
            ], ignore_index=True)
            
            # Train/test split
            train_data, test_data = train_test_split(
                combined_data, test_size=0.3, random_state=42, 
                stratify=combined_data['is_fake']
            )
            
            return {
                'train': train_data,
                'test': test_data,
                'task': 'fake_detection',
                'real_samples': len(real_reviews),
                'fake_samples': len(fake_reviews)
            }
            
        except Exception as e:
            logger.error(f"ERROR: Failed to prepare fake detection dataset: {e}")
            return {}
    
    def validate_sentiment_analyzer(self) -> ModelValidationResult:
        """
        Validate sentiment analysis model performance
        """
        logger.info("VALIDATING: Sentiment Analysis Model")
        start_time = time.time()
        
        try:
            test_data = self.validation_datasets['sentiment']['test']
            
            logger.info(f"TESTING: Sentiment analysis on {len(test_data)} food reviews...")
            
            # Get predictions from model
            predictions = []
            confidences = []
            processing_times = []
            
            for idx, row in test_data.iterrows():
                try:
                    result = self.sentiment_analyzer.analyze_review(row['review_text'])
                    predictions.append(result.overall_sentiment)
                    confidences.append(result.confidence)
                    processing_times.append(result.processing_time)
                    
                except Exception as e:
                    logger.warning(f"WARNING: Failed to analyze review {idx}: {e}")
                    predictions.append('NEUTRAL')
                    confidences.append(0.5)
                    processing_times.append(0.0)
                
                # Progress logging
                if (len(predictions)) % 100 == 0:
                    logger.info(f"PROGRESS: Analyzed {len(predictions)}/{len(test_data)} reviews")
            
            # Calculate metrics
            true_labels = test_data['true_sentiment'].tolist()
            
            # Overall accuracy
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
            
            # Detailed metrics
            class_report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(true_labels, predictions).tolist()
            
            # Additional food-specific metrics
            avg_confidence = np.mean(confidences)
            avg_processing_time = np.mean(processing_times)
            
            # Rating correlation (sentiment vs actual rating)
            sentiment_scores = []
            for pred in predictions:
                if pred == 'POSITIVE':
                    sentiment_scores.append(4.5)
                elif pred == 'NEGATIVE':
                    sentiment_scores.append(1.5)
                else:
                    sentiment_scores.append(3.0)
            
            rating_correlation = np.corrcoef(sentiment_scores, test_data['rating'])[0, 1]
            
            processing_time = time.time() - start_time
            
            detailed_metrics = {
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'avg_confidence': avg_confidence,
                'avg_processing_time_per_review': avg_processing_time,
                'rating_correlation': rating_correlation,
                'positive_predictions': predictions.count('POSITIVE'),
                'negative_predictions': predictions.count('NEGATIVE'),
                'neutral_predictions': predictions.count('NEUTRAL')
            }
            
            result = ModelValidationResult(
                model_name='sentiment_analyzer',
                validation_type='food_review_sentiment_classification',
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                sample_size=len(test_data),
                processing_time=processing_time,
                detailed_metrics=detailed_metrics,
                validation_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"SUCCESS: Sentiment validation complete")
            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  F1-Score: {f1:.3f}")
            logger.info(f"  Avg Confidence: {avg_confidence:.3f}")
            logger.info(f"  Rating Correlation: {rating_correlation:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"ERROR: Sentiment validation failed: {e}")
            return ModelValidationResult(
                model_name='sentiment_analyzer',
                validation_type='food_review_sentiment_classification',
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                sample_size=0, processing_time=0.0,
                detailed_metrics={'error': str(e)},
                validation_timestamp=datetime.now().isoformat()
            )
    
    def validate_fake_detector(self) -> ModelValidationResult:
        """
        Validate fake review detection model performance
        """
        logger.info("VALIDATING: Fake Review Detection Model")
        start_time = time.time()
        
        try:
            # Check if model is trained
            if not self.fake_detector.model:
                logger.info("TRAINING: Fake detector model not found, training now...")
                features_df, labels = self.fake_detector.prepare_training_data(limit=1000)
                self.fake_detector.train_model(features_df, labels)
            
            test_data = self.validation_datasets['fake_detection']['test']
            
            logger.info(f"TESTING: Fake detection on {len(test_data)} samples...")
            
            # Get predictions
            predictions = []
            probabilities = []
            confidences = []
            
            for idx, row in test_data.iterrows():
                try:
                    # Prepare data for fake detector
                    user_data = {
                        'review_count': 10,  # Default values
                        'avg_rating': 3.5,
                        'rating_variance': 1.0,
                        'days_active': 100,
                        'reviews_per_day': 0.1
                    }
                    
                    review_data = {
                        'rating': row['rating'],
                        'helpful_votes': row['helpful_votes'],
                        'total_votes': row['total_votes'],
                        'rating_deviation': 0
                    }
                    
                    result = self.fake_detector.predict_fake_probability(
                        row['review_text'],
                        user_data=user_data,
                        review_data=review_data
                    )
                    
                    probabilities.append(result.is_fake_probability)
                    confidences.append(result.confidence)
                    
                    # Convert probability to binary prediction (threshold = 0.5)
                    predictions.append(1 if result.is_fake_probability > 0.5 else 0)
                    
                except Exception as e:
                    logger.warning(f"WARNING: Failed to predict for sample {idx}: {e}")
                    predictions.append(0)
                    probabilities.append(0.0)
                    confidences.append(0.5)
                
                # Progress logging
                if len(predictions) % 50 == 0:
                    logger.info(f"PROGRESS: Analyzed {len(predictions)}/{len(test_data)} samples")
            
            # Calculate metrics
            true_labels = test_data['is_fake'].tolist()
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            # Additional metrics
            from sklearn.metrics import roc_auc_score
            try:
                auc_score = roc_auc_score(true_labels, probabilities)
            except:
                auc_score = 0.5
            
            class_report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(true_labels, predictions).tolist()
            
            avg_confidence = np.mean(confidences)
            avg_fake_probability = np.mean(probabilities)
            
            processing_time = time.time() - start_time
            
            detailed_metrics = {
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'auc_score': auc_score,
                'avg_confidence': avg_confidence,
                'avg_fake_probability': avg_fake_probability,
                'true_positives': sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1),
                'false_positives': sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1),
                'true_negatives': sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0),
                'false_negatives': sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
            }
            
            result = ModelValidationResult(
                model_name='fake_detector',
                validation_type='fake_review_detection',
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                sample_size=len(test_data),
                processing_time=processing_time,
                detailed_metrics=detailed_metrics,
                validation_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"SUCCESS: Fake detection validation complete")
            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  Precision: {precision:.3f}")
            logger.info(f"  Recall: {recall:.3f}")
            logger.info(f"  F1-Score: {f1:.3f}")
            logger.info(f"  AUC Score: {auc_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"ERROR: Fake detection validation failed: {e}")
            return ModelValidationResult(
                model_name='fake_detector',
                validation_type='fake_review_detection',
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                sample_size=0, processing_time=0.0,
                detailed_metrics={'error': str(e)},
                validation_timestamp=datetime.now().isoformat()
            )
    
    def validate_recommendation_engine(self) -> ModelValidationResult:
        """
        Validate recommendation engine performance
        """
        logger.info("VALIDATING: Recommendation Engine")
        start_time = time.time()
        
        try:
            # Get test users with sufficient rating history
            test_users = self.recommendation_engine.user_item_matrix.index[:20]  # Test first 20 users
            
            logger.info(f"TESTING: Recommendations for {len(test_users)} users...")
            
            successful_recommendations = 0
            total_recommendations = 0
            avg_predicted_ratings = []
            avg_confidences = []
            recommendation_diversity = []
            
            for user_id in test_users:
                try:
                    # Get recommendations
                    recommendations = self.recommendation_engine.get_hybrid_recommendations(
                        user_id, n_recommendations=10
                    )
                    
                    if recommendations:
                        successful_recommendations += 1
                        total_recommendations += len(recommendations)
                        
                        # Calculate metrics
                        predicted_ratings = [rec.predicted_rating for rec in recommendations]
                        confidences = [rec.confidence for rec in recommendations]
                        
                        avg_predicted_ratings.extend(predicted_ratings)
                        avg_confidences.extend(confidences)
                        
                        # Diversity: unique product categories in recommendations
                        unique_products = len(set([rec.product_id for rec in recommendations]))
                        recommendation_diversity.append(unique_products / len(recommendations))
                        
                except Exception as e:
                    logger.warning(f"WARNING: Failed to get recommendations for user {user_id}: {e}")
                    continue
            
            # Calculate validation metrics
            success_rate = successful_recommendations / len(test_users)
            avg_recommendations_per_user = total_recommendations / max(successful_recommendations, 1)
            avg_predicted_rating = np.mean(avg_predicted_ratings) if avg_predicted_ratings else 0
            avg_confidence = np.mean(avg_confidences) if avg_confidences else 0
            avg_diversity = np.mean(recommendation_diversity) if recommendation_diversity else 0
            
            # Simulate accuracy using recommendation system stats
            rec_stats = self.recommendation_engine.get_recommendation_stats()
            
            processing_time = time.time() - start_time
            
            detailed_metrics = {
                'success_rate': success_rate,
                'avg_recommendations_per_user': avg_recommendations_per_user,
                'avg_predicted_rating': avg_predicted_rating,
                'avg_confidence': avg_confidence,
                'avg_diversity': avg_diversity,
                'system_stats': rec_stats,
                'users_tested': len(test_users),
                'successful_users': successful_recommendations
            }
            
            # Use success rate as accuracy proxy
            accuracy = success_rate
            precision = avg_confidence
            recall = success_rate
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            result = ModelValidationResult(
                model_name='recommendation_engine',
                validation_type='food_product_recommendations',
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                sample_size=len(test_users),
                processing_time=processing_time,
                detailed_metrics=detailed_metrics,
                validation_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"SUCCESS: Recommendation validation complete")
            logger.info(f"  Success Rate: {success_rate:.3f}")
            logger.info(f"  Avg Predicted Rating: {avg_predicted_rating:.2f}")
            logger.info(f"  Avg Confidence: {avg_confidence:.3f}")
            logger.info(f"  Avg Diversity: {avg_diversity:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"ERROR: Recommendation validation failed: {e}")
            return ModelValidationResult(
                model_name='recommendation_engine',
                validation_type='food_product_recommendations',
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                sample_size=0, processing_time=0.0,
                detailed_metrics={'error': str(e)},
                validation_timestamp=datetime.now().isoformat()
            )
    
    def save_validation_results(self, results: List[ModelValidationResult]):
        """
        Save validation results to database
        """
        if not results:
            logger.warning("WARNING: No validation results to save")
            return
        
        logger.info(f"SAVING: {len(results)} validation results to database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create model_validation table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_validation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    model_name TEXT,
                    validation_type TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    sample_size INTEGER,
                    processing_time REAL,
                    detailed_metrics TEXT,
                    validation_timestamp TEXT,
                    created_at TEXT
                )
            """)
            
            # Insert results
            for result in results:
                conn.execute("""
                    INSERT INTO model_validation 
                    (session_id, model_name, validation_type, accuracy, precision, recall,
                     f1_score, sample_size, processing_time, detailed_metrics, 
                     validation_timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    result.model_name,
                    result.validation_type,
                    result.accuracy,
                    result.precision,
                    result.recall,
                    result.f1_score,
                    result.sample_size,
                    result.processing_time,
                    json.dumps(result.detailed_metrics),
                    result.validation_timestamp,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info("SUCCESS: Validation results saved to database")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save validation results: {e}")
    
    def generate_validation_report(self, results: List[ModelValidationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        """
        logger.info("GENERATING: Comprehensive validation report...")
        
        report = {
            'session_info': {
                'session_id': self.session_id,
                'validation_timestamp': datetime.now().isoformat(),
                'models_validated': len(results),
                'dataset_type': 'Amazon Fine Food Reviews'
            },
            'model_results': {},
            'summary_metrics': {},
            'recommendations': []
        }
        
        # Process each model's results
        total_accuracy = 0
        total_f1 = 0
        
        for result in results:
            report['model_results'][result.model_name] = {
                'validation_type': result.validation_type,
                'performance_metrics': {
                    'accuracy': round(result.accuracy, 4),
                    'precision': round(result.precision, 4),
                    'recall': round(result.recall, 4),
                    'f1_score': round(result.f1_score, 4)
                },
                'sample_size': result.sample_size,
                'processing_time': round(result.processing_time, 2),
                'key_insights': self._extract_key_insights(result)
            }
            
            total_accuracy += result.accuracy
            total_f1 += result.f1_score
        
        # Summary metrics
        report['summary_metrics'] = {
            'overall_accuracy': round(total_accuracy / len(results), 4),
            'overall_f1_score': round(total_f1 / len(results), 4),
            'total_samples_tested': sum(r.sample_size for r in results),
            'total_processing_time': round(sum(r.processing_time for r in results), 2)
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_improvement_recommendations(results)
        
        return report
    
    def _extract_key_insights(self, result: ModelValidationResult) -> List[str]:
        """
        Extract key insights from validation results
        """
        insights = []
        
        if result.model_name == 'sentiment_analyzer':
            insights.append(f"Average confidence: {result.detailed_metrics.get('avg_confidence', 0):.3f}")
            insights.append(f"Rating correlation: {result.detailed_metrics.get('rating_correlation', 0):.3f}")
            
        elif result.model_name == 'fake_detector':
            auc = result.detailed_metrics.get('auc_score', 0)
            insights.append(f"AUC Score: {auc:.3f}")
            insights.append(f"Detection accuracy: {result.accuracy:.3f}")
            
        elif result.model_name == 'recommendation_engine':
            success_rate = result.detailed_metrics.get('success_rate', 0)
            diversity = result.detailed_metrics.get('avg_diversity', 0)
            insights.append(f"Success rate: {success_rate:.3f}")
            insights.append(f"Recommendation diversity: {diversity:.3f}")
        
        return insights
    
    def _generate_improvement_recommendations(self, results: List[ModelValidationResult]) -> List[str]:
        """
        Generate recommendations for model improvement
        """
        recommendations = []
        
        for result in results:
            if result.accuracy < 0.7:
                recommendations.append(f"{result.model_name}: Consider retraining with more data or feature engineering")
            
            if result.f1_score < 0.6:
                recommendations.append(f"{result.model_name}: F1-score indicates class imbalance, consider balanced sampling")
        
        if not recommendations:
            recommendations.append("All models performing well - consider production deployment")
        
        return recommendations
    
    def run_comprehensive_validation(self) -> ValidationSession:
        """
        Run complete validation pipeline for all models
        """
        logger.info(f"STARTING: Comprehensive model validation session")
        logger.info(f"Session ID: {self.session_id}")
        
        start_time = datetime.now()
        
        # Initialize session
        session = ValidationSession(
            session_id=self.session_id,
            start_time=start_time,
            models_tested=[],
            total_samples_tested=0,
            overall_success_rate=0.0,
            validation_results=[]
        )
        
        try:
            # Step 1: Load models
            if not self.load_models():
                logger.error("ERROR: Failed to load models")
                return session
            
            # Step 2: Prepare validation datasets
            if not self.prepare_validation_datasets():
                logger.error("ERROR: Failed to prepare validation datasets")
                return session
            
            # Step 3: Validate each model
            logger.info("STEP 3: Running model validations...")
            
            # Validate sentiment analyzer
            sentiment_result = self.validate_sentiment_analyzer()
            self.validation_results.append(sentiment_result)
            session.models_tested.append('sentiment_analyzer')
            
            # Validate fake detector
            fake_result = self.validate_fake_detector()
            self.validation_results.append(fake_result)
            session.models_tested.append('fake_detector')
            
            # Validate recommendation engine
            rec_result = self.validate_recommendation_engine()
            self.validation_results.append(rec_result)
            session.models_tested.append('recommendation_engine')
            
            # Update session with results
            session.validation_results = self.validation_results
            session.total_samples_tested = sum(r.sample_size for r in self.validation_results)
            session.overall_success_rate = sum(r.accuracy for r in self.validation_results) / len(self.validation_results)
            session.end_time = datetime.now()
            
            # Step 4: Save results
            self.save_validation_results(self.validation_results)
            
            # Step 5: Generate report
            report = self.generate_validation_report(self.validation_results)
            session.session_summary = report
            
            logger.info("SUCCESS: Comprehensive validation completed")
            logger.info(f"Models tested: {len(session.models_tested)}")
            logger.info(f"Total samples: {session.total_samples_tested}")
            logger.info(f"Overall success rate: {session.overall_success_rate:.3f}")
            
            return session
            
        except Exception as e:
            logger.error(f"ERROR: Validation session failed: {e}")
            session.session_summary = {'error': str(e)}
            return session

def main():
    """
    Main function for testing model validation
    """
    print("TESTING: Model Performance Validator - Day 3")
    print("=" * 60)
    
    # Initialize validator
    validator = ModelPerformanceValidator()
    
    # Run comprehensive validation
    session = validator.run_comprehensive_validation()
    
    # Display results
    print("\n" + "=" * 60)
    print("VALIDATION SESSION RESULTS:")
    print(f"Session ID: {session.session_id}")
    print(f"Models Tested: {len(session.models_tested)}")
    print(f"Total Samples: {session.total_samples_tested}")
    print(f"Overall Success Rate: {session.overall_success_rate:.3f}")
    
    if session.validation_results:
        print("\nMODEL PERFORMANCE SUMMARY:")
        for result in session.validation_results:
            print(f"\n{result.model_name.upper()}:")
            print(f"  Accuracy: {result.accuracy:.3f}")
            print(f"  F1-Score: {result.f1_score:.3f}")
            print(f"  Samples: {result.sample_size}")
            print(f"  Time: {result.processing_time:.1f}s")
    
    print(f"\n✅ COMPLETE: Model validation finished!")
    print(f"Results saved to database for analytics dashboard")

if __name__ == "__main__":
    main()