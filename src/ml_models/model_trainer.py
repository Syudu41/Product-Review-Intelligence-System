"""
Model Training Pipeline
Orchestrates training and evaluation of all ML models in the system
Specialized for Amazon Fine Food Reviews dataset
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our custom models
from ml_models.sentiment_analyzer import SentimentAnalyzer
from ml_models.fake_detector import FakeReviewDetector
from ml_models.recommendation_engine import RecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/food_model_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """
    Comprehensive training pipeline for all ML models on Amazon Fine Food Reviews
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.sentiment_analyzer = None
        self.fake_detector = None
        self.recommendation_engine = None
        
        # Training configuration for food reviews
        self.config = {
            'sentiment_analysis': {
                'batch_size': 500,
                'enable_openai': False,  # Disabled due to quota
                'save_results': True,
                'dataset_type': 'food_reviews'
            },
            'fake_detection': {
                'training_samples': 2000,
                'test_split': 0.2,
                'save_model': True,
                'dataset_type': 'food_reviews'
            },
            'recommendations': {
                'min_user_ratings': 1,
                'n_recommendations': 10,
                'save_results': True,
                'dataset_type': 'food_recommendations'
            }
        }
        
        # Results storage
        self.training_results = {}
        
        logger.info("SUCCESS: ModelTrainingPipeline initialized for Amazon Fine Food Reviews")
    
    def setup_models(self):
        """Initialize all ML models for food reviews processing"""
        logger.info("SETUP: Initializing all ML models for food reviews...")
        
        try:
            # Initialize sentiment analyzer for food reviews
            logger.info("INITIALIZING: Sentiment Analyzer for food reviews...")
            self.sentiment_analyzer = SentimentAnalyzer(self.db_path)
            
            # Initialize fake detector for food reviews
            logger.info("INITIALIZING: Fake Detector for food reviews...")
            self.fake_detector = FakeReviewDetector(self.db_path)
            
            # Initialize recommendation engine for food products
            logger.info("INITIALIZING: Recommendation Engine for food products...")
            self.recommendation_engine = RecommendationEngine(self.db_path)
            
            logger.info("SUCCESS: All models initialized successfully for food reviews")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize models: {e}")
            raise
    
    def train_sentiment_analysis(self) -> Dict:
        """Train and evaluate sentiment analysis model for food reviews"""
        logger.info("TRAINING: Sentiment Analysis Model for Food Reviews")
        start_time = time.time()
        
        try:
            # Batch process food reviews for sentiment analysis
            batch_size = self.config['sentiment_analysis']['batch_size']
            results = self.sentiment_analyzer.batch_analyze_reviews(limit=batch_size)
            
            if results:
                # Save results to database
                if self.config['sentiment_analysis']['save_results']:
                    self.sentiment_analyzer.save_sentiment_results(results)
                
                # Get statistics
                stats = self.sentiment_analyzer.get_sentiment_stats()
                
                training_time = time.time() - start_time
                
                sentiment_results = {
                    'status': 'success',
                    'samples_processed': len(results),
                    'training_time': training_time,
                    'statistics': stats,
                    'dataset_type': 'Amazon Fine Food Reviews',
                    'performance_metrics': {
                        'avg_confidence': stats['overall_stats']['avg_confidence'],
                        'avg_sentiment_score': stats['overall_stats']['avg_sentiment_score'],
                        'positive_ratio': stats['overall_stats']['positive_count'] / stats['overall_stats']['total_analyzed'],
                        'processing_rate': len(results) / training_time
                    }
                }
                
                logger.info(f"SUCCESS: Food reviews sentiment analysis - {len(results)} samples in {training_time:.1f}s")
                return sentiment_results
            else:
                return {'status': 'failed', 'error': 'No results generated'}
                
        except Exception as e:
            logger.error(f"ERROR: Food reviews sentiment analysis training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_fake_detection(self) -> Dict:
        """Train and evaluate fake detection model for food reviews"""
        logger.info("TRAINING: Fake Detection Model for Food Reviews")
        start_time = time.time()
        
        try:
            # Prepare training data for food reviews
            training_samples = self.config['fake_detection']['training_samples']
            features_df, labels = self.fake_detector.prepare_training_data(limit=training_samples)
            
            # Train model
            training_results = self.fake_detector.train_model(features_df, labels)
            
            # Save model
            if self.config['fake_detection']['save_model']:
                self.fake_detector.save_model()
            
            # Test on real food reviews
            batch_results = self.fake_detector.batch_detect_fake_reviews(limit=100)
            if batch_results:
                self.fake_detector.save_detection_results(batch_results)
            
            training_time = time.time() - start_time
            
            fake_detection_results = {
                'status': 'success',
                'training_samples': len(features_df),
                'features_count': len(features_df.columns),
                'training_time': training_time,
                'dataset_type': 'Amazon Fine Food Reviews',
                'model_performance': {
                    'train_score': training_results['train_score'],
                    'test_score': training_results['test_score'],
                    'auc_score': training_results['auc_score'],
                    'cv_mean': training_results['cv_mean'],
                    'cv_std': training_results['cv_std']
                },
                'best_params': training_results['best_params'],
                'feature_importance': training_results['feature_importance'],
                'batch_test_results': {
                    'samples_tested': len(batch_results) if batch_results else 0,
                    'avg_fake_probability': np.mean([r['fake_probability'] for r in batch_results]) if batch_results else 0
                }
            }
            
            logger.info(f"SUCCESS: Food reviews fake detection - AUC: {training_results['auc_score']:.3f}")
            return fake_detection_results
            
        except Exception as e:
            logger.error(f"ERROR: Food reviews fake detection training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def train_recommendation_system(self) -> Dict:
        """Train and evaluate recommendation system for food products"""
        logger.info("TRAINING: Recommendation System for Food Products")
        start_time = time.time()
        
        try:
            # Get system statistics
            stats = self.recommendation_engine.get_recommendation_stats()
            
            # Test recommendations for sample users (food product recommendations)
            sample_users = list(self.recommendation_engine.user_item_matrix.index)[:10]
            recommendation_results = []
            
            for user_id in sample_users:
                try:
                    # Generate food product recommendations
                    recommendations = self.recommendation_engine.get_hybrid_recommendations(
                        user_id, 
                        n_recommendations=self.config['recommendations']['n_recommendations']
                    )
                    
                    if recommendations:
                        # Save food product recommendations
                        if self.config['recommendations']['save_results']:
                            self.recommendation_engine.save_user_recommendations(user_id, recommendations)
                        
                        # Calculate quality metrics for food recommendations
                        avg_predicted_rating = np.mean([rec.predicted_rating for rec in recommendations])
                        avg_confidence = np.mean([rec.confidence for rec in recommendations])
                        
                        recommendation_results.append({
                            'user_id': user_id,
                            'recommendations_count': len(recommendations),
                            'avg_predicted_rating': avg_predicted_rating,
                            'avg_confidence': avg_confidence
                        })
                        
                except Exception as e:
                    logger.warning(f"WARNING: Failed to generate food recommendations for user {user_id}: {e}")
                    continue
            
            training_time = time.time() - start_time
            
            recommendation_system_results = {
                'status': 'success',
                'training_time': training_time,
                'dataset_type': 'Amazon Fine Food Reviews',
                'system_statistics': stats,
                'sample_recommendations': {
                    'users_tested': len(recommendation_results),
                    'successful_recommendations': len([r for r in recommendation_results if r['recommendations_count'] > 0]),
                    'avg_predicted_rating': np.mean([r['avg_predicted_rating'] for r in recommendation_results]) if recommendation_results else 0,
                    'avg_confidence': np.mean([r['avg_confidence'] for r in recommendation_results]) if recommendation_results else 0
                },
                'recommendation_details': recommendation_results[:5]  # First 5 for brevity
            }
            
            logger.info(f"SUCCESS: Food product recommendations - {len(recommendation_results)} users processed")
            return recommendation_system_results
            
        except Exception as e:
            logger.error(f"ERROR: Food recommendation system training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def evaluate_system_performance(self) -> Dict:
        """Evaluate overall system performance for food reviews"""
        logger.info("EVALUATING: Overall system performance for food reviews")
        
        try:
            # Check database tables and record counts
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            table_stats = {}
            tables = ['reviews', 'sentiment_analysis', 'fake_detection', 'user_recommendations']
            
            for table in tables:
                try:
                    count_query = f"SELECT COUNT(*) FROM {table}"
                    count = conn.execute(count_query).fetchone()[0]
                    table_stats[table] = count
                except Exception as e:
                    table_stats[table] = f"Error: {e}"
            
            conn.close()
            
            # Calculate system-wide metrics for food reviews
            system_metrics = {
                'database_health': table_stats,
                'model_coverage': {
                    'sentiment_analysis': table_stats.get('sentiment_analysis', 0) > 0,
                    'fake_detection': table_stats.get('fake_detection', 0) > 0,
                    'recommendations': table_stats.get('user_recommendations', 0) > 0
                },
                'data_pipeline_status': 'operational' if table_stats.get('reviews', 0) > 0 else 'failed',
                'dataset_type': 'Amazon Fine Food Reviews'
            }
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"ERROR: System evaluation failed: {e}")
            return {'error': str(e)}
    
    def generate_training_report(self) -> Dict:
        """Generate comprehensive training report for food reviews"""
        logger.info("GENERATING: Comprehensive training report for food reviews")
        
        report = {
            'training_session': {
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'database_path': self.db_path,
                'dataset_type': 'Amazon Fine Food Reviews'
            },
            'results': self.training_results,
            'system_evaluation': self.evaluate_system_performance()
        }
        
        # Calculate overall success rate
        successful_models = sum(1 for model_results in self.training_results.values() 
                              if model_results.get('status') == 'success')
        total_models = len(self.training_results)
        
        report['summary'] = {
            'total_models_trained': total_models,
            'successful_models': successful_models,
            'success_rate': successful_models / total_models if total_models > 0 else 0,
            'overall_status': 'success' if successful_models == total_models else 'partial_success',
            'dataset_focus': 'Amazon Fine Food Reviews'
        }
        
        return report
    
    def save_training_report(self, report: Dict, filepath: str = "reports/food_training_report.json"):
        """Save training report to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"SUCCESS: Food reviews training report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save training report: {e}")
    
    def run_full_training_pipeline(self) -> Dict:
        """Run complete training pipeline for all models on Amazon Fine Food Reviews"""
        logger.info("STARTING: Full ML Training Pipeline for Amazon Fine Food Reviews")
        pipeline_start_time = time.time()
        
        print("ML MODEL TRAINING PIPELINE - AMAZON FINE FOOD REVIEWS")
        print("=" * 50)
        
        try:
            # Setup models
            print("\nSTEP 1: Setting up models for food reviews...")
            self.setup_models()
            print("SUCCESS: All models initialized for food reviews")
            
            # Train sentiment analysis for food reviews
            print("\nSTEP 2: Training sentiment analysis for food reviews...")
            self.training_results['sentiment_analysis'] = self.train_sentiment_analysis()
            if self.training_results['sentiment_analysis']['status'] == 'success':
                print(f"SUCCESS: Food sentiment analysis - {self.training_results['sentiment_analysis']['samples_processed']} samples")
            else:
                print(f"FAILED: Food sentiment analysis - {self.training_results['sentiment_analysis'].get('error', 'Unknown error')}")
            
            # Train fake detection for food reviews
            print("\nSTEP 3: Training fake detection for food reviews...")
            self.training_results['fake_detection'] = self.train_fake_detection()
            if self.training_results['fake_detection']['status'] == 'success':
                auc = self.training_results['fake_detection']['model_performance']['auc_score']
                print(f"SUCCESS: Food fake detection - AUC Score: {auc:.3f}")
            else:
                print(f"FAILED: Food fake detection - {self.training_results['fake_detection'].get('error', 'Unknown error')}")
            
            # Train recommendation system for food products
            print("\nSTEP 4: Training recommendation system for food products...")
            self.training_results['recommendation_system'] = self.train_recommendation_system()
            if self.training_results['recommendation_system']['status'] == 'success':
                users = self.training_results['recommendation_system']['sample_recommendations']['users_tested']
                print(f"SUCCESS: Food recommendations - {users} users processed")
            else:
                print(f"FAILED: Food recommendations - {self.training_results['recommendation_system'].get('error', 'Unknown error')}")
            
            # Generate report
            print("\nSTEP 5: Generating training report...")
            report = self.generate_training_report()
            self.save_training_report(report)
            
            total_time = time.time() - pipeline_start_time
            
            print(f"\nPIPELINE SUMMARY FOR AMAZON FINE FOOD REVIEWS:")
            print(f"Total Training Time: {total_time:.1f} seconds")
            print(f"Models Trained: {report['summary']['successful_models']}/{report['summary']['total_models_trained']}")
            print(f"Success Rate: {report['summary']['success_rate']*100:.1f}%")
            print(f"Overall Status: {report['summary']['overall_status'].upper()}")
            print(f"Dataset Focus: {report['summary']['dataset_focus']}")
            
            # Detailed results for food reviews
            print(f"\nDETAILED RESULTS FOR FOOD REVIEWS:")
            if 'sentiment_analysis' in self.training_results:
                sa_results = self.training_results['sentiment_analysis']
                if sa_results['status'] == 'success':
                    print(f"  Food Sentiment Analysis: {sa_results['performance_metrics']['avg_confidence']:.3f} avg confidence")
            
            if 'fake_detection' in self.training_results:
                fd_results = self.training_results['fake_detection']
                if fd_results['status'] == 'success':
                    print(f"  Food Fake Detection: {fd_results['model_performance']['auc_score']:.3f} AUC score")
            
            if 'recommendation_system' in self.training_results:
                rs_results = self.training_results['recommendation_system']
                if rs_results['status'] == 'success':
                    print(f"  Food Recommendations: {rs_results['sample_recommendations']['avg_predicted_rating']:.2f} avg rating")
            
            logger.info(f"SUCCESS: Full training pipeline for food reviews completed in {total_time:.1f}s")
            return report
            
        except Exception as e:
            logger.error(f"ERROR: Training pipeline failed: {e}")
            print(f"\nERROR: Training pipeline failed: {e}")
            return {'status': 'failed', 'error': str(e)}

def main():
    """
    Main function for running the training pipeline on Amazon Fine Food Reviews
    """
    print("STARTING: ML Model Training Pipeline for Amazon Fine Food Reviews")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Run full training
    report = pipeline.run_full_training_pipeline()
    
    print(f"\nCOMPLETE: Food reviews training pipeline finished!")
    print(f"Report saved to: reports/food_training_report.json")

if __name__ == "__main__":
    main()