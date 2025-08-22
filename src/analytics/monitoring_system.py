"""
Real-time Monitoring System - Day 3 Production Monitoring
Comprehensive system health monitoring and performance tracking
Feeds data to analytics dashboard and alert manager
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import time
import logging
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import schedule
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import our ML models for monitoring
try:
    from src.ml_models.sentiment_analyzer import SentimentAnalyzer
    from src.ml_models.fake_detector import FakeReviewDetector
    from src.ml_models.recommendation_engine import RecommendationEngine
except ImportError as e:
    print(f"Warning: Could not import models: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetric:
    """System performance metric"""
    metric_name: str
    metric_value: float
    metric_unit: str
    metric_type: str  # 'system', 'model', 'business'
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'metric_type': self.metric_type,
            'timestamp': self.timestamp.isoformat(),
            'metadata': json.dumps(self.metadata or {})
        }

@dataclass
class ModelPerformanceMetric:
    """Model-specific performance metric"""
    model_name: str
    metric_name: str
    metric_value: float
    sample_size: int
    processing_time: float
    confidence: float
    timestamp: datetime
    additional_data: Dict[str, Any] = None

@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    response_time: float
    error_message: Optional[str]
    timestamp: datetime
    details: Dict[str, Any] = None

class RealTimeMonitoringSystem:
    """
    Comprehensive real-time monitoring system for Product Review Intelligence
    Tracks system health, model performance, and business metrics
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Monitoring configuration
        self.monitoring_interval = 60  # seconds
        self.health_check_interval = 300  # 5 minutes
        self.model_test_interval = 1800  # 30 minutes
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'model_accuracy': 0.70,
            'error_rate': 0.05
        }
        
        # Initialize models for testing
        self.models = {}
        self.metrics_buffer = []
        self.max_buffer_size = 1000
        
        # Performance tracking
        self.performance_history = {
            'system': [],
            'models': [],
            'business': []
        }
        
        logger.info("SUCCESS: RealTimeMonitoringSystem initialized")
    
    def initialize_models(self):
        """Initialize ML models for performance monitoring"""
        logger.info("INITIALIZING: ML models for monitoring...")
        
        try:
            # Initialize sentiment analyzer
            self.models['sentiment'] = SentimentAnalyzer(self.db_path)
            logger.info("SUCCESS: Sentiment analyzer initialized for monitoring")
            
            # Initialize fake detector
            self.models['fake_detector'] = FakeReviewDetector(self.db_path)
            try:
                self.models['fake_detector'].load_model()
                logger.info("SUCCESS: Fake detector initialized for monitoring")
            except FileNotFoundError:
                logger.warning("WARNING: Fake detector model not found for monitoring")
            
            # Initialize recommendation engine
            self.models['recommendation'] = RecommendationEngine(self.db_path)
            logger.info("SUCCESS: Recommendation engine initialized for monitoring")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize models for monitoring: {e}")
            return False
    
    def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect comprehensive system performance metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(SystemMetric(
                metric_name='cpu_usage',
                metric_value=cpu_percent,
                metric_unit='percent',
                metric_type='system',
                timestamp=timestamp,
                metadata={'cores': psutil.cpu_count()}
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(SystemMetric(
                metric_name='memory_usage',
                metric_value=memory.percent,
                metric_unit='percent',
                metric_type='system',
                timestamp=timestamp,
                metadata={
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2)
                }
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(SystemMetric(
                metric_name='disk_usage',
                metric_value=disk_percent,
                metric_unit='percent',
                metric_type='system',
                timestamp=timestamp,
                metadata={
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2)
                }
            ))
            
            # Database metrics
            db_size = self._get_database_size()
            metrics.append(SystemMetric(
                metric_name='database_size',
                metric_value=db_size,
                metric_unit='mb',
                metric_type='system',
                timestamp=timestamp,
                metadata={'database_path': self.db_path}
            ))
            
            # Process metrics
            process = psutil.Process()
            metrics.append(SystemMetric(
                metric_name='process_memory',
                metric_value=process.memory_info().rss / (1024**2),  # MB
                metric_unit='mb',
                metric_type='system',
                timestamp=timestamp,
                metadata={'pid': process.pid}
            ))
            
            logger.info(f"SUCCESS: Collected {len(metrics)} system metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"ERROR: Failed to collect system metrics: {e}")
            return []
    
    def _get_database_size(self) -> float:
        """Get database file size in MB"""
        try:
            if os.path.exists(self.db_path):
                size_bytes = os.path.getsize(self.db_path)
                return round(size_bytes / (1024**2), 2)  # Convert to MB
            return 0.0
        except Exception:
            return 0.0
    
    def test_model_performance(self) -> List[ModelPerformanceMetric]:
        """Test model performance with sample data"""
        performance_metrics = []
        timestamp = datetime.now()
        
        if not self.models:
            logger.warning("WARNING: No models available for performance testing")
            return []
        
        logger.info("TESTING: Model performance...")
        
        # Get sample reviews for testing
        sample_reviews = self._get_sample_reviews(limit=10)
        
        if not sample_reviews:
            logger.warning("WARNING: No sample reviews available for model testing")
            return []
        
        # Test sentiment analyzer
        if 'sentiment' in self.models:
            sentiment_metrics = self._test_sentiment_performance(sample_reviews, timestamp)
            performance_metrics.extend(sentiment_metrics)
        
        # Test fake detector
        if 'fake_detector' in self.models and hasattr(self.models['fake_detector'], 'model') and self.models['fake_detector'].model:
            fake_metrics = self._test_fake_detector_performance(sample_reviews, timestamp)
            performance_metrics.extend(fake_metrics)
        
        # Test recommendation engine
        if 'recommendation' in self.models:
            rec_metrics = self._test_recommendation_performance(timestamp)
            performance_metrics.extend(rec_metrics)
        
        logger.info(f"SUCCESS: Collected {len(performance_metrics)} model performance metrics")
        return performance_metrics
    
    def _get_sample_reviews(self, limit: int = 10) -> List[Dict]:
        """Get sample reviews for model testing"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT Id, product_id, user_id, rating, review_text, helpful_votes, total_votes
                FROM reviews 
                WHERE review_text IS NOT NULL 
                AND LENGTH(review_text) > 20
                ORDER BY RANDOM()
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[limit])
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get sample reviews: {e}")
            return []
    
    def _test_sentiment_performance(self, sample_reviews: List[Dict], timestamp: datetime) -> List[ModelPerformanceMetric]:
        """Test sentiment analyzer performance"""
        metrics = []
        
        try:
            sentiment_analyzer = self.models['sentiment']
            start_time = time.time()
            
            accuracies = []
            confidences = []
            processing_times = []
            
            for review in sample_reviews:
                try:
                    review_start = time.time()
                    result = sentiment_analyzer.analyze_review(review['review_text'])
                    review_time = time.time() - review_start
                    
                    # Calculate accuracy against rating
                    predicted_sentiment = result.overall_sentiment
                    actual_rating = review['rating']
                    
                    if actual_rating >= 4 and predicted_sentiment == 'POSITIVE':
                        accuracy = 1.0
                    elif actual_rating <= 2 and predicted_sentiment == 'NEGATIVE':
                        accuracy = 1.0
                    elif actual_rating == 3 and predicted_sentiment == 'NEUTRAL':
                        accuracy = 1.0
                    else:
                        accuracy = 0.0
                    
                    accuracies.append(accuracy)
                    confidences.append(result.confidence)
                    processing_times.append(review_time)
                    
                except Exception as e:
                    logger.warning(f"WARNING: Failed to test sentiment for review: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            if accuracies:
                metrics.append(ModelPerformanceMetric(
                    model_name='sentiment_analyzer',
                    metric_name='accuracy',
                    metric_value=np.mean(accuracies),
                    sample_size=len(accuracies),
                    processing_time=total_time,
                    confidence=np.mean(confidences),
                    timestamp=timestamp,
                    additional_data={
                        'avg_processing_time_per_review': np.mean(processing_times),
                        'max_processing_time': max(processing_times),
                        'min_processing_time': min(processing_times)
                    }
                ))
                
                metrics.append(ModelPerformanceMetric(
                    model_name='sentiment_analyzer',
                    metric_name='throughput',
                    metric_value=len(accuracies) / total_time,
                    sample_size=len(accuracies),
                    processing_time=total_time,
                    confidence=np.mean(confidences),
                    timestamp=timestamp
                ))
        
        except Exception as e:
            logger.error(f"ERROR: Sentiment performance test failed: {e}")
        
        return metrics
    
    def _test_fake_detector_performance(self, sample_reviews: List[Dict], timestamp: datetime) -> List[ModelPerformanceMetric]:
        """Test fake detector performance"""
        metrics = []
        
        try:
            fake_detector = self.models['fake_detector']
            start_time = time.time()
            
            fake_probabilities = []
            confidences = []
            processing_times = []
            
            for review in sample_reviews:
                try:
                    review_start = time.time()
                    
                    # Prepare test data
                    user_data = {'review_count': 10, 'avg_rating': 3.5}
                    review_data = {
                        'rating': review['rating'],
                        'helpful_votes': review.get('helpful_votes', 0),
                        'total_votes': review.get('total_votes', 0)
                    }
                    
                    result = fake_detector.predict_fake_probability(
                        review['review_text'],
                        user_data=user_data,
                        review_data=review_data
                    )
                    
                    review_time = time.time() - review_start
                    
                    fake_probabilities.append(result.is_fake_probability)
                    confidences.append(result.confidence)
                    processing_times.append(review_time)
                    
                except Exception as e:
                    logger.warning(f"WARNING: Failed to test fake detection for review: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            if fake_probabilities:
                metrics.append(ModelPerformanceMetric(
                    model_name='fake_detector',
                    metric_name='avg_fake_probability',
                    metric_value=np.mean(fake_probabilities),
                    sample_size=len(fake_probabilities),
                    processing_time=total_time,
                    confidence=np.mean(confidences),
                    timestamp=timestamp,
                    additional_data={
                        'avg_processing_time_per_review': np.mean(processing_times),
                        'high_risk_count': sum(1 for p in fake_probabilities if p > 0.7),
                        'low_risk_count': sum(1 for p in fake_probabilities if p < 0.3)
                    }
                ))
                
                metrics.append(ModelPerformanceMetric(
                    model_name='fake_detector',
                    metric_name='throughput',
                    metric_value=len(fake_probabilities) / total_time,
                    sample_size=len(fake_probabilities),
                    processing_time=total_time,
                    confidence=np.mean(confidences),
                    timestamp=timestamp
                ))
        
        except Exception as e:
            logger.error(f"ERROR: Fake detector performance test failed: {e}")
        
        return metrics
    
    def _test_recommendation_performance(self, timestamp: datetime) -> List[ModelPerformanceMetric]:
        """Test recommendation engine performance"""
        metrics = []
        
        try:
            rec_engine = self.models['recommendation']
            start_time = time.time()
            
            # Test with a few sample users
            sample_users = list(rec_engine.user_item_matrix.index)[:5]
            
            successful_recommendations = 0
            total_users_tested = 0
            recommendation_counts = []
            processing_times = []
            
            for user_id in sample_users:
                try:
                    user_start = time.time()
                    recommendations = rec_engine.get_hybrid_recommendations(user_id, n_recommendations=5)
                    user_time = time.time() - user_start
                    
                    total_users_tested += 1
                    processing_times.append(user_time)
                    
                    if recommendations:
                        successful_recommendations += 1
                        recommendation_counts.append(len(recommendations))
                    else:
                        recommendation_counts.append(0)
                        
                except Exception as e:
                    logger.warning(f"WARNING: Failed to test recommendations for user {user_id}: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            if total_users_tested > 0:
                success_rate = successful_recommendations / total_users_tested
                avg_recommendations = np.mean(recommendation_counts) if recommendation_counts else 0
                
                metrics.append(ModelPerformanceMetric(
                    model_name='recommendation_engine',
                    metric_name='success_rate',
                    metric_value=success_rate,
                    sample_size=total_users_tested,
                    processing_time=total_time,
                    confidence=success_rate,
                    timestamp=timestamp,
                    additional_data={
                        'avg_recommendations_per_user': avg_recommendations,
                        'avg_processing_time_per_user': np.mean(processing_times) if processing_times else 0,
                        'successful_users': successful_recommendations
                    }
                ))
                
                if total_time > 0:
                    metrics.append(ModelPerformanceMetric(
                        model_name='recommendation_engine',
                        metric_name='throughput',
                        metric_value=total_users_tested / total_time,
                        sample_size=total_users_tested,
                        processing_time=total_time,
                        confidence=success_rate,
                        timestamp=timestamp
                    ))
        
        except Exception as e:
            logger.error(f"ERROR: Recommendation performance test failed: {e}")
        
        return metrics
    
    def perform_health_checks(self) -> List[HealthCheckResult]:
        """Perform comprehensive health checks"""
        health_results = []
        timestamp = datetime.now()
        
        logger.info("PERFORMING: System health checks...")
        
        # Database health check
        db_health = self._check_database_health(timestamp)
        health_results.append(db_health)
        
        # Model health checks
        model_health = self._check_models_health(timestamp)
        health_results.extend(model_health)
        
        # System resource health checks
        resource_health = self._check_resource_health(timestamp)
        health_results.extend(resource_health)
        
        logger.info(f"SUCCESS: Completed {len(health_results)} health checks")
        return health_results
    
    def _check_database_health(self, timestamp: datetime) -> HealthCheckResult:
        """Check database connectivity and basic operations"""
        start_time = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reviews")
            count = cursor.fetchone()[0]
            
            # Test write operation
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            conn.close()
            
            response_time = time.time() - start_time
            
            status = 'healthy' if response_time < 1.0 else 'warning'
            
            return HealthCheckResult(
                component='database',
                status=status,
                response_time=response_time,
                error_message=None,
                timestamp=timestamp,
                details={
                    'total_reviews': count,
                    'database_size_mb': self._get_database_size(),
                    'connection_timeout': '5s'
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component='database',
                status='critical',
                response_time=time.time() - start_time,
                error_message=str(e),
                timestamp=timestamp,
                details={'error_type': type(e).__name__}
            )
    
    def _check_models_health(self, timestamp: datetime) -> List[HealthCheckResult]:
        """Check model availability and basic functionality"""
        health_results = []
        
        for model_name, model in self.models.items():
            start_time = time.time()
            
            try:
                if model_name == 'sentiment':
                    # Test sentiment analyzer
                    result = model.analyze_review("This is a test review for monitoring.")
                    status = 'healthy' if result.confidence > 0.5 else 'warning'
                    details = {'confidence': result.confidence, 'sentiment': result.overall_sentiment}
                    
                elif model_name == 'fake_detector' and hasattr(model, 'model') and model.model:
                    # Test fake detector
                    result = model.predict_fake_probability("This is a test review for monitoring.")
                    status = 'healthy' if result.confidence > 0.5 else 'warning'
                    details = {'fake_probability': result.is_fake_probability, 'confidence': result.confidence}
                    
                elif model_name == 'recommendation':
                    # Test recommendation engine
                    stats = model.get_recommendation_stats()
                    status = 'healthy' if stats.get('total_users', 0) > 0 else 'warning'
                    details = {'total_users': stats.get('total_users', 0), 'total_products': stats.get('total_products', 0)}
                    
                else:
                    status = 'warning'
                    details = {'note': 'Model not fully initialized'}
                
                response_time = time.time() - start_time
                
                health_results.append(HealthCheckResult(
                    component=f'model_{model_name}',
                    status=status,
                    response_time=response_time,
                    error_message=None,
                    timestamp=timestamp,
                    details=details
                ))
                
            except Exception as e:
                health_results.append(HealthCheckResult(
                    component=f'model_{model_name}',
                    status='critical',
                    response_time=time.time() - start_time,
                    error_message=str(e),
                    timestamp=timestamp,
                    details={'error_type': type(e).__name__}
                ))
        
        return health_results
    
    def _check_resource_health(self, timestamp: datetime) -> List[HealthCheckResult]:
        """Check system resource health"""
        health_results = []
        
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = 'healthy' if cpu_percent < self.thresholds['cpu_usage'] else 'warning'
            if cpu_percent > 95:
                cpu_status = 'critical'
            
            health_results.append(HealthCheckResult(
                component='cpu',
                status=cpu_status,
                response_time=1.0,  # CPU check time
                error_message=None if cpu_status == 'healthy' else f'High CPU usage: {cpu_percent}%',
                timestamp=timestamp,
                details={'usage_percent': cpu_percent, 'threshold': self.thresholds['cpu_usage']}
            ))
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_status = 'healthy' if memory.percent < self.thresholds['memory_usage'] else 'warning'
            if memory.percent > 95:
                memory_status = 'critical'
            
            health_results.append(HealthCheckResult(
                component='memory',
                status=memory_status,
                response_time=0.1,
                error_message=None if memory_status == 'healthy' else f'High memory usage: {memory.percent}%',
                timestamp=timestamp,
                details={
                    'usage_percent': memory.percent,
                    'available_gb': round(memory.available / (1024**3), 2),
                    'threshold': self.thresholds['memory_usage']
                }
            ))
            
            # Disk check
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = 'healthy' if disk_percent < self.thresholds['disk_usage'] else 'warning'
            if disk_percent > 98:
                disk_status = 'critical'
            
            health_results.append(HealthCheckResult(
                component='disk',
                status=disk_status,
                response_time=0.1,
                error_message=None if disk_status == 'healthy' else f'High disk usage: {disk_percent:.1f}%',
                timestamp=timestamp,
                details={
                    'usage_percent': disk_percent,
                    'free_gb': round(disk.free / (1024**3), 2),
                    'threshold': self.thresholds['disk_usage']
                }
            ))
            
        except Exception as e:
            health_results.append(HealthCheckResult(
                component='system_resources',
                status='critical',
                response_time=0.0,
                error_message=f'Resource check failed: {e}',
                timestamp=timestamp,
                details={'error_type': type(e).__name__}
            ))
        
        return health_results
    
    def save_metrics_to_database(self, system_metrics: List[SystemMetric], 
                                performance_metrics: List[ModelPerformanceMetric],
                                health_results: List[HealthCheckResult]):
        """Save all monitoring metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create monitoring tables if they don't exist
            self._create_monitoring_tables(conn)
            
            # Save system metrics
            for metric in system_metrics:
                conn.execute("""
                    INSERT INTO system_metrics 
                    (metric_name, metric_value, metric_unit, metric_type, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_name,
                    metric.metric_value,
                    metric.metric_unit,
                    metric.metric_type,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.metadata or {})
                ))
            
            # Save model performance metrics
            for metric in performance_metrics:
                conn.execute("""
                    INSERT INTO model_performance_metrics
                    (model_name, metric_name, metric_value, sample_size, processing_time,
                     confidence, timestamp, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.model_name,
                    metric.metric_name,
                    metric.metric_value,
                    metric.sample_size,
                    metric.processing_time,
                    metric.confidence,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.additional_data or {})
                ))
            
            # Save health check results
            for result in health_results:
                conn.execute("""
                    INSERT INTO health_checks
                    (component, status, response_time, error_message, timestamp, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.component,
                    result.status,
                    result.response_time,
                    result.error_message,
                    result.timestamp.isoformat(),
                    json.dumps(result.details or {})
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"SUCCESS: Saved {len(system_metrics)} system metrics, "
                       f"{len(performance_metrics)} performance metrics, "
                       f"{len(health_results)} health checks to database")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save monitoring metrics: {e}")
    
    def _create_monitoring_tables(self, conn):
        """Create monitoring tables in database"""
        # System metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                metric_type TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                sample_size INTEGER,
                processing_time REAL,
                confidence REAL,
                timestamp TEXT NOT NULL,
                additional_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Health checks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time REAL,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_performance_metrics(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_health_checks_timestamp ON health_checks(timestamp)")
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        logger.info("STARTING: Monitoring cycle")
        cycle_start = time.time()
        
        try:
            # Collect system metrics
            system_metrics = self.collect_system_metrics()
            
            # Test model performance
            performance_metrics = self.test_model_performance()
            
            # Perform health checks
            health_results = self.perform_health_checks()
            
            # Save to database
            self.save_metrics_to_database(system_metrics, performance_metrics, health_results)
            
            # Check for alerts
            alerts = self._check_for_alerts(system_metrics, performance_metrics, health_results)
            
            cycle_time = time.time() - cycle_start
            
            logger.info(f"SUCCESS: Monitoring cycle completed in {cycle_time:.2f}s")
            logger.info(f"Collected: {len(system_metrics)} system metrics, "
                       f"{len(performance_metrics)} performance metrics, "
                       f"{len(health_results)} health checks")
            
            if alerts:
                logger.warning(f"ALERTS: {len(alerts)} alerts generated")
                
            return {
                'success': True,
                'cycle_time': cycle_time,
                'metrics_collected': len(system_metrics) + len(performance_metrics),
                'health_checks': len(health_results),
                'alerts_generated': len(alerts)
            }
            
        except Exception as e:
            logger.error(f"ERROR: Monitoring cycle failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'cycle_time': time.time() - cycle_start
            }
    
    def _check_for_alerts(self, system_metrics: List[SystemMetric], 
                         performance_metrics: List[ModelPerformanceMetric],
                         health_results: List[HealthCheckResult]) -> List[Dict]:
        """Check for conditions that should trigger alerts"""
        alerts = []
        
        # Check system metrics against thresholds
        for metric in system_metrics:
            if metric.metric_name in self.thresholds:
                threshold = self.thresholds[metric.metric_name]
                if metric.metric_value > threshold:
                    alerts.append({
                        'type': 'system_threshold',
                        'severity': 'warning',
                        'component': metric.metric_name,
                        'message': f'{metric.metric_name} ({metric.metric_value:.1f}%) exceeds threshold ({threshold}%)',
                        'metric_value': metric.metric_value,
                        'threshold': threshold,
                        'timestamp': metric.timestamp.isoformat()
                    })
        
        # Check model performance
        for metric in performance_metrics:
            if metric.metric_name == 'accuracy' and metric.metric_value < self.thresholds['model_accuracy']:
                alerts.append({
                    'type': 'model_performance',
                    'severity': 'warning',
                    'component': metric.model_name,
                    'message': f'{metric.model_name} accuracy ({metric.metric_value:.1%}) below threshold ({self.thresholds["model_accuracy"]:.1%})',
                    'metric_value': metric.metric_value,
                    'threshold': self.thresholds['model_accuracy'],
                    'timestamp': metric.timestamp.isoformat()
                })
        
        # Check health status
        for health in health_results:
            if health.status in ['warning', 'critical']:
                alerts.append({
                    'type': 'health_check',
                    'severity': health.status,
                    'component': health.component,
                    'message': health.error_message or f'{health.component} health check status: {health.status}',
                    'response_time': health.response_time,
                    'timestamp': health.timestamp.isoformat()
                })
        
        return alerts
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        if self.monitoring_active:
            logger.warning("WARNING: Monitoring already active")
            return
        
        logger.info("STARTING: Continuous monitoring system")
        
        # Initialize models
        if not self.initialize_models():
            logger.error("ERROR: Failed to initialize models, monitoring may be limited")
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Run monitoring cycle
                    result = self.run_monitoring_cycle()
                    
                    # Wait for next cycle
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"ERROR: Monitoring loop error: {e}")
                    time.sleep(30)  # Wait before retrying
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"SUCCESS: Monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.monitoring_active:
            logger.warning("WARNING: Monitoring not active")
            return
        
        logger.info("STOPPING: Continuous monitoring system")
        
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        logger.info("SUCCESS: Monitoring stopped")
    
    def get_recent_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get recent monitoring metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Get recent system metrics
            system_query = """
                SELECT metric_name, metric_value, metric_unit, timestamp
                FROM system_metrics 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """
            system_df = pd.read_sql_query(system_query, conn, params=[since_time])
            
            # Get recent model metrics
            model_query = """
                SELECT model_name, metric_name, metric_value, timestamp
                FROM model_performance_metrics 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """
            model_df = pd.read_sql_query(model_query, conn, params=[since_time])
            
            # Get recent health checks
            health_query = """
                SELECT component, status, response_time, timestamp
                FROM health_checks 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """
            health_df = pd.read_sql_query(health_query, conn, params=[since_time])
            
            conn.close()
            
            return {
                'system_metrics': system_df.to_dict('records'),
                'model_metrics': model_df.to_dict('records'),
                'health_checks': health_df.to_dict('records'),
                'timeframe_hours': hours,
                'total_records': len(system_df) + len(model_df) + len(health_df)
            }
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get recent metrics: {e}")
            return {'error': str(e)}

def main():
    """Main function for testing monitoring system"""
    print("TESTING: Real-time Monitoring System - Day 3")
    print("=" * 60)
    
    # Initialize monitoring system
    monitoring = RealTimeMonitoringSystem()
    
    print("\nTESTING: Single monitoring cycle...")
    
    # Initialize models
    if monitoring.initialize_models():
        print("✅ Models initialized successfully")
    else:
        print("⚠️ Model initialization had issues")
    
    # Run single monitoring cycle
    result = monitoring.run_monitoring_cycle()
    
    if result['success']:
        print(f"✅ Monitoring cycle completed successfully")
        print(f"   Cycle time: {result['cycle_time']:.2f}s")
        print(f"   Metrics collected: {result['metrics_collected']}")
        print(f"   Health checks: {result['health_checks']}")
        print(f"   Alerts generated: {result['alerts_generated']}")
    else:
        print(f"❌ Monitoring cycle failed: {result['error']}")
    
    # Test getting recent metrics
    print(f"\nTESTING: Recent metrics retrieval...")
    recent_metrics = monitoring.get_recent_metrics(hours=1)
    
    if 'error' not in recent_metrics:
        print(f"✅ Retrieved {recent_metrics['total_records']} recent metrics")
    else:
        print(f"❌ Failed to retrieve metrics: {recent_metrics['error']}")
    
    # Test short-term continuous monitoring
    print(f"\nTESTING: Short-term continuous monitoring (30 seconds)...")
    
    # Set short interval for testing
    monitoring.monitoring_interval = 10  # 10 seconds for testing
    
    monitoring.start_monitoring()
    print("✅ Monitoring started")
    
    # Let it run for 30 seconds
    time.sleep(30)
    
    monitoring.stop_monitoring()
    print("✅ Monitoring stopped")
    
    print(f"\n✅ COMPLETE: Monitoring system test finished!")
    print(f"Check database for monitoring metrics: system_metrics, model_performance_metrics, health_checks")

if __name__ == "__main__":
    main()