"""
Fake Review Detection System using Traditional ML
Uses features from review text, user behavior, and sentiment patterns
Specialized for Amazon Fine Food Reviews
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib

# Text processing libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Installing NLTK...")
    os.system("pip install nltk")
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/food_fake_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FakeDetectionResult:
    """Result of fake review detection"""
    is_fake_probability: float  # 0-1, higher = more likely fake
    confidence: float
    risk_level: str  # LOW, MEDIUM, HIGH
    features_scores: Dict[str, float]
    processing_time: float

class FakeReviewDetector:
    """
    Advanced fake review detection using traditional ML features
    Specialized for Amazon Fine Food Reviews
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.feature_names = []
        self.sia = SentimentIntensityAnalyzer()
        
        # Fake review patterns and keywords for food reviews
        self.fake_patterns = self._initialize_food_fake_patterns()
        
        logger.info("SUCCESS: FakeReviewDetector initialized successfully for Amazon Fine Food Reviews")
    
    def _initialize_food_fake_patterns(self) -> Dict:
        """Initialize patterns commonly found in fake food reviews"""
        return {
            'generic_food_phrases': [
                'great taste', 'amazing flavor', 'delicious food', 'highly recommend',
                'fresh ingredients', 'excellent quality', 'love this product',
                'worth the money', 'perfect snack', 'good value', 'tasty treat'
            ],
            'spam_indicators': [
                'buy now', 'click here', 'visit my', 'check out',
                'follow me', 'discount code', 'free shipping', 'limited time'
            ],
            'excessive_punctuation': [
                '!!!', '???', '!!', '...', '***'
            ],
            'competitor_food_keywords': [
                'better than', 'compared to', 'unlike other brands',
                'worst food', 'dont buy', 'terrible taste', 'overpriced food',
                'much better brands', 'competitor tastes better'
            ],
            'food_specific_spam': [
                'changed my life', 'best food ever', 'amazing amazing',
                'perfect perfect', 'love love love', 'yummy yummy',
                'so good so good', 'fresh fresh fresh'
            ]
        }
    
    def extract_text_features(self, review_text: str) -> Dict[str, float]:
        """
        Extract text-based features for fake detection in food reviews
        """
        if not review_text or pd.isna(review_text):
            return {f'text_{key}': 0.0 for key in [
                'length', 'word_count', 'sentence_count', 'avg_word_length',
                'caps_ratio', 'punct_ratio', 'exclamation_ratio',
                'generic_food_phrase_count', 'spam_indicator_count',
                'excessive_punct_count', 'competitor_keyword_count',
                'food_spam_count', 'sentiment_compound', 'sentiment_extreme'
            ]}
        
        text = str(review_text).lower()
        
        # Basic text statistics
        text_length = len(text)
        words = word_tokenize(text)
        word_count = len(words)
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Advanced text features
        caps_count = sum(1 for c in review_text if c.isupper())
        caps_ratio = caps_count / len(review_text) if len(review_text) > 0 else 0
        
        punct_count = sum(1 for c in text if c in '!?.,;:')
        punct_ratio = punct_count / len(text) if len(text) > 0 else 0
        
        exclamation_count = text.count('!')
        exclamation_ratio = exclamation_count / len(text) if len(text) > 0 else 0
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Fake food review pattern detection
        generic_food_phrase_count = sum(1 for phrase in self.fake_patterns['generic_food_phrases'] if phrase in text)
        spam_indicator_count = sum(1 for phrase in self.fake_patterns['spam_indicators'] if phrase in text)
        excessive_punct_count = sum(1 for punct in self.fake_patterns['excessive_punctuation'] if punct in text)
        competitor_keyword_count = sum(1 for phrase in self.fake_patterns['competitor_food_keywords'] if phrase in text)
        food_spam_count = sum(1 for phrase in self.fake_patterns['food_specific_spam'] if phrase in text)
        
        # Sentiment analysis
        sentiment_scores = self.sia.polarity_scores(text)
        sentiment_compound = sentiment_scores['compound']
        sentiment_extreme = 1 if abs(sentiment_compound) > 0.8 else 0
        
        return {
            'text_length': text_length,
            'text_word_count': word_count,
            'text_sentence_count': sentence_count,
            'text_avg_word_length': avg_word_length,
            'text_caps_ratio': caps_ratio,
            'text_punct_ratio': punct_ratio,
            'text_exclamation_ratio': exclamation_ratio,
            'text_generic_food_phrase_count': generic_food_phrase_count,
            'text_spam_indicator_count': spam_indicator_count,
            'text_excessive_punct_count': excessive_punct_count,
            'text_competitor_keyword_count': competitor_keyword_count,
            'text_food_spam_count': food_spam_count,
            'text_sentiment_compound': sentiment_compound,
            'text_sentiment_extreme': sentiment_extreme
        }
    
    def extract_user_features(self, user_data: Dict) -> Dict[str, float]:
        """
        Extract user behavior features for fake detection
        """
        return {
            'user_review_count': user_data.get('review_count', 0),
            'user_avg_rating': user_data.get('avg_rating', 3.0),
            'user_rating_variance': user_data.get('rating_variance', 1.0),
            'user_days_active': user_data.get('days_active', 30),
            'user_reviews_per_day': user_data.get('reviews_per_day', 0.1)
        }
    
    def extract_review_features(self, review_data: Dict) -> Dict[str, float]:
        """
        Extract review-specific features for fake detection
        """
        return {
            'review_rating': review_data.get('rating', 3.0),
            'review_helpful_votes': review_data.get('helpful_votes', 0),
            'review_total_votes': review_data.get('total_votes', 0),
            'review_helpfulness_ratio': (
                review_data.get('helpful_votes', 0) / max(review_data.get('total_votes', 1), 1)
            ),
            'review_rating_deviation': review_data.get('rating_deviation', 0)
        }
    
    def extract_all_features(self, review_text: str, user_data: Dict = None, 
                           review_data: Dict = None) -> Dict[str, float]:
        """
        Extract comprehensive feature set for fake detection in food reviews
        """
        features = {}
        
        # Text features
        text_features = self.extract_text_features(review_text)
        features.update(text_features)
        
        # User features (with defaults if not provided)
        user_features = self.extract_user_features(user_data or {})
        features.update(user_features)
        
        # Review features (with defaults if not provided)
        review_features = self.extract_review_features(review_data or {})
        features.update(review_features)
        
        return features
    
    def prepare_training_data(self, limit: int = 2000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from real food reviews and synthetic fake reviews
        """
        logger.info(f"PREPARING: Training data for food reviews (limit: {limit})")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get real food reviews (labeled as authentic)
        real_reviews_query = """
            SELECT Id, review_text, rating, helpful_votes, total_votes,
                   rating_deviation, user_id, product_id
            FROM reviews
            ORDER BY RANDOM()
            LIMIT ?
        """
        real_reviews = pd.read_sql_query(real_reviews_query, conn, params=[limit//2])
        real_reviews['is_fake'] = 0
        
        # Try to get synthetic fake food reviews
        try:
            fake_reviews_query = """
                SELECT review_text, rating, 0 as helpful_votes, 0 as total_votes,
                       0 as rating_deviation, user_id, product_id, 
                       ROW_NUMBER() OVER() + 100000 as Id
                FROM synthetic_reviews
                WHERE is_fake = 1
                LIMIT ?
            """
            fake_reviews = pd.read_sql_query(fake_reviews_query, conn, params=[limit//2])
            fake_reviews['is_fake'] = 1
            
            logger.info(f"SUCCESS: Found {len(fake_reviews)} synthetic fake food reviews")
            
        except Exception as e:
            logger.warning(f"WARNING: Could not load synthetic food reviews: {e}")
            # Create some artificial fake food reviews for training
            fake_reviews = self._create_artificial_fake_food_reviews(limit//2)
        
        conn.close()
        
        # Combine datasets
        all_reviews = pd.concat([real_reviews, fake_reviews], ignore_index=True)
        
        # Extract features for each food review
        feature_list = []
        labels = []
        
        logger.info(f"EXTRACTING: Features from {len(all_reviews)} food reviews...")
        
        for idx, row in all_reviews.iterrows():
            try:
                # Prepare data dictionaries
                user_data = {
                    'review_count': np.random.randint(1, 50),  # Simulated user data
                    'avg_rating': np.random.uniform(2.0, 5.0),
                    'rating_variance': np.random.uniform(0.5, 2.0),
                    'days_active': np.random.randint(30, 365),
                    'reviews_per_day': np.random.uniform(0.01, 1.0)
                }
                
                review_data = {
                    'rating': row.get('rating', 3.0),
                    'helpful_votes': row.get('helpful_votes', 0),
                    'total_votes': row.get('total_votes', 0),
                    'rating_deviation': row.get('rating_deviation', 0)
                }
                
                features = self.extract_all_features(
                    row['review_text'], 
                    user_data, 
                    review_data
                )
                
                feature_list.append(features)
                labels.append(row['is_fake'])
                
                if (idx + 1) % 200 == 0:
                    logger.info(f"PROGRESS: Processed {idx + 1}/{len(all_reviews)} food reviews")
                    
            except Exception as e:
                logger.error(f"ERROR: Failed to process food review {idx}: {e}")
                continue
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_list)
        labels_series = pd.Series(labels)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"SUCCESS: Prepared {len(features_df)} food review samples with {len(features_df.columns)} features")
        return features_df, labels_series
    
    def _create_artificial_fake_food_reviews(self, count: int) -> pd.DataFrame:
        """
        Create artificial fake food reviews for training when synthetic data is not available
        """
        logger.info(f"CREATING: {count} artificial fake food reviews for training")
        
        fake_food_templates = [
            "Great food! Highly recommend! Amazing taste and fast shipping!",
            "Perfect snack! Love this product! Worth the money! Excellent flavor!",
            "Best purchase ever! Fantastic quality! Fast delivery! Highly recommended!",
            "Amazing! Perfect! Great! Excellent! Fantastic! Wonderful! Outstanding taste!",
            "This food is the best! Better than any other brand! Buy now!",
            "Terrible food! Worst quality! Don't buy! Save your money!",
            "Perfect 5 stars! No complaints! Fast shipping! Great price! Delicious!",
            "Love it love it love it! Best food ever! Amazing amazing amazing taste!",
            "Great food great price great service great shipping great everything!",
            "Awesome product! Click here for more! Visit my profile! Discount code! Yummy!",
            "Fresh fresh fresh! Tasty tasty! Delicious delicious! Perfect perfect!",
            "Changed my life completely! Best food investment ever! Amazing flavor!"
        ]
        
        fake_reviews = []
        for i in range(count):
            template = np.random.choice(fake_food_templates)
            
            # Add some food-specific variation
            if np.random.random() < 0.3:
                template += " " + np.random.choice([
                    "Definitely recommend for food lovers!", "Buy now for great taste!", "Perfect flavor!",
                    "Amazing quality food!", "Fast shipping fresh!", "Great value for taste!"
                ])
            
            fake_reviews.append({
                'Id': 100000 + i,
                'review_text': template,
                'rating': np.random.choice([1, 5]),  # Extreme ratings
                'helpful_votes': 0,
                'total_votes': 0,
                'rating_deviation': np.random.uniform(-2, 2),
                'user_id': f'fake_food_user_{i}',
                'product_id': f'fake_food_product_{i % 100}',
                'is_fake': 1
            })
        
        return pd.DataFrame(fake_reviews)
    
    def train_model(self, features_df: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Train the fake food review detection model
        """
        logger.info("TRAINING: Fake food review detection model")
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        logger.info("TRAINING: Running hyperparameter optimization for food reviews...")
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'best_params': grid_search.best_params_,
            'train_score': train_score,
            'test_score': test_score,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': feature_importance.head(10).to_dict('records')
        }
        
        logger.info(f"SUCCESS: Food review fake detection model trained successfully")
        logger.info(f"Train Score: {train_score:.3f}")
        logger.info(f"Test Score: {test_score:.3f}")
        logger.info(f"AUC Score: {auc_score:.3f}")
        logger.info(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def predict_fake_probability(self, review_text: str, user_data: Dict = None, 
                                review_data: Dict = None) -> FakeDetectionResult:
        """
        Predict if a food review is fake
        """
        start_time = datetime.now()
        
        if not self.model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Extract features
            features = self.extract_all_features(review_text, user_data, review_data)
            features_df = pd.DataFrame([features])
            
            # Ensure all training features are present
            for feature_name in self.feature_names:
                if feature_name not in features_df.columns:
                    features_df[feature_name] = 0
            
            # Reorder columns to match training data
            features_df = features_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            fake_probability = self.model.predict_proba(features_scaled)[0][1]
            confidence = max(fake_probability, 1 - fake_probability)
            
            # Determine risk level
            if fake_probability < 0.3:
                risk_level = 'LOW'
            elif fake_probability < 0.7:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            # Get feature contributions (simplified)
            feature_scores = dict(zip(self.feature_names, features_scaled[0]))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FakeDetectionResult(
                is_fake_probability=fake_probability,
                confidence=confidence,
                risk_level=risk_level,
                features_scores=feature_scores,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"ERROR: Food review prediction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FakeDetectionResult(
                is_fake_probability=0.5,
                confidence=0.5,
                risk_level='MEDIUM',
                features_scores={},
                processing_time=processing_time
            )
    
    def batch_detect_fake_reviews(self, review_ids: List[int] = None, limit: int = 500) -> List[Dict]:
        """
        Batch process food reviews for fake detection
        """
        if not self.model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"STARTING: Batch fake detection for food reviews (limit: {limit})")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        # Query food reviews
        if review_ids:
            placeholders = ','.join(['?' for _ in review_ids])
            query = f"SELECT Id, review_text, rating, helpful_votes, total_votes, rating_deviation FROM reviews WHERE Id IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=review_ids)
        else:
            query = "SELECT Id, review_text, rating, helpful_votes, total_votes, rating_deviation FROM reviews ORDER BY Id LIMIT ?"
            df = pd.read_sql_query(query, conn, params=[limit])
        
        conn.close()
        
        logger.info(f"PROCESSING: {len(df)} food reviews for fake detection...")
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                review_data = {
                    'rating': row.get('rating', 3.0),
                    'helpful_votes': row.get('helpful_votes', 0),
                    'total_votes': row.get('total_votes', 0),
                    'rating_deviation': row.get('rating_deviation', 0)
                }
                
                result = self.predict_fake_probability(
                    row['review_text'], 
                    review_data=review_data
                )
                
                result_data = {
                    'review_id': row['Id'],
                    'fake_probability': result.is_fake_probability,
                    'confidence': result.confidence,
                    'risk_level': result.risk_level,
                    'processing_time': result.processing_time,
                    'analyzed_at': datetime.now().isoformat()
                }
                
                results.append(result_data)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"PROGRESS: Processed {idx + 1}/{len(df)} food reviews")
                    
            except Exception as e:
                logger.error(f"ERROR: Failed to process food review {row['Id']}: {e}")
                continue
        
        logger.info(f"COMPLETE: Batch fake detection complete: {len(results)} food reviews")
        return results
    
    def save_detection_results(self, results: List[Dict]):
        """
        Save fake detection results to database
        """
        if not results:
            logger.warning("WARNING: No food fake detection results to save")
            return
        
        logger.info(f"SAVING: {len(results)} food fake detection results to database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create fake_detection table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fake_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id INTEGER REFERENCES reviews(Id),
                    fake_probability REAL,
                    confidence REAL,
                    risk_level TEXT,
                    processing_time REAL,
                    analyzed_at TEXT,
                    UNIQUE(review_id)
                )
            """)
            
            # Insert results
            for result in results:
                conn.execute("""
                    INSERT OR REPLACE INTO fake_detection 
                    (review_id, fake_probability, confidence, risk_level, processing_time, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result['review_id'],
                    result['fake_probability'],
                    result['confidence'],
                    result['risk_level'],
                    result['processing_time'],
                    result['analyzed_at']
                ))
            
            conn.commit()
            conn.close()
            
            logger.info("SUCCESS: Food fake detection results saved successfully")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save food fake detection results: {e}")
            raise
    
    def save_model(self, filepath: str = "models/food_fake_detector.pkl"):
        """
        Save the trained model to disk
        """
        if not self.model:
            raise ValueError("No model to save. Train model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'fake_patterns': self.fake_patterns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"SUCCESS: Food fake detection model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/food_fake_detector.pkl"):
        """
        Load a trained model from disk
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.fake_patterns = model_data['fake_patterns']
        
        logger.info(f"SUCCESS: Food fake detection model loaded from {filepath}")

def main():
    """
    Main function for testing fake food review detection
    """
    print("TESTING: Fake Review Detection System for Amazon Fine Food Reviews")
    print("=" * 50)
    
    # Initialize detector
    detector = FakeReviewDetector()
    
    # Prepare training data
    print("\nPREPARING: Training data for food reviews...")
    features_df, labels = detector.prepare_training_data(limit=1000)
    
    print(f"Dataset: {len(features_df)} samples, {len(features_df.columns)} features")
    print(f"Real food reviews: {sum(labels == 0)}, Fake food reviews: {sum(labels == 1)}")
    
    # Train model
    print("\nTRAINING: Model for food reviews...")
    results = detector.train_model(features_df, labels)
    
    print("\nMODEL RESULTS:")
    print(f"AUC Score: {results['auc_score']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std'] * 2:.3f})")
    
    print("\nTop Important Features for Food Reviews:")
    for feature in results['feature_importance'][:5]:
        print(f"  {feature['feature']}: {feature['importance']:.3f}")
    
    # Test single prediction
    print("\nTESTING: Single predictions on food reviews...")
    
    # Test authentic food review
    authentic_food_review = """
    I ordered this coffee last month and have been enjoying it daily. 
    The flavor is rich and bold, exactly what I was looking for. The beans 
    taste fresh and the aroma is wonderful. The packaging kept everything 
    secure during shipping. The only minor issue is that it's a bit pricey, 
    but overall I'm satisfied with the taste and would recommend it to other coffee lovers.
    """
    
    result_authentic = detector.predict_fake_probability(authentic_food_review)
    print(f"\nAuthentic Food Review Test:")
    print(f"  Fake Probability: {result_authentic.is_fake_probability:.3f}")
    print(f"  Risk Level: {result_authentic.risk_level}")
    print(f"  Confidence: {result_authentic.confidence:.3f}")
    
    # Test suspicious food review
    suspicious_food_review = """
    Great food! Amazing taste! Fast shipping! Highly recommend! 
    Perfect snack! Love this product! Worth the money! Excellent flavor!
    Best purchase ever! Five stars! Buy now! Amazing amazing amazing taste!
    Fresh fresh fresh! Delicious delicious! Perfect perfect perfect!
    """
    
    result_suspicious = detector.predict_fake_probability(suspicious_food_review)
    print(f"\nSuspicious Food Review Test:")
    print(f"  Fake Probability: {result_suspicious.is_fake_probability:.3f}")
    print(f"  Risk Level: {result_suspicious.risk_level}")
    print(f"  Confidence: {result_suspicious.confidence:.3f}")
    
    # Batch processing test
    print(f"\nTESTING: Batch processing (first 100 food reviews)...")
    batch_results = detector.batch_detect_fake_reviews(limit=100)
    
    if batch_results:
        print(f"SUCCESS: Processed {len(batch_results)} food reviews")
        
        # Save results
        detector.save_detection_results(batch_results)
        
        # Statistics
        fake_probs = [r['fake_probability'] for r in batch_results]
        risk_levels = [r['risk_level'] for r in batch_results]
        
        print(f"\nFAKE DETECTION STATISTICS FOR FOOD REVIEWS:")
        print(f"Average fake probability: {np.mean(fake_probs):.3f}")
        print(f"High risk reviews: {risk_levels.count('HIGH')}")
        print(f"Medium risk reviews: {risk_levels.count('MEDIUM')}")
        print(f"Low risk reviews: {risk_levels.count('LOW')}")
    
    # Save model
    print(f"\nSAVING: Model for food reviews...")
    detector.save_model()
    
    print(f"\nCOMPLETE: Fake Food Review Detection System test complete!")

if __name__ == "__main__":
    main()