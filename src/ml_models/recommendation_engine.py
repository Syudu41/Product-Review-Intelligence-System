"""
Product Recommendation Engine using Collaborative Filtering
Implements user-based and item-based recommendations with content filtering
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/recommendations.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RecommendationResult:
    """Result of product recommendation"""
    product_id: str
    product_name: str
    predicted_rating: float
    confidence: float
    recommendation_score: float
    reason: str
    similar_products: List[str]

class RecommendationEngine:
    """
    Advanced recommendation system using multiple techniques:
    1. Collaborative Filtering (User-based, Item-based)
    2. Content-based Filtering
    3. Matrix Factorization (NMF)
    4. Hybrid approach combining all methods
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.user_item_matrix = None
        self.item_features_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.nmf_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        
        # Load data and build matrices
        self._load_data()
        self._build_user_item_matrix()
        self._build_item_features()
        self._build_similarity_matrices()
        self._train_matrix_factorization()
        
        logger.info("SUCCESS: RecommendationEngine initialized successfully")
    
    def _load_data(self):
        """Load reviews and product data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load reviews with user and product information
            reviews_query = """
                SELECT r.Id, r.user_id, r.product_id, r.rating, r.review_text,
                       r.ProfileName, r.helpful_votes, r.total_votes
                FROM reviews r
                WHERE r.rating IS NOT NULL AND r.product_id IS NOT NULL
            """
            self.reviews_df = pd.read_sql_query(reviews_query, conn)
            
            # Load products (create product summaries if products table doesn't exist)
            try:
                products_query = "SELECT * FROM products"
                self.products_df = pd.read_sql_query(products_query, conn)
                logger.info(f"SUCCESS: Loaded {len(self.products_df)} products from products table")
            except:
                # Create products summary from reviews
                self.products_df = self.reviews_df.groupby('product_id').agg({
                    'rating': ['mean', 'count'],
                    'review_text': lambda x: ' '.join(str(text) for text in x if pd.notna(text))
                }).reset_index()
                
                # Flatten column names
                self.products_df.columns = ['product_id', 'avg_rating', 'review_count', 'combined_text']
                self.products_df['product_name'] = self.products_df['product_id'].apply(
                    lambda x: f"Product {x}"
                )
                logger.info(f"SUCCESS: Created {len(self.products_df)} product summaries from reviews")
            
            conn.close()
            
            logger.info(f"SUCCESS: Loaded {len(self.reviews_df)} reviews")
            logger.info(f"Users: {self.reviews_df['user_id'].nunique()}")
            logger.info(f"Products: {self.reviews_df['product_id'].nunique()}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to load data: {e}")
            raise
    
    def _build_user_item_matrix(self):
        """Build user-item rating matrix"""
        try:
            # Create user-item matrix
            self.user_item_matrix = self.reviews_df.pivot_table(
                index='user_id',
                columns='product_id',
                values='rating',
                fill_value=0
            )
            
            # Convert to sparse matrix for efficiency
            self.user_item_sparse = csr_matrix(self.user_item_matrix.values)
            
            logger.info(f"SUCCESS: Built user-item matrix: {self.user_item_matrix.shape}")
            logger.info(f"Sparsity: {(1 - np.count_nonzero(self.user_item_matrix.values) / self.user_item_matrix.size) * 100:.2f}%")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to build user-item matrix: {e}")
            raise
    
    def _build_item_features(self):
        """Build item feature matrix using content information"""
        try:
            # Prepare text data for TF-IDF
            if 'combined_text' in self.products_df.columns:
                product_texts = self.products_df['combined_text'].fillna('')
            elif 'review_text' in self.products_df.columns:
                product_texts = self.products_df['review_text'].fillna('')
            else:
                # Create basic text from product names
                product_texts = self.products_df['product_name'].fillna('').astype(str)
            
            # Create TF-IDF features
            tfidf_features = self.tfidf_vectorizer.fit_transform(product_texts)
            
            # Create additional features
            numerical_features = []
            
            # Rating-based features
            if 'avg_rating' in self.products_df.columns:
                numerical_features.append(self.products_df['avg_rating'].fillna(3.0))
            
            if 'review_count' in self.products_df.columns:
                numerical_features.append(np.log1p(self.products_df['review_count'].fillna(1)))
            
            # Combine features
            if numerical_features:
                numerical_array = np.column_stack(numerical_features)
                numerical_scaled = self.scaler.fit_transform(numerical_array)
                
                # Combine TF-IDF and numerical features
                from scipy.sparse import hstack
                self.item_features_matrix = hstack([tfidf_features, csr_matrix(numerical_scaled)])
            else:
                self.item_features_matrix = tfidf_features
            
            logger.info(f"SUCCESS: Built item features matrix: {self.item_features_matrix.shape}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to build item features: {e}")
            # Fallback to simple features
            self.item_features_matrix = csr_matrix((len(self.products_df), 1))
    
    def _build_similarity_matrices(self):
        """Build user and item similarity matrices"""
        try:
            # User similarity (based on rating patterns)
            logger.info("COMPUTING: User similarity matrix...")
            if self.user_item_matrix.shape[0] < 2000:  # Only for reasonable sizes
                user_similarity = cosine_similarity(self.user_item_matrix.values)
                self.user_similarity_matrix = pd.DataFrame(
                    user_similarity,
                    index=self.user_item_matrix.index,
                    columns=self.user_item_matrix.index
                )
                logger.info(f"SUCCESS: User similarity matrix: {self.user_similarity_matrix.shape}")
            else:
                logger.info("INFO: Skipping user similarity matrix (too many users)")
                self.user_similarity_matrix = None
            
            # Item similarity (based on content features)
            logger.info("COMPUTING: Item similarity matrix...")
            if self.item_features_matrix.shape[0] < 2000:  # Only for reasonable sizes
                item_similarity = cosine_similarity(self.item_features_matrix)
                self.item_similarity_matrix = pd.DataFrame(
                    item_similarity,
                    index=self.products_df['product_id'],
                    columns=self.products_df['product_id']
                )
                logger.info(f"SUCCESS: Item similarity matrix: {self.item_similarity_matrix.shape}")
            else:
                logger.info("INFO: Skipping item similarity matrix (too many items)")
                self.item_similarity_matrix = None
                
        except Exception as e:
            logger.error(f"ERROR: Failed to build similarity matrices: {e}")
            self.user_similarity_matrix = None
            self.item_similarity_matrix = None
    
    def _train_matrix_factorization(self):
        """Train NMF for matrix factorization recommendations"""
        try:
            logger.info("TRAINING: Matrix factorization model...")
            
            # Use NMF for non-negative matrix factorization
            n_components = min(50, min(self.user_item_matrix.shape) - 1)
            
            self.nmf_model = NMF(
                n_components=n_components,
                init='random',
                random_state=42,
                max_iter=200,
                alpha_W=0.1,
                alpha_H=0.1,
                l1_ratio=0.1
            )
            
            # Fit model
            self.user_features = self.nmf_model.fit_transform(self.user_item_matrix.values)
            self.item_features = self.nmf_model.components_.T
            
            # Reconstruct matrix for predictions
            self.predicted_ratings = np.dot(self.user_features, self.item_features.T)
            
            logger.info(f"SUCCESS: NMF model trained with {n_components} components")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to train matrix factorization: {e}")
            self.nmf_model = None
    
    def get_user_based_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on similar users"""
        try:
            if self.user_similarity_matrix is None or user_id not in self.user_similarity_matrix.index:
                return []
            
            # Find similar users
            user_similarities = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False)
            similar_users = user_similarities.head(20).index[1:]  # Exclude self
            
            # Get items rated by similar users but not by target user
            user_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
            
            recommendations = {}
            
            for similar_user in similar_users:
                similarity_score = user_similarities[similar_user]
                if similarity_score < 0.1:  # Skip users with low similarity
                    continue
                
                similar_user_items = self.user_item_matrix.loc[similar_user]
                for item, rating in similar_user_items.items():
                    if rating > 0 and item not in user_items:
                        if item not in recommendations:
                            recommendations[item] = {'weighted_sum': 0, 'similarity_sum': 0}
                        recommendations[item]['weighted_sum'] += rating * similarity_score
                        recommendations[item]['similarity_sum'] += similarity_score
            
            # Calculate predicted ratings
            results = []
            for item, data in recommendations.items():
                if data['similarity_sum'] > 0:
                    predicted_rating = data['weighted_sum'] / data['similarity_sum']
                    confidence = min(data['similarity_sum'], 1.0)
                    
                    results.append({
                        'product_id': item,
                        'predicted_rating': predicted_rating,
                        'confidence': confidence,
                        'method': 'user_based'
                    })
            
            return sorted(results, key=lambda x: x['predicted_rating'], reverse=True)[:n_recommendations]
            
        except Exception as e:
            logger.error(f"ERROR: User-based recommendations failed: {e}")
            return []
    
    def get_item_based_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on item similarity"""
        try:
            if self.item_similarity_matrix is None or user_id not in self.user_item_matrix.index:
                return []
            
            # Get items rated by user
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0]
            
            if len(rated_items) == 0:
                return []
            
            recommendations = {}
            
            for item, rating in rated_items.items():
                if item in self.item_similarity_matrix.index:
                    # Find similar items
                    similar_items = self.item_similarity_matrix.loc[item].sort_values(ascending=False)
                    
                    for similar_item, similarity in similar_items.head(20).items():
                        if similar_item != item and similarity > 0.1 and user_ratings[similar_item] == 0:
                            if similar_item not in recommendations:
                                recommendations[similar_item] = {'weighted_sum': 0, 'similarity_sum': 0}
                            recommendations[similar_item]['weighted_sum'] += rating * similarity
                            recommendations[similar_item]['similarity_sum'] += similarity
            
            # Calculate predicted ratings
            results = []
            for item, data in recommendations.items():
                if data['similarity_sum'] > 0:
                    predicted_rating = data['weighted_sum'] / data['similarity_sum']
                    confidence = min(data['similarity_sum'], 1.0)
                    
                    results.append({
                        'product_id': item,
                        'predicted_rating': predicted_rating,
                        'confidence': confidence,
                        'method': 'item_based'
                    })
            
            return sorted(results, key=lambda x: x['predicted_rating'], reverse=True)[:n_recommendations]
            
        except Exception as e:
            logger.error(f"ERROR: Item-based recommendations failed: {e}")
            return []
    
    def get_matrix_factorization_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using matrix factorization"""
        try:
            if self.nmf_model is None or user_id not in self.user_item_matrix.index:
                return []
            
            user_idx = list(self.user_item_matrix.index).index(user_id)
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # Get predicted ratings for all items
            predicted_ratings = self.predicted_ratings[user_idx]
            
            # Filter out already rated items
            recommendations = []
            for item_idx, (item_id, rating) in enumerate(user_ratings.items()):
                if rating == 0:  # User hasn't rated this item
                    predicted_rating = predicted_ratings[item_idx]
                    if predicted_rating > 0:
                        # Calculate confidence based on how well we can reconstruct known ratings
                        confidence = min(predicted_rating / 5.0, 1.0)
                        
                        recommendations.append({
                            'product_id': item_id,
                            'predicted_rating': predicted_rating,
                            'confidence': confidence,
                            'method': 'matrix_factorization'
                        })
            
            return sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)[:n_recommendations]
            
        except Exception as e:
            logger.error(f"ERROR: Matrix factorization recommendations failed: {e}")
            return []
    
    def get_content_based_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on content similarity"""
        try:
            if user_id not in self.user_item_matrix.index:
                return []
            
            # Get user's rating history
            user_ratings = self.user_item_matrix.loc[user_id]
            liked_items = user_ratings[user_ratings >= 4].index.tolist()
            
            if not liked_items:
                # Fallback to highest rated items
                return self.get_popular_recommendations(n_recommendations)
            
            # Find products similar to liked items
            recommendations = {}
            
            for liked_item in liked_items:
                if liked_item in self.products_df['product_id'].values:
                    item_idx = self.products_df[self.products_df['product_id'] == liked_item].index[0]
                    
                    # Calculate similarity with all other items
                    if self.item_similarity_matrix is not None and liked_item in self.item_similarity_matrix.index:
                        similarities = self.item_similarity_matrix.loc[liked_item]
                        
                        for similar_item, similarity in similarities.items():
                            if (similar_item != liked_item and 
                                similarity > 0.1 and 
                                user_ratings[similar_item] == 0):
                                
                                if similar_item not in recommendations:
                                    recommendations[similar_item] = 0
                                recommendations[similar_item] += similarity
            
            # Convert to list format
            results = []
            for item_id, score in recommendations.items():
                # Get product info
                product_info = self.products_df[self.products_df['product_id'] == item_id]
                if not product_info.empty:
                    avg_rating = product_info['avg_rating'].iloc[0] if 'avg_rating' in product_info.columns else 3.5
                    
                    results.append({
                        'product_id': item_id,
                        'predicted_rating': min(avg_rating, 5.0),
                        'confidence': min(score, 1.0),
                        'method': 'content_based'
                    })
            
            return sorted(results, key=lambda x: x['confidence'], reverse=True)[:n_recommendations]
            
        except Exception as e:
            logger.error(f"ERROR: Content-based recommendations failed: {e}")
            return []
    
    def get_popular_recommendations(self, n_recommendations: int = 10) -> List[Dict]:
        """Get popular item recommendations as fallback"""
        try:
            # Calculate popularity score
            item_stats = self.reviews_df.groupby('product_id').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            
            item_stats.columns = ['product_id', 'avg_rating', 'rating_count']
            
            # Calculate popularity score (weighted rating)
            min_reviews = 5
            item_stats['popularity_score'] = (
                (item_stats['rating_count'] / (item_stats['rating_count'] + min_reviews)) * item_stats['avg_rating'] +
                (min_reviews / (item_stats['rating_count'] + min_reviews)) * 3.5
            )
            
            # Filter items with sufficient reviews
            popular_items = item_stats[item_stats['rating_count'] >= min_reviews].sort_values(
                'popularity_score', ascending=False
            ).head(n_recommendations)
            
            results = []
            for _, row in popular_items.iterrows():
                results.append({
                    'product_id': row['product_id'],
                    'predicted_rating': row['avg_rating'],
                    'confidence': min(row['rating_count'] / 100, 1.0),
                    'method': 'popular'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"ERROR: Popular recommendations failed: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[RecommendationResult]:
        """Get hybrid recommendations combining all methods"""
        try:
            logger.info(f"GENERATING: Hybrid recommendations for user {user_id}")
            
            # Get recommendations from all methods
            user_based = self.get_user_based_recommendations(user_id, n_recommendations * 2)
            item_based = self.get_item_based_recommendations(user_id, n_recommendations * 2)
            matrix_fact = self.get_matrix_factorization_recommendations(user_id, n_recommendations * 2)
            content_based = self.get_content_based_recommendations(user_id, n_recommendations * 2)
            popular = self.get_popular_recommendations(n_recommendations)
            
            # Combine recommendations with weights
            combined_scores = {}
            method_weights = {
                'user_based': 0.3,
                'item_based': 0.3,
                'matrix_factorization': 0.25,
                'content_based': 0.1,
                'popular': 0.05
            }
            
            for recommendations, weight in [
                (user_based, method_weights['user_based']),
                (item_based, method_weights['item_based']),
                (matrix_fact, method_weights['matrix_factorization']),
                (content_based, method_weights['content_based']),
                (popular, method_weights['popular'])
            ]:
                for rec in recommendations:
                    product_id = rec['product_id']
                    if product_id not in combined_scores:
                        combined_scores[product_id] = {
                            'weighted_rating': 0,
                            'weight_sum': 0,
                            'methods': [],
                            'confidences': []
                        }
                    
                    combined_scores[product_id]['weighted_rating'] += rec['predicted_rating'] * weight
                    combined_scores[product_id]['weight_sum'] += weight
                    combined_scores[product_id]['methods'].append(rec['method'])
                    combined_scores[product_id]['confidences'].append(rec['confidence'])
            
            # Calculate final scores and create results
            final_recommendations = []
            
            for product_id, data in combined_scores.items():
                if data['weight_sum'] > 0:
                    final_rating = data['weighted_rating'] / data['weight_sum']
                    final_confidence = np.mean(data['confidences'])
                    recommendation_score = final_rating * final_confidence
                    
                    # Get product information
                    product_info = self.products_df[self.products_df['product_id'] == product_id]
                    if not product_info.empty:
                        product_name = product_info['product_name'].iloc[0] if 'product_name' in product_info.columns else f"Product {product_id}"
                    else:
                        product_name = f"Product {product_id}"
                    
                    # Create reason string
                    methods = list(set(data['methods']))
                    reason = f"Recommended by {', '.join(methods)}"
                    
                    # Find similar products
                    similar_products = []
                    if self.item_similarity_matrix is not None and product_id in self.item_similarity_matrix.index:
                        similar = self.item_similarity_matrix.loc[product_id].sort_values(ascending=False).head(4)
                        similar_products = [pid for pid in similar.index if pid != product_id][:3]
                    
                    final_recommendations.append(RecommendationResult(
                        product_id=product_id,
                        product_name=product_name,
                        predicted_rating=final_rating,
                        confidence=final_confidence,
                        recommendation_score=recommendation_score,
                        reason=reason,
                        similar_products=similar_products
                    ))
            
            # Sort by recommendation score and return top N
            final_recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
            
            logger.info(f"SUCCESS: Generated {len(final_recommendations)} hybrid recommendations")
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"ERROR: Hybrid recommendations failed: {e}")
            return []
    
    def save_user_recommendations(self, user_id: str, recommendations: List[RecommendationResult]):
        """Save recommendations to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Drop and recreate user_recommendations table to ensure correct schema
            conn.execute("DROP TABLE IF EXISTS user_recommendations")
            conn.execute("""
                CREATE TABLE user_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    product_id TEXT,
                    predicted_rating REAL,
                    confidence REAL,
                    recommendation_score REAL,
                    reason TEXT,
                    similar_products TEXT,
                    generated_at TEXT,
                    UNIQUE(user_id, product_id)
                )
            """)
            
            # Clear existing recommendations for this user
            conn.execute("DELETE FROM user_recommendations WHERE user_id = ?", (user_id,))
            
            # Insert new recommendations
            for rec in recommendations:
                conn.execute("""
                    INSERT INTO user_recommendations 
                    (user_id, product_id, predicted_rating, confidence, recommendation_score, 
                     reason, similar_products, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    rec.product_id,
                    rec.predicted_rating,
                    rec.confidence,
                    rec.recommendation_score,
                    rec.reason,
                    json.dumps(rec.similar_products),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"SUCCESS: Saved {len(recommendations)} recommendations for user {user_id}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save recommendations: {e}")
            raise
    
    def get_recommendation_stats(self) -> Dict:
        """Get recommendation system statistics"""
        try:
            stats = {
                'total_users': len(self.user_item_matrix.index),
                'total_products': len(self.user_item_matrix.columns),
                'total_ratings': np.count_nonzero(self.user_item_matrix.values),
                'matrix_sparsity': (1 - np.count_nonzero(self.user_item_matrix.values) / self.user_item_matrix.size) * 100,
                'avg_user_ratings': np.mean(np.count_nonzero(self.user_item_matrix.values, axis=1)),
                'avg_product_ratings': np.mean(np.count_nonzero(self.user_item_matrix.values, axis=0)),
                'has_user_similarity': self.user_similarity_matrix is not None,
                'has_item_similarity': self.item_similarity_matrix is not None,
                'has_matrix_factorization': self.nmf_model is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get recommendation stats: {e}")
            return {}

def main():
    """
    Main function for testing recommendation system
    """
    print("TESTING: Recommendation Engine System")
    print("=" * 50)
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine()
    
    # Get system statistics
    stats = rec_engine.get_recommendation_stats()
    print(f"\nSYSTEM STATISTICS:")
    print(f"Total Users: {stats['total_users']}")
    print(f"Total Products: {stats['total_products']}")
    print(f"Total Ratings: {stats['total_ratings']}")
    print(f"Matrix Sparsity: {stats['matrix_sparsity']:.2f}%")
    print(f"Avg User Ratings: {stats['avg_user_ratings']:.1f}")
    print(f"User Similarity Available: {stats['has_user_similarity']}")
    print(f"Item Similarity Available: {stats['has_item_similarity']}")
    print(f"Matrix Factorization Available: {stats['has_matrix_factorization']}")
    
    # Test recommendations for a sample user
    sample_users = list(rec_engine.user_item_matrix.index)[:5]
    
    for user_id in sample_users:
        print(f"\nTESTING: Recommendations for user {user_id}")
        
        # Get user's rating history
        user_ratings = rec_engine.user_item_matrix.loc[user_id]
        rated_products = user_ratings[user_ratings > 0]
        print(f"User has rated {len(rated_products)} products (avg: {rated_products.mean():.2f})")
        
        # Generate hybrid recommendations
        recommendations = rec_engine.get_hybrid_recommendations(user_id, n_recommendations=5)
        
        if recommendations:
            print(f"TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec.product_name}")
                print(f"     Predicted Rating: {rec.predicted_rating:.2f}")
                print(f"     Confidence: {rec.confidence:.2f}")
                print(f"     Reason: {rec.reason}")
                if rec.similar_products:
                    print(f"     Similar: {', '.join(rec.similar_products[:2])}")
                print()
            
            # Save recommendations
            rec_engine.save_user_recommendations(user_id, recommendations)
            print(f"SUCCESS: Recommendations saved for user {user_id}")
        else:
            print(f"WARNING: No recommendations generated for user {user_id}")
        
        print("-" * 40)
    
    print(f"\nCOMPLETE: Recommendation Engine test complete!")

if __name__ == "__main__":
    main()