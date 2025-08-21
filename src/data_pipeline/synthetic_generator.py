"""
Synthetic Data Generator for Product Review Intelligence System
Generates realistic fake reviews for ML model training
"""

import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from pathlib import Path
import json
import re
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticReviewGenerator:
    """
    Generate synthetic fake reviews with realistic patterns for ML training
    """
    
    def __init__(self):
        self.db_path = DATABASE_URL.replace('sqlite:///', '')
        
        # Fake review patterns and templates
        self.fake_patterns = {
            'generic_positive': [
                "Great food! Highly recommend!",
                "Amazing quality and fast shipping!",
                "Perfect! Exactly what I needed.",
                "Love it! Will buy again!",
                "Excellent value for money!",
                "Best purchase ever! Five stars!",
                "Outstanding quality and service!",
                "Perfect item, perfect seller!"
            ],
            'generic_negative': [
                "Terrible quality! Don't buy!",
                "Waste of money! Very disappointed.",
                "Poor quality and bad service.",
                "Not as described! Scam!",
                "Awful product! Returned immediately.",
                "Don't waste your time or money!",
                "Very poor quality for the price.",
                "Completely useless! Avoid!"
            ],
            'duplicate_content': [
                "This product changed my life completely",
                "Best investment I have ever made",
                "Cannot believe how amazing this is",
                "Everyone needs to buy this now",
                "Perfect for everyday use and more"
            ]
        }
        
        # Real review templates for variation
        self.realistic_templates = {
            'positive': [
                "I've been using this product for {duration} and I'm really impressed. The {feature} is great and {detail}. Would definitely recommend to {audience}.",
                "Initially I was skeptical, but after {duration} of use, I can say this is {opinion}. The {feature} is particularly good. {conclusion}.",
                "Bought this as a {purpose} and it exceeded my expectations. {detail} The only minor issue is {minor_issue}, but overall very satisfied.",
                "Great value for the price! {feature} works as expected and {detail}. Shipping was fast and packaging was secure."
            ],
            'negative': [
                "Unfortunately this didn't work out for me. The {feature} is {issue} and {detail}. Maybe it works for others but not my use case.",
                "Had high hopes but disappointed. {issue} is a major problem and {detail}. Considering returning it.",
                "The product itself is okay but {issue}. Also, {detail}. For the price, I expected better quality.",
                "Works but has several issues. {issue} and the {feature} is {problem}. Customer service was {service_quality}."
            ]
        }
        
        # Food-specific features and details for template filling
        self.features = ['taste', 'freshness', 'packaging', 'texture', 'flavor', 'ingredients']
        self.durations = ['2 weeks', '1 month', '3 months', '6 months', 'a year']
        self.audiences = ['anyone', 'food lovers', 'families', 'health conscious people', 'cooking enthusiasts']
        self.issues = ['stale', 'expired', 'poorly packaged', 'overpriced', 'artificial tasting']
        
        # Food product categories and IDs (based on actual Amazon Fine Food Reviews dataset)
        self.product_categories = {
            'snacks': ['B000E7L2R4', 'B00I5F7N9K', 'B0002IY5ZY', 'B000EMJHGY'],
            'beverages': ['B007JBXKFW', 'B003GTR8IO', 'B00C7J5084', 'B004U49QU2'],
            'condiments': ['B000E63LRO', 'B000LKTTTW', 'B000H25SOU', 'B004RF6TT0'],
            'baking': ['B00032CZUM', 'B000J3R7Y6', 'B000E8T5TO', 'B001E5E2QS'],
            'organic': ['B000GZS9XY', 'B004VLINS4', 'B000FPN8TK', 'B006N3IG0E'],
            'specialty': ['B0001XO398', 'B000I5F8NS', 'B004RF6TT0', 'B000Y7FYSU']
        }
        
        # User behavior patterns for fake accounts
        self.fake_user_patterns = {
            'burst_reviewer': {'reviews_per_day': (5, 15), 'active_days': (1, 3)},
            'copy_paster': {'duplicate_ratio': 0.8, 'minor_variations': True},
            'extreme_rater': {'rating_variance': 0.1, 'extreme_bias': True},
            'new_account': {'account_age_days': (0, 30), 'low_activity': True}
        }
    
    def generate_fake_reviews(self, num_reviews: int = 1000) -> pd.DataFrame:
        """
        Generate comprehensive fake review dataset
        """
        logger.info(f"Generating {num_reviews} fake reviews...")
        
        fake_reviews = []
        
        # Generate different types of fake reviews
        fake_types = {
            'generic_spam': int(num_reviews * 0.3),
            'duplicate_content': int(num_reviews * 0.25),
            'bot_generated': int(num_reviews * 0.2),
            'incentivized': int(num_reviews * 0.15),
            'competitor_attack': int(num_reviews * 0.1)
        }
        
        for fake_type, count in fake_types.items():
            logger.info(f"Generating {count} {fake_type} reviews...")
            
            if fake_type == 'generic_spam':
                reviews = self._generate_generic_spam(count)
            elif fake_type == 'duplicate_content':
                reviews = self._generate_duplicate_content(count)
            elif fake_type == 'bot_generated':
                reviews = self._generate_bot_reviews(count)
            elif fake_type == 'incentivized':
                reviews = self._generate_incentivized_reviews(count)
            elif fake_type == 'competitor_attack':
                reviews = self._generate_competitor_attacks(count)
            
            fake_reviews.extend(reviews)
        
        # Convert to DataFrame and add metadata
        df = pd.DataFrame(fake_reviews)
        df = self._add_fake_metadata(df)
        
        logger.info(f"Generated {len(df)} fake reviews")
        logger.info(f"Fake review types: {df['fake_type'].value_counts().to_dict()}")
        
        return df
    
    def _generate_generic_spam(self, count: int) -> List[Dict]:
        """Generate generic spam reviews"""
        reviews = []
        
        for i in range(count):
            is_positive = random.random() > 0.3  # 70% positive spam
            pattern_key = 'generic_positive' if is_positive else 'generic_negative'
            
            review = {
                'product_id': random.choice(self._get_all_products()),
                'user_id': f'SPAM_{i}_{random.randint(1000, 9999)}',
                'rating': 5 if is_positive else 1,
                'review_text': random.choice(self.fake_patterns[pattern_key]),
                'review_title': 'Great!' if is_positive else 'Terrible!',
                'date': self._random_date(),
                'helpful_votes': random.randint(0, 2),
                'total_votes': random.randint(0, 5),
                'fake_type': 'generic_spam',
                'is_fake': True
            }
            
            reviews.append(review)
        
        return reviews
    
    def _generate_duplicate_content(self, count: int) -> List[Dict]:
        """Generate reviews with duplicate/near-duplicate content"""
        reviews = []
        base_templates = self.fake_patterns['duplicate_content']
        
        for i in range(count):
            base_text = random.choice(base_templates)
            
            # Create variations
            if random.random() < 0.6:  # 60% exact duplicates
                review_text = base_text
            else:  # 40% minor variations
                review_text = self._create_minor_variation(base_text)
            
            review = {
                'product_id': random.choice(self._get_all_products()),
                'user_id': f'DUP_{i}_{random.randint(1000, 9999)}',
                'rating': random.choice([4, 5]) if 'amazing' in base_text.lower() else random.choice([1, 2]),
                'review_text': review_text,
                'review_title': base_text[:20] + '...',
                'date': self._random_date(),
                'helpful_votes': 0,
                'total_votes': random.randint(0, 3),
                'fake_type': 'duplicate_content',
                'is_fake': True
            }
            
            reviews.append(review)
        
        return reviews
    
    def _generate_bot_reviews(self, count: int) -> List[Dict]:
        """Generate bot-like reviews with unnatural patterns"""
        reviews = []
        
        for i in range(count):
            # Bot characteristics: unnatural language, repetitive structure
            sentiment = random.choice(['positive', 'negative'])
            template = random.choice(self.realistic_templates[sentiment])
            
            # Fill template with robotic choices for food products
            review_text = template.format(
                duration=random.choice(self.durations),
                feature=random.choice(self.features),
                detail='It meets nutritional specifications.',
                audience=random.choice(self.audiences),
                opinion='satisfactory for the specified dietary requirements',
                conclusion='Recommend based on nutritional analysis.',
                purpose='dietary supplement',
                minor_issue='minor flavor variations',
                issue=random.choice(self.issues),
                problem='suboptimal',
                service_quality='adequate'
            )
            
            review = {
                'product_id': random.choice(self._get_all_products()),
                'user_id': f'BOT_{i}_{random.randint(10000, 99999)}',
                'rating': 4 if sentiment == 'positive' else 2,
                'review_text': review_text,
                'review_title': 'Product Evaluation Report' if sentiment == 'positive' else 'Issues Identified',
                'date': self._random_date(),
                'helpful_votes': random.randint(0, 1),
                'total_votes': random.randint(1, 3),
                'fake_type': 'bot_generated',
                'is_fake': True
            }
            
            reviews.append(review)
        
        return reviews
    
    def _generate_incentivized_reviews(self, count: int) -> List[Dict]:
        """Generate incentivized/paid reviews"""
        reviews = []
        
        incentive_phrases = [
            "received this product for free in exchange for honest review",
            "got this at discount for review",
            "company provided this for testing",
            "free sample in exchange for feedback"
        ]
        
        for i in range(count):
            base_sentiment = random.choice(['positive', 'negative'])
            template = random.choice(self.realistic_templates[base_sentiment])
            
            review_text = template.format(
                duration=random.choice(self.durations),
                feature=random.choice(self.features),
                detail=random.choice(incentive_phrases),
                audience=random.choice(self.audiences),
                opinion='quite good',
                conclusion='Overall positive experience.',
                purpose='testing',
                minor_issue='flavor intensity',
                issue='occasionally inconsistent',
                problem='not ideal',
                service_quality='responsive'
            )
            
            review = {
                'product_id': random.choice(self._get_all_products()),
                'user_id': f'INC_{i}_{random.randint(1000, 9999)}',
                'rating': random.choice([4, 5]) if base_sentiment == 'positive' else random.choice([2, 3]),
                'review_text': review_text,
                'review_title': 'Honest Review' if base_sentiment == 'positive' else 'Mixed Experience',
                'date': self._random_date(),
                'helpful_votes': random.randint(1, 5),
                'total_votes': random.randint(2, 8),
                'fake_type': 'incentivized',
                'is_fake': True
            }
            
            reviews.append(review)
        
        return reviews
    
    def _generate_competitor_attacks(self, count: int) -> List[Dict]:
        """Generate competitor attack reviews"""
        reviews = []
        
        attack_phrases = [
            "much better food brands available",
            "competitor product X tastes better",
            "overpriced compared to similar foods",
            "misleading nutritional claims",
            "better options for the same price"
        ]
        
        for i in range(count):
            review_text = f"Disappointed with this purchase. {random.choice(attack_phrases)}. " + \
                         f"The {random.choice(self.features)} is {random.choice(self.issues)} and " + \
                         f"I would not recommend this to anyone."
            
            review = {
                'product_id': random.choice(self._get_all_products()),
                'user_id': f'COMP_{i}_{random.randint(1000, 9999)}',
                'rating': random.choice([1, 2]),
                'review_text': review_text,
                'review_title': 'Better alternatives exist',
                'date': self._random_date(),
                'helpful_votes': random.randint(0, 2),
                'total_votes': random.randint(1, 4),
                'fake_type': 'competitor_attack',
                'is_fake': True
            }
            
            reviews.append(review)
        
        return reviews
    
    def _create_minor_variation(self, text: str) -> str:
        """Create minor variations of text"""
        variations = [
            lambda t: t.replace('.', '!'),
            lambda t: t.replace(' ', '  '),
            lambda t: t + ' Really!',
            lambda t: t.replace('this', 'this item'),
            lambda t: t.replace('great', 'amazing'),
            lambda t: t.lower(),
            lambda t: t.upper()
        ]
        
        variation_func = random.choice(variations)
        return variation_func(text)
    
    def _random_date(self) -> datetime:
        """Generate random date within last 2 years"""
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        return start_date + timedelta(days=random_days)
    
    def _get_all_products(self) -> List[str]:
        """Get all available product IDs"""
        all_products = []
        for category_products in self.product_categories.values():
            all_products.extend(category_products)
        return all_products
    
    def _add_fake_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata and features to identify fake reviews"""
        
        # Text-based features
        df['review_length'] = df['review_text'].str.len()
        df['word_count'] = df['review_text'].str.split().str.len()
        df['exclamation_count'] = df['review_text'].str.count('!')
        df['capital_ratio'] = df['review_text'].apply(self._calculate_capital_ratio)
        df['repeated_words'] = df['review_text'].apply(self._count_repeated_words)
        
        # Rating-based features
        df['extreme_rating'] = df['rating'].isin([1, 5])
        
        # Temporal features
        df['review_hour'] = pd.to_datetime(df['date']).dt.hour
        df['is_weekend'] = pd.to_datetime(df['date']).dt.dayofweek.isin([5, 6])
        
        # User-based features (simplified)
        user_review_counts = df['user_id'].value_counts()
        df['user_review_count'] = df['user_id'].map(user_review_counts)
        
        # Add source
        df['source'] = 'synthetic_fake'
        df['generation_timestamp'] = datetime.now()
        
        return df
    
    def _calculate_capital_ratio(self, text: str) -> float:
        """Calculate ratio of capital letters"""
        if not text:
            return 0
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0
        capitals = [c for c in letters if c.isupper()]
        return len(capitals) / len(letters)
    
    def _count_repeated_words(self, text: str) -> int:
        """Count repeated words in text"""
        if not text:
            return 0
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return sum(1 for count in word_counts.values() if count > 1)
    
    def generate_mixed_dataset(self, fake_reviews: int = 1000, real_reviews: int = 2000) -> pd.DataFrame:
        """
        Generate mixed dataset with both fake and real-looking reviews
        """
        logger.info(f"Generating mixed dataset: {fake_reviews} fake + {real_reviews} real-looking")
        
        # Generate fake reviews
        fake_df = self.generate_fake_reviews(fake_reviews)
        
        # Generate real-looking reviews
        real_df = self._generate_realistic_reviews(real_reviews)
        
        # Combine and shuffle
        mixed_df = pd.concat([fake_df, real_df], ignore_index=True)
        mixed_df = mixed_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Mixed dataset created: {len(mixed_df)} total reviews")
        logger.info(f"Fake ratio: {fake_df['is_fake'].sum() / len(mixed_df):.2%}")
        
        return mixed_df
    
    def _generate_realistic_reviews(self, count: int) -> pd.DataFrame:
        """Generate realistic-looking reviews for contrast"""
        reviews = []
        
        realistic_texts = [
            "I ordered this food item last month and have been enjoying it regularly. The taste is excellent and it's exactly what I expected. The packaging was secure and everything arrived fresh. The expiration date gives plenty of time to use it. Overall, I'm satisfied with this purchase and would consider buying from this brand again.",
            "After reading several reviews, I decided to try this product. It arrived quickly and was well-packaged. The flavor is rich and it fits perfectly into my meal planning. The taste has been consistent over the past few weeks. My only minor complaint is that it could be slightly less salty during preparation.",
            "This food product has both pros and cons. On the positive side, it's fresh and does what it claims to do. The ingredients are clearly listed and I was able to prepare it without any issues. However, for the price point, I expected some additional flavor complexity that is missing. Still, it's a decent value overall.",
            "I've been eating this for about three months now and wanted to share my experience. Initially, I had some concerns about the taste, but it turned out to be perfect for my dietary needs. The freshness has been reliable and I haven't encountered any major issues. The customer support was helpful when I had a question about ingredients."
        ]
        
        for i in range(count):
            review = {
                'product_id': random.choice(self._get_all_products()),
                'user_id': f'REAL_{i}_{random.randint(10000, 99999)}',
                'rating': random.choice([2, 3, 4, 4, 4, 5]),  # Realistic distribution
                'review_text': random.choice(realistic_texts),
                'review_title': f'Good product overall' if random.random() > 0.5 else f'Mixed experience',
                'date': self._random_date(),
                'helpful_votes': random.randint(0, 10),
                'total_votes': random.randint(0, 15),
                'fake_type': 'realistic',
                'is_fake': False
            }
            
            reviews.append(review)
        
        return pd.DataFrame(reviews)
    
    def save_synthetic_reviews(self, df: pd.DataFrame, table_name: str = 'synthetic_reviews') -> bool:
        """Save synthetic reviews to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(df)} synthetic reviews to {table_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to save synthetic reviews: {e}")
            return False
    
    def export_training_data(self, df: pd.DataFrame, output_dir: str = 'data') -> Dict[str, str]:
        """Export training data for ML models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        files_created = {}
        
        try:
            # Full dataset
            full_path = output_path / 'synthetic_reviews_full.csv'
            df.to_csv(full_path, index=False)
            files_created['full_dataset'] = str(full_path)
            
            # Features only (for ML training)
            feature_columns = [
                'review_length', 'word_count', 'exclamation_count', 
                'capital_ratio', 'repeated_words', 'extreme_rating',
                'user_review_count', 'is_fake'
            ]
            
            features_df = df[feature_columns]
            features_path = output_path / 'fake_detection_features.csv'
            features_df.to_csv(features_path, index=False)
            files_created['features'] = str(features_path)
            
            # Summary statistics
            summary = {
                'total_reviews': len(df),
                'fake_reviews': df['is_fake'].sum(),
                'fake_ratio': df['is_fake'].mean(),
                'fake_types': df['fake_type'].value_counts().to_dict(),
                'rating_distribution': df['rating'].value_counts().to_dict(),
                'avg_review_length': df['review_length'].mean(),
                'generation_date': datetime.now().isoformat()
            }
            
            summary_path = output_path / 'synthetic_data_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            files_created['summary'] = str(summary_path)
            
            logger.info(f"Exported training data to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
        
        return files_created


def main():
    """
    Main function for testing synthetic generator
    """
    generator = SyntheticReviewGenerator()
    
    logger.info("🤖 Generating synthetic fake reviews for ML training...")
    
    try:
        # Generate mixed dataset
        mixed_df = generator.generate_mixed_dataset(fake_reviews=500, real_reviews=500)
        
        # Display statistics
        print("\n=== Synthetic Data Statistics ===")
        print(f"Total reviews: {len(mixed_df)}")
        print(f"Fake reviews: {mixed_df['is_fake'].sum()}")
        print(f"Fake ratio: {mixed_df['is_fake'].mean():.2%}")
        print(f"Fake types: {mixed_df['fake_type'].value_counts().to_dict()}")
        print(f"Rating distribution: {mixed_df['rating'].value_counts().to_dict()}")
        
        # Save to database
        success = generator.save_synthetic_reviews(mixed_df)
        
        if success:
            print("✅ Synthetic reviews saved to database")
        
        # Export training data
        files_created = generator.export_training_data(mixed_df)
        print(f"📁 Training data exported: {list(files_created.keys())}")
        
        print("\n✅ Synthetic data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Synthetic data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()