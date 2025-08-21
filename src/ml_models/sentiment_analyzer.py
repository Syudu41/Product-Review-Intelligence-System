"""
Sentiment Analysis System using Hugging Face Transformers and OpenAI API
Handles overall sentiment + aspect-based sentiment analysis
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# ML and NLP imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("⚠️  Installing transformers...")
    os.system("pip install transformers torch")
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

try:
    import openai
except ImportError:
    print("⚠️  Installing openai...")
    os.system("pip install openai")
    import openai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging (without emojis for Windows compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentiment_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    overall_sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL
    confidence: float
    score: float  # 1-5 scale
    aspects: Dict[str, Dict[str, float]]  # aspect -> {sentiment, confidence}
    processing_time: float

class SentimentAnalyzer:
    """
    Advanced sentiment analysis system combining Hugging Face transformers
    with OpenAI API for aspect-based sentiment analysis
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.openai_client = None
        self.hf_sentiment_pipeline = None
        self.aspects = ["price", "quality", "shipping", "service", "packaging", "value"]
        
        # Initialize OpenAI client if API key is available
        self._setup_openai()
        
        # Initialize Hugging Face pipeline
        self._setup_huggingface()
        
        logger.info("SUCCESS: SentimentAnalyzer initialized successfully")
    
    def _setup_openai(self):
        """Setup OpenAI client with API key validation"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                # Use new OpenAI client syntax (v1.0+)
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
                
                # Test API connection
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                logger.info("SUCCESS: OpenAI API connected successfully")
            else:
                logger.warning("WARNING: OpenAI API key not found. Aspect analysis will use rule-based approach.")
                
        except Exception as e:
            logger.error(f"ERROR: OpenAI setup failed: {e}")
            self.openai_client = None
    
    def _setup_huggingface(self):
        """Setup Hugging Face sentiment analysis pipeline"""
        try:
            # Use RoBERTa model fine-tuned for sentiment analysis
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            logger.info(f"LOADING: Hugging Face model: {model_name}")
            
            # Initialize pipeline with error handling
            self.hf_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                truncation=True,
                max_length=512
            )
            
            logger.info("SUCCESS: Hugging Face pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"ERROR: Hugging Face setup failed: {e}")
            # Fallback to default sentiment pipeline
            try:
                self.hf_sentiment_pipeline = pipeline("sentiment-analysis")
                logger.info("SUCCESS: Fallback sentiment pipeline loaded")
            except Exception as e2:
                logger.error(f"ERROR: Fallback pipeline also failed: {e2}")
                raise Exception("Could not initialize any sentiment analysis pipeline")
    
    def analyze_sentiment_hf(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using Hugging Face transformers
        Returns: (sentiment_label, confidence_score)
        """
        if not self.hf_sentiment_pipeline:
            raise ValueError("Hugging Face pipeline not initialized")
        
        try:
            # Clean and truncate text
            text = str(text)[:512]  # Limit to model's max length
            
            # Get prediction
            result = self.hf_sentiment_pipeline(text)[0]
            
            # Normalize labels (different models use different labels)
            label = result['label'].upper()
            confidence = result['score']
            
            # Map labels to standard format
            if label in ['LABEL_2', 'POSITIVE', 'POS']:
                sentiment = 'POSITIVE'
            elif label in ['LABEL_0', 'NEGATIVE', 'NEG']:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"ERROR: HF sentiment analysis failed: {e}")
            return 'NEUTRAL', 0.5
    
    def extract_aspects_openai(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Extract aspect-based sentiment using OpenAI API
        Returns: {aspect: {sentiment: score, confidence: float}}
        """
        if not self.openai_client:
            return self._extract_aspects_rule_based(text)
        
        try:
            prompt = f"""
            Analyze this product review for sentiment about specific aspects.
            
            Review: "{text}"
            
            For each aspect mentioned, rate the sentiment from 1-5 (1=very negative, 3=neutral, 5=very positive).
            Only include aspects that are clearly mentioned or implied.
            
            Aspects to consider: price, quality, shipping, service, packaging, value
            
            Return JSON format:
            {{
                "price": {{"sentiment": 4, "confidence": 0.8}},
                "quality": {{"sentiment": 5, "confidence": 0.9}}
            }}
            
            If an aspect isn't mentioned, don't include it.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                aspects_data = json.loads(json_str)
                return aspects_data
            else:
                logger.warning("Could not parse JSON from OpenAI response")
                return self._extract_aspects_rule_based(text)
                
        except Exception as e:
            logger.error(f"ERROR: OpenAI aspect extraction failed: {e}")
            return self._extract_aspects_rule_based(text)
    
    def _extract_aspects_rule_based(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Fallback rule-based aspect extraction
        """
        text_lower = text.lower()
        aspects_found = {}
        
        # Keywords for each aspect
        aspect_keywords = {
            "price": ["price", "cost", "expensive", "cheap", "value", "money", "dollar", "affordable"],
            "quality": ["quality", "build", "material", "durable", "sturdy", "flimsy", "solid"],
            "shipping": ["shipping", "delivery", "arrived", "fast", "slow", "quick", "package"],
            "service": ["service", "support", "help", "staff", "customer", "response"],
            "packaging": ["packaging", "box", "wrapped", "protected", "damaged", "secure"],
            "value": ["worth", "value", "recommend", "satisfied", "happy", "disappointed"]
        }
        
        # Simple sentiment words
        positive_words = ["good", "great", "excellent", "amazing", "perfect", "love", "awesome", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst", "disappointed", "poor"]
        
        for aspect, keywords in aspect_keywords.items():
            # Check if aspect is mentioned
            if any(keyword in text_lower for keyword in keywords):
                # Simple sentiment calculation
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment_score = 4.0 + min(positive_count * 0.2, 1.0)
                    confidence = min(0.7 + positive_count * 0.1, 0.9)
                elif negative_count > positive_count:
                    sentiment_score = 2.0 - min(negative_count * 0.2, 1.0)
                    confidence = min(0.7 + negative_count * 0.1, 0.9)
                else:
                    sentiment_score = 3.0
                    confidence = 0.5
                
                aspects_found[aspect] = {
                    "sentiment": max(1.0, min(5.0, sentiment_score)),
                    "confidence": confidence
                }
        
        return aspects_found
    
    def sentiment_score_to_rating(self, sentiment: str, confidence: float) -> float:
        """
        Convert sentiment label and confidence to 1-5 scale
        """
        base_scores = {
            'POSITIVE': 4.0,
            'NEUTRAL': 3.0,
            'NEGATIVE': 2.0
        }
        
        base_score = base_scores.get(sentiment, 3.0)
        
        # Adjust based on confidence
        if sentiment == 'POSITIVE':
            score = base_score + (confidence - 0.5) * 2  # Range: 3.0 to 5.0
        elif sentiment == 'NEGATIVE':
            score = base_score - (confidence - 0.5) * 2  # Range: 1.0 to 3.0
        else:
            score = base_score  # Neutral stays at 3.0
        
        return max(1.0, min(5.0, score))
    
    def analyze_review(self, review_text: str) -> SentimentResult:
        """
        Complete sentiment analysis for a single review
        """
        start_time = time.time()
        
        try:
            # Overall sentiment analysis
            sentiment, confidence = self.analyze_sentiment_hf(review_text)
            score = self.sentiment_score_to_rating(sentiment, confidence)
            
            # Aspect-based sentiment analysis
            aspects = self.extract_aspects_openai(review_text)
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                overall_sentiment=sentiment,
                confidence=confidence,
                score=score,
                aspects=aspects,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"ERROR: Review analysis failed: {e}")
            processing_time = time.time() - start_time
            
            return SentimentResult(
                overall_sentiment='NEUTRAL',
                confidence=0.5,
                score=3.0,
                aspects={},
                processing_time=processing_time
            )
    
    def batch_analyze_reviews(self, review_ids: List[int] = None, limit: int = 500) -> List[Dict]:
        """
        Batch process reviews from database
        """
        logger.info(f"STARTING: Batch sentiment analysis (limit: {limit})")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        # First, check what columns exist in the reviews table
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(reviews)")
        columns_info = cursor.fetchall()
        available_columns = [col[1] for col in columns_info]
        logger.info(f"Available columns in reviews table: {available_columns}")
        
        # Determine the correct column names
        id_column = None
        text_column = None
        
        # Look for ID column variations
        for col in available_columns:
            if col.lower() in ['id', 'review_id', 'rowid']:
                id_column = col
                break
        
        # Look for text column variations
        for col in available_columns:
            if col.lower() in ['review_text', 'text', 'content', 'body']:
                text_column = col
                break
        
        if not id_column or not text_column:
            logger.error(f"ERROR: Could not find required columns. Available: {available_columns}")
            conn.close()
            return []
        
        logger.info(f"Using columns: ID='{id_column}', TEXT='{text_column}'")
        
        # Query reviews with correct column names
        if review_ids:
            placeholders = ','.join(['?' for _ in review_ids])
            query = f"SELECT {id_column}, {text_column} FROM reviews WHERE {id_column} IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=review_ids)
        else:
            query = f"SELECT {id_column}, {text_column} FROM reviews ORDER BY {id_column} LIMIT ?"
            df = pd.read_sql_query(query, conn, params=[limit])
        
        conn.close()
        
        # Rename columns to standard names for processing
        df = df.rename(columns={id_column: 'id', text_column: 'review_text'})
        
        logger.info(f"PROCESSING: {len(df)} reviews...")
        
        results = []
        start_time = time.time()
        
        for idx, row in df.iterrows():
            try:
                result = self.analyze_review(row['review_text'])
                
                # Prepare data for database storage
                result_data = {
                    'review_id': row['id'],
                    'overall_sentiment': result.overall_sentiment,
                    'confidence': result.confidence,
                    'sentiment_score': result.score,
                    'aspects_json': json.dumps(result.aspects),
                    'processing_time': result.processing_time,
                    'analyzed_at': datetime.now().isoformat()
                }
                
                results.append(result_data)
                
                # Progress logging
                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (len(df) - idx - 1) / rate if rate > 0 else 0
                    logger.info(f"PROGRESS: Processed {idx + 1}/{len(df)} reviews | Rate: {rate:.1f}/sec | ETA: {eta:.0f}s")
                
            except Exception as e:
                logger.error(f"ERROR: Failed to process review {row['id']}: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(f"COMPLETE: Batch analysis complete: {len(results)} reviews in {total_time:.1f}s")
        
        return results
    
    def save_sentiment_results(self, results: List[Dict]):
        """
        Save sentiment analysis results to database
        """
        if not results:
            logger.warning("WARNING: No results to save")
            return
        
        logger.info(f"SAVING: {len(results)} sentiment results to database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create sentiment_analysis table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id INTEGER REFERENCES reviews(id),
                    overall_sentiment TEXT,
                    confidence REAL,
                    sentiment_score REAL,
                    aspects_json TEXT,
                    processing_time REAL,
                    analyzed_at TEXT,
                    UNIQUE(review_id)
                )
            """)
            
            # Insert results (replace existing)
            for result in results:
                conn.execute("""
                    INSERT OR REPLACE INTO sentiment_analysis 
                    (review_id, overall_sentiment, confidence, sentiment_score, 
                     aspects_json, processing_time, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['review_id'],
                    result['overall_sentiment'],
                    result['confidence'],
                    result['sentiment_score'],
                    result['aspects_json'],
                    result['processing_time'],
                    result['analyzed_at']
                ))
            
            conn.commit()
            conn.close()
            
            logger.info("SUCCESS: Sentiment results saved successfully")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save sentiment results: {e}")
            raise
    
    def get_sentiment_stats(self) -> Dict:
        """
        Get sentiment analysis statistics from database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Overall statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_analyzed,
                    AVG(confidence) as avg_confidence,
                    AVG(sentiment_score) as avg_sentiment_score,
                    AVG(processing_time) as avg_processing_time,
                    COUNT(CASE WHEN overall_sentiment = 'POSITIVE' THEN 1 END) as positive_count,
                    COUNT(CASE WHEN overall_sentiment = 'NEGATIVE' THEN 1 END) as negative_count,
                    COUNT(CASE WHEN overall_sentiment = 'NEUTRAL' THEN 1 END) as neutral_count
                FROM sentiment_analysis
            """
            
            stats = pd.read_sql_query(stats_query, conn).iloc[0].to_dict()
            
            # Aspect statistics
            aspect_query = """
                SELECT aspects_json 
                FROM sentiment_analysis 
                WHERE aspects_json IS NOT NULL AND aspects_json != '{}'
            """
            
            aspect_data = pd.read_sql_query(aspect_query, conn)
            
            # Parse aspects and calculate averages
            aspect_stats = {}
            for _, row in aspect_data.iterrows():
                try:
                    aspects = json.loads(row['aspects_json'])
                    for aspect, data in aspects.items():
                        if aspect not in aspect_stats:
                            aspect_stats[aspect] = []
                        aspect_stats[aspect].append(data['sentiment'])
                except:
                    continue
            
            # Calculate aspect averages
            for aspect in aspect_stats:
                aspect_stats[aspect] = {
                    'avg_sentiment': np.mean(aspect_stats[aspect]),
                    'count': len(aspect_stats[aspect])
                }
            
            conn.close()
            
            return {
                'overall_stats': stats,
                'aspect_stats': aspect_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get sentiment stats: {e}")
            return {}

def main():
    """
    Main function for testing sentiment analysis
    """
    print("TESTING: Sentiment Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test single review
    test_review = """
    This product is absolutely amazing! The quality is outstanding and the price is very reasonable. 
    The shipping was incredibly fast - arrived in just 2 days. Customer service was also very helpful 
    when I had questions. The packaging was secure and everything arrived in perfect condition. 
    I would definitely recommend this to anyone looking for great value!
    """
    
    print("\nTESTING: Single review analysis...")
    result = analyzer.analyze_review(test_review)
    
    print(f"Overall Sentiment: {result.overall_sentiment}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Score (1-5): {result.score:.2f}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print("\nAspect Analysis:")
    for aspect, data in result.aspects.items():
        print(f"  {aspect}: {data['sentiment']:.2f} (confidence: {data['confidence']:.2f})")
    
    # Test batch processing
    print("\nTESTING: Batch processing (first 100 reviews)...")
    batch_results = analyzer.batch_analyze_reviews(limit=100)
    
    if batch_results:
        print(f"SUCCESS: Successfully processed {len(batch_results)} reviews")
        
        # Save results
        analyzer.save_sentiment_results(batch_results)
        
        # Get statistics
        stats = analyzer.get_sentiment_stats()
        print("\nSTATISTICS: Analysis Statistics:")
        print(f"Total Analyzed: {stats['overall_stats']['total_analyzed']}")
        print(f"Average Confidence: {stats['overall_stats']['avg_confidence']:.3f}")
        print(f"Average Sentiment Score: {stats['overall_stats']['avg_sentiment_score']:.2f}")
        print(f"Positive Reviews: {stats['overall_stats']['positive_count']}")
        print(f"Negative Reviews: {stats['overall_stats']['negative_count']}")
        print(f"Neutral Reviews: {stats['overall_stats']['neutral_count']}")
        
        if stats['aspect_stats']:
            print("\nTop Aspects:")
            for aspect, data in sorted(stats['aspect_stats'].items(), 
                                     key=lambda x: x[1]['count'], reverse=True)[:5]:
                print(f"  {aspect}: {data['avg_sentiment']:.2f} ({data['count']} mentions)")
    
    print("\nCOMPLETE: Sentiment Analysis System test complete!")

if __name__ == "__main__":
    main()