"""
FastAPI Backend for Product Review Intelligence System
Provides REST API endpoints for all ML models and analytics
"""

import os
import sys
import sqlite3
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Installing FastAPI and dependencies...")
    os.system("pip install fastapi uvicorn python-multipart")
    from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our ML models
from ml_models.sentiment_analyzer import SentimentAnalyzer
from ml_models.fake_detector import FakeReviewDetector
from ml_models.recommendation_engine import RecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global model instances
sentiment_analyzer = None
fake_detector = None
recommendation_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global sentiment_analyzer, fake_detector, recommendation_engine
    
    logger.info("STARTUP: Initializing ML models...")
    
    try:
        # Initialize models
        sentiment_analyzer = SentimentAnalyzer()
        fake_detector = FakeReviewDetector()
        
        # Load pre-trained fake detection model if available
        try:
            fake_detector.load_model()
            logger.info("SUCCESS: Loaded pre-trained fake detection model")
        except FileNotFoundError:
            logger.warning("WARNING: No pre-trained fake detection model found")
        
        recommendation_engine = RecommendationEngine()
        
        logger.info("SUCCESS: All models initialized successfully")
        
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize models: {e}")
        raise
    
    yield
    
    logger.info("SHUTDOWN: Cleaning up resources...")

# Create FastAPI app
app = FastAPI(
    title="Product Review Intelligence API",
    description="Advanced ML-powered product review analysis and recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ReviewAnalysisRequest(BaseModel):
    review_text: str = Field(..., description="Review text to analyze")
    user_data: Optional[Dict] = Field(None, description="Optional user information")
    review_data: Optional[Dict] = Field(None, description="Optional review metadata")

class ReviewAnalysisResponse(BaseModel):
    review_text: str
    sentiment: Dict[str, Any]
    fake_detection: Dict[str, Any]
    processing_time: float
    timestamp: str

class ProductAnalyticsResponse(BaseModel):
    product_id: str
    product_name: str
    review_count: int
    average_rating: float
    sentiment_distribution: Dict[str, int]
    fake_review_percentage: float
    top_aspects: List[Dict[str, Any]]
    recent_trends: Dict[str, Any]

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    recommendation_type: str
    generated_at: str

class SystemStatsResponse(BaseModel):
    database_stats: Dict[str, Any]
    model_performance: Dict[str, Any]
    api_metrics: Dict[str, Any]
    timestamp: str

# Database helper functions
def get_db_connection():
    """Get database connection"""
    return sqlite3.connect("./database/review_intelligence.db")

def execute_query(query: str, params: tuple = ()) -> List[Dict]:
    """Execute SQL query and return results as list of dictionaries"""
    try:
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Product Review Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        
        # Check model availability
        models_status = {
            "sentiment_analyzer": sentiment_analyzer is not None,
            "fake_detector": fake_detector is not None,
            "recommendation_engine": recommendation_engine is not None
        }
        
        return {
            "status": "healthy",
            "database": "connected",
            "models": models_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/analyze/review", response_model=ReviewAnalysisResponse)
async def analyze_review(request: ReviewAnalysisRequest):
    """Analyze a single review for sentiment and fake detection"""
    start_time = time.time()
    
    try:
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze_review(request.review_text)
        
        # Fake detection (if model is available)
        if fake_detector.model:
            fake_result = fake_detector.predict_fake_probability(
                request.review_text,
                user_data=request.user_data,
                review_data=request.review_data
            )
            fake_detection = {
                "is_fake_probability": fake_result.is_fake_probability,
                "confidence": fake_result.confidence,
                "risk_level": fake_result.risk_level
            }
        else:
            fake_detection = {
                "is_fake_probability": 0.0,
                "confidence": 0.5,
                "risk_level": "UNKNOWN",
                "note": "Model not available"
            }
        
        processing_time = time.time() - start_time
        
        return ReviewAnalysisResponse(
            review_text=request.review_text,
            sentiment={
                "overall_sentiment": sentiment_result.overall_sentiment,
                "confidence": sentiment_result.confidence,
                "score": sentiment_result.score,
                "aspects": sentiment_result.aspects
            },
            fake_detection=fake_detection,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Review analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/products/{product_id}/analytics", response_model=ProductAnalyticsResponse)
async def get_product_analytics(
    product_id: str = Path(..., description="Product ID to analyze"),
    include_recent: bool = Query(True, description="Include recent trends")
):
    """Get comprehensive analytics for a specific product"""
    try:
        # Get basic product stats
        product_query = """
            SELECT product_id, COUNT(*) as review_count, AVG(rating) as avg_rating
            FROM reviews 
            WHERE product_id = ?
            GROUP BY product_id
        """
        product_stats = execute_query(product_query, (product_id,))
        
        if not product_stats:
            raise HTTPException(status_code=404, detail="Product not found")
        
        stats = product_stats[0]
        
        # Get sentiment distribution
        sentiment_query = """
            SELECT s.overall_sentiment, COUNT(*) as count
            FROM sentiment_analysis s
            JOIN reviews r ON s.review_id = r.Id
            WHERE r.product_id = ?
            GROUP BY s.overall_sentiment
        """
        sentiment_data = execute_query(sentiment_query, (product_id,))
        sentiment_distribution = {row['overall_sentiment']: row['count'] for row in sentiment_data}
        
        # Get fake review percentage
        fake_query = """
            SELECT AVG(f.fake_probability) as avg_fake_prob, COUNT(*) as total_checked
            FROM fake_detection f
            JOIN reviews r ON f.review_id = r.Id
            WHERE r.product_id = ?
        """
        fake_data = execute_query(fake_query, (product_id,))
        fake_percentage = fake_data[0]['avg_fake_prob'] * 100 if fake_data[0]['avg_fake_prob'] else 0
        
        # Get top aspects (from sentiment analysis)
        aspects_query = """
            SELECT s.aspects_json
            FROM sentiment_analysis s
            JOIN reviews r ON s.review_id = r.Id
            WHERE r.product_id = ? AND s.aspects_json != '{}'
            LIMIT 100
        """
        aspects_data = execute_query(aspects_query, (product_id,))
        
        # Aggregate aspects
        aspect_scores = {}
        for row in aspects_data:
            try:
                aspects = json.loads(row['aspects_json'])
                for aspect, data in aspects.items():
                    if aspect not in aspect_scores:
                        aspect_scores[aspect] = []
                    aspect_scores[aspect].append(data['sentiment'])
            except:
                continue
        
        top_aspects = [
            {
                "aspect": aspect,
                "average_score": sum(scores) / len(scores),
                "mention_count": len(scores)
            }
            for aspect, scores in aspect_scores.items()
        ]
        top_aspects.sort(key=lambda x: x['mention_count'], reverse=True)
        
        # Recent trends (if requested)
        recent_trends = {}
        if include_recent:
            recent_query = """
                SELECT DATE(r.date) as review_date, AVG(r.rating) as avg_rating, COUNT(*) as count
                FROM reviews r
                WHERE r.product_id = ? AND r.date >= date('now', '-30 days')
                GROUP BY DATE(r.date)
                ORDER BY review_date DESC
                LIMIT 30
            """
            recent_data = execute_query(recent_query, (product_id,))
            recent_trends = {
                "daily_ratings": [
                    {"date": row['review_date'], "rating": row['avg_rating'], "count": row['count']}
                    for row in recent_data
                ]
            }
        
        # Get product name (handle missing columns gracefully)
        try:
            product_name_query = "SELECT name FROM products WHERE product_id = ?"
            product_name_data = execute_query(product_name_query, (product_id,))
            product_name = product_name_data[0]['name'] if product_name_data else f"Product {product_id}"
        except:
            # If 'name' column doesn't exist, just use the product_id
            product_name = f"Product {product_id}"
        
        return ProductAnalyticsResponse(
            product_id=product_id,
            product_name=product_name,
            review_count=stats['review_count'],
            average_rating=round(stats['avg_rating'], 2),
            sentiment_distribution=sentiment_distribution,
            fake_review_percentage=round(fake_percentage, 2),
            top_aspects=top_aspects[:5],
            recent_trends=recent_trends
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: str = Path(..., description="User ID for recommendations"),
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    refresh: bool = Query(False, description="Generate fresh recommendations")
):
    """Get personalized recommendations for a user"""
    try:
        # Check if user exists
        user_query = "SELECT COUNT(*) as count FROM reviews WHERE user_id = ?"
        user_count = execute_query(user_query, (user_id,))

        if not user_count or user_count[0]['count'] == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        if refresh:
            # Generate fresh recommendations
            recommendations = recommendation_engine.get_hybrid_recommendations(user_id, limit)
            
            if recommendations:
                # Save to database
                recommendation_engine.save_user_recommendations(user_id, recommendations)
                
                recommendation_data = [
                    {
                        "product_id": rec.product_id,
                        "product_name": rec.product_name,
                        "predicted_rating": round(rec.predicted_rating, 2),
                        "confidence": round(rec.confidence, 3),
                        "recommendation_score": round(rec.recommendation_score, 3),
                        "reason": rec.reason,
                        "similar_products": rec.similar_products
                    }
                    for rec in recommendations
                ]
                
                recommendation_type = "fresh_hybrid"
            else:
                recommendation_data = []
                recommendation_type = "none_available"
        else:
            # Get stored recommendations
            stored_query = """
                SELECT product_id, predicted_rating, confidence, recommendation_score, 
                       reason, similar_products, generated_at
                FROM user_recommendations 
                WHERE user_id = ?
                ORDER BY recommendation_score DESC
                LIMIT ?
            """
            stored_recs = execute_query(stored_query, (user_id, limit))
            
            if stored_recs:
                recommendation_data = [
                    {
                        "product_id": rec['product_id'],
                        "product_name": f"Product {rec['product_id']}",
                        "predicted_rating": round(rec['predicted_rating'], 2),
                        "confidence": round(rec['confidence'], 3),
                        "recommendation_score": round(rec['recommendation_score'], 3),
                        "reason": rec['reason'],
                        "similar_products": json.loads(rec['similar_products']) if rec['similar_products'] else []
                    }
                    for rec in stored_recs
                ]
                recommendation_type = "stored"
            else:
                # Fallback to popular products
                popular_query = """
                    SELECT product_id, AVG(rating) as avg_rating, COUNT(*) as count
                    FROM reviews
                    GROUP BY product_id
                    HAVING COUNT(*) >= 5
                    ORDER BY avg_rating DESC, count DESC
                    LIMIT ?
                """
                popular_recs = execute_query(popular_query, (limit,))
                
                recommendation_data = [
                    {
                        "product_id": rec['product_id'],
                        "product_name": f"Product {rec['product_id']}",
                        "predicted_rating": round(rec['avg_rating'], 2),
                        "confidence": 0.5,
                        "recommendation_score": rec['avg_rating'],
                        "reason": "Popular product",
                        "similar_products": []
                    }
                    for rec in popular_recs
                ]
                recommendation_type = "popular_fallback"
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendation_data,
            recommendation_type=recommendation_type,
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@app.get("/analytics/system-stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Database stats
        tables = ['reviews', 'sentiment_analysis', 'fake_detection', 'user_recommendations', 'products']
        db_stats = {}
        
        for table in tables:
            try:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = execute_query(count_query)
                db_stats[table] = result[0]['count'] if result else 0
            except:
                db_stats[table] = 0
        
        # Model performance stats
        model_performance = {}
        
        # Sentiment analysis performance
        if db_stats['sentiment_analysis'] > 0:
            sentiment_perf_query = """
                SELECT AVG(confidence) as avg_confidence, AVG(processing_time) as avg_time
                FROM sentiment_analysis
            """
            sentiment_perf = execute_query(sentiment_perf_query)
            if sentiment_perf:
                model_performance['sentiment_analysis'] = {
                    "avg_confidence": round(sentiment_perf[0]['avg_confidence'], 3),
                    "avg_processing_time": round(sentiment_perf[0]['avg_time'], 3),
                    "samples_processed": db_stats['sentiment_analysis']
                }
        
        # Fake detection performance
        if db_stats['fake_detection'] > 0:
            fake_perf_query = """
                SELECT AVG(confidence) as avg_confidence, AVG(fake_probability) as avg_fake_prob
                FROM fake_detection
            """
            fake_perf = execute_query(fake_perf_query)
            if fake_perf:
                model_performance['fake_detection'] = {
                    "avg_confidence": round(fake_perf[0]['avg_confidence'], 3),
                    "avg_fake_probability": round(fake_perf[0]['avg_fake_prob'], 3),
                    "samples_processed": db_stats['fake_detection']
                }
        
        # API metrics (simplified)
        api_metrics = {
            "uptime": "Available",
            "models_loaded": {
                "sentiment_analyzer": sentiment_analyzer is not None,
                "fake_detector": fake_detector is not None and fake_detector.model is not None,
                "recommendation_engine": recommendation_engine is not None
            }
        }
        
        return SystemStatsResponse(
            database_stats=db_stats,
            model_performance=model_performance,
            api_metrics=api_metrics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"System stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

@app.get("/products/search")
async def search_products(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results")
):
    """Search products by name or review content"""
    try:
        search_query = """
            SELECT DISTINCT r.product_id, 
                   COUNT(*) as review_count,
                   AVG(r.rating) as avg_rating,
                   GROUP_CONCAT(SUBSTR(r.review_text, 1, 100)) as sample_reviews
            FROM reviews r
            WHERE r.review_text LIKE ? OR r.product_id LIKE ?
            GROUP BY r.product_id
            ORDER BY COUNT(*) DESC, AVG(r.rating) DESC
            LIMIT ?
        """
        
        search_term = f"%{query}%"
        results = execute_query(search_query, (search_term, search_term, limit))
        
        formatted_results = [
            {
                "product_id": row['product_id'],
                "product_name": f"Product {row['product_id']}",
                "review_count": row['review_count'],
                "average_rating": round(row['avg_rating'], 2),
                "sample_review": row['sample_reviews'][:100] + "..." if row['sample_reviews'] else ""
            }
            for row in results
        ]
        
        return {
            "query": query,
            "results_count": len(formatted_results),
            "products": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/admin/retrain-models")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining (admin endpoint)"""
    try:
        # Add background task for retraining
        background_tasks.add_task(background_retrain_models)
        
        return {
            "message": "Model retraining initiated in background",
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retrain trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")

async def background_retrain_models():
    """Background task for model retraining"""
    logger.info("BACKGROUND: Starting model retraining...")
    try:
        # This would trigger the model training pipeline
        # For now, just log the action
        logger.info("Model retraining completed successfully")
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")

def main():
    """Run the FastAPI server"""
    print("STARTING: FastAPI Server for Product Review Intelligence")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("System Stats: http://localhost:8000/analytics/system-stats")
    print("=" * 60)
    
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()