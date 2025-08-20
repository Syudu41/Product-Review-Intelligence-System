"""
SQLAlchemy models for Review Intelligence Engine
"""
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, 
    DateTime, Date, ForeignKey, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(Text, nullable=False)
    category = Column(String(100))
    avg_rating = Column(Float, default=0.0)
    total_reviews = Column(Integer, default=0)
    scrape_url = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")
    live_reviews = relationship("LiveReview", back_populates="product", cascade="all, delete-orphan")
    recommendations = relationship("UserRecommendation", back_populates="product")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    username = Column(String(100))
    review_count = Column(Integer, default=0)
    avg_rating_given = Column(Float, default=0.0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    reviews = relationship("Review", back_populates="user")
    recommendations = relationship("UserRecommendation", back_populates="user")

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(100), unique=True, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), index=True)
    rating = Column(Integer, nullable=False)
    review_text = Column(Text)
    review_title = Column(String(500))
    helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    verified_purchase = Column(Boolean, default=False)
    review_date = Column(Date)
    scrape_date = Column(DateTime, default=func.now())
    
    # ML Analysis Fields
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    sentiment_confidence = Column(Float)
    is_fake = Column(Boolean)
    fake_confidence = Column(Float)
    
    # Metadata
    data_source = Column(String(50), default='kaggle', index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('rating >= 1 AND rating <= 5', name='valid_rating'),
    )
    
    # Relationships
    product = relationship("Product", back_populates="reviews")
    user = relationship("User", back_populates="reviews")
    aspect_sentiments = relationship("AspectSentiment", back_populates="review", cascade="all, delete-orphan")

class LiveReview(Base):
    __tablename__ = "live_reviews"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    review_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    review_title = Column(String(500))
    scraper_source = Column(String(50))
    scrape_date = Column(DateTime, default=func.now())
    raw_html = Column(Text)
    
    # Processing status
    processed = Column(Boolean, default=False, index=True)
    error_message = Column(Text)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('rating >= 1 AND rating <= 5', name='valid_live_rating'),
    )
    
    # Relationships
    product = relationship("Product", back_populates="live_reviews")

class AspectSentiment(Base):
    __tablename__ = "aspect_sentiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(100), ForeignKey("reviews.review_id"), nullable=False, index=True)
    aspect = Column(String(50), nullable=False)  # price, quality, shipping, service
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    confidence = Column(Float)
    
    # Relationships
    review = relationship("Review", back_populates="aspect_sentiments")

class UserRecommendation(Base):
    __tablename__ = "user_recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False)
    recommended_product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False)
    recommendation_score = Column(Float)
    recommendation_reason = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="recommendations")
    product = relationship("Product", back_populates="recommendations")

class SystemMetric(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float)
    metric_metadata = Column(Text)  # JSON string
    recorded_at = Column(DateTime, default=func.now())