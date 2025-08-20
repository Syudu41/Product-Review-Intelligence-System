-- Review Intelligence Engine Database Schema
-- SQLite compatible schema

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(50) UNIQUE NOT NULL,  -- ASIN or unique identifier
    name TEXT NOT NULL,
    category VARCHAR(100),
    avg_rating REAL DEFAULT 0.0,
    total_reviews INTEGER DEFAULT 0,
    scrape_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) UNIQUE NOT NULL,
    username VARCHAR(100),
    review_count INTEGER DEFAULT 0,
    avg_rating_given REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reviews table (main table)
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id VARCHAR(100) UNIQUE,
    product_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    review_title VARCHAR(500),
    helpful_votes INTEGER DEFAULT 0,
    total_votes INTEGER DEFAULT 0,
    verified_purchase BOOLEAN DEFAULT FALSE,
    review_date DATE,
    scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- ML Analysis Fields (populated later)
    sentiment_score REAL,  -- -1 to 1
    sentiment_label VARCHAR(20),  -- positive, negative, neutral
    sentiment_confidence REAL,
    is_fake BOOLEAN,
    fake_confidence REAL,
    
    -- Metadata
    data_source VARCHAR(50) DEFAULT 'kaggle',  -- kaggle, scraped, synthetic
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (product_id) REFERENCES products (product_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- Live reviews table (for real-time scraped data)
CREATE TABLE IF NOT EXISTS live_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(50) NOT NULL,
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_title VARCHAR(500),
    scraper_source VARCHAR(50),  -- amazon, bestbuy, etc.
    scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_html TEXT,  -- Store original HTML for debugging
    
    -- Processing status
    processed BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

-- Aspect sentiments table (for detailed analysis)
CREATE TABLE IF NOT EXISTS aspect_sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id VARCHAR(100) NOT NULL,
    aspect VARCHAR(50) NOT NULL,  -- price, quality, shipping, service
    sentiment_score REAL,
    sentiment_label VARCHAR(20),
    confidence REAL,
    
    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
);

-- User recommendations table
CREATE TABLE IF NOT EXISTS user_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) NOT NULL,
    recommended_product_id VARCHAR(50) NOT NULL,
    recommendation_score REAL,
    recommendation_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (recommended_product_id) REFERENCES products (product_id)
);

-- System metrics table (for monitoring)
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL,
    metric_metadata TEXT,  -- JSON string
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews (product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews (user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews (rating);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews (sentiment_score);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews (review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_source ON reviews (data_source);
CREATE INDEX IF NOT EXISTS idx_live_reviews_product ON live_reviews (product_id);
CREATE INDEX IF NOT EXISTS idx_live_reviews_processed ON live_reviews (processed);
CREATE INDEX IF NOT EXISTS idx_aspect_sentiments_review ON aspect_sentiments (review_id);

-- Create triggers to update timestamps
CREATE TRIGGER IF NOT EXISTS update_products_timestamp 
    AFTER UPDATE ON products
    BEGIN
        UPDATE products SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_users_timestamp 
    AFTER UPDATE ON users
    BEGIN
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_reviews_timestamp 
    AFTER UPDATE ON reviews
    BEGIN
        UPDATE reviews SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;