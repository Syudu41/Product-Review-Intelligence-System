#!/usr/bin/env python3
"""
FIXED Setup Script for Day 1 - Review Intelligence Engine
This script creates all necessary files AND folder structure
"""
import os
import sys
from pathlib import Path
import subprocess

def create_all_files():
    """Create all necessary files with their content"""
    print("Creating all project files...")
    
    # 1. Create config.py
    config_content = '''"""
Configuration management for Review Intelligence Engine
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
LIVE_DATA_DIR = DATA_DIR / "live"
DATABASE_DIR = PROJECT_ROOT / "database"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 SYNTHETIC_DATA_DIR, LIVE_DATA_DIR, DATABASE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

class Config:
    """Base configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_DIR}/review_intelligence.db")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "app.log"
    
    # Scraping Settings
    SCRAPING_DELAY_MIN = int(os.getenv("SCRAPING_DELAY_MIN", "1"))
    SCRAPING_DELAY_MAX = int(os.getenv("SCRAPING_DELAY_MAX", "3"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    USER_AGENT = os.getenv("USER_AGENT", 
                          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # Development
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Rate Limiting
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    API_RATE_LIMIT_PER_HOUR = int(os.getenv("API_RATE_LIMIT_PER_HOUR", "100"))
    
    # Data Processing
    SAMPLE_SIZE_FOR_TESTING = 1000
    BATCH_SIZE = 100

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    DATABASE_URL = f"sqlite:///{DATABASE_DIR}/test_review_intelligence.db"
    SAMPLE_SIZE_FOR_TESTING = 100

# Select configuration based on environment
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

current_config = config_map.get(
    os.getenv("ENVIRONMENT", "development"), 
    DevelopmentConfig
)
'''
    
    with open("config.py", "w") as f:
        f.write(config_content)
    print("✓ Created config.py")

    # 2. Create database files
    Path("database").mkdir(exist_ok=True)
    
    # Database schema
    schema_content = '''-- Review Intelligence Engine Database Schema
-- SQLite compatible schema

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(50) UNIQUE NOT NULL,
    name TEXT NOT NULL,
    category VARCHAR(100),
    avg_rating REAL DEFAULT 0.0,
    total_reviews INTEGER DEFAULT 0,
    scrape_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(50) UNIQUE NOT NULL,
    username VARCHAR(100),
    review_count INTEGER DEFAULT 0,
    avg_rating_given REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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
    sentiment_score REAL,
    sentiment_label VARCHAR(20),
    sentiment_confidence REAL,
    is_fake BOOLEAN,
    fake_confidence REAL,
    data_source VARCHAR(50) DEFAULT 'kaggle',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products (product_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS live_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(50) NOT NULL,
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_title VARCHAR(500),
    scraper_source VARCHAR(50),
    scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_html TEXT,
    processed BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

CREATE TABLE IF NOT EXISTS aspect_sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id VARCHAR(100) NOT NULL,
    aspect VARCHAR(50) NOT NULL,
    sentiment_score REAL,
    sentiment_label VARCHAR(20),
    confidence REAL,
    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
);

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

CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL,
    metric_metadata TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews (product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews (user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews (rating);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews (sentiment_score);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews (review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_source ON reviews (data_source);
'''
    
    with open("database/schema.sql", "w") as f:
        f.write(schema_content)
    print("✓ Created database/schema.sql")

    # Database setup
    db_setup_content = '''"""
Database setup and initialization for Review Intelligence Engine
"""
import sqlite3
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import current_config, DATABASE_DIR
except ImportError:
    # Fallback if config not available
    DATABASE_DIR = Path(__file__).parent.parent / "database"
    DATABASE_DIR.mkdir(exist_ok=True)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Simple SQLite database manager"""
    
    def __init__(self):
        self.db_path = DATABASE_DIR / "review_intelligence.db"
        self.schema_path = DATABASE_DIR / "schema.sql"
        
    def create_database(self):
        """Create database and tables"""
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Read and execute schema
            if self.schema_path.exists():
                with open(self.schema_path, 'r') as f:
                    schema_sql = f.read()
                
                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        cursor.execute(statement)
                
                conn.commit()
                logger.info("Database and tables created successfully")
            else:
                logger.error(f"Schema file not found: {self.schema_path}")
                return False
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database creation failed: {e}")
            return False
    
    def test_connection(self):
        """Test database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            logger.info(f"Database connection successful. Tables: {[t[0] for t in tables]}")
            return True, [t[0] for t in tables]
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False, []

def initialize_database():
    """Main function to initialize database"""
    logger.info("Starting database initialization...")
    
    db_manager = DatabaseManager()
    
    # Create database
    if not db_manager.create_database():
        return False
    
    # Test connection
    success, tables = db_manager.test_connection()
    if success:
        logger.info("Database initialization complete!")
        logger.info(f"Tables created: {tables}")
        return True
    else:
        return False

if __name__ == "__main__":
    success = initialize_database()
    if success:
        print("✓ Database setup completed successfully")
    else:
        print("❌ Database setup failed")
'''
    
    with open("database/db_setup.py", "w") as f:
        f.write(db_setup_content)
    print("✓ Created database/db_setup.py")

    # 3. Create src/utils files
    Path("src/utils").mkdir(parents=True, exist_ok=True)
    
    # Logger utility
    logger_content = '''"""
Simple logging configuration for Review Intelligence Engine
"""
import logging
import os
from pathlib import Path

def setup_logger(name=None):
    """Setup basic logger"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = logs_dir / "app.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name=None):
    """Get configured logger instance"""
    return setup_logger(name)
'''
    
    with open("src/utils/logger.py", "w") as f:
        f.write(logger_content)
    print("✓ Created src/utils/logger.py")

    # 4. Create simple validation test
    Path("tests").mkdir(exist_ok=True)
    
    validation_content = '''"""
Simple Day 1 Validation Script
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports"""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config.py import successful")
    except ImportError as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from database.db_setup import DatabaseManager
        print("✓ database.db_setup import successful")
    except ImportError as e:
        print(f"❌ database.db_setup import failed: {e}")
        return False
    
    try:
        from src.utils.logger import get_logger
        print("✓ src.utils.logger import successful")
    except ImportError as e:
        print(f"❌ src.utils.logger import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database setup"""
    print("Testing database...")
    
    try:
        from database.db_setup import initialize_database
        success = initialize_database()
        if success:
            print("✓ Database initialization successful")
            return True
        else:
            print("❌ Database initialization failed")
            return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_logging():
    """Test logging system"""
    print("Testing logging...")
    
    try:
        from src.utils.logger import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        
        # Check if log file was created
        log_file = Path("logs/app.log")
        if log_file.exists():
            print("✓ Logging system working")
            return True
        else:
            print("❌ Log file not created")
            return False
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def run_day1_validation():
    """Run all Day 1 validation tests"""
    print("="*50)
    print("DAY 1 SIMPLE VALIDATION")
    print("="*50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Database Tests", test_database),
        ("Logging Tests", test_logging)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        success = test_func()
        results.append(success)
    
    print("\\n" + "="*50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        return 0
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        return 1

if __name__ == "__main__":
    exit_code = run_day1_validation()
    sys.exit(exit_code)
'''
    
    with open("tests/day1_validation.py", "w") as f:
        f.write(validation_content)
    print("✓ Created tests/day1_validation.py")

    # 5. Create sample data
    Path("tests/sample_data").mkdir(parents=True, exist_ok=True)
    
    sample_csv = '''reviewerID,asin,reviewerName,helpful,reviewText,overall,summary,unixReviewTime,reviewTime
A2SUAM1J3GNN3B,0000013714,J. McDonald,"[0, 0]",I bought this for my husband who plays the piano.,5,Works with Yamaha piano,1393545600,01 28 2014
A14VAT5EAX3D9S,0000013714,guitar god,,The product does exactly as it should and is quite affordable.,5,Exactly what I expected,1363392000,03 16 2013
A195EZSQDW3E21,0000013714,Rick Bennette,"[0, 0]","A useful cable that works as expected.",4,Good product,1377648000,08 28 2013
'''
    
    with open("tests/sample_data/sample_reviews.csv", "w") as f:
        f.write(sample_csv)
    print("✓ Created tests/sample_data/sample_reviews.csv")

def install_dependencies():
    """Install dependencies with fixed requirements"""
    print("Installing dependencies...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("FIXED REVIEW INTELLIGENCE ENGINE - DAY 1 SETUP")
    print("="*60)
    
    # Create all files
    create_all_files()
    
    print("\\nInstalling dependencies...")
    install_dependencies()
    
    print("\\nRunning validation...")
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        from tests.day1_validation import run_day1_validation
        exit_code = run_day1_validation()
        
        if exit_code == 0:
            print("\\n🎉 Setup completed successfully!")
        else:
            print("\\n⚠️  Setup completed with some issues")
    except Exception as e:
        print(f"\\n❌ Validation failed: {e}")

if __name__ == "__main__":
    main()