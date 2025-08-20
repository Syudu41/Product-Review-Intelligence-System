"""
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
    SAMPLE_SIZE_FOR_TESTING = 1000  # Use smaller dataset for testing
    BATCH_SIZE = 100
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
            
        if not DATABASE_DIR.exists():
            errors.append(f"Database directory {DATABASE_DIR} does not exist")
            
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    DATABASE_URL = os.getenv("DATABASE_URL_PROD", Config.DATABASE_URL)

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    DATABASE_URL = f"sqlite:///{DATABASE_DIR}/test_review_intelligence.db"
    SAMPLE_SIZE_FOR_TESTING = 100  # Very small for fast tests

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