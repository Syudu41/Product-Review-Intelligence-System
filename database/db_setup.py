"""
Database setup and initialization for Review Intelligence Engine - FIXED VERSION
"""
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import sys
import os
import logging

# Add project root to Python path - FIXED
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import current_config, DATABASE_DIR
except ImportError:
    # Fallback if config not available - FIXED
    print("Warning: Could not import config, using defaults")
    DATABASE_DIR = Path(__file__).parent
    DATABASE_DIR.mkdir(exist_ok=True)
    
    class FallbackConfig:
        DATABASE_URL = f"sqlite:///{DATABASE_DIR}/review_intelligence.db"
        DEBUG = True
    
    current_config = FallbackConfig()

from database.models import Base

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database initialization and connections"""
    
    def __init__(self, config=None):
        self.config = config or current_config
        self.engine = None
        self.SessionLocal = None
        
    def create_engine_and_session(self):
        """Create SQLAlchemy engine and session factory"""
        try:
            # Extract database path from URL for SQLite
            if "sqlite" in self.config.DATABASE_URL:
                db_path = self.config.DATABASE_URL.replace("sqlite:///", "")
                db_file = Path(db_path)
                db_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.engine = create_engine(
                self.config.DATABASE_URL,
                echo=self.config.DEBUG,  # Log SQL queries in debug mode
                connect_args={"check_same_thread": False} if "sqlite" in self.config.DATABASE_URL else {}
            )
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info(f"Database engine created: {self.config.DATABASE_URL}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            return False
    
    def create_tables(self):
        """Create all tables using SQLAlchemy models"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    def execute_schema_file(self, schema_file="schema.sql"):
        """Execute SQL schema file directly (backup method)"""
        try:
            schema_path = DATABASE_DIR / schema_file
            if not schema_path.exists():
                logger.error(f"Schema file not found: {schema_path}")
                return False
            
            with open(schema_path, 'r') as file:
                schema_sql = file.read()
            
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            with self.engine.connect() as connection:
                for statement in statements:
                    if statement:
                        connection.execute(text(statement))
                connection.commit()
            
            logger.info(f"Schema file executed successfully: {schema_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute schema file: {e}")
            return False
    
    def get_session(self):
        """Get database session"""
        if not self.SessionLocal:
            self.create_engine_and_session()
        return self.SessionLocal()
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_table_info(self):
        """Get information about existing tables"""
        try:
            with self.engine.connect() as connection:
                # Get table names
                if "sqlite" in self.config.DATABASE_URL:
                    result = connection.execute(text(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ))
                else:
                    result = connection.execute(text(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                    ))
                
                tables = [row[0] for row in result.fetchall()]
                
                # Get table counts
                table_info = {}
                for table in tables:
                    if not table.startswith('sqlite_'):  # Skip SQLite system tables
                        count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = count_result.fetchone()[0]
                        table_info[table] = count
                
                return table_info
                
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return {}
    
    def reset_database(self):
        """Drop all tables and recreate them (USE WITH CAUTION)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("All tables dropped")
            
            Base.metadata.create_all(bind=self.engine)
            logger.info("All tables recreated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

def initialize_database(config=None, reset=False):
    """Main function to initialize database"""
    logger.info("Starting database initialization...")
    
    db_manager = DatabaseManager(config)
    
    # Create engine and session
    if not db_manager.create_engine_and_session():
        return False
    
    # Reset database if requested
    if reset:
        logger.warning("Resetting database...")
        if not db_manager.reset_database():
            return False
    
    # Create tables
    if not db_manager.create_tables():
        return False
    
    # Test connection
    if not db_manager.test_connection():
        return False
    
    # Show table information
    table_info = db_manager.get_table_info()
    logger.info("Database initialization complete!")
    logger.info(f"Tables created: {list(table_info.keys())}")
    
    return db_manager

def get_database_stats(db_manager):
    """Get current database statistics"""
    table_info = db_manager.get_table_info()
    
    stats = {
        "total_tables": len(table_info),
        "table_counts": table_info,
        "total_records": sum(table_info.values())
    }
    
    return stats

if __name__ == "__main__":
    # Initialize database
    db_manager = initialize_database()
    
    if db_manager:
        stats = get_database_stats(db_manager)
        print("\n=== Database Statistics ===")
        print(f"Total tables: {stats['total_tables']}")
        print(f"Total records: {stats['total_records']}")
        print("\nTable counts:")
        for table, count in stats['table_counts'].items():
            print(f"  {table}: {count}")
        print("\n✓ Database setup completed successfully")
    else:
        print("❌ Database initialization failed!")