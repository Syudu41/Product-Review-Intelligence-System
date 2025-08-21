#!/usr/bin/env python3
"""
Database Migration Script
Updates existing database schema to match current models
"""

import sqlite3
import sys
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_column_exists(cursor, table_name, column_name):
    """Check if column exists in table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    return column_name in columns

def migrate_live_reviews_table(db_path):
    """Migrate live_reviews table to include missing columns"""
    
    logger.info("Starting live_reviews table migration...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if live_reviews table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='live_reviews'
            """)
            
            if not cursor.fetchone():
                logger.info("live_reviews table doesn't exist, will be created by models")
                return True
            
            # List of columns to add
            columns_to_add = [
                ("user_id", "TEXT"),
                ("helpful_votes", "INTEGER DEFAULT 0"),
                ("total_votes", "INTEGER DEFAULT 0"), 
                ("verified_purchase", "BOOLEAN DEFAULT 0"),
                ("date", "DATETIME"),
                ("scrape_timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("source", "TEXT DEFAULT 'amazon_scraper'")
            ]
            
            # Add missing columns
            for column_name, column_def in columns_to_add:
                if not check_column_exists(cursor, 'live_reviews', column_name):
                    try:
                        cursor.execute(f"ALTER TABLE live_reviews ADD COLUMN {column_name} {column_def}")
                        logger.info(f"✅ Added column: {column_name}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e).lower():
                            logger.error(f"Failed to add column {column_name}: {e}")
                        else:
                            logger.info(f"⚠️  Column {column_name} already exists")
                else:
                    logger.info(f"✅ Column {column_name} already exists")
            
            # Create foreign key index if it doesn't exist
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_live_reviews_user_id ON live_reviews(user_id)")
                logger.info("✅ Created user_id index")
            except sqlite3.OperationalError as e:
                logger.warning(f"Index creation warning: {e}")
            
            conn.commit()
            logger.info("live_reviews table migration completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

def verify_migration(db_path):
    """Verify migration was successful"""
    
    logger.info("Verifying migration...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check live_reviews schema
            cursor.execute("PRAGMA table_info(live_reviews)")
            columns = cursor.fetchall()
            
            logger.info("live_reviews table schema:")
            for column in columns:
                logger.info(f"  - {column[1]} ({column[2]})")
            
            # Check for required columns
            column_names = [column[1] for column in columns]
            required_columns = ['user_id', 'helpful_votes', 'total_votes', 'verified_purchase', 'date', 'source']
            
            missing_columns = [col for col in required_columns if col not in column_names]
            
            if missing_columns:
                logger.error(f"❌ Missing columns: {missing_columns}")
                return False
            else:
                logger.info("✅ All required columns present")
                return True
                
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def test_scraper_compatibility(db_path):
    """Test that scraper data format works with new schema"""
    
    logger.info("Testing scraper compatibility...")
    
    # Sample data that scraper would insert
    test_data = {
        'product_id': 'TEST_PRODUCT_123',
        'user_id': 'TEST_USER_123', 
        'rating': 5,
        'review_text': 'Test review for migration',
        'review_title': 'Test Review',
        'helpful_votes': 0,
        'total_votes': 0,
        'verified_purchase': True,
        'source': 'amazon_scraper',
        'scrape_timestamp': datetime.now()
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Try to insert test data
            columns = ', '.join(test_data.keys())
            placeholders = ', '.join(['?' for _ in test_data])
            values = list(test_data.values())
            
            cursor.execute(f"""
                INSERT INTO live_reviews ({columns}) 
                VALUES ({placeholders})
            """, values)
            
            # Get the inserted record
            cursor.execute("SELECT * FROM live_reviews WHERE product_id = ?", ('TEST_PRODUCT_123',))
            result = cursor.fetchone()
            
            if result:
                logger.info("✅ Test insert successful")
                
                # Clean up test data
                cursor.execute("DELETE FROM live_reviews WHERE product_id = ?", ('TEST_PRODUCT_123',))
                conn.commit()
                
                return True
            else:
                logger.error("❌ Test insert failed - no data returned")
                return False
                
    except Exception as e:
        logger.error(f"❌ Scraper compatibility test failed: {e}")
        return False

def backup_database(db_path):
    """Create backup of database before migration"""
    
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Simple file copy for SQLite
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"✅ Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None

def main():
    """Main migration function"""
    
    print("🔄 DATABASE MIGRATION TOOL")
    print("=" * 50)
    
    # Database path
    db_path = "database/review_intelligence.db"
    
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    
    logger.info(f"Migrating database: {db_path}")
    
    # Create backup
    backup_path = backup_database(db_path)
    if not backup_path:
        logger.error("Failed to create backup, aborting migration")
        sys.exit(1)
    
    # Run migration
    if migrate_live_reviews_table(db_path):
        logger.info("✅ Migration completed")
        
        # Verify migration
        if verify_migration(db_path):
            logger.info("✅ Migration verification passed")
            
            # Test scraper compatibility
            if test_scraper_compatibility(db_path):
                logger.info("✅ Scraper compatibility test passed")
                
                print("\n" + "=" * 50)
                print("🎉 MIGRATION SUCCESSFUL!")
                print("=" * 50)
                print("✅ Database schema updated")
                print("✅ Scraper compatibility verified")
                print(f"✅ Backup created: {backup_path}")
                print("✅ Ready to run scraper!")
                
            else:
                logger.error("❌ Scraper compatibility test failed")
                print("\n⚠️  Migration completed but scraper test failed")
                
        else:
            logger.error("❌ Migration verification failed")
            print("\n❌ Migration verification failed")
            
    else:
        logger.error("❌ Migration failed")
        print(f"\n❌ Migration failed. Restore from backup: {backup_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()