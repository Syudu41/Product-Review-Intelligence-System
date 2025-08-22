"""
Quick fix for monitoring database schema conflicts
Run this to fix the database tables for monitoring system
"""

import sqlite3
import os

def fix_monitoring_schema(db_path="./database/review_intelligence.db"):
    """Fix database schema for monitoring system"""
    
    print("🔧 FIXING: Database schema for monitoring system...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Drop existing conflicting tables
        print("STEP 1: Dropping existing conflicting tables...")
        cursor.execute("DROP TABLE IF EXISTS system_metrics")
        cursor.execute("DROP TABLE IF EXISTS model_performance_metrics") 
        cursor.execute("DROP TABLE IF EXISTS health_checks")
        
        # Create fresh monitoring tables with correct schema
        print("STEP 2: Creating fresh monitoring tables...")
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                metric_type TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance metrics table
        cursor.execute("""
            CREATE TABLE model_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                sample_size INTEGER,
                processing_time REAL,
                confidence REAL,
                timestamp TEXT NOT NULL,
                additional_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Health checks table
        cursor.execute("""
            CREATE TABLE health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time REAL,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for performance
        print("STEP 3: Creating performance indices...")
        cursor.execute("CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp)")
        cursor.execute("CREATE INDEX idx_model_metrics_timestamp ON model_performance_metrics(timestamp)")
        cursor.execute("CREATE INDEX idx_health_checks_timestamp ON health_checks(timestamp)")
        
        conn.commit()
        conn.close()
        
        print("✅ SUCCESS: Database schema fixed for monitoring system")
        print("✅ Tables created: system_metrics, model_performance_metrics, health_checks")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to fix schema: {e}")
        return False

if __name__ == "__main__":
    success = fix_monitoring_schema()
    if success:
        print("\n🚀 READY: Run monitoring system again - it should work now!")
    else:
        print("\n❌ FAILED: Manual database inspection needed")