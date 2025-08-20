#!/usr/bin/env python3
"""
Simple standalone validation script for Day 1 setup
This script validates that all basic components are working correctly
"""
import sys
import os
from pathlib import Path
import sqlite3

def test_project_structure():
    """Test that all required folders exist"""
    print("🔍 Testing project structure...")
    
    required_folders = [
        "data/raw", "data/processed", "data/synthetic", "data/live",
        "database", "src/data_pipeline", "src/scraping", "src/utils",
        "tests", "logs"
    ]
    
    missing_folders = []
    for folder in required_folders:
        if not Path(folder).exists():
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"❌ Missing folders: {missing_folders}")
        return False
    else:
        print("✅ All required folders exist")
        return True

def test_required_files():
    """Test that all required files exist"""
    print("🔍 Testing required files...")
    
    required_files = [
        "config.py",
        "requirements.txt",
        ".env",
        ".gitignore",
        "database/schema.sql",
        "database/models.py",
        "database/db_setup.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_python_imports():
    """Test that basic Python imports work"""
    print("🔍 Testing Python imports...")
    
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())
    
    try:
        import config
        print("✅ config.py imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import config.py: {e}")
        return False
    
    try:
        from database.models import Base, Product, User, Review
        print("✅ database.models imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import database.models: {e}")
        return False
    
    return True

def test_database_file():
    """Test that database file exists and has tables"""
    print("🔍 Testing database...")
    
    db_path = Path("database/review_intelligence.db")
    
    if not db_path.exists():
        print("❌ Database file does not exist")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Expected tables
        expected_tables = [
            'products', 'users', 'reviews', 'live_reviews',
            'aspect_sentiments', 'user_recommendations', 'system_metrics'
        ]
        
        missing_tables = [table for table in expected_tables if table not in tables]
        
        if missing_tables:
            print(f"❌ Missing database tables: {missing_tables}")
            return False
        
        print(f"✅ Database exists with {len(tables)} tables: {tables}")
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_environment_config():
    """Test environment configuration"""
    print("🔍 Testing environment configuration...")
    
    if not Path(".env").exists():
        print("❌ .env file not found")
        return False
    
    # Read .env file
    with open(".env", "r") as f:
        env_content = f.read()
    
    if "your_openai_api_key_here" in env_content:
        print("⚠️  OpenAI API key not set in .env file")
        print("   Please set OPENAI_API_KEY=your_actual_key")
        return False
    
    if "OPENAI_API_KEY=" in env_content:
        print("✅ Environment configuration looks good")
        return True
    else:
        print("❌ OPENAI_API_KEY not found in .env file")
        return False

def test_logging():
    """Test that logging works"""
    print("🔍 Testing logging system...")
    
    try:
        # Add to path and import
        sys.path.insert(0, os.getcwd())
        from src.utils.logger import get_logger
        
        # Create a test logger
        logger = get_logger("validation_test")
        logger.info("Test log message from validation")
        
        # Check if log file was created
        log_file = Path("logs/app.log")
        if log_file.exists():
            print("✅ Logging system works - log file created")
            return True
        else:
            print("❌ Log file not created")
            return False
            
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are installed"""
    print("🔍 Testing dependencies...")
    
    # Use correct import names (not pip package names)
    required_packages = [
        'pandas', 'numpy', 'sqlalchemy', 'fastapi', 'streamlit',
        'transformers', 'requests', 'bs4',  # beautifulsoup4 imports as bs4
        'openai', 'dotenv',  # python-dotenv imports as dotenv
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def run_comprehensive_validation():
    """Run all validation tests"""
    print("=" * 60)
    print("DAY 1 SETUP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Required Files", test_required_files),
        ("Python Imports", test_python_imports),
        ("Database", test_database_file),
        ("Environment Config", test_environment_config),
        ("Dependencies", test_dependencies),
        ("Logging System", test_logging)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Day 1 setup is complete!")
        print("\nYou're ready to proceed to Day 2!")
        return 0
    elif passed >= total * 0.8:
        print("✅ MOSTLY SUCCESSFUL - Minor issues need attention")
        print("\nYou can proceed to Day 2, but fix issues when possible")
        return 0
    else:
        print("❌ SETUP INCOMPLETE - Please fix issues before proceeding")
        print("\nResolve the failed tests before moving to Day 2")
        return 1

if __name__ == "__main__":
    exit_code = run_comprehensive_validation()
    sys.exit(exit_code)