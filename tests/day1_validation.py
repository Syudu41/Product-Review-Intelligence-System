"""
Day 1 Validation Script - Test all Day 1 components
Run this script to validate database setup, configuration, and basic functionality
"""
import sys
import os
from pathlib import Path
import pytest
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import current_config, DevelopmentConfig, TestingConfig
from database.db_setup import DatabaseManager, initialize_database, get_database_stats
from src.utils.logger import get_logger, setup_logger
from src.utils.validators import DataValidator, create_validation_report

logger = get_logger(__name__)

class Day1ValidationSuite:
    """Complete validation suite for Day 1 deliverables"""
    
    def __init__(self):
        self.test_results = {
            "config_tests": {},
            "database_tests": {},
            "logging_tests": {},
            "validation_tests": {},
            "overall_status": "PENDING"
        }
    
    def run_all_tests(self):
        """Run all Day 1 validation tests"""
        logger.info("Starting Day 1 validation suite...")
        
        try:
            # Test configuration
            self.test_configuration()
            
            # Test database setup
            self.test_database_setup()
            
            # Test logging system
            self.test_logging_system()
            
            # Test data validation
            self.test_data_validation()
            
            # Generate final report
            results = self.generate_final_report()
            return results
            
        except Exception as e:
            logger.error(f"Validation suite failed: {e}")
            self.test_results["overall_status"] = "FAILED"
            return self.test_results
    
    def test_configuration(self):
        """Test configuration management"""
        logger.info("Testing configuration...")
        
        config_tests = {}
        
        # Test config loading
        try:
            assert hasattr(current_config, 'DATABASE_URL'), "DATABASE_URL missing"
            assert hasattr(current_config, 'LOG_LEVEL'), "LOG_LEVEL missing"
            assert hasattr(current_config, 'DEBUG'), "DEBUG missing"
            config_tests["config_loading"] = "PASS"
        except Exception as e:
            config_tests["config_loading"] = f"FAIL: {e}"
        
        # Test directory creation
        try:
            from config import DATA_DIR, DATABASE_DIR, LOGS_DIR
            assert DATA_DIR.exists(), "Data directory not created"
            assert DATABASE_DIR.exists(), "Database directory not created"
            assert LOGS_DIR.exists(), "Logs directory not created"
            config_tests["directory_creation"] = "PASS"
        except Exception as e:
            config_tests["directory_creation"] = f"FAIL: {e}"
        
        # Test environment configurations
        try:
            dev_config = DevelopmentConfig()
            test_config = TestingConfig()
            assert dev_config.DEBUG == True, "Development config incorrect"
            assert test_config.DATABASE_URL != dev_config.DATABASE_URL, "Test DB should be different"
            config_tests["environment_configs"] = "PASS"
        except Exception as e:
            config_tests["environment_configs"] = f"FAIL: {e}"
        
        self.test_results["config_tests"] = config_tests
        logger.info(f"Configuration tests: {config_tests}")
    
    def test_database_setup(self):
        """Test database initialization and models"""
        logger.info("Testing database setup...")
        
        db_tests = {}
        
        # Test database manager creation
        try:
            db_manager = DatabaseManager(TestingConfig())
            assert db_manager is not None, "Database manager creation failed"
            db_tests["manager_creation"] = "PASS"
        except Exception as e:
            db_tests["manager_creation"] = f"FAIL: {e}"
            return
        
        # Test engine and session creation
        try:
            success = db_manager.create_engine_and_session()
            assert success, "Engine/session creation failed"
            assert db_manager.engine is not None, "Engine is None"
            assert db_manager.SessionLocal is not None, "SessionLocal is None"
            db_tests["engine_session"] = "PASS"
        except Exception as e:
            db_tests["engine_session"] = f"FAIL: {e}"
        
        # Test table creation
        try:
            success = db_manager.create_tables()
            assert success, "Table creation failed"
            db_tests["table_creation"] = "PASS"
        except Exception as e:
            db_tests["table_creation"] = f"FAIL: {e}"
        
        # Test connection
        try:
            success = db_manager.test_connection()
            assert success, "Connection test failed"
            db_tests["connection_test"] = "PASS"
        except Exception as e:
            db_tests["connection_test"] = f"FAIL: {e}"
        
        # Test table info
        try:
            table_info = db_manager.get_table_info()
            expected_tables = ['products', 'users', 'reviews', 'live_reviews', 
                             'aspect_sentiments', 'user_recommendations', 'system_metrics']
            
            for table in expected_tables:
                assert table in table_info, f"Table {table} not found"
            
            db_tests["table_verification"] = "PASS"
            db_tests["tables_found"] = list(table_info.keys())
        except Exception as e:
            db_tests["table_verification"] = f"FAIL: {e}"
        
        self.test_results["database_tests"] = db_tests
        logger.info(f"Database tests: {db_tests}")
    
    def test_logging_system(self):
        """Test logging configuration"""
        logger.info("Testing logging system...")
        
        logging_tests = {}
        
        # Test logger creation
        try:
            test_logger = setup_logger("test_logger")
            assert test_logger is not None, "Logger creation failed"
            logging_tests["logger_creation"] = "PASS"
        except Exception as e:
            logging_tests["logger_creation"] = f"FAIL: {e}"
        
        # Test log file creation
        try:
            test_logger.info("Test log message")
            log_file = current_config.LOG_FILE
            assert Path(log_file).exists(), "Log file not created"
            logging_tests["log_file_creation"] = "PASS"
        except Exception as e:
            logging_tests["log_file_creation"] = f"FAIL: {e}"
        
        # Test different log levels
        try:
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")
            logging_tests["log_levels"] = "PASS"
        except Exception as e:
            logging_tests["log_levels"] = f"FAIL: {e}"
        
        self.test_results["logging_tests"] = logging_tests
        logger.info(f"Logging tests: {logging_tests}")
    
    def test_data_validation(self):
        """Test data validation utilities"""
        logger.info("Testing data validation...")
        
        validation_tests = {}
        
        # Create sample data for testing
        sample_reviews = pd.DataFrame({
            'product_id': ['PROD001', 'PROD002', 'PROD003', '', 'PROD005'],
            'rating': [5, 4, 3, 2, 1],
            'review_text': ['Great product!', 'Good quality', '', 'Poor quality', 'Terrible'],
            'user_id': ['USER001', 'USER002', 'USER003', 'USER004', 'USER005'],
            'review_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        # Test review validation
        try:
            report = create_validation_report(sample_reviews, "review")
            assert report is not None, "Validation report creation failed"
            assert "total_rows" in report, "Missing total_rows in report"
            assert "valid_rows" in report, "Missing valid_rows in report"
            assert "errors" in report, "Missing errors in report"
            validation_tests["review_validation"] = "PASS"
            validation_tests["sample_validation_results"] = {
                "total_rows": report["total_rows"],
                "valid_rows": report["valid_rows"],
                "error_count": len(report["errors"]),
                "warning_count": len(report["warnings"])
            }
        except Exception as e:
            validation_tests["review_validation"] = f"FAIL: {e}"
        
        # Test data cleaning
        try:
            cleaned_data = DataValidator.clean_review_data(sample_reviews)
            assert len(cleaned_data) > 0, "Data cleaning removed all rows"
            assert 'review_text' in cleaned_data.columns, "Missing review_text after cleaning"
            validation_tests["data_cleaning"] = "PASS"
        except Exception as e:
            validation_tests["data_cleaning"] = f"FAIL: {e}"
        
        # Test text quality validation
        try:
            quality_result = DataValidator.validate_review_text_quality("This is a good product with excellent quality!")
            assert "valid" in quality_result, "Missing valid field in quality result"
            assert "quality_score" in quality_result, "Missing quality_score"
            validation_tests["text_quality"] = "PASS"
        except Exception as e:
            validation_tests["text_quality"] = f"FAIL: {e}"
        
        self.test_results["validation_tests"] = validation_tests
        logger.info(f"Validation tests: {validation_tests}")
    
    def generate_final_report(self):
        """Generate final validation report"""
        logger.info("Generating final validation report...")
        
        # Count total tests and passes
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if category == "overall_status":
                continue
            
            for test_name, result in tests.items():
                if test_name.endswith("_results") or test_name.endswith("_found"):
                    continue  # Skip metadata
                
                total_tests += 1
                if isinstance(result, str) and result == "PASS":
                    passed_tests += 1
        
        # Determine overall status
        if passed_tests == total_tests:
            self.test_results["overall_status"] = "ALL_PASS"
        elif passed_tests >= total_tests * 0.8:  # 80% pass rate
            self.test_results["overall_status"] = "MOSTLY_PASS"
        else:
            self.test_results["overall_status"] = "FAILED"
        
        # Add summary
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Validation complete: {passed_tests}/{total_tests} tests passed")
        return self.test_results

def run_day1_validation():
    """Main function to run Day 1 validation"""
    print("="*60)
    print("DAY 1 VALIDATION SUITE")
    print("="*60)
    
    validator = Day1ValidationSuite()
    results = validator.run_all_tests()
    
    # Check if results is valid
    if not results or "overall_status" not in results:
        print("\n❌ Validation failed to complete properly")
        return 1
    
    # Print results
    print(f"\nOVERALL STATUS: {results['overall_status']}")
    print(f"SUMMARY: {results['summary']}")
    
    print("\nDETAILED RESULTS:")
    for category, tests in results.items():
        if category in ["overall_status", "summary"]:
            continue
            
        print(f"\n{category.upper()}:")
        for test_name, result in tests.items():
            if isinstance(result, str):
                status = "✓" if result == "PASS" else "✗"
                print(f"  {status} {test_name}: {result}")
            elif isinstance(result, dict):
                print(f"  📊 {test_name}: {result}")
            else:
                print(f"  📋 {test_name}: {result}")
    
    # Return exit code for CI/CD
    return 0 if results["overall_status"] in ["ALL_PASS", "MOSTLY_PASS"] else 1

if __name__ == "__main__":
    exit_code = run_day1_validation()
    sys.exit(exit_code)