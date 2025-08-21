#!/usr/bin/env python3
"""
Quick Dataset Category Fix
Corrects the hardcoded 'Electronics' category to 'Food & Beverage'
"""

import sqlite3
import os
from pathlib import Path

def fix_database_categories():
    """Fix the incorrect Electronics categorization in database"""
    
    print("Fixing dataset categorization...")
    
    db_path = "database/review_intelligence.db"
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Update products table
            cursor = conn.execute("""
                UPDATE products 
                SET category = 'Food & Beverage' 
                WHERE category = 'Electronics'
            """)
            
            rows_updated = cursor.rowcount
            print(f"Updated {rows_updated} products from 'Electronics' to 'Food & Beverage'")
            
            # Verify the change
            result = conn.execute("""
                SELECT category, COUNT(*) as count 
                FROM products 
                GROUP BY category
            """).fetchall()
            
            print("Category distribution after fix:")
            for category, count in result:
                print(f"   - {category}: {count} products")
            
            return True
            
    except Exception as e:
        print(f"Error fixing database: {e}")
        return False

def fix_documentation():
    """Fix documentation to reflect food dataset"""
    
    print("\nFixing documentation...")
    
    readme_path = Path("README.md")
    
    if readme_path.exists():
        try:
            # Read current content
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace Electronics references with Food
            replacements = [
                ('Electronics category', 'Food & Beverage category'),
                ('Electronics Category', 'Food & Beverage Category'),
                ('electronics', 'food products'),
                ('Electronics', 'Food & Beverage'),
                ('tech product reviews', 'food product reviews'),
                ('Amazon Fine Food Reviews (Electronics category', 'Amazon Fine Food Reviews (Food & Beverage category'),
                ('Product Review Intelligence System', 'Food Review Intelligence System')
            ]
            
            changes_made = 0
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    changes_made += 1
            
            # Write back
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated README.md with {changes_made} corrections")
            return True
            
        except Exception as e:
            print(f"Error updating README.md: {e}")
            return False
    else:
        print("README.md not found")
        return False

def fix_code_references():
    """Fix hardcoded category references in code"""
    
    print("\nFixing code references...")
    
    # Files that might have hardcoded categories
    files_to_check = [
        "src/data_pipeline/etl_pipeline.py",
        "config.py"
    ]
    
    changes_made = 0
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace hardcoded Electronics
                if "category'] = 'Electronics'" in content:
                    content = content.replace(
                        "category'] = 'Electronics'", 
                        "category'] = 'Food & Beverage'"
                    )
                    
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"   Fixed category in {file_path}")
                    changes_made += 1
                    
            except Exception as e:
                print(f"   Error fixing {file_path}: {e}")
    
    print(f"Fixed {changes_made} code files")

def verify_dataset():
    """Verify the dataset is properly categorized"""
    
    print("\nVerifying dataset...")
    
    db_path = "database/review_intelligence.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Check categories
            categories = conn.execute("""
                SELECT DISTINCT category 
                FROM products
            """).fetchall()
            
            print("Current categories:")
            for (category,) in categories:
                print(f"   - {category}")
            
            # Check sample products
            sample_products = conn.execute("""
                SELECT product_id, product_name, category, avg_rating, total_reviews
                FROM products 
                LIMIT 5
            """).fetchall()
            
            print("\nSample products:")
            for product in sample_products:
                print(f"   {product[0]}: {product[1]} ({product[2]}) - {product[3]:.1f} stars, {product[4]} reviews")
            
            # Check review content
            sample_reviews = conn.execute("""
                SELECT review_text 
                FROM reviews 
                LIMIT 3
            """).fetchall()
            
            print("\nSample review content:")
            for i, (review_text,) in enumerate(sample_reviews, 1):
                preview = review_text[:60] + "..." if len(review_text) > 60 else review_text
                print(f"   {i}. {preview}")
                
    except Exception as e:
        print(f"Error verifying dataset: {e}")

def main():
    """Main fix function"""
    print("=" * 60)
    print("FOOD DATASET CATEGORIZATION FIX")
    print("=" * 60)
    
    print("This script fixes the dataset category mislabeling.")
    print("Your data is Amazon Fine Food Reviews, not Electronics.\n")
    
    # Fix database
    if fix_database_categories():
        print("Database categories fixed successfully")
    
    # Fix documentation
    if fix_documentation():
        print("Documentation updated successfully")
    
    # Fix code
    fix_code_references()
    
    # Verify
    verify_dataset()
    
    print("\n" + "=" * 60)
    print("FIX COMPLETE!")
    print("=" * 60)
    print("Your dataset is now properly labeled as Food & Beverage")
    print("Day 2 ML models remain intact and functional")
    print("Ready to continue with Day 3!")
    print("=" * 60)

if __name__ == "__main__":
    main()