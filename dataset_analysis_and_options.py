#!/usr/bin/env python3
"""
Dataset Analysis and Migration Options
Analyzes current food dataset and provides options for Day 2
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter

def analyze_current_dataset():
    """Analyze the actual dataset we have"""
    print("🔍 ANALYZING CURRENT DATASET")
    print("=" * 50)
    
    db_path = "database/review_intelligence.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Sample reviews analysis
            reviews_sample = pd.read_sql("""
                SELECT product_id, review_text, rating 
                FROM reviews 
                LIMIT 100
            """, conn)
            
            # Product ID analysis
            product_ids = pd.read_sql("""
                SELECT DISTINCT product_id 
                FROM reviews 
                LIMIT 50
            """, conn)
            
            print("📊 PRODUCT ID PATTERNS:")
            amazon_asins = []
            numeric_ids = []
            
            for pid in product_ids['product_id']:
                if re.match(r'^B[0-9A-Z]{9}$', str(pid)):
                    amazon_asins.append(pid)
                elif str(pid).isdigit():
                    numeric_ids.append(pid)
            
            print(f"   Amazon ASINs: {len(amazon_asins)} (e.g., {amazon_asins[:3]})")
            print(f"   Numeric IDs: {len(numeric_ids)} (e.g., {numeric_ids[:3]})")
            
            # Content analysis
            print(f"\n🍯 REVIEW CONTENT ANALYSIS:")
            all_text = ' '.join(reviews_sample['review_text'].astype(str))
            
            # Food-related keywords
            food_keywords = {
                'taste': ['taste', 'flavor', 'delicious', 'yummy', 'sweet', 'sour', 'salty'],
                'food_types': ['coffee', 'tea', 'chocolate', 'syrup', 'cookies', 'snack', 'candy'],
                'food_quality': ['fresh', 'stale', 'expired', 'organic', 'natural'],
                'cooking': ['cook', 'bake', 'recipe', 'ingredient', 'kitchen'],
                'health': ['healthy', 'diet', 'nutrition', 'calories', 'fat', 'sugar']
            }
            
            keyword_counts = {}
            text_lower = all_text.lower()
            
            for category, keywords in food_keywords.items():
                count = sum(text_lower.count(keyword) for keyword in keywords)
                keyword_counts[category] = count
                
            print("   Food keyword frequency:")
            for category, count in keyword_counts.items():
                print(f"   - {category}: {count} mentions")
            
            # Electronics keywords for comparison
            electronics_keywords = ['battery', 'screen', 'device', 'electronic', 'tech', 'digital', 'wireless']
            electronics_count = sum(text_lower.count(keyword) for keyword in electronics_keywords)
            
            print(f"   Electronics keywords: {electronics_count} mentions")
            
            # Rating distribution
            rating_dist = pd.read_sql("""
                SELECT rating, COUNT(*) as count 
                FROM reviews 
                GROUP BY rating 
                ORDER BY rating
            """, conn)
            
            print(f"\n⭐ RATING DISTRIBUTION:")
            for _, row in rating_dist.iterrows():
                print(f"   {row['rating']} stars: {row['count']:,} reviews")
            
            # Sample review content
            print(f"\n📝 SAMPLE REVIEW CONTENT:")
            for i, text in enumerate(reviews_sample['review_text'].head(5)):
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"   {i+1}. {preview}")
                
            return {
                'domain': 'food',
                'amazon_asins': len(amazon_asins),
                'total_products': len(product_ids),
                'food_keyword_score': sum(keyword_counts.values()),
                'electronics_keyword_score': electronics_count,
                'rating_distribution': rating_dist.to_dict('records')
            }
                
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None

def get_electronics_dataset_options():
    """Research available electronics datasets"""
    print("\n🔌 ELECTRONICS DATASET OPTIONS")
    print("=" * 50)
    
    electronics_options = {
        'kaggle_electronics': {
            'name': 'Amazon Electronics Reviews',
            'kaggle_id': 'datafiniti/consumer-reviews-of-amazon-products',
            'size': '~100K reviews',
            'pros': ['Real Amazon electronics', 'Product metadata', 'Recent reviews'],
            'cons': ['Smaller than food dataset', 'May need cleaning'],
            'effort': 'Medium - New ETL pipeline run'
        },
        'synthetic_electronics': {
            'name': 'Generated Electronics Reviews', 
            'kaggle_id': 'Custom generation',
            'size': '20K reviews (configurable)',
            'pros': ['Perfectly clean', 'Controlled distribution', 'Custom features'],
            'cons': ['Not real data', 'Less realistic patterns'],
            'effort': 'Low - Modify synthetic generator'
        },
        'keep_food': {
            'name': 'Keep Current Food Dataset',
            'kaggle_id': 'snap/amazon-fine-food-reviews',
            'size': '19,997 reviews (current)',
            'pros': ['Data already processed', 'No rework needed', 'Rich domain'],
            'cons': ['Documentation inconsistency', 'Different from original plan'],
            'effort': 'Minimal - Fix documentation only'
        }
    }
    
    print("Available options:")
    for key, option in electronics_options.items():
        print(f"\n📦 {option['name']}")
        print(f"   Dataset: {option['kaggle_id']}")
        print(f"   Size: {option['size']}")
        print(f"   Effort: {option['effort']}")
        print(f"   Pros: {', '.join(option['pros'])}")
        print(f"   Cons: {', '.join(option['cons'])}")
    
    return electronics_options

def migration_impact_analysis():
    """Analyze impact of each option on Day 2"""
    print(f"\n🎯 DAY 2 ML IMPACT ANALYSIS")
    print("=" * 50)
    
    impacts = {
        'sentiment_analysis': {
            'food': 'Taste, freshness, delivery sentiment patterns',
            'electronics': 'Performance, durability, feature sentiment patterns'
        },
        'fake_detection': {
            'food': 'Health claims, taste subjectivity, dietary spam',
            'electronics': 'Technical jargon, fake specs, bot reviews'
        },
        'recommendations': {
            'food': 'Taste preferences, dietary needs, brand trust',
            'electronics': 'Feature requirements, price points, compatibility'
        },
        'business_value': {
            'food': 'Food e-commerce, grocery platforms, recipe sites',
            'electronics': 'Tech retail, product comparison, tech blogs'
        }
    }
    
    for aspect, domains in impacts.items():
        print(f"\n📊 {aspect.upper()}:")
        for domain, description in domains.items():
            print(f"   {domain.title()}: {description}")

def recommend_best_option():
    """Provide recommendation based on analysis"""
    print(f"\n💡 RECOMMENDATION")
    print("=" * 50)
    
    print("🎯 BEST OPTION: **KEEP FOOD DATASET**")
    print()
    print("✅ REASONS:")
    print("   1. Data is already processed and validated (19,997 reviews)")
    print("   2. Rich, diverse domain with interesting ML challenges")
    print("   3. No time lost re-running ETL pipeline")
    print("   4. Food review intelligence has strong business value")
    print("   5. Sentiment patterns in food are actually more complex")
    print()
    print("🔧 REQUIRED FIXES:")
    print("   1. Update documentation (5 minutes)")
    print("   2. Fix hardcoded 'Electronics' category")
    print("   3. Adjust ML model features for food domain")
    print()
    print("🚀 DAY 2 ADVANTAGE:")
    print("   - Food reviews have richer sentiment vocabulary")
    print("   - More diverse fake review patterns") 
    print("   - Stronger business case (food e-commerce)")
    print("   - Unique differentiator in portfolio")

def create_quick_fix_script():
    """Create script to fix the dataset categorization"""
    print(f"\n🔧 QUICK FIX IMPLEMENTATION")
    print("=" * 50)
    
    fix_script = '''
# Quick fix script to correct dataset categorization
import sqlite3

def fix_dataset_categorization():
    """Fix the incorrect Electronics categorization"""
    
    db_path = "database/review_intelligence.db"
    
    with sqlite3.connect(db_path) as conn:
        # Update products table
        conn.execute("""
            UPDATE products 
            SET category = 'Food & Beverage' 
            WHERE category = 'Electronics'
        """)
        
        print("✅ Updated products table category")
        
        # Verify the change
        result = conn.execute("""
            SELECT category, COUNT(*) as count 
            FROM products 
            GROUP BY category
        """).fetchall()
        
        print("📊 Category distribution:")
        for category, count in result:
            print(f"   - {category}: {count} products")

if __name__ == "__main__":
    fix_dataset_categorization()
'''
    
    with open("fix_dataset_category.py", "w") as f:
        f.write(fix_script)
    
    print("📁 Created: fix_dataset_category.py")
    print("🔄 Run: python fix_dataset_category.py")

def main():
    """Main analysis function"""
    print("🔍 DATASET DISCREPANCY ANALYSIS")
    print("=" * 60)
    
    # Analyze current dataset
    analysis = analyze_current_dataset()
    
    if analysis:
        food_score = analysis['food_keyword_score']
        electronics_score = analysis['electronics_keyword_score']
        
        print(f"\n🎯 DOMAIN CONFIDENCE:")
        print(f"   Food domain score: {food_score}")
        print(f"   Electronics domain score: {electronics_score}")
        
        if food_score > electronics_score * 3:
            print("   ✅ CONFIRMED: This is a FOOD REVIEW dataset")
        else:
            print("   ⚠️  Mixed or unclear domain")
    
    # Show options
    get_electronics_dataset_options()
    
    # Impact analysis  
    migration_impact_analysis()
    
    # Recommendation
    recommend_best_option()
    
    # Quick fix
    create_quick_fix_script()
    
    print(f"\n" + "=" * 60)
    print("🎯 DECISION NEEDED:")
    print("1. Keep food dataset (RECOMMENDED)")
    print("2. Switch to electronics dataset") 
    print("3. Generate synthetic electronics data")
    print("=" * 60)

if __name__ == "__main__":
    main()