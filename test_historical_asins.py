"""
Quick Test Script: Find Active Historical Food Product ASINs
Tests multiple historical ASINs to find which ones are still available on Amazon
Run this from the root project directory
"""

import sys
import time
from src.validation.live_scraper import AmazonFoodScraper

def test_historical_asins():
    """
    Test multiple historical ASINs to find active ones
    """
    print("🔍 TESTING: Historical Amazon Food Product ASINs")
    print("=" * 60)
    
    # Initialize scraper
    scraper = AmazonFoodScraper()
    
    # Get more historical products
    print("📊 FETCHING: Historical food products from database...")
    products = scraper.get_existing_food_products(limit=25, min_reviews=3)
    
    if not products:
        print("❌ ERROR: No historical products found!")
        return
    
    print(f"✅ FOUND: {len(products)} historical food products to test")
    print()
    
    active_products = []
    tested_count = 0
    
    for i, product in enumerate(products, 1):
        product_id = product['product_id']
        historical_reviews = product['historical_review_count']
        historical_rating = product['historical_avg_rating']
        
        print(f"🧪 TESTING {i:2d}/{len(products)}: {product_id}")
        print(f"   📈 Historical: {historical_reviews} reviews, {historical_rating}⭐")
        
        try:
            # Test with just 1 page, max 3 reviews for speed
            reviews = scraper.scrape_product_reviews(
                product_id, 
                max_pages=1, 
                max_reviews=3
            )
            
            tested_count += 1
            
            if reviews:
                print(f"   ✅ ACTIVE: Found {len(reviews)} current reviews!")
                
                # Show sample review
                sample = reviews[0]
                print(f"      👤 User: {sample.user_id}")
                print(f"      ⭐ Rating: {sample.rating}/5")
                print(f"      📅 Date: {sample.review_date.strftime('%Y-%m-%d')}")
                print(f"      💬 Preview: {sample.review_text[:80]}...")
                
                active_products.append({
                    'product_id': product_id,
                    'historical_reviews': historical_reviews,
                    'current_reviews_found': len(reviews),
                    'sample_review': sample
                })
                
                # Save these successful reviews
                saved = scraper.save_scraped_reviews(reviews)
                print(f"      💾 SAVED: {saved} reviews to database")
                
            else:
                print(f"   ❌ INACTIVE: Product not available (404)")
            
        except Exception as e:
            print(f"   ⚠️  ERROR: {str(e)[:50]}...")
        
        print()
        
        # Stop if we found 3 active products (enough for validation)
        if len(active_products) >= 3:
            print("🎉 SUCCESS: Found enough active products for validation!")
            break
        
        # Don't test too many at once
        if tested_count >= 15:
            print("⏰ LIMIT: Tested 15 products, stopping to avoid rate limits")
            break
    
    # Summary
    print("=" * 60)
    print("📋 SUMMARY RESULTS:")
    print(f"🧪 Products Tested: {tested_count}")
    print(f"✅ Active Products Found: {len(active_products)}")
    print(f"❌ Inactive Products: {tested_count - len(active_products)}")
    
    if active_products:
        print(f"\n🎯 ACTIVE PRODUCTS FOR VALIDATION:")
        for i, product in enumerate(active_products, 1):
            print(f"  {i}. {product['product_id']}")
            print(f"     📊 Historical: {product['historical_reviews']} reviews")
            print(f"     🆕 Current: {product['current_reviews_found']} reviews scraped")
            print(f"     📅 Latest Review: {product['sample_review'].review_date.strftime('%Y-%m-%d')}")
            print()
        
        print("🚀 RECOMMENDATION: Run full validation on these active products!")
        print("   Use: scraper.run_live_validation_session(max_products=3)")
        
        return active_products
        
    else:
        print("😞 NO ACTIVE PRODUCTS FOUND")
        print("🔄 NEXT STEPS:")
        print("   1. Try testing more products (increase limit)")
        print("   2. Switch to scraping current popular food products")
        print("   3. Use existing data for drift analysis")
        
        return []

def run_validation_on_active_products(active_products):
    """
    Run full validation session on the active products we found
    """
    if not active_products:
        print("❌ No active products to validate")
        return
    
    print("🚀 STARTING: Full validation session on active products")
    print("=" * 60)
    
    scraper = AmazonFoodScraper()
    
    # Extract just the product IDs
    active_asins = [p['product_id'] for p in active_products]
    
    print(f"🎯 TARGETING: {len(active_asins)} active food products")
    for asin in active_asins:
        print(f"   - {asin}")
    
    # Run validation session
    session = scraper.run_live_validation_session(
        max_products=len(active_asins), 
        reviews_per_product=20
    )
    
    print("\n📊 VALIDATION SESSION RESULTS:")
    print(f"✅ Products Successful: {session.products_successful}/{session.products_attempted}")
    print(f"📝 Reviews Scraped: {session.reviews_scraped}")
    print(f"⚠️  Errors: {session.errors_encountered}")
    print(f"📈 Status: {session.status}")
    
    return session

if __name__ == "__main__":
    print("🍎 AMAZON FOOD PRODUCT VALIDATION TEST")
    print("Testing historical ASINs from 2012 dataset...")
    print()
    
    # Phase 1: Find active products
    active_products = test_historical_asins()
    
    # Phase 2: Ask user if they want full validation
    if active_products:
        print("\n" + "="*60)
        response = input("🤔 Run full validation on active products? (y/n): ").lower()
        
        if response == 'y':
            session = run_validation_on_active_products(active_products)
            print(f"\n🎉 COMPLETE: Validation finished!")
            print(f"Check your database for {session.reviews_scraped} new live reviews!")
        else:
            print("👍 Skipped full validation. You can run it later.")
    
    print("\n✨ Test complete! Ready for Day 3 File 2: Model Validator")