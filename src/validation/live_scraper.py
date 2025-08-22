"""
Live Scraper for Fresh Amazon Food Reviews - Day 3 Validation
Scrapes current Amazon food product reviews to validate against 2012 historical data
Specialized for Amazon Fine Food Reviews dataset validation
"""

import os
import sys
import sqlite3
import requests
import time
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import re

# Web scraping imports
try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    print("Installing required packages...")
    os.system("pip install beautifulsoup4 requests lxml")
    from bs4 import BeautifulSoup
    import requests

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from database.models import get_or_create_user, get_or_create_product
    from config import current_config
except ImportError:
    print("Warning: Could not import database models or config")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedReview:
    """Data structure for scraped review"""
    product_id: str
    user_id: str
    rating: int
    review_text: str
    review_title: str
    helpful_votes: int
    total_votes: int
    verified_purchase: bool
    review_date: datetime
    source: str = "amazon_live_scraper"

@dataclass
class ScrapingSession:
    """Data structure for scraping session statistics"""
    session_id: str
    start_time: datetime
    products_attempted: int
    products_successful: int
    reviews_scraped: int
    errors_encountered: int
    avg_delay: float
    status: str

class AmazonFoodScraper:
    """
    Advanced Amazon food product scraper for live validation
    Targets existing ASINs from historical food review dataset
    """
    
    def __init__(self, db_path: str = "./database/review_intelligence.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session_stats = None
        
        # Setup request headers for Amazon scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        self.session.headers.update(self.headers)
        
        # Rate limiting settings
        self.base_delay = 2.0  # Base delay between requests
        self.random_delay_range = (1.0, 4.0)  # Additional random delay
        self.request_count = 0
        self.start_time = time.time()
        
        # Amazon domain settings
        self.amazon_domains = [
            'amazon.com',
            'www.amazon.com'
        ]
        
        logger.info("SUCCESS: AmazonFoodScraper initialized for live validation")
    
    def get_existing_food_products(self, limit: int = 50, min_reviews: int = 5) -> List[Dict]:
        """
        Get existing food product ASINs from historical dataset
        Prioritizes products with more reviews for better validation
        """
        logger.info(f"FETCHING: Existing food product ASINs (limit: {limit}, min_reviews: {min_reviews})")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get food products with sufficient historical reviews
            query = """
                SELECT product_id, COUNT(*) as review_count, AVG(rating) as avg_rating,
                       MAX(date) as last_review_date
                FROM reviews 
                WHERE product_id IS NOT NULL 
                GROUP BY product_id
                HAVING COUNT(*) >= ?
                ORDER BY COUNT(*) DESC, AVG(rating) DESC
                LIMIT ?
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (min_reviews, limit))
            results = cursor.fetchall()
            
            products = []
            for row in results:
                products.append({
                    'product_id': row[0],
                    'historical_review_count': row[1],
                    'historical_avg_rating': round(row[2], 2),
                    'last_review_date': row[3]
                })
            
            conn.close()
            
            logger.info(f"SUCCESS: Found {len(products)} food products for live validation")
            return products
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get existing food products: {e}")
            return []
    
    def build_amazon_review_url(self, product_id: str, page: int = 1) -> str:
        """
        Build Amazon review URL for a specific product ASIN
        """
        # Amazon review URL pattern
        base_url = f"https://www.amazon.com/product-reviews/{product_id}"
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'sortBy': 'recent',
            'pageNumber': page
        }
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{param_string}"
    
    def extract_reviews_from_page(self, html: str, product_id: str) -> List[ScrapedReview]:
        """
        Extract reviews from Amazon product review page HTML
        """
        reviews = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find review containers (Amazon's current structure)
            review_containers = soup.find_all('div', {'data-hook': 'review'})
            
            if not review_containers:
                # Fallback: try alternative selectors
                review_containers = soup.find_all('div', class_=lambda x: x and 'review' in x.lower())
            
            logger.info(f"PARSING: Found {len(review_containers)} review containers for {product_id}")
            
            for container in review_containers:
                try:
                    review = self._parse_single_review(container, product_id)
                    if review:
                        reviews.append(review)
                except Exception as e:
                    logger.warning(f"WARNING: Failed to parse individual review: {e}")
                    continue
            
            logger.info(f"SUCCESS: Extracted {len(reviews)} valid reviews for {product_id}")
            return reviews
            
        except Exception as e:
            logger.error(f"ERROR: Failed to extract reviews from page: {e}")
            return []
    
    def _parse_single_review(self, container, product_id: str) -> Optional[ScrapedReview]:
        """
        Parse a single review from its HTML container
        """
        try:
            # Extract user ID/name
            user_element = container.find('span', class_='a-profile-name') or container.find('a', class_='a-profile')
            user_id = user_element.get_text(strip=True) if user_element else f"user_{random.randint(10000, 99999)}"
            
            # Clean user ID
            user_id = re.sub(r'[^\w\-_]', '_', user_id)[:50]
            
            # Extract rating
            rating_element = container.find('i', class_=lambda x: x and 'a-icon-star' in x)
            rating = 3  # Default
            if rating_element:
                rating_text = rating_element.get('class', [])
                for cls in rating_text:
                    if 'a-star-' in cls:
                        try:
                            rating = int(cls.split('-')[-1])
                            break
                        except:
                            continue
            
            # Extract review title
            title_element = container.find('a', {'data-hook': 'review-title'}) or container.find('span', {'data-hook': 'review-title'})
            review_title = title_element.get_text(strip=True) if title_element else "No Title"
            
            # Clean title
            review_title = review_title.replace('5.0 out of 5 stars', '').replace('4.0 out of 5 stars', '').replace('3.0 out of 5 stars', '').replace('2.0 out of 5 stars', '').replace('1.0 out of 5 stars', '').strip()
            
            # Extract review text
            text_element = container.find('span', {'data-hook': 'review-body'})
            review_text = ""
            if text_element:
                # Get all text, clean it
                review_text = text_element.get_text(strip=True)
                # Remove "Read more" links and other artifacts
                review_text = re.sub(r'Read more.*$', '', review_text, flags=re.IGNORECASE)
                review_text = review_text.strip()
            
            # Skip if no meaningful text
            if len(review_text) < 10:
                return None
            
            # Extract helpful votes
            helpful_element = container.find('span', {'data-hook': 'helpful-vote-statement'})
            helpful_votes = 0
            total_votes = 0
            
            if helpful_element:
                helpful_text = helpful_element.get_text(strip=True)
                # Parse "X people found this helpful" or "One person found this helpful"
                if 'people found this helpful' in helpful_text:
                    try:
                        helpful_votes = int(re.search(r'(\d+)', helpful_text).group(1))
                        total_votes = helpful_votes  # Simplified
                    except:
                        pass
                elif 'One person found this helpful' in helpful_text:
                    helpful_votes = 1
                    total_votes = 1
            
            # Extract date
            date_element = container.find('span', {'data-hook': 'review-date'})
            review_date = datetime.now()  # Default to now
            
            if date_element:
                date_text = date_element.get_text(strip=True)
                # Parse "Reviewed in the United States on January 15, 2025"
                try:
                    # Extract date part after "on "
                    if ' on ' in date_text:
                        date_part = date_text.split(' on ')[-1]
                        review_date = datetime.strptime(date_part, '%B %d, %Y')
                except:
                    pass
            
            # Check for verified purchase
            verified_element = container.find('span', {'data-hook': 'avp-badge'})
            verified_purchase = verified_element is not None
            
            return ScrapedReview(
                product_id=product_id,
                user_id=user_id,
                rating=rating,
                review_text=review_text,
                review_title=review_title,
                helpful_votes=helpful_votes,
                total_votes=total_votes,
                verified_purchase=verified_purchase,
                review_date=review_date
            )
            
        except Exception as e:
            logger.warning(f"WARNING: Failed to parse single review: {e}")
            return None
    
    def scrape_product_reviews(self, product_id: str, max_pages: int = 3, max_reviews: int = 50) -> List[ScrapedReview]:
        """
        Scrape reviews for a specific Amazon food product
        """
        logger.info(f"SCRAPING: Reviews for food product {product_id} (max_pages: {max_pages})")
        
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            try:
                # Rate limiting
                self._apply_rate_limiting()
                
                # Build URL
                url = self.build_amazon_review_url(product_id, page)
                logger.info(f"REQUESTING: {url}")
                
                # Make request
                response = self.session.get(url, timeout=15)
                self.request_count += 1
                
                if response.status_code == 200:
                    # Extract reviews from page
                    page_reviews = self.extract_reviews_from_page(response.text, product_id)
                    all_reviews.extend(page_reviews)
                    
                    logger.info(f"SUCCESS: Page {page} - {len(page_reviews)} reviews extracted")
                    
                    # Check if we have enough reviews
                    if len(all_reviews) >= max_reviews:
                        logger.info(f"COMPLETE: Reached max reviews limit ({max_reviews})")
                        break
                    
                    # Check if this was the last page (no reviews found)
                    if len(page_reviews) == 0:
                        logger.info(f"COMPLETE: No more reviews found on page {page}")
                        break
                        
                elif response.status_code == 503:
                    logger.warning(f"WARNING: Rate limited on page {page}, increasing delay")
                    time.sleep(10)  # Back off
                    continue
                    
                else:
                    logger.warning(f"WARNING: Page {page} returned status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"ERROR: Failed to scrape page {page} for {product_id}: {e}")
                continue
        
        logger.info(f"COMPLETE: Scraped {len(all_reviews)} total reviews for {product_id}")
        return all_reviews[:max_reviews]  # Ensure we don't exceed limit
    
    def _apply_rate_limiting(self):
        """
        Apply intelligent rate limiting to avoid detection
        """
        # Base delay
        time.sleep(self.base_delay)
        
        # Random additional delay
        additional_delay = random.uniform(*self.random_delay_range)
        time.sleep(additional_delay)
        
        # Progressive delay if making many requests
        if self.request_count > 20:
            progressive_delay = min(self.request_count * 0.1, 5.0)
            time.sleep(progressive_delay)
        
        # Rotate user agent occasionally
        if self.request_count % 10 == 0:
            self._rotate_user_agent()
    
    def _rotate_user_agent(self):
        """
        Rotate user agent to avoid detection
        """
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        
        new_user_agent = random.choice(user_agents)
        self.session.headers.update({'User-Agent': new_user_agent})
        logger.info(f"ROTATED: User agent updated")
    
    def save_scraped_reviews(self, reviews: List[ScrapedReview]) -> int:
        """
        Save scraped reviews to database
        """
        if not reviews:
            logger.warning("WARNING: No reviews to save")
            return 0
        
        logger.info(f"SAVING: {len(reviews)} scraped food reviews to database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            
            for review in reviews:
                try:
                    # Insert review into live_reviews table
                    cursor.execute("""
                        INSERT OR IGNORE INTO live_reviews 
                        (product_id, user_id, rating, review_text, review_title,
                         helpful_votes, total_votes, verified_purchase, date, 
                         source, scrape_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        review.product_id,
                        review.user_id,
                        review.rating,
                        review.review_text,
                        review.review_title,
                        review.helpful_votes,
                        review.total_votes,
                        review.verified_purchase,
                        review.review_date.isoformat(),
                        review.source,
                        datetime.now().isoformat()
                    ))
                    
                    if cursor.rowcount > 0:
                        saved_count += 1
                        
                except Exception as e:
                    logger.warning(f"WARNING: Failed to save individual review: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"SUCCESS: Saved {saved_count}/{len(reviews)} food reviews to database")
            return saved_count
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save scraped reviews: {e}")
            return 0
    
    def run_live_validation_session(self, max_products: int = 10, reviews_per_product: int = 20) -> ScrapingSession:
        """
        Run a complete live validation session for Amazon food products
        """
        session_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"STARTING: Live validation session {session_id}")
        logger.info(f"TARGET: {max_products} food products, {reviews_per_product} reviews each")
        
        # Initialize session stats
        self.session_stats = ScrapingSession(
            session_id=session_id,
            start_time=start_time,
            products_attempted=0,
            products_successful=0,
            reviews_scraped=0,
            errors_encountered=0,
            avg_delay=0.0,
            status="running"
        )
        
        try:
            # Get existing food products to validate
            products = self.get_existing_food_products(limit=max_products, min_reviews=5)
            
            if not products:
                logger.error("ERROR: No food products found for validation")
                self.session_stats.status = "failed"
                return self.session_stats
            
            logger.info(f"VALIDATION: Processing {len(products)} food products...")
            
            total_reviews_scraped = 0
            
            for i, product in enumerate(products, 1):
                product_id = product['product_id']
                
                logger.info(f"PRODUCT {i}/{len(products)}: {product_id} (historical: {product['historical_review_count']} reviews)")
                
                self.session_stats.products_attempted += 1
                
                try:
                    # Scrape reviews for this product
                    reviews = self.scrape_product_reviews(
                        product_id, 
                        max_pages=3, 
                        max_reviews=reviews_per_product
                    )
                    
                    if reviews:
                        # Save reviews
                        saved_count = self.save_scraped_reviews(reviews)
                        total_reviews_scraped += saved_count
                        
                        if saved_count > 0:
                            self.session_stats.products_successful += 1
                        
                        logger.info(f"SUCCESS: Product {product_id} - {saved_count} new reviews saved")
                    else:
                        logger.warning(f"WARNING: No reviews found for product {product_id}")
                        
                except Exception as e:
                    logger.error(f"ERROR: Failed to process product {product_id}: {e}")
                    self.session_stats.errors_encountered += 1
                    continue
                
                # Progress logging
                if i % 5 == 0 or i == len(products):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = total_reviews_scraped / elapsed if elapsed > 0 else 0
                    logger.info(f"PROGRESS: {i}/{len(products)} products, {total_reviews_scraped} reviews, {rate:.2f} reviews/sec")
            
            # Update final session stats
            self.session_stats.reviews_scraped = total_reviews_scraped
            self.session_stats.status = "completed"
            
            # Calculate average delay
            total_time = (datetime.now() - start_time).total_seconds()
            self.session_stats.avg_delay = total_time / max(self.request_count, 1)
            
            # Save session stats
            self._save_session_stats()
            
            logger.info(f"COMPLETE: Validation session finished")
            logger.info(f"SUMMARY: {self.session_stats.products_successful}/{self.session_stats.products_attempted} products successful")
            logger.info(f"SUMMARY: {self.session_stats.reviews_scraped} total reviews scraped")
            
            return self.session_stats
            
        except Exception as e:
            logger.error(f"ERROR: Validation session failed: {e}")
            self.session_stats.status = "failed"
            self.session_stats.errors_encountered += 1
            return self.session_stats
    
    def _save_session_stats(self):
        """
        Save session statistics to database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create scraping_sessions table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scraping_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    start_time TEXT,
                    end_time TEXT,
                    products_attempted INTEGER,
                    products_successful INTEGER,
                    reviews_scraped INTEGER,
                    errors_encountered INTEGER,
                    avg_delay REAL,
                    status TEXT,
                    created_at TEXT
                )
            """)
            
            # Insert session stats
            conn.execute("""
                INSERT OR REPLACE INTO scraping_sessions 
                (session_id, start_time, end_time, products_attempted, products_successful,
                 reviews_scraped, errors_encountered, avg_delay, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_stats.session_id,
                self.session_stats.start_time.isoformat(),
                datetime.now().isoformat(),
                self.session_stats.products_attempted,
                self.session_stats.products_successful,
                self.session_stats.reviews_scraped,
                self.session_stats.errors_encountered,
                self.session_stats.avg_delay,
                self.session_stats.status,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("SUCCESS: Session statistics saved to database")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to save session stats: {e}")

def main():
    """
    Main function for testing live scraper
    """
    print("TESTING: Live Amazon Food Product Scraper - Day 3 Validation")
    print("=" * 60)
    
    # Initialize scraper
    scraper = AmazonFoodScraper()
    
    # Test getting existing products
    print("\nTESTING: Getting existing food products...")
    products = scraper.get_existing_food_products(limit=5)
    
    if products:
        print(f"SUCCESS: Found {len(products)} food products for validation")
        for product in products[:3]:
            print(f"  - {product['product_id']}: {product['historical_review_count']} historical reviews")
    else:
        print("WARNING: No food products found")
        return
    
    # Test scraping a single product
    test_product_id = products[0]['product_id']
    print(f"\nTESTING: Scraping reviews for {test_product_id}...")
    
    reviews = scraper.scrape_product_reviews(test_product_id, max_pages=1, max_reviews=5)
    
    if reviews:
        print(f"SUCCESS: Scraped {len(reviews)} reviews")
        
        # Show sample review
        sample_review = reviews[0]
        print(f"\nSAMPLE REVIEW:")
        print(f"  User: {sample_review.user_id}")
        print(f"  Rating: {sample_review.rating}/5")
        print(f"  Title: {sample_review.review_title}")
        print(f"  Text: {sample_review.review_text[:100]}...")
        print(f"  Date: {sample_review.review_date}")
        print(f"  Verified: {sample_review.verified_purchase}")
        
        # Save reviews
        saved_count = scraper.save_scraped_reviews(reviews)
        print(f"SUCCESS: Saved {saved_count} reviews to database")
    else:
        print("WARNING: No reviews scraped")
    
    print(f"\nCOMPLETE: Live scraper test finished!")
    print(f"NEXT: Run full validation session with: scraper.run_live_validation_session()")

if __name__ == "__main__":
    main()